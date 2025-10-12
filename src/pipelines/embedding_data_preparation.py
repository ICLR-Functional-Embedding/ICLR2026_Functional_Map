import numpy as np
import random
import torch
from torch.utils.data import Dataset


def robust_std(x):
    """
    Compute a robust estimate of standard deviation using the median absolute deviation (MAD).
    
    Args:
        x (array-like): Input data.
    
    Returns:
        float: Robust estimate of standard deviation.
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

def segment_data(data, segment_length, shuffle=True, threshold=10, min_segments_for_filtering=100):
    """
    Segment time-series data into fixed-length chunks and remove segments with outliers.
    
    Args:
        data (array-like): Input 1D time-series data.
        segment_length (int): Length of each segment(in samples).
        shuffle (bool): Whether to shuffle segments after filtering.
        threshold (float): Outlier rejection threshold in MAD units.
    
    Returns:
        np.ndarray: Filtered and optionally shuffled segments.
    """
    num_segments = len(data) // segment_length
    if num_segments == 0:
        print("⚠️ Not enough data to form even one segment.")
        return np.array([])

    segments = np.array(np.split(data[:num_segments * segment_length], num_segments))

    if num_segments < min_segments_for_filtering:
        print(f"⚠️ Only {num_segments} segments available. Skipping noise filtering.")
        clean_segments = segments
    else:
        # Compute robust threshold
        median_val = np.median(data)
        std_est = robust_std(data)
        upper_thresh = median_val + threshold * std_est
        lower_thresh = median_val - threshold * std_est

        # Vectorized filtering: find segments where all values are within thresholds
        mask = np.all((segments >= lower_thresh) & (segments <= upper_thresh), axis=1)
        clean_segments = segments[mask]
        print(f"Segments before cleaning: {len(segments)} | After cleaning: {len(clean_segments)}")

    if shuffle:
        np.random.shuffle(clean_segments)

    return clean_segments


def create_balanced_pairs(region_segments_dict, max_dataset_size=20000, seed=42):
    """
    Create a balanced dataset of similar and dissimilar pairs for contrastive training.
    
    Args:
        region_segments_dict (dict): Keys are region names, values are np.arrays of segments.
        max_dataset_size (int): Max total number of training pairs.
        seed (int): Random seed for reproducibility.
    
    Returns:
        (list of tuple, list of int): List of (segment1, segment2) pairs and corresponding binary labels (0 = similar, 1 = dissimilar).
    """
    random.seed(seed)
    np.random.seed(seed)
    region_names = list(region_segments_dict.keys())
    num_regions = len(region_names)

    # ---- SIMILAR PAIRS ----
    similar_pairs = []
    per_region_quota = max_dataset_size // 2 // num_regions

    for region in region_names:
        segments = region_segments_dict[region]
        indices = list(range(len(segments)))
        random.shuffle(indices)  # Avoid sorted bias

        # Sample random non-repeating index pairs (like combinations but shuffled)
        seen_pairs = set()
        max_unique_pairs = len(indices) * (len(indices) - 1) // 2
        while len(similar_pairs) < per_region_quota * (region_names.index(region) + 1):
            i, j = random.sample(indices, 2)
            if (i, j) not in seen_pairs and (j, i) not in seen_pairs:
                similar_pairs.append((segments[i], segments[j]))
                seen_pairs.add((i, j))
            if len(seen_pairs) >= len(indices) * (len(indices) - 1) // 2 or len(seen_pairs) >= max_unique_pairs:
                break

    similar_labels = [0] * len(similar_pairs)

    # ---- DISSIMILAR PAIRS ----
    dissimilar_pairs = []
    seen_dissimilar = set()
    per_region_combo_quota = max_dataset_size // 2 // (num_regions * (num_regions - 1) // 2)

    for i, region_a in enumerate(region_names):
        for region_b in region_names[i+1:]:
            segs_a = region_segments_dict[region_a]
            segs_b = region_segments_dict[region_b]

            count = 0
            while count < per_region_combo_quota:
                a = random.choice(segs_a)
                b = random.choice(segs_b)
                key = (id(a), id(b))
                key_rev = (id(b), id(a))
                if key not in seen_dissimilar and key_rev not in seen_dissimilar:
                    dissimilar_pairs.append((a, b))
                    seen_dissimilar.add(key)
                    count += 1

    dissimilar_labels = [1] * len(dissimilar_pairs)

    # ---- Final Combine & Shuffle ----
    total_pairs = min(len(similar_pairs), len(dissimilar_pairs))
    final_pairs = similar_pairs[:total_pairs] + dissimilar_pairs[:total_pairs]
    final_labels = [0] * total_pairs + [1] * total_pairs

    combined = list(zip(final_pairs, final_labels))
    random.shuffle(combined)
    final_pairs, final_labels = zip(*combined)

    print(f"Created {len(final_pairs)} total pairs ({total_pairs} similar, {total_pairs} dissimilar) from {num_regions} regions.")

    return list(final_pairs), list(final_labels)


def create_balanced_pairs_for_3stage_training(data_dict, stage='inter_subject', max_dataset_size=20000, seed=42):
    """
    Create a balanced dataset of similar and dissimilar pairs for contrastive training.
    
    Args:
        data_dict (dict): Dictionary of segments, structure depends on stage.
        stage (str): One of 'intra_session', 'intra_subject', 'inter_subject'.
        max_dataset_size (int): Max total number of training pairs.
        seed (int): Random seed for reproducibility.
    
    Returns:
        (list of tuple, list of int): List of (segment1, segment2) pairs and labels (0=similar, 1=dissimilar).
    """

    random.seed(seed)
    np.random.seed(seed)

    similar_pairs, dissimilar_pairs = [], []

    if stage == 'inter_subject':
        # Format: dict[region] = list of segments
        region_names = list(data_dict.keys())
        num_regions = len(region_names)
        per_region_quota = max_dataset_size // 2 // num_regions

        # Similar pairs
        for region in region_names:
            segments = data_dict[region]
            if len(segments) < 2:
                continue
            indices = list(range(len(segments)))
            random.shuffle(indices)
            seen = set()
            while len(similar_pairs) < per_region_quota * (region_names.index(region) + 1):
                i, j = random.sample(indices, 2)
                if (i, j) not in seen and (j, i) not in seen:
                    similar_pairs.append((segments[i], segments[j]))
                    seen.add((i, j))
                if len(seen) > len(indices) * (len(indices) - 1) // 2:
                    break

        # Dissimilar pairs
        per_combo_quota = max_dataset_size // 2 // (num_regions * (num_regions - 1) // 2)
        seen = set()
        for i, r1 in enumerate(region_names):
            for r2 in region_names[i+1:]:
                s1 = data_dict[r1]
                s2 = data_dict[r2]
                count = 0
                while count < per_combo_quota:
                    a = random.choice(s1)
                    b = random.choice(s2)
                    key = (id(a), id(b))
                    if key not in seen:
                        dissimilar_pairs.append((a, b))
                        seen.add(key)
                        count += 1

    elif stage == 'intra_subject':
        # Format: dict[subject][region] = list of segments
        for subj, region_dict in data_dict.items():
            region_names = list(region_dict.keys())
            if len(region_names) < 2:
                continue

            # Similar
            for region in region_names:
                segments = region_dict[region]
                if len(segments) < 2:
                    continue
                indices = list(range(len(segments)))
                random.shuffle(indices)
                seen = set()
                while len(seen) < len(indices) * (len(indices) - 1) // 2:
                    i, j = random.sample(indices, 2)
                    if (i, j) not in seen and (j, i) not in seen:
                        similar_pairs.append((segments[i], segments[j]))
                        seen.add((i, j))
                    if len(similar_pairs) > max_dataset_size // 2:
                        break

            # Dissimilar
            for i, r1 in enumerate(region_names):
                for r2 in region_names[i+1:]:
                    s1 = region_dict[r1]
                    s2 = region_dict[r2]
                    count = 0
                    while len(dissimilar_pairs) < max_dataset_size // 2:  # conservative
                        a = random.choice(s1)
                        b = random.choice(s2)
                        dissimilar_pairs.append((a, b))
                        count += 1

    elif stage == 'intra_session':
        # Format: dict[subject][session][region] = list of segments
        for subj in data_dict:
            count_similar = 0
            count_dissimilar = 0
            for sess in data_dict[subj]:
                print(f"\nLoading {subj} {sess} segments...") 
                region_dict = data_dict[subj][sess]
                region_names = list(region_dict.keys())
                if len(region_names) < 2:
                    continue

                # Similar
                for region in region_names:
                    segments = region_dict[region]
                    if len(segments) < 2:
                        continue
                    indices = list(range(len(segments)))
                    random.shuffle(indices)
                    seen = set()
                    while len(seen) < len(indices) * (len(indices) - 1) // 2 and count_similar < max_dataset_size//34:
                        i, j = random.sample(indices, 2)
                        if (i, j) not in seen and (j, i) not in seen:
                            similar_pairs.append((segments[i], segments[j]))
                            count_similar += 1
                            seen.add((i, j))
                        if len(similar_pairs) > max_dataset_size // 2:
                            break

                # Dissimilar
                for i, r1 in enumerate(region_names):
                    for r2 in region_names[i+1:]:
                        s1 = region_dict[r1]
                        s2 = region_dict[r2]
                        count = 0
                        while len(dissimilar_pairs) < max_dataset_size // 2 and count_dissimilar < max_dataset_size//34:  
                            a = random.choice(s1)
                            b = random.choice(s2)
                            dissimilar_pairs.append((a, b))
                            count += 1
                            count_dissimilar += 1
            print(f"added {count_similar} similar pairs and {count_dissimilar} dissimilar pairs from subject {subj}")

    else:
        raise ValueError(f"Unknown stage: {stage}")

    # Final Combine & Shuffle
    total_pairs = min(len(similar_pairs), len(dissimilar_pairs))
    similar_pairs = similar_pairs[:total_pairs]
    dissimilar_pairs = dissimilar_pairs[:total_pairs]
    final_pairs = similar_pairs + dissimilar_pairs
    final_labels = [0] * len(similar_pairs) + [1] * len(dissimilar_pairs)

    combined = list(zip(final_pairs, final_labels))
    random.shuffle(combined)
    final_pairs, final_labels = zip(*combined)

    print(f"[{stage.upper()}] Created {len(final_pairs)} total pairs "
          f"({len(similar_pairs)} similar, {len(dissimilar_pairs)} dissimilar).")

    return list(final_pairs), list(final_labels)


class LFPDataset(Dataset):
    """
    PyTorch Dataset class for loading LFP contrastive pairs.
    
    Each item is a tuple of two time-series segments and a binary label:
    - (segment1, segment2, label)
    - label = 0 for similar (same region), 1 for dissimilar (different regions)
    """
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1, x2 = self.pairs[idx]
        x1 = torch.tensor(x1, dtype=torch.float32).unsqueeze(0)
        x2 = torch.tensor(x2, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x1, x2, label
