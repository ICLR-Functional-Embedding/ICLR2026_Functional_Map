import torch
import torch.nn as nn
import torch.nn.functional as F



class SiameseNetSingleSubject(nn.Module):
    """
    Siamese network for learning similarity between pairs of time-series segments.
    
    Uses a shared CNN encoder to embed each segment and compares them using distance.
    
    Args:
        embedding_dim (int): Dimensionality of the latent embedding space.
    """
    def __init__(self, embedding_dim, dropout=0.3, normalized_output=False):
        super(SiameseNetSingleSubject, self).__init__()
        self.encoder = CNNEncoderSingleSubject(embedding_dim, dropout, normalized_output)

    def forward(self, x1, x2):
        embed1 = self.encoder(x1)
        embed2 = self.encoder(x2)
        distance = (embed1 - embed2).norm(p=2, dim=1)   # shape: (batch,)
        
        return distance 
    
class CNNEncoderSingleSubject(nn.Module): 
    """
    CNN-based encoder for embedding 1D neural time-series signals into a functional latent space.
    
    This version uses sequential 1D convolutions, GELU activations, dropout, and pooling layers.
    The output is a fixed-size embedding vector.
    
    Args:
        embedding_dim (int): The dimensionality of the final embedding vector.
    """
    def __init__(self, embedding_dim, dropout=0.3, normalized_output = False):
        super(CNNEncoderSingleSubject, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(1, 64, kernel_size=15, stride=4, padding=0),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(3, stride=2),
            nn.Dropout(dropout),

            nn.Conv1d(64, 64, kernel_size=11, stride=2, padding=0), 
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(3, stride=2),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(3, stride=2),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(3, stride=2),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(10)
            
        ])

        self.fc = nn.Sequential(
            nn.Linear(10*64, embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),

        )

        self.normalized_output = normalized_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.normalized_output:
            return F.normalize(x, p=2, dim=1)
        else:
            return x
    
    def embed(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.normalized_output:
            return F.normalize(x, p=2, dim=1)
        else:
            return x
