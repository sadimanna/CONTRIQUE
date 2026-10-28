import torch
import torch.nn as nn
from typing import Optional, Tuple


class CONTRIQUE_model(nn.Module):
    """
    CONTRIQUE Model: Contrastive learning model for Image Quality Assessment (IQA).
    Uses a ResNet backbone encoder without final pooling/FC layer and an MLP projector head.
    Preserves exact layer structure and state_dict naming for compatibility with pretrained checkpoints.
    """
    def __init__(
        self,
        args=None,
        encoder: Optional[nn.Module] = None,
        n_features: int = 2048,
        patch_dim: Tuple[int, int] = (2, 2),
        normalize: bool = True,
        projection_dim: int = 128,
    ):
        super().__init__()

        self.normalize = normalize
        if encoder is not None:
            self.encoder = nn.Sequential(*list(encoder.children())[:-2])
        else:
            self.encoder = None
            
        self.n_features = n_features
        self.patch_dim = patch_dim
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_patch = nn.AdaptiveAvgPool2d(patch_dim)

        # MLP projector (Linear -> BN -> ReLU -> Linear -> BN)
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.BatchNorm1d(self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
        )
        
    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor):
        # global features from encoder
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        
        # local features
        h_i_patch = self.avgpool_patch(h_i)
        h_j_patch = self.avgpool_patch(h_j)
        
        num_patches = self.patch_dim[0] * self.patch_dim[1]
        h_i_patch = h_i_patch.reshape(-1, self.n_features, num_patches)
        h_j_patch = h_j_patch.reshape(-1, self.n_features, num_patches)
        
        h_i_patch = torch.transpose(h_i_patch, 2, 1)
        h_i_patch = h_i_patch.reshape(-1, self.n_features)
        
        h_j_patch = torch.transpose(h_j_patch, 2, 1)
        h_j_patch = h_j_patch.reshape(-1, self.n_features)
        
        h_i = self.avgpool(h_i)
        h_j = self.avgpool(h_j)
        
        h_i = torch.flatten(h_i, 1)
        h_j = torch.flatten(h_j, 1)
        
        if self.normalize:
            h_i = nn.functional.normalize(h_i, dim=1)
            h_j = nn.functional.normalize(h_j, dim=1)
            
            h_i_patch = nn.functional.normalize(h_i_patch, dim=1)
            h_j_patch = nn.functional.normalize(h_j_patch, dim=1)
        
        # global projections
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        
        # local projections
        z_i_patch = self.projector(h_i_patch)
        z_j_patch = self.projector(h_j_patch)
        
        return z_i, z_j, z_i_patch, z_j_patch, h_i, h_j, h_i_patch, h_j_patch