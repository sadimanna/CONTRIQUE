"""
CONTRIQUE Image Quality Assessment (IQA) Wrapper.

This module provides a modular, production-ready PyTorch wrapper (CONTRIQUE)
for Blind (No-Reference) and Full-Reference Image Quality Assessment.

Models are loaded once in __init__, and quality assessment or feature extraction
is performed seamlessly via the forward() method.
"""

import os
import pickle
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from modules.network import get_network
from modules.CONTRIQUE_model import CONTRIQUE_model


def _safe_load_checkpoint(model: nn.Module, checkpoint_path: Union[str, Path], device: torch.device):
    """
    Safely load model state_dict handling PyTorch 2.6+ weights_only,
    wrapper dictionaries, and DataParallel/DDP 'module.' prefixes.
    """
    checkpoint_path = str(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'.")

    try:
        # PyTorch 2.6+ defaults to weights_only=True
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)

    # Unwrap if stored in a metadata container
    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

    # Strip 'module.' prefix if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict)
    return model


class CONTRIQUE(nn.Module):
    """
    Wrapper for CONTRIQUE Image Quality Assessment.

    Features:
    - Pretrained model loaded during initialization
    - Supports Blind (No-Reference) and Full-Reference (FR) scoring
    - Accepts file paths, PIL Images, or PyTorch tensors
    - Provides forward() method returning predicted quality scores
    - Extract raw 4096-D CONTRIQUE features with extract_features()

    Args:
        model_path: Path to CONTRIQUE pretrained checkpoint (.tar).
        regressor_path: Path to pretrained linear regressor (.save).
                        Examples:
                        - Blind/NR: 'models/CLIVE.save', 'models/LIVE.save',
                                    'models/Koniq.save', 'models/CSIQ.save'
                        - FR:       'models/LIVE_FR.save', 'models/CSIQ_FR.save'
        device: 'cuda', 'cpu', or torch.device. If None, auto-selects.
        network: Backbone architecture (default: 'resnet50').
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/CONTRIQUE_checkpoint25.tar",
        regressor_path: Optional[Union[str, Path]] = "models/CLIVE.save",
        device: Optional[Union[str, torch.device]] = None,
        network: str = "resnet50",
    ):
        super().__init__()

        # Setup device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build backbone and CONTRIQUE model
        encoder = get_network(network, pretrained=False)
        self.n_features = 2048
        self.model = CONTRIQUE_model(args=None, encoder=encoder, n_features=self.n_features)

        # Load weights
        if model_path is not None:
            _safe_load_checkpoint(self.model, model_path, self.device)

        self.model.to(self.device)
        self.model.eval()

        # Freeze encoder and projector parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Load linear regressor if specified
        self.regressor = None
        self.regressor_path = regressor_path
        if regressor_path is not None:
            self.load_regressor(regressor_path)

        # Standard ToTensor transform and Bicubic resampling
        self.to_tensor = transforms.ToTensor()
        self.resample_mode = (
            Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
        )

    def to(self, *args, **kwargs):
        """Override to update self.device and model device synchronously."""
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self.device = torch.device(device)
        self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def load_regressor(self, regressor_path: Union[str, Path]):
        """Load or switch the linear regressor model (.save)."""
        reg_path = str(regressor_path)
        if not os.path.isfile(reg_path):
            raise FileNotFoundError(f"Regressor file not found at '{reg_path}'.")

        with warnings.catch_warnings():
            # Suppress unpickling scikit-learn version mismatch warnings
            warnings.simplefilter("ignore")
            with open(reg_path, "rb") as f:
                self.regressor = pickle.load(f)
        self.regressor_path = reg_path

    def _prepare_inputs(
        self, image: Union[str, Path, Image.Image, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert input image into full-resolution and half-resolution tensors
        ready for the CONTRIQUE encoder.
        """
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
            sz = pil_img.size
            pil_img_2 = pil_img.resize((sz[0] // 2, sz[1] // 2), resample=self.resample_mode)
            x1 = self.to_tensor(pil_img).unsqueeze(0).to(self.device)
            x2 = self.to_tensor(pil_img_2).unsqueeze(0).to(self.device)
            return x1, x2

        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            sz = pil_img.size
            pil_img_2 = pil_img.resize((sz[0] // 2, sz[1] // 2), resample=self.resample_mode)
            x1 = self.to_tensor(pil_img).unsqueeze(0).to(self.device)
            x2 = self.to_tensor(pil_img_2).unsqueeze(0).to(self.device)
            return x1, x2

        elif isinstance(image, torch.Tensor):
            x = image.to(self.device)
            if x.dim() == 3:
                x = x.unsqueeze(0)  # [1, C, H, W]
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)  # Expand grayscale to 3 channels

            # Ensure values are float in [0, 1]
            if x.dtype == torch.uint8:
                x = x.float() / 255.0

            # Downsample by 2 using bicubic interpolation
            h, w = x.shape[2], x.shape[3]
            x2 = F.interpolate(x, size=(h // 2, w // 2), mode="bicubic", align_corners=False)
            return x, x2

        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}. Expected file path, PIL.Image, or torch.Tensor."
            )

    @torch.inference_mode()
    def extract_features(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
        as_tensor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract 4096-dimensional CONTRIQUE feature vector for an image.
        Concatenates full-scale (2048-D) and half-scale (2048-D) normalized features.

        Args:
            image: Input image (file path, PIL Image, or tensor).
            as_tensor: If True, returns torch.Tensor on self.device.
                       If False (default), returns numpy.ndarray of shape (1, 4096).
        """
        x1, x2 = self._prepare_inputs(image)
        _, _, _, _, feat1, feat2, _, _ = self.model(x1, x2)

        feat_tensor = torch.cat((feat1, feat2), dim=1)  # [B, 4096]
        if as_tensor:
            return feat_tensor
        return feat_tensor.detach().cpu().numpy()

    @torch.inference_mode()
    def forward(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
        ref_image: Optional[Union[str, Path, Image.Image, torch.Tensor]] = None,
        return_features: bool = False,
    ) -> Union[float, np.ndarray, Tuple[Union[float, np.ndarray], np.ndarray]]:
        """
        Assess image quality.

        - If ref_image is None: Blind / No-Reference (NR) IQA.
        - If ref_image is provided: Full-Reference (FR) IQA.

        Args:
            image: Distorted image (or single image for NR evaluation).
            ref_image: Optional pristine reference image for FR evaluation.
            return_features: If True, returns (score, features).

        Returns:
            Predicted quality score (float if single sample, ndarray if batch).
            If no regressor is loaded, returns the feature vector.
        """
        dist_feat = self.extract_features(image, as_tensor=False)

        if ref_image is not None:
            # Full-Reference mode: compute absolute feature difference
            ref_feat = self.extract_features(ref_image, as_tensor=False)
            features = np.abs(ref_feat - dist_feat)
        else:
            # Blind / No-Reference mode
            features = dist_feat

        if self.regressor is None:
            warnings.warn("No regressor loaded; returning raw 4096-D feature representation.")
            return features

        pred = self.regressor.predict(features)
        score = float(pred[0]) if len(pred) == 1 else pred

        if return_features:
            return score, features
        return score


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CONTRIQUE Quality Assessment Wrapper CLI")
    parser.add_argument("--image", "-i", type=str, default="sample_images/60.bmp",
                        help="Path to image to assess")
    parser.add_argument("--ref", "-r", type=str, default=None,
                        help="Optional reference image path for Full-Reference mode")
    parser.add_argument("--model_path", "-m", type=str, default="models/CONTRIQUE_checkpoint25.tar",
                        help="Path to CONTRIQUE model weights (.tar)")
    parser.add_argument("--regressor", "-g", type=str, default=None,
                        help="Path to regressor (.save). Defaults to CLIVE.save (NR) or CSIQ_FR.save (FR)")
    parser.add_argument("--device", "-d", type=str, default=None,
                        help="Device to use ('cuda', 'cpu')")

    args = parser.parse_args()

    # Select appropriate default regressor if not specified
    if args.regressor is None:
        if args.ref is not None:
            args.regressor = "models/CSIQ_FR.save"
        else:
            args.regressor = "models/CLIVE.save"

    mode_str = "Full-Reference (FR)" if args.ref is not None else "Blind / No-Reference (NR)"
    print(f"Initializing CONTRIQUE wrapper [{mode_str}]...")

    wrapper = CONTRIQUE(
        model_path=args.model_path,
        regressor_path=args.regressor,
        device=args.device,
    )

    print(f"Running inference on: {args.image}")
    if args.ref:
        print(f"Reference image:      {args.ref}")
        score = wrapper(args.image, ref_image=args.ref)
    else:
        score = wrapper(args.image)

    print(f"Predicted Quality Score: {score}")


if __name__ == "__main__":
    main()
