import os
import argparse
import pickle
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from modules.network import get_network
from modules.CONTRIQUE_model import CONTRIQUE_model


def load_contrique_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    """
    Safely load CONTRIQUE checkpoint handling map_location, nested dicts, and module prefixes.
    """
    try:
        # PyTorch 2.6+ defaults to weights_only=True, but checkpoints may contain legacy structures
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)

    # If checkpoint is wrapped in a dictionary (e.g. {'state_dict': ...})
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    # Strip any DDP 'module.' prefix if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict)
    return model


def main(args):
    device = torch.device(args.device)

    # Load image and ensure RGB mode
    image = Image.open(args.im_path).convert("RGB")
    
    # Downscale image by 2 using BICUBIC resampling (matching original CONTRIQUE evaluation)
    sz = image.size
    resample_mode = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    image_2 = image.resize((sz[0] // 2, sz[1] // 2), resample=resample_mode)
    
    # Transform to tensor and send to target device
    to_tensor = transforms.ToTensor()
    image_t = to_tensor(image).unsqueeze(0).to(device)
    image_2_t = to_tensor(image_2).unsqueeze(0).to(device)
    
    # Load CONTRIQUE model
    encoder = get_network("resnet50", pretrained=False)
    model = CONTRIQUE_model(args=args, encoder=encoder, n_features=2048)
    load_contrique_checkpoint(model, args.model_path, device)
    model = model.to(device)
    model.eval()
    
    # Extract features using modern inference_mode
    with torch.inference_mode():
        _, _, _, _, model_feat, model_feat_2, _, _ = model(image_t, image_2_t)
    
    feat = np.hstack((model_feat.detach().cpu().numpy(), model_feat_2.detach().cpu().numpy()))
    
    # Load linear regressor model
    with open(args.linear_regressor_path, "rb") as f:
        regressor = pickle.load(f)
        
    score = regressor.predict(feat)[0]
    print(score)


def parse_args():
    parser = argparse.ArgumentParser(description="CONTRIQUE Blind Image Quality Prediction")
    
    parser.add_argument("--im_path", type=str, default="sample_images/33.bmp",
                        help="Path to image", metavar="")
    parser.add_argument("--model_path", type=str, default="models/CONTRIQUE_checkpoint25.tar",
                        help="Path to trained CONTRIQUE model", metavar="")
    parser.add_argument("--linear_regressor_path", type=str, default="models/CLIVE.save",
                        help="Path to trained linear regressor", metavar="")
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (e.g. 'cuda:0', 'cpu')", metavar="")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)