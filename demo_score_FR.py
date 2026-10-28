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
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

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

    # Load reference and distorted images in RGB mode
    ref_image = Image.open(args.ref_path).convert("RGB")
    dist_image = Image.open(args.dist_path).convert("RGB")
    
    # Downscale images by 2 using BICUBIC resampling
    resample_mode = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    ref_sz = ref_image.size
    dist_sz = dist_image.size
    ref_image_2 = ref_image.resize((ref_sz[0] // 2, ref_sz[1] // 2), resample=resample_mode)
    dist_image_2 = dist_image.resize((dist_sz[0] // 2, dist_sz[1] // 2), resample=resample_mode)
    
    # Transform to tensor
    to_tensor = transforms.ToTensor()
    ref_image_t = to_tensor(ref_image).unsqueeze(0).to(device)
    ref_image_2_t = to_tensor(ref_image_2).unsqueeze(0).to(device)
    dist_image_t = to_tensor(dist_image).unsqueeze(0).to(device)
    dist_image_2_t = to_tensor(dist_image_2).unsqueeze(0).to(device)
    
    # Load CONTRIQUE Model
    encoder = get_network("resnet50", pretrained=False)
    model = CONTRIQUE_model(args=args, encoder=encoder, n_features=2048)
    load_contrique_checkpoint(model, args.model_path, device)
    model = model.to(device)
    model.eval()
    
    # Extract features using modern inference_mode
    with torch.inference_mode():
        _, _, _, _, ref_feat, ref_feat_2, _, _ = model(ref_image_t, ref_image_2_t)
        _, _, _, _, dist_feat, dist_feat_2, _, _ = model(dist_image_t, dist_image_2_t)
    
    ref = np.hstack((ref_feat.detach().cpu().numpy(), ref_feat_2.detach().cpu().numpy()))
    dist = np.hstack((dist_feat.detach().cpu().numpy(), dist_feat_2.detach().cpu().numpy()))
    feat = np.abs(ref - dist)
    
    # Load regressor model
    with open(args.linear_regressor_path, "rb") as f:
        regressor = pickle.load(f)
        
    score = regressor.predict(feat)[0]
    print(score)


def parse_args():
    parser = argparse.ArgumentParser(description="CONTRIQUE Full-Reference Image Quality Prediction")
    
    parser.add_argument("--ref_path", type=str, default="sample_images/womanhat.bmp",
                        help="Path to reference image", metavar="")
    parser.add_argument("--dist_path", type=str, default="sample_images/img191.bmp",
                        help="Path to distorted image", metavar="")
    parser.add_argument("--model_path", type=str, default="models/CONTRIQUE_checkpoint25.tar",
                        help="Path to trained CONTRIQUE model", metavar="")
    parser.add_argument("--linear_regressor_path", type=str, default="models/LIVE_FR.save",
                        help="Path to trained linear regressor", metavar="")
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (e.g. 'cuda:0', 'cpu')", metavar="")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)