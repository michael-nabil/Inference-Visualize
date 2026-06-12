import os
import numpy as np

import cv2
import torch
import torch.nn as nn

from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from PIL import Image

# Setup standard weights and the torchvision preprocessing pipeline
WEIGHTS = ConvNeXt_Base_Weights.DEFAULT
PREPROCESSOR = WEIGHTS.transforms()
INPUT_VID_PATH = r"/video/path"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the pre-trained ConvNeXt-Base model
        self.model = convnext_base(weights=WEIGHTS)
        
        # ConvNeXt's 'classifier' is a Sequential block with 3 steps:
        # (0): LayerNorm2d 
        # (1): Flatten 
        # (2): Linear (the actual 1000-class output)
        # We replace ONLY the final Linear layer. This allows us to keep the 
        # LayerNorm and Flattening operations intact for a pristine feature vector.
        self.model.classifier[2] = nn.Identity()
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Cleanly outputs the feature vector
        return self.model(x)


def frames_sampling_and_preprocessing(video_path):
    if not os.path.exists(video_path):
        print(f'{video_path} NOT FOUND!')

    cap = cv2.VideoCapture(video_path)

    stride = 15 # we use fixed stride  (Equivalent to 2fps sampling for a 30fps video)
    frames = []
    count = 0

    while True:
        success, frame = cap.read()
        if not success: break
        
        if count % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert NumPy array to PIL Image for the transform
            pil_img = Image.fromarray(frame)
            
            # Apply EfficientNetV2 processing
            frame_tensor = PREPROCESSOR(pil_img)
            frames.append(frame_tensor)
        count += 1
        
    cap.release()
    preprocessed_frames_tensor = torch.stack(frames)
    return preprocessed_frames_tensor

# 1
# frames_sampling_and_preprocessing(INPUT_VID_PATH)

def extract_features(video_path, extractor, device, batch_size=32):
    
    extractor.eval()
    with torch.no_grad():
        frames = frames_sampling_and_preprocessing(video_path)
        T = frames.shape[0]
        feats = []

        for start in range(0, T, batch_size):
            batch = frames[start : start + batch_size].to(device)
            f = extractor(batch)
            feats.append(f.cpu())

        video_feats = torch.cat(feats, dim=0)

    return video_feats

# 1. instantiating the extraction model and moving it to GPU
# extractor = ConvNeXtFeatureExtractor().to(DEVICE)
# extractor.eval()

# 2. runnning feature extraction
# video_features = extract_features(INPUT_VID_PATH, extractor, DEVICE,) #batch_size=16)


