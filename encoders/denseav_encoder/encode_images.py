import os
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torchaudio.functional import resample
import argparse
import numpy as np

from denseav.shared import norm, crop_to_divisor, blur_dim

def run():

    model_name = "sound_and_language"
    video_path = "/home/data_gen/3DAVS_benchmark/generation_pipeline/generated_dataset/part_1/5ZKStnWn8Zo/1/audio_video_violin.mp4"
    result_dir = "/home/data_gen/3DAVS_benchmark/generation_pipeline/generated_dataset/part_1/5ZKStnWn8Zo/1/visual_features_denseav"

    os.makedirs(result_dir, exist_ok=True)

    load_size = 224

    model = torch.hub.load('mhamilton723/DenseAV', model_name).cuda()

    original_frames, audio, info = torchvision.io.read_video(video_path, pts_unit='sec')

    img_transform = T.Compose([
        T.Resize(load_size, Image.BILINEAR),
        lambda x: crop_to_divisor(x, 8),
        lambda x: x.to(torch.float32) / 255,
        norm])

    frames = torch.cat([img_transform(f.permute(2, 0, 1)).unsqueeze(0) for f in original_frames], axis=0)

    with torch.no_grad():
        image_feats = model.forward_image({"frames": frames.unsqueeze(0).cuda()}, max_batch_size=2)
        image_feats = {k: v.cpu() for k,v in image_feats.items()}

    print(f'Feature set length: {image_feats["image_feats"].size()}')

    for i in range(len(image_feats["image_feats"])):

        fmap = image_feats["image_feats"][i]

        print(f'Feature size: {fmap.shape}')

        torch.save(fmap.half(), os.path.join(result_dir, f"frame_{i:05d}_fmap_CxHxW.pt")) 
    

if __name__ == "__main__":
    torch.manual_seed(69)
    run()