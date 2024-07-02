from os.path import join
import os
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torchaudio.functional import resample
import argparse
import numpy as np

from denseav.plotting import plot_attention_video, plot_2head_attention_video, plot_feature_video, display_video_in_notebook
from denseav.shared import norm, crop_to_divisor, blur_dim

###

def test(args):

    model_name = "sound"
    video_path = "/home/data_gen/3DAVS_benchmark/generation_pipeline/generated_dataset/part_1/5ZKStnWn8Zo/small_for_splats/audio_video_vacuum_cleaner.mp4"
    result_dir = "results"

    load_size = 224
    plot_size = 224

    model = torch.hub.load('mhamilton723/DenseAV', model_name).cuda()

    _, audio, info = torchvision.io.read_video(video_path, pts_unit='sec')

    audio = torch.mean(audio, dim=0, keepdim=True)
    sample_rate = 16000

    if info["audio_fps"] != sample_rate:
        audio = resample(audio, info["audio_fps"], sample_rate)
    audio = audio[0].unsqueeze(0)

    with torch.no_grad():
        audio_feats = model.forward_audio({"audio": audio.cuda()})
        audio_feats = {k: v.cpu() for k,v in audio_feats.items()}

    for i in range(len(image_feats["image_feats"])):

        fmap = image_feats["image_feats"][i].cpu().numpy().astype(np.float16)
        torch.save(torch.tensor(fmap).half(), os.path.join('../../data/feature_maps', f"frame_{i:05d}_fmap_CxHxW.pt")) 


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count() 
    test(args)
