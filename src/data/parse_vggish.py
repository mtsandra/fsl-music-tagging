import subprocess
import os
from tqdm import tqdm
import numpy as np
import torch

model = torch.hub.load("harritaylor/torchvggish", "vggish", device="cpu", postprocess=True)
model.eval()
WAV_FOLDER = "/ssddata/tma98/data/mtat/wav/"
VGG_FOLDER = "/ssddata/tma98/data/mtat/pretrained_features/vggish/"
NPY_SAVE_MODE = False

def convert_to_wav(mp3_path):
    wav_path = mp3_path.replace(".mp3", ".wav")
    if not os.path.exists(WAV_FOLDER):
        os.makedirs(WAV_FOLDER, exist_ok=True)
    
    wav_path = os.path.join(WAV_FOLDER, os.path.basename(wav_path))
    subprocess.run(["ffmpeg", "-i", mp3_path, wav_path])
    return wav_path

def get_vggish_embeddings(wav_path, sr=16000):

    output = model(wav_path, fs=sr)
    output = output.detach().cpu().numpy()
    if NPY_SAVE_MODE:
        if not os.path.exists(VGG_FOLDER):
            os.makedirs(VGG_FOLDER, exist_ok=True)
        npy_path = wav_path.replace(WAV_FOLDER, VGG_FOLDER).replace(".wav", ".npy")
        
        np.save(npy_path, output)
        print(f"Saved VGGish embeddings to  {npy_path}")
    else:
        return output

    
def get_multiple_vggish_embeddings(input_dir, output_dir, sr=16000):
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files):
            if file.endswith(".mp3"):
                mp3_path = os.path.join(root, file)
                wav_path = os.path.join(WAV_FOLDER, os.path.basename(mp3_path).replace(".mp3", ".wav"))
                #wav_path = convert_to_wav(mp3_path)
                get_vggish_embeddings(wav_path,  sr)
                
if __name__ == "__main__":
    import fire
    fire.Fire()