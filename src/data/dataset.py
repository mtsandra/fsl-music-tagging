import os
import numpy as np
import json
import torchaudio as ta
import torch
from torch.utils import data
import soundfile as sf

from data.helper import get_n_class_k_shot_subset, get_n_class_all_subset, get_n_class_k_shot_top50, get_eval_data_top50
from data.parse_vggish import get_vggish_embeddings
from data.parse_passt import extract_passt_feature
from data.parse_openl3 import extract_openl3_feature


class MTATDataset(data.Dataset):
    def __init__(self, data_root, mode):
        self.data_root = data_root
        self.path = self._get_split_path(mode)

        self.audio_path = None
        print(os.getcwd())
        with open(self.path, 'r') as f:
            self.data = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self._get_audio_samples(self.data[index]['mp3_path'])
        #breakpoint()
        tags = self.data[index]['tags']
        tags = np.array(tags) 

        audio_path = self.data[index]['mp3_path']
        vggish = self._get_vggish_embeddings(audio_path)
        vggish_norm = self._standardize_embeddings(vggish)

        vggish = self._summarize_embeddings(vggish)
        vggish_norm = self._summarize_embeddings(vggish_norm)
        
        assert vggish.shape == (128*2,), f"Expected (256,), but got {vggish.shape}"
        # print(f"When LOADING, vggish shape: ", vggish.shape)
        passt = self._get_passt_embeddings(audio_path)
        passt_norm = self._standardize_embeddings(passt)
        passt = self._summarize_embeddings(passt)
        passt_norm = self._summarize_embeddings(passt_norm)
        assert passt.shape == (768*2,), f"Expected (1536,), but got {passt.shape}"
        # openl3 = 0
        openl3 = self._get_openl3_embeddings(audio_path).squeeze(0)
        openl3_norm = self._standardize_embeddings(openl3)
        openl3 = self._summarize_embeddings(openl3)
        openl3_norm = self._summarize_embeddings(openl3_norm)
        assert openl3.shape == (512*2,), f"Expected (1024), but got {openl3.shape}"
        
        combined = np.concatenate((vggish_norm, passt_norm, openl3_norm), axis=0)
        assert combined.shape == (2816, ), f"Expected (2816,), but got {combined.shape}"
        
        # try:
        #     # tag_index = self.data[index]['tag_index']
            
        #     

            
        #     # print(f"When LOADING, passt shape: ", passt.shape)
        # except Exception as e:
        #     # tag_index = -1
        #     
        
        # try:
        

        # except Exception as e:
        #     openl3 = self._download_openl3_embeddings(audio_path).squeeze(0)
        

        return {"x": x, "tags": tags,  "vggish": vggish, "passt": passt, "openl3": openl3, "combined": combined}

    def _get_audio_samples(self, mp3_path):
        # joins the mp3_path with data_root
            
        mp3_path = os.path.join(self.data_root,"mp3" ,mp3_path)
        x,fs = ta.load(mp3_path)
        assert fs == 16000, f"Sampling rate is {fs}, but expected 16000"
        x = x.squeeze(0)
        return x
    
    def _save_clip_to_wav(self, samples, path):
        sf.write(path, samples.numpy(), 16000)
        self.audio_path = path
        
    def _delete_wav(self, path):
        os.remove(path)
        
    def _summarize_embeddings(self, embedding):
        embed_mean = embedding.mean(axis=0)
        embed_std = embedding.std(axis=0)
        return np.concatenate((embed_mean, embed_std), axis=0)
    
    def _get_vggish_embeddings(self, audio_path):

        audio_path = os.path.join(self.data_root, "mp3", audio_path)
        vggish_path = os.path.join(self.data_root, "pretrained_features/vggish", os.path.basename(audio_path).replace(".mp3", ".npy"))
    
        return np.load(vggish_path)
    
    def _get_passt_embeddings(self, mp3_path):
        

        mp3_path = os.path.join(self.data_root, "mp3", mp3_path)
        passt_path = os.path.join(self.data_root, "pretrained_features/passt", os.path.basename(mp3_path).replace(".mp3", ".npy"))
        return np.load(passt_path)
        
        
    def _download_passt_embeddings(self, mp3_path):
        wav_path = mp3_path.replace(".mp3", ".wav")
        wav_path = wav_path.split("/")[-1]
        wav_path = os.path.join(self.data_root, "wav", wav_path)
        return extract_passt_feature(wav_path)
    
    def _download_openl3_embeddings(self, mp3_path):
        wav_path = mp3_path.replace(".mp3", ".wav")
        wav_path = wav_path.split("/")[-1]
        wav_path = os.path.join(self.data_root, "wav", wav_path)
        return extract_openl3_feature(wav_path)
    
    def _get_openl3_embeddings(self, mp3_path):
        
        mp3_path = os.path.join(self.data_root, "mp3", mp3_path)
        openl3_path = os.path.join(self.data_root, "pretrained_features/openl3", os.path.basename(mp3_path).replace(".mp3", ".npy"))
        return np.load(openl3_path)
    
    def _normalize_embeddings(self, embeddings):
        min_val = np.min(embeddings)
        max_val = np.max(embeddings)

        return (embeddings - min_val) / (max_val - min_val) 
        
    def _standardize_embeddings(self, embeddings):
        # standardize per embedding, take the mean and std of each embedding row
        embed_mean = embeddings.mean(axis=1, keepdims=True)
        embed_std = embeddings.std(axis=1, keepdims=True)
        return (embeddings - embed_mean) / embed_std
    
    def _get_split_path(self, mode):
        if mode == 'probe':
            return "./data/fiftyplus.json"
        
        if mode == 'top50':
            return "./data/top50.json"

        else:
            raise ValueError(f"Invalid mode: {mode}")
        

class MTATNovelSet(MTATDataset):
        
    def __init__(self, data_root, mode, n_way, k_shot):
        super().__init__(data_root, mode)
        self.n_way = n_way
        self.k_shot = k_shot
        self.data, self.tag_names = self._get_novel_k_shot_data()

    
    def _get_novel_k_shot_data(self):
        novel_df, tag_names = get_n_class_k_shot_subset(self.n_way, self.k_shot)
        desired_clip_ids = novel_df['clip_id'].tolist()
        
        novel_json = []
        for data in self.data:
            if data['clip_id'] in desired_clip_ids:
                data['tags'] = novel_df[novel_df['clip_id'] == data['clip_id']].iloc[:, 2:].values.tolist()
                novel_json.append(data)
        

        return novel_json, tag_names
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
class MTATNovelAllData(MTATDataset):
        
    def __init__(self, data_root, mode, n_way):
        super().__init__(data_root, mode)
        self.n_way = n_way

        self.data, self.tag_names = self._get_novel_all_data()

    
    def _get_novel_all_data(self):
        novel_df, tag_names = get_n_class_all_subset(self.n_way)
        desired_clip_ids = novel_df['clip_id'].tolist()
        
        novel_json = []
        for data in self.data:
            if data['clip_id'] in desired_clip_ids:
                data['tags'] = novel_df[novel_df['clip_id'] == data['clip_id']].iloc[:, 2:].values.tolist()
                novel_json.append(data)
        

        return novel_json, tag_names
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    
    
class MTATNovelTop50(MTATDataset):
        
    def __init__(self, data_root, mode, n_way, k_shot, load_mode):
        super().__init__(data_root, mode)
        self.load_mode = load_mode
        self.n_way = n_way
        self.k_shot = k_shot
        self.data, self.tag_names = self._get_novel_k_shot_data()

    
    def _get_novel_k_shot_data(self):
        if self.load_mode == "train":
            novel_df, tag_names = get_n_class_k_shot_top50(self.n_way, self.k_shot)
        else:
            novel_df, tag_names = get_eval_data_top50(self.n_way, self.k_shot, self.load_mode)
        desired_clip_ids = novel_df['clip_id'].tolist()
        
        novel_json = []
        for data in self.data:
            if data['clip_id'] in desired_clip_ids:
                data['tags'] = novel_df[novel_df['clip_id'] == data['clip_id']].iloc[:, 2:].values.tolist()
                novel_json.append(data)
        

        return novel_json, tag_names
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return super().__getitem__(index)

