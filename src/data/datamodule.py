import pytorch_lightning as pl
from torch.utils import data
from .dataset import MTATDataset, MTATNovelSet, MTATNovelAllData, MTATNovelTop50

class MTATDataModule(pl.LightningDataModule):
    def __init__(self, data_root, n_way=5, k_shot=5):
        super().__init__()
        self.data_root = data_root
        self.n_way = n_way
        self.k_shot = k_shot
    
    def setup(self, stage=None):
        self.probe_dataset = MTATNovelSet(self.data_root, 'probe',  n_way=self.n_way, k_shot=self.k_shot)
    
    def setup_all(self, stage=None):
        self.probe_all_dataset = MTATNovelAllData(self.data_root, 'probe',  n_way=self.n_way)
        
    def setup_top50(self, stage=None):
        self.train_dataset = MTATNovelTop50(self.data_root, 'top50',  n_way=self.n_way, k_shot=self.k_shot, load_mode="train")
        self.valid_dataset = MTATNovelTop50(self.data_root, 'top50',  n_way=self.n_way, k_shot=self.k_shot, load_mode="valid")
        self.test_dataset = MTATNovelTop50(self.data_root, 'top50',  n_way=self.n_way, k_shot=self.k_shot, load_mode="test")
    
    def top50_train_dataloader(self, batch_size, num_workers):
        return data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)
    
    def top50_valid_dataloader(self, batch_size, num_workers):
        return data.DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)
    
    def top50_test_dataloader(self, batch_size, num_workers):
        return data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)
    
    def get_tag_names_top50(self):
        return self.train_dataset.tag_names
    
    def probe_dataloader(self, batch_size, num_workers):
        return data.DataLoader(self.probe_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)
    
    def get_tag_names(self):
        return self.probe_dataset.tag_names
    
    def get_tag_names_all(self):
        return self.probe_all_dataset.tag_names
    
    def probe_all_dataloader(self, batch_size, num_workers):
        return data.DataLoader(self.probe_all_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)
    
    
    