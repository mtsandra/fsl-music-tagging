import csv
import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from datetime import datetime
from tqdm import tqdm

from data.datamodule import MTATDataModule
from models import LightningBaseline

def get_input_dim(f):
    if f == "vggish":
        return 256
    elif f == "passt":
        return 1536
    elif f == "openl3":
        return 1024
    elif f == "combined":
        return 2816




def ablation_study(continue_from_ckpt=False):
    seed_everything(42)
    if continue_from_ckpt:
        ckpt_csv = pd.read_csv("./logs/ablation_study/ablation_results_tbc.csv")
        n_max = ckpt_csv["n"].max()
        k_max = ckpt_csv["k"].max()
    else:
        n_max = 0
        k_max = 0
    
    f = "combined"
    input_dim = get_input_dim(f)
    
    

    with open(f'./logs/ablation_study/ablation_results_{f}_{datetime.now()}.csv', mode='w', newline='') as csv_file:
        fieldnames = ['feature', 'n', 'k', 'mAP', 'roc_auc']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        

        writer.writeheader()

        for n in tqdm( [2,5,15,35,50]):
            for k in [1,5,10,20]:
                if k <= k_max and n <= n_max:
                    continue
                
                baseline_model = LightningBaseline(input_dim=input_dim, output_dim=n, pretrained_features=f)
                NovelDataModule = MTATDataModule(data_root="/ssddata/tma98/data/mtat", n_way=n, k_shot=k)
                NovelDataModule.setup_top50()
                train_loader = NovelDataModule.top50_train_dataloader(batch_size=8, num_workers=4)
                val_loader = NovelDataModule.top50_valid_dataloader(batch_size=8, num_workers=4)
                test_loader = NovelDataModule.top50_test_dataloader(batch_size=8, num_workers=4)
                early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=10)
                logger = TensorBoardLogger(f"./logs/ablation_study/tensorboard_logs/{f}", name=f"{n}way_{k}shot_baseline")
                trainer = Trainer(log_every_n_steps=10, max_epochs=400, callbacks=[early_stopping], logger =logger)

                trainer.fit(model=baseline_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
                test_result_dic = trainer.test(model=baseline_model, dataloaders=test_loader)[0]
                
                # Write the results to the CSV file
                writer.writerow({
                    'feature': f,
                    'n': n,
                    'k': k,
                    'mAP': test_result_dic["test_mAP"],
                    'roc_auc': test_result_dic["test_auc_roc"]
                })
                csv_file.flush()



ablation_study()
