import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from models import baseline, metrics
from data.datamodule import MTATDataModule


class LightningBaseline(L.LightningModule):
    def __init__(self, input_dim, output_dim, pretrained_features):
        super().__init__()
        self.model = baseline.LinearBCE(input_dim, output_dim)
        self.pretrained_features = pretrained_features
        self.criterion = torch.nn.BCELoss()
        self.train_metrics = metrics.MultiLabelBinaryEval(output_dim)
        self.val_metrics = metrics.MultiLabelBinaryEval(output_dim)
        self.test_metrics = metrics.MultiLabelBinaryEval(output_dim)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        loss, logits, targets = self.common_step(batch)
        self.train_metrics.update(logits, targets)
        self.log('train_loss', loss,  prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-4)
    
    def validation_step(self, batch, batch_idx):
        loss, logits, targets = self.common_step(batch)
        print(f"Logits: {logits.shape}, Targets: {targets.shape}")
        self.val_metrics.update(logits, targets)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, logits, targets = self.common_step(batch)
        self.test_metrics.update(logits, targets)
        self.log('test_loss', loss)
        return loss
    
    def common_step(self, batch):
        x, y = batch[self.pretrained_features], batch["tags"].squeeze()
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y.float())
        return loss, y_hat, y
    
    def on_train_epoch_start(self):
        self.train_metrics.reset()


    
    def on_train_epoch_end(self) -> None:
        metric_dic = self.train_metrics.compute()
        self.log('train_mAP', metric_dic["mAP"], on_epoch=True)
        self.log('train_auc_roc', metric_dic["auc_roc"], on_epoch=True)
        
    def on_validation_epoch_start(self):
        self.val_metrics.reset()
    
    def on_validation_epoch_end(self) -> None:
        metric_dic = self.val_metrics.compute()
        self.log('val_mAP', metric_dic["mAP"], on_epoch=True)
        self.log('val_auc_roc', metric_dic["auc_roc"], on_epoch=True)
        
    def on_test_epoch_start(self):
        self.test_metrics.reset()
        
    def on_test_epoch_end(self) -> None:
        metric_dic = self.test_metrics.compute()
        self.log('test_mAP', metric_dic["mAP"], on_epoch=True)
        self.log('test_auc_roc', metric_dic["auc_roc"], on_epoch=True)
    
        
        


if __name__ == "__main__":
    n_way = 5
    k_shot = 5
    pretrained_features = "openl3"
    baseline_model = LightningBaseline(input_dim=1024, output_dim=n_way, pretrained_features="openl3")
    NovelDataModule = MTATDataModule(data_root="/ssddata/tma98/data/mtat", n_way=n_way, k_shot=k_shot)
    NovelDataModule.setup_top50()
    train_loader = NovelDataModule.top50_train_dataloader(batch_size=5, num_workers=0)
    val_loader = NovelDataModule.top50_valid_dataloader(batch_size=5, num_workers=0)
    test_loader = NovelDataModule.top50_test_dataloader(batch_size=5, num_workers=0)
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    logger = TensorBoardLogger("tensorboard_logs", name=f"{pretrained_features}_{n_way}way_{k_shot}shot_baseline")
    trainer = L.Trainer(max_epochs=20, callbacks=[early_stopping], logger=logger)

    trainer.fit(model=baseline_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test = trainer.test(model=baseline_model, dataloaders=test_loader) 

    
    