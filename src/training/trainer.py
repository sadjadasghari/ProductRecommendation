import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.models.two_tower import TwoTowerRecSys
from src.training.loss import InfoNCELoss
from src.data.multimodal_dataset import MultimodalRetailDataset, collate_multimodal_batch

class LitTwoTower(pl.LightningModule):
    def __init__(self, embed_dim=128, lr=1e-4, batch_size=64):
        super().__init__()
        self.save_hyperparameters()
        self.model = TwoTowerRecSys(embed_dim=embed_dim)
        self.loss_fn = InfoNCELoss(temperature=0.07)
        self.lr = lr
        self.batch_size = batch_size

    def forward(self, history, img, txt_ids, txt_mask, context):
        return self.model(history, img, txt_ids, txt_mask, context)

    def training_step(self, batch, batch_idx):
        history, target_img, target_txt_ids, target_txt_mask, context = batch
        
        # 1. Forward pass
        user_emb, item_emb = self(history, target_img, target_txt_ids, target_txt_mask, context)
        
        # 2. Compute symmetric InfoNCE contrastive loss
        loss = self.loss_fn(user_emb, item_emb)
        
        self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        # Using AdamW with weight decay typical for representation learning
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

def run_training():
    # Setup dataset
    train_ds = MultimodalRetailDataset(num_samples=2000) # Small for demonstration
    train_loader = DataLoader(
        train_ds, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=collate_multimodal_batch,
        num_workers=0
    )
    
    # Initialize the Lightning Module
    model = LitTwoTower(batch_size=32)
    
    # Initialize Trainer (Fast dev run for pipeline verification)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        fast_dev_run=True  # Ensure it runs quickly as a functional test
    )
    
    print("🚀 Starting large-scale training pipeline execution...")
    trainer.fit(model, train_dataloaders=train_loader)
    print("✅ Training pipeline completed successfully.")
    
    # Save the PyTorch state dict for export later
    torch.save(model.model.state_dict(), "two_tower_weights.pt")
    print("💾 Saved base model weights to two_tower_weights.pt")

if __name__ == "__main__":
    run_training()
