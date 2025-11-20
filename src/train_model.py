import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
from data import ChordDataModule
from model import ChordTransformer
list

def train_model():

    # Initialize data module
    dm = ChordDataModule(
        batch_size=64,
        max_seq_len=256,
        min_freq=5,
        num_workers=4
    )
    
    # Setup data
    dm.setup()
    
    # Initialize model
    model = ChordTransformer(
        vocab_size=len(dm.vocab),
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        learning_rate=1e-4
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator='gpu',
        devices=2,
        precision='16-mixed',  # Use mixed precision for faster training
        gradient_clip_val=1.0,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                filename='chord-transformer-{epoch:02d}-{val_loss:.2f}'
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ],
        logger=pl.loggers.TensorBoardLogger('logs/', name='chord_transformer')
    )
    
    # Train
    trainer.fit(model, dm)
    
    # Test
    trainer.test(model, dm)
    
    # Generate some chord progressions
    print("\n=== Generated Chord Progressions ===")
    for i in range(5):
        progression = model.generate(dm.vocab, max_len=50, temperature=0.9)
        print(f"{i+1}. {progression}")

# Training script
if __name__ == '__main__':
    train_model()