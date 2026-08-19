#!/usr/bin/env python
# coding: utf-8

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

import xarray as xr
import pandas as pd

from dataset import EnsembleDataset, LogMinMaxScale
from model   import VariationalAutoencoder, VAELoss

## Training function
def train_epoch(model, loader, criterion, optimizer, device):
    # Set training mode
    model.train()

    total_loss  = 0.0
    total_recon = 0.0
    total_kl    = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        # Model prediction
        prediction, mu, logvar = model(batch)
        # Compute loss
        loss, recon, kl = criterion(prediction,batch,mu,logvar)

        # Update weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss  += loss.item()
        total_recon += recon.item()
        total_kl    += kl.item()
        num_batches += 1
    return {'loss': total_loss/num_batches, 
            'recon': total_recon/num_batches,
            'kl': total_kl/num_batches}

## Evaluation function
def evaluate_epoch(model, loader, criterion, device):
    # Set inference mode
    model.eval()

    total_loss  = 0.0
    total_recon = 0.0
    total_kl    = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # Model prediction
            prediction, mu, logvar = model(batch)
            # Compute loss
            loss, recon, kl = criterion(prediction,batch,mu,logvar)

            # Update metrics
            total_loss  += loss.item()
            total_recon += recon.item()
            total_kl    += kl.item()
            num_batches += 1
    return {'loss': total_loss/num_batches, 
            'recon': total_recon/num_batches,
            'kl': total_kl/num_batches}

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    fname_train = 'data/helens64.ens.nc'
    fname_val   = 'data/helens48.ens.nc'

    ## 1. Loading raw data and normaliation
    ds1 = xr.open_dataset(fname_train)
    da1 = ds1["tephra_grn_load"]
    ds2 = xr.open_dataset(fname_val)
    da2 = ds2["tephra_grn_load"]

    H = da1.sizes['lat']
    W = da1.sizes['lon']

    ## 2. Create a custom Dataset
    min_value = 0
    max_value = 6
    transform = LogMinMaxScale(min_value, max_value)

    train_dataset = EnsembleDataset(da1, transform)
    val_dataset   = EnsembleDataset(da2, transform)

    ## 3. Create a DataLoader
    train_loader = DataLoader(train_dataset, 
                              batch_size=config['BATCH_SIZE'], 
                              shuffle=True)
    val_loader   = DataLoader(val_dataset,
                              batch_size=config['BATCH_SIZE'],
                              shuffle=False)

    ## 4. Define a model
    model = VariationalAutoencoder(config['LATENT_DIM'], in_shape = (H,W))
    model = model.to(device)
#    summary(model, (1,H,W))

    ## 5. Loss function
    criterion = VAELoss(beta=config['BETA'])

    ## 6. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])

    ## Training loop
    train_losses = []
    val_losses   = []

    best_val_loss = float('inf')
    best_state    = None
    best_epoch    = -1

    for epoch in range(config['NUM_EPOCHS']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = evaluate_epoch(model, val_loader, criterion, device)

        # Store current losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Store best state
        if val_loss['loss'] < best_val_loss:
            best_val_loss = val_loss['loss']
            best_epoch    = epoch
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch%10 == 0 or epoch == config['NUM_EPOCHS']-1:
            print(f"-> Epoch {epoch+1:02d} \n"
                  f"   Train loss {train_loss} \n"
                  f"   Validation loss: {val_loss}")
    print(f"Done! Best epoch: {best_epoch+1} (val_loss={best_val_loss:.4f})")

    ## Save metrics
    df_train = pd.DataFrame(train_losses)
    df_val   = pd.DataFrame(val_losses)
    df_train.to_csv("training_loss.csv", index=False)
    df_val.to_csv("validation_loss.csv", index=False)

    ## Save trained model
    torch.save({
        'model_state_dict': best_state,                # Trained Model parameters
        'LATENT_DIM': config['LATENT_DIM'],            # Dimension of the latent space (=2)
        'MINVAL': min_value,                           # Min value used for normalization (=0)
        'MAXVAL': max_value,                           # Max value used for normalization (=20)
        'EPOCH':  best_epoch,
        'LOSS':  best_val_loss, 
        'IN_SHAPE': (H, W)
        }, config['FNAME_MODEL'])

if __name__ == "__main__":

    ### Configuration ###
    config = {
        'BATCH_SIZE':    16,
        'LATENT_DIM':    32,
        'LEARNING_RATE': 2E-4,
        'NUM_EPOCHS':    100,
        'BETA':          1,
        'FNAME_MODEL':   'vae.pt',
        }

    main(config)
