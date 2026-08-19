#!/usr/bin/env python
# coding: utf-8

import torch
from model import VariationalAutoencoder
from dataset import LogMinMaxScale
import xarray as xr

## Reconstruct samples from a dataset/loader
@torch.no_grad()
def reconstruct_samples(model, loader, transform=None, device='cpu'):
    model.eval()
    reconstructions = []
    targets = []

    for batch in loader:
        batch = batch.to(device)

        recon = model(batch)

        if transform is not None:
            recon = transform.invert(recon)
            batch = transform.invert(batch)

        reconstructions.append(recon.detach().cpu())
        targets.append(batch.detach().cpu())

    reconstructions = torch.cat(reconstructions, dim=0).squeeze(1).numpy()
    targets = torch.cat(targets, dim=0).squeeze(1).numpy()
    return reconstructions, targets

def save_reconstructions(fname_out, x_recon, x_true=None):
    data_vars = {
        "reconstruction": (("ens", "lat", "lon"), x_recon),
    }
    if x_true is not None:
        data_vars["target"] = (("ens", "lat", "lon"), x_true)

    ds = xr.Dataset(
        data_vars,
        coords={"ens": range(x_recon.shape[0])},
    )
    ds.to_netcdf(fname_out)

## Generate nens new samples
def generate_samples(latent_dim, model, transform, nens = 2048):
    z = torch.randn(nens, latent_dim)
    with torch.no_grad():
        new_sample = model.decode(z)
        x = transform.invert(new_sample).squeeze()
    return x.numpy()

def save_samples(x, fname_out):
    da = xr.DataArray(
        x,
        dims=("ens", "lat", "lon"),
        coords={"ens": range(x.shape[0])},
        name="tephra_grn_load",  # match your variable name from training
    )
    da.to_netcdf(fname_out)

def main(checkpoint, fname_out):

    ## Normalization
    min_value = checkpoint['MINVAL']
    max_value = checkpoint['MAXVAL']
    transform = LogMinMaxScale(min_value, max_value)

    # Model
    model = VariationalAutoencoder(
                checkpoint['LATENT_DIM'], 
                in_shape=checkpoint['IN_SHAPE']
                )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    ## Generate new samples
    x = generate_samples(checkpoint['LATENT_DIM'], model, transform, nens = 12)

    ## Save to disk
    save_samples(x, fname_out)
    print(f"Saved {x.shape[0]} samples to {fname_out}")

if __name__ == "__main__":
    ## Load weight parameters and some metadata
    fname = 'vae.pt'
    checkpoint = torch.load(fname, map_location='cpu')

    main(checkpoint, fname_out='generated_samples.nc')
