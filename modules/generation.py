#!/usr/bin/env python
# coding: utf-8

import torch
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

@torch.no_grad()
def reconstruct_vae_samples(model, loader, transform=None, device='cpu'):
    model.eval()
    reconstructions = []
    targets = []

    for batch in loader:
        batch = batch.to(device)

        recon, _, _ = model(batch)

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

def save_samples_with_mean(x_generated_raw, z, lat, lon, fname):
    # 1. Create DataArray with explicit coordinates for lat and lon
    da = xr.DataArray(
        x_generated_raw.numpy(),
        dims=("ens", "lat", "lon"),
        coords={
            "lat": lat,
            "lon": lon,
        },
        name="samples",
    )

    da_z = xr.DataArray(
        z.numpy(),
        dims=("ens", "latent_dim"),
        name="z",
    )

    # 2. Compute the ensemble mean (coordinates for lat/lon are automatically preserved)
    da_mean = da.mean(dim="ens").rename("mean")
    
    # 3. Merge both into an xr.Dataset and save to NetCDF
    ds = xr.Dataset(
        {
            da.name: da,
            da_mean.name: da_mean,
            da_z.name: da_z,
        }
    )
    
    # Optional: Add metadata attributes to latitude and longitude
    ds["lat"].attrs = {"units": "degrees_north", "standard_name": "latitude"}
    ds["lon"].attrs = {"units": "degrees_east", "standard_name": "longitude"}
    
    ds.to_netcdf(fname)