import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_output_size(size, kernel_size, stride, padding):
    """Output size of a single Conv2d dimension."""
    return (size + 2 * padding - kernel_size) // stride + 1

def conv_transpose2d_output_padding(in_size, target_size, kernel_size, stride, padding):
    """output_padding needed for ConvTranspose2d to hit an exact target size."""
    base = (in_size - 1) * stride - 2 * padding + kernel_size
    out_pad = target_size - base
    if not (0 <= out_pad < stride):
        raise ValueError(
            f"Invalid output_padding={out_pad} for stride={stride} "
            f"(in_size={in_size}, target_size={target_size})"
        )
    return out_pad

##############
### Models ###
##############
class Autoencoder(nn.Module):
    def __init__(self,
                 latent_dim,
                 in_shape,
                 in_channels=1,
                 channels=(16, 32, 64, 128),
                 kernel_size=3,
                 stride=2,
                 padding=1):
        super().__init__()

        # --- Compute shapes analytically, layer by layer ---
        shapes = []
        h, w = in_shape
        for _ in channels:
            h = conv2d_output_size(h, kernel_size, stride, padding)
            w = conv2d_output_size(w, kernel_size, stride, padding)
            shapes.append((h, w))

        final_c, final_h, final_w = channels[-1], h, w
        flatten_size = final_c * final_h * final_w
        decoder_in_channels  = list(reversed(channels))
        decoder_out_channels = decoder_in_channels[1:] + [in_channels]
        decoder_in_shapes    = list(reversed(shapes))
        decoder_out_shapes   = decoder_in_shapes[1:] + [in_shape]

        # --- Encoder ---
        encoder_layers = []
        c_in = in_channels
        for c_out in channels:
            encoder_layers += [
                nn.Conv2d(c_in, c_out, kernel_size, stride, padding),
                nn.ReLU(True),
            ]
            c_in = c_out
        encoder_layers += [
            nn.Flatten(),
            nn.Linear(flatten_size, latent_dim),
        ]
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Decoder ---
        decoder_layers = [
            nn.Linear(latent_dim, flatten_size),
            nn.ReLU(True),
            nn.Unflatten(1, (final_c, final_h, final_w)),
        ]
        for i, (c_in, c_out) in enumerate(zip(decoder_in_channels, decoder_out_channels)):
#            in_h,  in_w  = decoder_in_shapes[i]
#            out_h, out_w = decoder_out_shapes[i]
#            out_pad_h = conv_transpose2d_output_padding(in_h, out_h, kernel_size, stride, padding)
#            out_pad_w = conv_transpose2d_output_padding(in_w, out_w, kernel_size, stride, padding)

#            decoder_layers.append(
#                nn.ConvTranspose2d(c_in, c_out,
#                                   kernel_size,
#                                   stride,
#                                   padding,
#                                   output_padding=(out_pad_h, out_pad_w))
#            )
            decoder_layers.append(
                nn.Upsample(size=decoder_out_shapes[i], 
                            mode="bilinear", 
                            align_corners=False)
                )
            decoder_layers.append(
                nn.Conv2d(c_in, c_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )                                                                                     
            )
            if i < len(decoder_in_channels) - 1:
                decoder_layers.append(nn.ReLU(True))
            else:
                decoder_layers.append(nn.Softplus())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

class VariationalAutoencoder(nn.Module):
    def __init__(self, 
                 latent_dim, 
                 in_shape, 
                 in_channels=1,
                 channels=(16, 32, 64), 
                 kernel_size=3, 
                 stride=2, 
                 padding=1):
        super().__init__()

        # --- Compute shapes analytically, layer by layer ---
        shapes = []
        h, w = in_shape
        for _ in channels:
            h = conv2d_output_size(h, kernel_size, stride, padding)
            w = conv2d_output_size(w, kernel_size, stride, padding)
            shapes.append((h, w))

        final_c, final_h, final_w = channels[-1], h, w
        flatten_size = final_c * final_h * final_w
        decoder_in_channels  = list(reversed(channels))
        decoder_out_channels = decoder_in_channels[1:] + [in_channels]
        decoder_in_shapes    = list(reversed(shapes))
        decoder_out_shapes   = decoder_in_shapes[1:] + [in_shape]

        # --- Encoder ---
        encoder_layers = []
        c_in = in_channels
        for c_out in channels:
            encoder_layers += [
                nn.Conv2d(c_in, c_out, kernel_size, stride, padding),
                nn.ReLU(True),
            ]
            c_in = c_out
        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu     = nn.Linear(flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(flatten_size, latent_dim)

        # --- Decoder ---
        decoder_layers = [
            nn.Linear(latent_dim, flatten_size),
            nn.ReLU(True),
            nn.Unflatten(1, (final_c, final_h, final_w)),
        ]
        for c_in, c_out, layer_in_shape, layer_out_shape in zip(decoder_in_channels,
                                                    decoder_out_channels,
                                                    decoder_in_shapes,
                                                    decoder_out_shapes):
            in_h,  in_w  = layer_in_shape
            out_h, out_w = layer_out_shape
            out_pad_h = conv_transpose2d_output_padding(in_h, out_h, kernel_size, stride, padding)
            out_pad_w = conv_transpose2d_output_padding(in_w, out_w, kernel_size, stride, padding)

            decoder_layers.append(
                nn.ConvTranspose2d(c_in, c_out, 
                                   kernel_size, 
                                   stride, 
                                   padding,
                                   output_padding=(out_pad_h, out_pad_w))
            )
            decoder_layers.append(nn.ReLU(True))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

##################
### Criterions ###
##################
class AELoss(nn.Module):
    """
    Reconstruction loss for a standard Autoencoder.

    Args:
        reduction (str): 'mean' or 'sum'
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        if reduction not in {'mean', 'sum'}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, recon_x, x):
        return F.mse_loss(recon_x, x, reduction=self.reduction)


class VAELoss(nn.Module):
    """
    Loss function for a Variational Autoencoder (VAE).

    Args:
        beta (float): weight for KL divergence (β-VAE)
        reduction (str): 'mean' or 'sum'
    """
    def __init__(self, beta=1.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def reconstruction_loss(self, recon_x, x):
        return F.mse_loss(recon_x, x, reduction=self.reduction)

    def kl_divergence(self, mu, logvar):
        # KL Divergence: D_KL(N(mu, σ) || N(0, I))
        # = -0.5 * sum(1 + log(σ^2) - μ^2 - σ^2)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        if self.reduction == 'mean':
            kl = kl.mean()
        else:
            kl = kl.sum()

        return kl

    def forward(self, recon_x, x, mu, logvar):
        recon = self.reconstruction_loss(recon_x, x)
        kl = self.kl_divergence(mu, logvar)
        loss = recon + self.beta * kl
        return loss, recon, kl
