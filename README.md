# Variational Autoencoder for Volcanic Ash / Satellite Data Assimilation

## Python Dependencies

All required Python packages are listed in `requirements.txt`. Before installing them, it is good practice to create a **virtual environment**: an isolated space where these packages are installed, so they don't interfere with other Python projects on your computer.

The example below uses [`uv`](https://github.com/astral-sh/uv), a fast, modern tool for managing Python environments and packages. This is just one option, other popular alternatives include `venv` (built into Python) and `conda`. Feel free to use whichever tool you are most comfortable with; the commands below are specific to `uv`, but the same general steps (create an environment, activate it, install dependencies) apply to any of them.

To create and activate an environment with `uv`:

```bash
uv venv my_env
source my_env/bin/activate
```

Then install the dependencies into the environment:

```bash
uv pip install -r requirements.txt
```

## Training Dataset

The training and validation datasets, along with the satellite data used in the assimilation procedure, can be obtained by running the following bash script:

```bash
./fetch_data.sh
```

## Training the Variational Autoencoder

The VAE training procedure is implemented in `VAE_training.py`. Configuration and hyperparameters are defined in separate sections of the `config.ini` file. For example, to train the network using the configuration defined in the `ETNA2018` section, run:

```bash
./VAE_training.py --section ETNA2018
```

For further information, see the help menu:

```bash
./VAE_training.py --help
```

## Training Configuration

An example configuration section from `config.ini` is shown below:

```ini
[ETNA2018]
LEARNING_RATE = 1E-3
NUM_EPOCHS = 400
WARMUP_EPOCHS = 50
LATENT_DIM = 32
MINVAL = 0
MAXVAL = 8
BIN_WEIGHTS = [0.0244, 0.59, 0.542, 0.393, 0.354, 0.799, 1.12, 4.18]
BIN_EDGES = [0, 0.13091, 0.26182, 0.39273, 0.52364, 0.65454, 0.78545, 0.91636, 1.0473]
BETA_MAX = 3E-3
```

### General Hyperparameters

* `LEARNING_RATE`: Learning rate.
* `NUM_EPOCHS`: Total number of training epochs.
* `LATENT_DIM`: Dimensionality of the latent space.

### Beta Variational Autoencoder

A $\beta$-VAE (beta variational autoencoder) modifies the standard VAE loss function by introducing a hyperparameter $\beta$ that weights the KL divergence (Kullback-Leibler divergence) term:

$$ Loss = \text{Reconstruction Loss} + \beta \times \text{KL Divergence} $$

An annealing schedule is used, in which $\beta$ is progressively increased from 0 to `BETA_MAX` over the first `WARMUP_EPOCHS` epochs.

* `BETA_MAX`: Maximum value of $\beta$.
* `WARMUP_EPOCHS`: Number of epochs over which the annealing schedule is applied.

### Normalization

Raw data is normalized according to the transformation:

$$ x = \frac{\operatorname{log1p}(x_{raw}) - \text{MINVAL}}{\text{MAXVAL}} $$

* `MINVAL`: Minimum value used for normalization.
* `MAXVAL`: Scale (range) value used for normalization.

### Weighted Loss Function

The loss function is based on a mean squared error (MSE) function with inverse-frequency pixel weighting applied to the reconstruction term. This weighting is important for handling highly skewed data distributions.

* `BIN_EDGES`: List of bin edges, determined from a histogram spanning the full range of the data.
* `BIN_WEIGHTS`: List of per-bin weights, with length `len(BIN_EDGES) - 1`. If not provided, a standard (unweighted) MSE loss is used instead.
