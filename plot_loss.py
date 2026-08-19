import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv("training_loss.csv")
df_val   = pd.read_csv("validation_loss.csv")

fig, axs = plt.subplots(nrows=2, sharex=True)

axs[0].set(ylabel='Reconstruction loss', yscale='log')
axs[1].set(ylabel='KL divergence', xlabel='Epoch')

df_train.plot(y='recon', 
              label='Training', 
              ax=axs[0])

df_val.plot(y='recon',
            label='Validation',
            ax=axs[0])

df_train.plot(y='kl',
            label='Training',
            ax=axs[1])

df_val.plot(y='kl',
            label='Validation',
            ax=axs[1])

plt.savefig("loss.png")
