"""LatentLab: an inspectable finite Gaussian variational autoencoder."""
from .core import VAE, train
from .demo import run_demo, evaluate, sample, sliced_w2
__all__ = ['VAE','train','run_demo','evaluate','sample','sliced_w2']
