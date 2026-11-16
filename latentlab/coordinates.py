"""Signed permutations of the two latent axes, not arbitrary basis changes."""
from .core import VAE


def _plan(model, order, signs):
    if not isinstance(model, VAE):
        raise ValueError('model must be a VAE')
    # Revalidate the public checkpoint before deriving offsets or copying it.
    model = VAE.from_dict(model.to_dict())
    if (not isinstance(order, (list, tuple)) or len(order) != 2 or
            any(type(v) is not int for v in order) or sorted(order) != [0, 1]):
        raise ValueError('order must be an integer permutation of [0,1]')
    if (not isinstance(signs, (list, tuple)) or len(signs) != 2 or
            any(type(v) is not int or v not in (-1, 1) for v in signs)):
        raise ValueError('signs must contain two integer -1 or 1 values')
    indices = list(range(len(model.params)))
    multipliers = [1] * len(indices)
    off, width, _ = model.layers[1]
    # mu changes sign and order; diagonal raw logvariance changes order only.
    for new, old in enumerate(order):
        for i in range(width + 1):
            indices[off + new*(width+1) + i] = off + old*(width+1) + i
            multipliers[off + new*(width+1) + i] = signs[new]
            indices[off + (2+new)*(width+1) + i] = off + (2+old)*(width+1) + i
    off, width, count = model.layers[2]
    for j in range(count):
        for new, old in enumerate(order):
            indices[off + j*(width+1) + new] = off + j*(width+1) + old
            multipliers[off + j*(width+1) + new] = signs[new]
    return model, indices, multipliers


def relabel_latents(model, order, signs):
    """Return a detached model under z'_j = signs[j]*z[order[j]].

    Explicit latent/noise draws must be transformed by the same rule when
    comparing realizations. Identical RNG seeds do not supply that coupling.
    This changes checkpoint coordinates, not the represented distribution.
    """
    other, indices, multipliers = _plan(model, order, signs)
    other.params = [other.params[i]*s for i, s in zip(indices, multipliers)]
    return other
