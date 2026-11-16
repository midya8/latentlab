# Signed latent coordinates and coupled noise

`relabel_latents(model, order, signs)` returns a detached VAE checkpoint with the two latent axes permuted and optionally sign-flipped. It does not retrain or mutate the input model. This is a coordinate transformation of the same represented observation distribution, not a new generative model or a quality improvement.

The finite tests compare coupled computations at tolerance `2e-10`. They are consistency checks, not an independent gradient/evidence oracle or a bitwise cross-platform guarantee.

## Parameter mapping

Write `z'_j = s_j z_(order[j])`, with `order` a permutation of `[0,1]` and `s_j` either -1 or 1. There are eight maps.

The encoder mean output rows change order and sign. Its raw diagonal-logvariance rows change order **without** sign flips. The elementwise bounded-logvariance map commutes with permutations, so the transformed posterior variance is the reordered original variance. Applying signs to logvariance would change the model, not rename axes.

For every decoder hidden unit, its two input weights change order and sign in the same way. Its bias is unchanged. Encoder hidden and decoder output layers are unchanged. Because signed permutation matrices are orthogonal, the standard isotropic Gaussian prior is invariant and has absolute Jacobian determinant one. Diagonal Gaussian posteriors remain diagonal under these maps.

This is not an arbitrary orthogonal change of latent basis. A general rotation of an unequal diagonal posterior covariance is no longer diagonal. The code does not add a full-covariance posterior or claim arbitrary basis invariance.

The resulting flat parameter transformation is a signed permutation. By the chain rule, the full parameter-gradient vector follows that same signed permutation when the explicit latent noise is coupled as below. Tests reconstruct that mapping by separate layer-block unpack/repack code rather than calling the production index plan.

## The epsilon coupling matters

For a held-fixed explicit Gaussian draw,

```text
z = mu(x) + exp(logvar(x)/2) * epsilon.
```

Matching the same realized latent point in new coordinates requires

```text
epsilon'_j = s_j epsilon_(order[j]).
```

Then `z'=S z`, the decoded observation mean is unchanged, and reconstruction loss, KL, objective and negative-ELBO estimator agree up to arithmetic rounding. All proposal/joint densities, importance log weights and cloud summaries agree too, while the latent/proposal/noise coordinate fields follow the map.

The same untransformed epsilon vector generally produces a **different realization** after renaming latent axes. The same RNG seed is therefore not a sample-by-sample invariance guarantee. The distribution of fresh isotropic noise is unchanged, but that distributional fact does not mean an identical finite pseudo-random stream is already coupled. A negative control explicitly shows an objective difference when noise is left unchanged. A separate one-sided parameter change also breaks the model function.

The stochastic Adam consistency test keeps the same sampled data indices and transforms each supplied epsilon before the mapped model sees it. Signed permutations also map Adam's first moments, while second moments are permuted without signs. Eight short five-update tests compare losses and final parameters under that explicit coupling. The public `train` interface and its random streams remain unchanged; no same-seed promise is added to it.

## Use

```python
from latentlab import VAE, relabel_latents

model = VAE(hidden=3, seed=72)
other = relabel_latents(model, order=[1,0], signs=[-1,1])
x, epsilon = [.8,-.1], [.5,-1.2]
mapped_epsilon = [1.2,.5]
a, _ = model.loss_grad(x, epsilon)
b, _ = other.loss_grad(x, mapped_epsilon)
assert abs(a['objective']-b['objective']) < 1e-10
```

`order` and `signs` must each be a list or tuple of two actual integers. Bools, repeated/out-of-range axes, zero signs, floats and strings are rejected. The model's checkpoint is revalidated before a detached copy is made. Nonfinite or malformed model parameters fail rather than producing an apparently valid new checkpoint. The original model and global RNG state do not change.

The returned object uses the existing checkpoint schema. It carries no extra mapping metadata, so keep the chosen order/signs if later explicit latent draws or gradient vectors need to be compared. No new CLI or checkpoint version is added.

## Verification and limits

Six new tests cover:

- All eight signed maps on both archived trained checkpoints and twelve generated models, hidden widths one, two, three and twelve, three observation/noise pairs and beta values 0, .3, 1 and 4. Every parameter, mean/logvariance coordinate, decoder output, loss term and gradient coordinate is checked.
- Thirty-six complete importance audits, including every proposal/latent/noise coordinate, density, log weight and summary field, at prefixes one, four and sixteen.
- Inverse recovery and all 64 pairwise map compositions, detached checkpoints, input isolation and unchanged global RNG state.
- Eight explicitly coupled five-update Adam runs, untransformed-noise and one-sided-parameter negative controls, strict map validation and the maximum hidden width 64.

All 47 earlier tests remain. The 53-test full installed-wheel verifier runs every test outside checkout, checks all six package origins before and after, compares three complete module CLI reports and regenerates both original figures. Model arithmetic, training code, checkpoint values, both saved demos and both SVG files are unchanged. These checks do not add independent evidence truth, disentanglement, latent identifiability, statistical calibration or generation-quality claims.

Signed coordinate permutations use the existing Gaussian prior/posterior and reparameterization equations documented in [DESIGN.md](DESIGN.md); the VAE methods remain attributed to Kingma and Welling there. This is a small exact parameter-index transformation with finite floating-point consistency checks, not a new symmetry theorem or a general neural-network equivalence checker.
