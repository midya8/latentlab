# Gaussian VAE objective and derivative audit

## Explicit model

Observation x and latent z both have dimension2. Each network has a tanh hidden layer, with default width12. Dense layers use row-major weights followed by each row bias. Encoder output has four coordinates: two posterior means followed by two raw logvariances. The final decoder layer is linear. All output likelihood variances are fixed, not learned.

For encoder outputs mu, raw, define

```
t = tanh(raw / 4)
ell = 4 t
s = exp(ell / 2)
z = mu + s * epsilon,  epsilon ~ N(0,I)
y = decoder(z)
p(z) = N(0,I)
q(z|x) = N(mu, diag(exp(ell)))
p(x|z) = N(y, sigma_obs² I)
```

Smooth bounded logvariance restricts this posterior family to ell in(-4,4). Its derivative is `dell/draw = 1-t²`, not1 and not a hard-clipping mask. Sampling and all derivatives use this same family.

For D=2, fixed observation sigma and one fixed-noise sample, the minimized objective is

```
NLL = ||y-x||² / (2 sigma_obs²) + log(2 pi sigma_obs²)
KL  = 1/2 sum_j [mu_j² + exp(ell_j) - 1 - ell_j]
J_beta = NLL + beta KL
```

At beta1, the expectation of `J_beta` over posterior noise is the negative ELBO. This follows by adding and subtracting log q in the evidence identity; the gap to negative log evidence is posterior KL. The loss evaluated at a finite set of noise draws is an estimator of that expectation. It need not bound negative log evidence on each random realization. For beta other than1, the implemented weighted objective is not labeled the ordinary ELBO. No actual marginal likelihood is computed.

## Reverse pass

With g_z obtained by differentiating Gaussian reconstruction loss through the decoder,

```
g_y   = (y-x) / sigma_obs²
g_mu  = g_z + beta mu
g_ell = 0.5 g_z * epsilon * exp(ell/2) + 0.5 beta (exp(ell)-1)
g_raw = g_ell * (1-t²)
```

The encoder output upstream vector is `concat(g_mu,g_raw)`. Every tanh layer applies `(1-output²)`. The same gradient supports beta0, beta1 and disclosed weighted objectives. Constants in Gaussian NLL do not affect weights, but do affect reported likelihood/ELBO numbers and are therefore retained.

The independent test objective reconstructs the full flat parameter vector using a separate cursor-based dense function, derives posterior samples and scalar loss without calling production forward/loss helpers, and central-differences every parameter. Nine whole-gradient settings cover hidden widths2,3,12 and beta0,1,4. A second check averages two independent fixed-noise/example pairs. The test is about differentiating a fixed-noise estimator, not an empirical proof of unbiasedness for all functions. Analytic reparameterization justifies the pathwise estimator under the standard differentiation/interchange conditions; these bounded tanh networks, finite parameters and Gaussian noise give finite relevant moments.

## Training and audit are separate

Training draws data with replacement and fresh independent two-coordinate Gaussian noise per example. Adam maintains first/second moment estimates and applies explicit bias correction. Invalid input data is rejected before any update. Steps0 returns an unchanged checkpoint. Shared training seeds across beta arms preserve data/noise streams, but the parameter trajectories differ; this is a matched fixture comparison, not an isolation of every mechanism or proof about all optimizers.

Evaluation consumes its own fixed noise seed and four draws per held-out point. NLL is averaged across points/draws; KL is analytic per point. Mean reconstruction MSE is computed separately from `decoder(mu(x))`. Prior generation uses z drawn from a standard normal, then adds independent sigma_obs Gaussian observation noise. Separate streams keep latent draws aligned if observation noise is disabled. Plot panels explicitly show prior noisy samples versus posterior-mean reconstructions.

Sliced W2 squared sorts projections of two equally sized empirical point clouds and averages squared order-statistic differences. Directions are the32 deterministic angles `pi*j/32`. This is an empirical angular approximation, not full Wasserstein, a statistical significance test or a population-density guarantee. The mode diagnostic assigns a sample to its nearest one of eight known centers only when distance<=0.3; otherwise it is outside. This leverages the synthetic generating mechanism and is not a general support estimator.

## Failure witness and limits

The beta0 arm has slightly better deterministic mean reconstruction MSE than beta1, while its analytic KL and prior-sample sliced distance are worse. Even beta1 has51.56% of generated noisy observations outside the chosen radius0.3 regions. All these are measured on the default single-seed run. The128 data points contain Gaussian jitter0.06 around eight equally represented unit-circle centers. The fixed model observation noise0.2 is wider, so the diagnostic also reflects likelihood-family mismatch. No claim that KL weighting alone is universally beneficial, that training converged, or that posterior collapse was established.

The origin-cloud control passes shape/finite validation but fails the disclosed sliced-distance threshold0.25. The zero-update demo fails its improvement gate. Tests exercise both success and failure, not always-true validation labels.

## Provenance

Kingma & Welling, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), Sections2-3 and AppendicesB-C, supplies the standard reparameterization and Gaussian KL formulas. The bounded-logvariance choice, tiny ring fixture, explicit scalar code, independent tests, matched control comparison, JSON checkpoint and measured figure are this repository's implementation choices. No paper benchmark numbers, original algorithm claim or copied code. Python standard library only at runtime.
