# Importance evidence audit: expectation is not a realization

The VAE's reconstruction loss is not marginal likelihood. This extension evaluates proper importance weights, shows what is and is not bounded, and supplies references where the true answer is known. It does not add IWAE training or claim an exact neural evidence value.

## Log-space computation

For fixed observation x, full-support proposal q(z|x), and independent z_i drawn from q,

```
log w_i = log p(z_i) + log p(x|z_i) - log q(z_i|x)
M_K = mean_i w_i
log M_K = max(log w) + log(mean(exp(log w-max(log w))))
L_K = E[log M_K]
```

Evidence p(x) is a density for continuous x, so its logarithm may be positive. With the actual normalized joint and valid full-support IID draws, `E[M_K] = p(x)` and Jensen gives `L_K <= log p(x)`. The standard IWAE result gives nondecreasing **expectation bounds**, `L_K <= L_(K+1)`. It does not say that a nested finite noise prefix has an increasing estimate, that every log M_K is a lower bound, or that a larger observed estimate proves a better model. No asymptotic convergence-rate or finite-sample accuracy claim is made here.

For each realized sample set, Jensen also gives

```
log mean(w) >= mean(log w)
```

This sample-wise inequality compares two quantities from the **same weights**. The ordinary VAE's analytic-KL-plus-sampled-NLL estimator is different sample by sample from `-mean(log w)`. They agree in expectation under the correct proposal, not on each finite set. `audit_importance` returns both, with distinct names.

Normalized weights, ESS `1/sum(normalized_weight²)`, largest weight, and the count of numerically zero normalized weights describe the observed cloud. Log-space arithmetic prevents overflow from direct exponentiation; very small relative weights can still underflow to numerical zero. This is reported rather than interpreted as missing mathematical support.

## Inspectable API and CLI

```sh
python -m latentlab importance-demo > /tmp/importance.json
python -m latentlab importance examples/importance-input.json > /tmp/anchor.json
python examples/render_importance.py --check
```

The input has exactly `version`, `checkpoint`, `x`, and `epsilon` fields. Version must be integer 1, not a boolean. Checkpoint import reuses the VAE's strict schema. Observation and each epsilon are finite two-coordinate vectors; 1-1,024 draws are supported. Input is bounded to one MB. This command emits a complete weight witness without changing weights or inventing random noise. Supplying epsilon values alone does not establish that they were IID standard normal draws.

The lower-level `gaussian_importance(log_joint,mu,logvar,epsilons)` accepts a normalized joint-log-density callback. It cannot verify whether a callback omitted a constant or whether supplied noise is correctly sampled. Its Gaussian proposal logvariances must lie in [-40,40]. `summarize_log_weights` supports finite log weights in [-1e6,1e6]. No posterior normalization is silently dropped.

## Neural prefix fixture, no known evidence target

The original 800-update model training, observations and checkpoints are unchanged. We audit anchor x=(1,0) for the untrained model and the original beta 0/beta 1 models. All use prefixes of 64 IID standard normal draws from independent local RNG seed 107. Sample counts are 1, 4, 16, 64. Model parameters are frozen during the audit.

For trained beta 1:

| K | log mean w | mean log w | ESS |
|---:|---:|---:|---:|
| 1 | -2.0464 | -2.0464 | 1.00 |
| 4 | 0.7106 | -0.6811 | 1.18 |
| 16 | -0.2701 | -1.2470 | 2.51 |
| 64 | -0.9271 | -1.4323 | 9.84 |

The random estimate rises then falls. Every row satisfies the same-sample Jensen inequality within floating-point tolerance. There is no true neural evidence value plotted, and no claim that 0.7106 is an actual lower bound. Complete prior/joint/proposal log densities, latent samples, epsilon, normalized weights and diagnostics for all three models are in [examples/importance-demo.json](examples/importance-demo.json).

## Independent linear-Gaussian reference

For a separate diagonal model, `z~N(0,I)` and `x|z~N(a*z+b,sigma²I)`. Its evidence is `N(x;b,diag(a²+sigma²))`. Coordinate posterior mean is `a(x-b)/(a²+sigma²)`, posterior variance is `sigma²/(a²+sigma²)`. These are not claims that a nonlinear tanh decoder has a Gaussian exact posterior.

The fixture uses x=(0.8,-0.3), a=(1.2,-0.7), b=(0.1,0.2), sigma=0.3. True log evidence is **-2.15379530672424**. Using the exact posterior as proposal makes every importance weight equal p(x), for any epsilon values; the 64-draw estimate matches with ESS 64. Expected negative ELBO plus log evidence equals KL(q||true posterior), with explicit residual returned. Independent scalar Simpson integrals of prior-times-likelihood confirm evidence for four different linear model settings, including zero and negative scales. They are tests of this analytic reference, not numerical evidence for the nonlinear VAE.

## High ESS, wrong evidence, common event

A two-atom example uses joint masses `(0.000001,0.999999)`, so total evidence is 1 and true log evidence is 0. Proposal probabilities are `(0.999,0.001)`, so support is correct but badly matched. A common-atom importance weight is `0.000001/0.999 = 1/999000`.

On the all-common event with64 independent draws:

- ESS is 64, its maximum possible value.
- Log evidence estimate is **-13.81451005763069**, rather than 0.
- The evidence estimate is too small by factor **999,000**.
- The event probability is `(999/1000)^64 = 93.79749638258456%`.

This is not a seed-picked accident or a zero-support violation. The missed rare atom carries nearly all joint mass. Relative weight balance says nothing about that unseen mass.

`rare_atom_reference(K)` enumerates all K+1 possible rare-atom counts. Binomial probabilities, their sum, expected evidence estimate and the all-common event probability use exact rational arithmetic. Log values and their probability-weighted sum use floating point. For K=1,4,16,64, expected log estimates are about -13.793788,-13.737278,-13.529377,-12.785808: increasing and below 0, yet still far away. Mean evidence estimate is exactly 1 for every K. The rare large weights restore unbiasedness, not accuracy of the typical finite run.

## Verification and preservation

The original 25 tests remain. Fifteen importance tests cover independent normal density products, extreme log weights/underflow, translation/permutation, one/constant-weight limits, exhaustive IID binary sequences through K=8, exact rational rare-atom expectations, analytic evidence checked by independent quadrature, constant exact-posterior weights, posterior-KL identity, direct VAE joint/proposal ratios, analytic-KL-versus-sampled-log distinction, no mutation/global RNG effects, strict validation, CLI failures and the actual nonmonotone neural prefix fixture. Three more figure tests check all 48 neural table cells and coordinates, exact controls, byte reproduction and full witness metadata, including an undisplayed joint-log-density change. Total: 43 tests.

Original `demo` output and original figure are unchanged byte-for-byte. The five-module v0.2 wheel runs both demos outside checkout. CI compares installed/source outputs as complete JSON objects and checks both figures.

## Attribution and limits

Importance weights and the expectation hierarchy follow Burda, Grosse & Salakhutdinov, [Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519), Section 3, Equations 8-10 and Theorem 1/Appendix A. The standard Gaussian VAE still credits Kingma & Welling. The scalar audit, exact finite failure fixture, independent tests, CLI and measured display are original code, not a new importance estimator, IWAE optimizer or reproduction of paper experiments. No confidence interval, effective-sample-size accuracy certificate, nonlinear exact evidence, monotone realized estimate, resampling-based posterior sampler or high-dimensional reliability claim.
