"""Synthetic fixture, held-fixed Monte Carlo audits, and descriptive controls."""
import math
import random
from .core import VAE, train, integer, number, vector


def ring(n=128, seed=11):
    integer(n, 8, 4096, 'n')
    integer(seed, 0, 2**32-1, 'seed')
    r = random.Random(seed)
    return [[math.cos(2*math.pi*(i%8)/8) + r.gauss(0,0.06),
             math.sin(2*math.pi*(i%8)/8) + r.gauss(0,0.06)] for i in range(n)]


def sliced_w2(a, b, directions=32):
    """Empirical equal-size sliced W2 squared, deterministic angular grid."""
    integer(directions, 1, 1024, 'directions')
    if len(a) != len(b) or not a:
        raise ValueError('need nonempty equal-size point sets')
    a = [vector(x, 2, 'a') for x in a]
    b = [vector(x, 2, 'b') for x in b]
    total = 0.0
    for j in range(directions):
        c, s = math.cos(math.pi*j/directions), math.sin(math.pi*j/directions)
        u = sorted(c*x+s*y for x,y in a)
        v = sorted(c*x+s*y for x,y in b)
        total += sum((x-y)**2 for x,y in zip(u,v))/len(u)
    return total/directions


def sample(model, n=128, seed=31, observation_noise=True):
    integer(n, 1, 4096, 'n')
    integer(seed, 0, 2**32-1, 'seed')
    if type(observation_noise) is not bool:
        raise ValueError('observation_noise must be boolean')
    z_rng, obs_rng = random.Random(seed), random.Random(seed+1)
    out = []
    for _ in range(n):
        y = model.decode([z_rng.gauss(0,1), z_rng.gauss(0,1)])
        if observation_noise:
            y = [v+model.sigma*obs_rng.gauss(0,1) for v in y]
        out.append(y)
    return out


def evaluate(model, data, seed=23, noise_draws=4):
    integer(noise_draws, 1, 32, 'noise_draws')
    integer(seed, 0, 2**32-1, 'seed')
    if not data or len(data)>4096:
        raise ValueError('invalid evaluation data')
    data = [vector(x,2,'data') for x in data]
    r = random.Random(seed)
    nll = kl = mse = 0.0
    reconstructed = []
    for x in data:
        mu, _ = model.encode(x)
        y = model.decode(mu)
        reconstructed.append(y)
        mse += sum((u-v)**2 for u,v in zip(x,y))/(2*len(data))
        for _ in range(noise_draws):
            terms,_ = model.loss_grad(x, [r.gauss(0,1),r.gauss(0,1)])
            nll += terms['nll']/(len(data)*noise_draws)
            kl += terms['kl']/(len(data)*noise_draws)
    generated = sample(model,len(data))
    centers = [(math.cos(2*math.pi*j/8),math.sin(2*math.pi*j/8)) for j in range(8)]
    counts = [0]*8
    outside = 0
    for p in generated:
        ds = [sum((p[i]-c[i])**2 for i in range(2)) for c in centers]
        j = min(range(8),key=ds.__getitem__)
        if ds[j] <= 0.3**2:
            counts[j] += 1
        else:
            outside += 1
    return {'nll_mc': nll, 'kl_analytic': kl, 'negative_elbo_mc': nll+kl,
            'posterior_mean_mse_per_coordinate': mse,
            'prior_sample_sliced_w2_squared': sliced_w2(data,generated),
            'prior_sample_mode_counts_within_0_3': counts,
            'prior_sample_outside_fraction': outside/len(data),
            'reconstructed': reconstructed, 'prior_samples': generated}


def run_demo(steps=800):
    integer(steps,0,10000,'steps')
    data, heldout = ring(128,11), ring(128,17)
    initial = VAE()
    result = {'version':1, 'config': {'hidden':12,'sigma':0.2,'train_n':128,'heldout_n':128,
              'steps':steps,'batch_size':24,'learning_rate':0.006,'model_seed':7,
              'train_seed':19,'train_data_seed':11,'heldout_data_seed':17,
              'evaluation_noise_seed':23,'evaluation_noise_draws':4,'prior_seed':31,
              'prior_observation_noise_seed':32,'sliced_directions':32},
              'heldout':heldout,'initial':evaluate(initial,heldout),'arms':[]}
    for beta in [0.0,1.0]:
        model = VAE()
        history = train(model,data,steps=steps,beta=beta)
        result['arms'].append({'beta':beta,'evaluation':evaluate(model,heldout),
                               'history':history,'checkpoint':model.to_dict()})
    result['gates'] = {'trained_beta1_improves_fixed_mc_negative_elbo':
                          result['arms'][1]['evaluation']['negative_elbo_mc'] < result['initial']['negative_elbo_mc'],
                      'collapsed_origin_rejected_by_distribution_audit':
                          sliced_w2(heldout,[[0,0]]*len(heldout)) > 0.25}
    return result
