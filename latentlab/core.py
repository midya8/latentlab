"""Small Gaussian VAE with explicit pathwise gradients, no array library."""
import math
import random


def number(x, name):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(name + ' must be finite numeric')
    return float(x)


def integer(x, low, high, name):
    if type(x) is not int or not low <= x <= high:
        raise ValueError(name + ' outside supported integer range')
    return x


def vector(x, size, name):
    if not isinstance(x, (list, tuple)) or len(x) != size:
        raise ValueError(name + ' has wrong shape')
    return [number(v, name) for v in x]


class VAE:
    """2-D observations/latents, one tanh hidden layer each way.

    Diagonal posterior logvariance is smoothly bounded by 4*tanh(raw/4).
    The observation likelihood is N(decoder(z), sigma**2 I), fixed sigma.
    Parameter ordering: encoder hidden, encoder output (mu then raw logvar),
    decoder hidden, decoder output. Each dense layer stores row weights/bias.
    """
    def __init__(self, hidden=12, sigma=0.2, seed=7):
        self.hidden = integer(hidden, 1, 64, 'hidden')
        self.sigma = number(sigma, 'sigma')
        if not 0.01 <= self.sigma <= 10:
            raise ValueError('sigma outside [0.01,10]')
        integer(seed, 0, 2**32 - 1, 'seed')
        rng = random.Random(seed)
        self.layers = []
        self.params = []
        for ni, no in [(2, hidden), (hidden, 4), (2, hidden), (hidden, 2)]:
            self.layers.append((len(self.params), ni, no))
            for _ in range(no):
                self.params.extend(rng.gauss(0, 1 / math.sqrt(ni)) for _ in range(ni))
                self.params.append(0.0)

    def _dense(self, x, layer, tanh):
        offset, ni, no = self.layers[layer]
        y = []
        for j in range(no):
            k = offset + j * (ni + 1)
            a = self.params[k + ni] + sum(self.params[k + i] * x[i] for i in range(ni))
            y.append(math.tanh(a) if tanh else a)
        return y

    def _back(self, x, y, gy, layer, tanh, grad):
        offset, ni, no = self.layers[layer]
        gx = [0.0] * ni
        for j in range(no):
            k = offset + j * (ni + 1)
            d = gy[j] * (1 - y[j]**2) if tanh else gy[j]
            for i in range(ni):
                grad[k + i] += d * x[i]
                gx[i] += self.params[k + i] * d
            grad[k + ni] += d
        return gx

    def encode(self, x):
        x = vector(x, 2, 'x')
        a = self._dense(x, 0, True)
        out = self._dense(a, 1, False)
        return out[:2], [4 * math.tanh(v / 4) for v in out[2:]]

    def decode(self, z):
        z = vector(z, 2, 'z')
        return self._dense(self._dense(z, 2, True), 3, False)

    def loss_grad(self, x, epsilon, beta=1.0):
        """One-noise negative-ELBO estimator when beta=1; weighted otherwise.

        Noise is an explicit held-fixed argument for independent derivatives.
        KL is analytic, not sampled. Returns scalar terms and a flat gradient.
        """
        x = vector(x, 2, 'x')
        epsilon = vector(epsilon, 2, 'epsilon')
        beta = number(beta, 'beta')
        if not 0 <= beta <= 100:
            raise ValueError('beta outside [0,100]')
        a = self._dense(x, 0, True)
        out = self._dense(a, 1, False)
        mu = out[:2]
        t = [math.tanh(v / 4) for v in out[2:]]
        lv = [4 * v for v in t]
        sd = [math.exp(v / 2) for v in lv]
        z = [mu[i] + sd[i] * epsilon[i] for i in range(2)]
        b = self._dense(z, 2, True)
        y = self._dense(b, 3, False)
        s2 = self.sigma**2
        nll = sum((y[i] - x[i])**2 for i in range(2)) / (2*s2) + math.log(2*math.pi*s2)
        kl = sum((mu[i]**2 + math.exp(lv[i]) - 1 - lv[i]) / 2 for i in range(2))
        grad = [0.0] * len(self.params)
        gb = self._back(b, y, [(y[i] - x[i]) / s2 for i in range(2)], 3, False, grad)
        gz = self._back(z, b, gb, 2, True, grad)
        gout = [gz[i] + beta*mu[i] for i in range(2)]
        gout += [(gz[i]*epsilon[i]*sd[i]/2 + beta*(math.exp(lv[i])-1)/2)*(1-t[i]**2)
                 for i in range(2)]
        ga = self._back(a, out, gout, 1, False, grad)
        self._back(x, a, ga, 0, True, grad)
        terms = {'objective': nll + beta*kl, 'nll': nll, 'kl': kl, 'negative_elbo': nll+kl}
        if not all(math.isfinite(v) for v in list(terms.values()) + grad):
            raise ValueError('nonfinite model computation')
        return terms, grad

    def to_dict(self):
        return {'version': 1, 'hidden': self.hidden, 'sigma': self.sigma, 'params': list(self.params)}

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or set(payload) != {'version','hidden','sigma','params'}:
            raise ValueError('invalid checkpoint fields')
        if type(payload['version']) is not int or payload['version'] != 1:
            raise ValueError('unsupported checkpoint version')
        model = cls(payload['hidden'], payload['sigma'])
        model.params = vector(payload['params'], len(model.params), 'params')
        return model


def train(model, data, steps=800, batch_size=24, beta=1.0, lr=0.006, seed=19):
    """Adam, with replacement batches and fresh independent Gaussian noise.

    No clipping, held-out selection or beta scheduling. Mutates model in place.
    All input data is validated before the first update.
    """
    integer(steps, 0, 10000, 'steps')
    integer(batch_size, 1, 256, 'batch_size')
    integer(seed, 0, 2**32-1, 'seed')
    lr = number(lr, 'lr')
    beta = number(beta, 'beta')
    if not 0 < lr <= 0.1 or not 0 <= beta <= 100:
        raise ValueError('unsupported lr/beta')
    if not isinstance(data, (list, tuple)) or not data or len(data) > 10000:
        raise ValueError('invalid training data')
    data = [vector(x, 2, 'data') for x in data]
    rng = random.Random(seed)
    m, v = [0.0]*len(model.params), [0.0]*len(model.params)
    history = []
    for step in range(1, steps+1):
        g = [0.0]*len(m)
        loss = 0.0
        for _ in range(batch_size):
            x = data[rng.randrange(len(data))]
            terms, grad = model.loss_grad(x, [rng.gauss(0,1), rng.gauss(0,1)], beta)
            loss += terms['objective']/batch_size
            for j in range(len(g)):
                g[j] += grad[j]/batch_size
        for j in range(len(g)):
            m[j] = 0.9*m[j] + 0.1*g[j]
            v[j] = 0.999*v[j] + 0.001*g[j]**2
            model.params[j] -= lr*(m[j]/(1-0.9**step))/(math.sqrt(v[j]/(1-0.999**step))+1e-8)
        if step == 1 or step % 100 == 0 or step == steps:
            history.append({'step': step, 'training_batch_objective': loss})
    return history
