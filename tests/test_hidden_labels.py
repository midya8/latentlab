"""Signed hidden-unit labels: finite consistency, not an independent VAE oracle."""
import copy
import json
import math
from pathlib import Path
import random
import unittest

from latentlab import VAE, train
from latentlab.importance import audit_importance


def mapping(model, seed):
    rng = random.Random(seed)
    indices = list(range(len(model.params)))
    signs = [1] * len(indices)
    for incoming, outgoing in ((0, 1), (2, 3)):
        off, ni, no = model.layers[incoming]
        tail, width, outputs = model.layers[outgoing]
        order = list(range(no));rng.shuffle(order)
        flips = [rng.choice((-1, 1)) for _ in order]
        for new, old in enumerate(order):
            for i in range(ni + 1):
                indices[off + new*(ni+1) + i] = off + old*(ni+1) + i
                signs[off + new*(ni+1) + i] = flips[new]
            for j in range(outputs):
                indices[tail + j*(width+1) + new] = tail + j*(width+1) + old
                signs[tail + j*(width+1) + new] = flips[new]
    return indices, signs


def transform(values, plan):
    indices, signs = plan
    return [values[i]*s for i, s in zip(indices, signs)]


def relabel(model, plan):
    other = VAE.from_dict(model.to_dict())
    other.params = transform(model.params, plan)
    return other


class HiddenLabelTests(unittest.TestCase):
    def close(self, a, b):
        if isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a:self.close(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):self.close(x, y)
        elif isinstance(a, (int, float)):
            self.assertTrue(math.isclose(a, b, rel_tol=2e-10, abs_tol=2e-10), (a, b))
        else:self.assertEqual(a, b)

    def check_model(self, model, seed):
        plan = mapping(model, seed);other = relabel(model, plan)
        before = copy.deepcopy((model.to_dict(), other.to_dict()))
        for x, e in [([.8, -.2], [.5, -1.2]), ([-.3, .7], [-.1, .9]), ([0, 0], [0, 0])]:
            self.close(model.encode(x), other.encode(x))
            self.close(model.decode(e), other.decode(e))
            for beta in (0, .3, 1, 4):
                t, g = model.loss_grad(x, e, beta)
                u, h = other.loss_grad(x, e, beta)
                self.close(t, u);self.close(transform(g, plan), h)
        self.assertEqual((model.to_dict(), other.to_dict()), tuple(before))

    def test_all_gradients_archive_and_generated_models(self):
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root/'examples/demo.json').read_text())
        for seed, arm in enumerate(report['arms']):
            self.check_model(VAE.from_dict(arm['checkpoint']), seed + 800)
        for seed in range(18):
            self.check_model(VAE(hidden=(1, 2, 3, 12)[seed % 4], seed=seed), seed + 100)

    def test_same_training_stream_maps_hidden_parameters(self):
        for seed in range(8):
            a = VAE(hidden=2+seed%3, seed=seed)
            plan = mapping(a, seed+300);b = relabel(a, plan)
            data = [[.8, .1], [-.4, .7], [0, -.6]]
            ha = train(a, data, steps=5, batch_size=3, beta=(0, 1)[seed%2], seed=seed+40)
            hb = train(b, data, steps=5, batch_size=3, beta=(0, 1)[seed%2], seed=seed+40)
            self.close(ha, hb);self.close(transform(a.params, plan), b.params)

    def test_every_importance_witness_and_summary(self):
        report = json.loads((Path(__file__).resolve().parents[1]/'examples/demo.json').read_text())
        models = [VAE()] + [VAE.from_dict(arm['checkpoint']) for arm in report['arms']]
        rng = random.Random(830)
        eps = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(16)]
        for seed, a in enumerate(models):
            b = relabel(a, mapping(a, seed+900))
            for n in (1, 4, 16):
                self.close(audit_importance(a, [.8, -.1], eps[:n]), audit_importance(b, [.8, -.1], eps[:n]))

    def test_one_sided_sign_change_breaks_function(self):
        a = VAE(hidden=3, seed=72)
        b = VAE.from_dict(a.to_dict())
        off, ni, _ = b.layers[2]
        b.params[off:off+ni+1] = [-v for v in b.params[off:off+ni+1]]
        z = [.4, -.7]
        self.assertGreater(max(abs(x-y) for x, y in zip(a.decode(z), b.decode(z))), 1e-4)


if __name__ == '__main__':unittest.main()
