import copy
from itertools import permutations, product
import json
import math
from pathlib import Path
import random
import unittest
from latentlab import VAE, train
from latentlab.coordinates import relabel_latents
from latentlab.importance import audit_importance


def axes(values, order, signs):
    return [signs[j]*values[order[j]] for j in range(2)]


def expected_parameters(values, model, order, signs):
    # Independent block unpack/repack, not the production flat index plan.
    blocks=[]
    for off, ni, no in model.layers:
        blocks.append([values[off+j*(ni+1):off+(j+1)*(ni+1)]for j in range(no)])
    hidden, encoder, decoder, output=blocks
    encoder=[[s*v for v in encoder[old]]for old,s in zip(order,signs)]+[encoder[2+old][:]for old in order]
    decoder=[[row[old]*s for old,s in zip(order,signs)]+[row[-1]]for row in decoder]
    return [v for layer in [hidden,encoder,decoder,output]for row in layer for v in row]


class LatentCoordinateTests(unittest.TestCase):
    def close(self,a,b):
        if isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for key in a:self.close(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.close(x,y)
        elif isinstance(a,(int,float)):
            self.assertTrue(math.isclose(a,b,rel_tol=2e-10,abs_tol=2e-10),(a,b))
        else:self.assertEqual(a,b)

    def models(self):
        report=json.loads((Path(__file__).resolve().parents[1]/'examples/demo.json').read_text())
        return [VAE.from_dict(arm['checkpoint'])for arm in report['arms']]+[VAE(hidden=(1,2,3,12)[seed%4],seed=seed)for seed in range(12)]

    def test_all_eight_maps_encoder_decoder_terms_and_every_gradient(self):
        for a in self.models():
            for order in permutations(range(2)):
                for signs in product([-1,1],repeat=2):
                    b=relabel_latents(a,order,signs)
                    self.assertEqual(b.params,expected_parameters(a.params,a,order,signs))
                    for x,e in [([.8,-.2],[.5,-1.2]),([-.3,.7],[-.1,.9]),([0,0],[0,0])]:
                        mu,lv=a.encode(x);m2,l2=b.encode(x)
                        self.close(m2,axes(mu,order,signs));self.close(l2,[lv[i]for i in order])
                        self.close(a.decode(e),b.decode(axes(e,order,signs)))
                        for beta in [0,.3,1,4]:
                            terms,g=a.loss_grad(x,e,beta);other,h=b.loss_grad(x,axes(e,order,signs),beta)
                            self.close(terms,other);self.close(h,expected_parameters(g,a,order,signs))

    def test_complete_importance_witnesses_require_coupled_epsilon(self):
        rng=random.Random(630);eps=[[rng.gauss(0,1),rng.gauss(0,1)]for _ in range(16)]
        for a in self.models()[:6]:
            for order,signs in [([1,0],[-1,1]),([0,1],[1,-1])]:
                b=relabel_latents(a,order,signs)
                for size in [1,4,16]:
                    r=audit_importance(a,[.8,-.1],eps[:size]);s=audit_importance(b,[.8,-.1],[axes(e,order,signs)for e in eps[:size]])
                    expected=copy.deepcopy(r)
                    for key in ['proposal_mu']:
                        expected[key]=axes(r[key],order,signs)
                    expected['proposal_logvar']=[r['proposal_logvar'][i]for i in order]
                    for key in ['epsilon','z']:expected[key]=[axes(row,order,signs)for row in r[key]]
                    self.close(expected,s)

    def test_inverse_and_group_composition_detached_checkpoint(self):
        a=VAE(hidden=3,seed=91);before=copy.deepcopy(a.to_dict());state=random.getstate()
        maps=list(product(permutations(range(2)),product([-1,1],repeat=2)))
        for order,signs in maps:
            b=relabel_latents(a,order,signs);inverse=[order.index(j)for j in range(2)];inverse_signs=[signs[inverse[j]]for j in range(2)]
            self.assertEqual(relabel_latents(b,inverse,inverse_signs).to_dict(),a.to_dict())
            for order2,signs2 in maps:
                combined_order=[order[order2[j]]for j in range(2)];combined_signs=[signs2[j]*signs[order2[j]]for j in range(2)]
                self.assertEqual(relabel_latents(b,order2,signs2).to_dict(),relabel_latents(a,combined_order,combined_signs).to_dict())
        self.assertEqual(a.to_dict(),before);self.assertEqual(random.getstate(),state)
        b.params[0]+=1;self.assertEqual(a.to_dict(),before)

    def test_coupled_training_stream_and_gradient_moments(self):
        data=[[.8,.1],[-.4,.7],[0,-.6]]
        for seed in range(8):
            a=VAE(hidden=2+seed%3,seed=seed);order=[1,0];signs=[-1,1]if seed%2 else [1,-1];b=relabel_latents(a,order,signs)
            original=b.loss_grad
            def coupled(x,epsilon,beta=1):
                return original(x,axes(epsilon,order,signs),beta)
            b.loss_grad=coupled
            ha=train(a,data,steps=5,batch_size=3,beta=seed%2,seed=seed+60)
            hb=train(b,data,steps=5,batch_size=3,beta=seed%2,seed=seed+60)
            self.close(ha,hb);self.close(b.params,expected_parameters(a.params,a,order,signs))

    def test_unchanged_noise_is_not_a_realization_invariant(self):
        a=VAE(hidden=3,seed=72);b=relabel_latents(a,[1,0],[-1,1]);x=[.8,-.1];e=[.5,-1.2]
        self.close(a.loss_grad(x,e)[0],b.loss_grad(x,axes(e,[1,0],[-1,1]))[0])
        self.assertGreater(abs(a.loss_grad(x,e)[0]['objective']-b.loss_grad(x,e)[0]['objective']),1e-3)
        unpaired=VAE.from_dict(a.to_dict());off,width,_=unpaired.layers[1]
        unpaired.params[off:off+width+1]=[-v for v in unpaired.params[off:off+width+1]]
        self.assertGreater(abs(a.loss_grad(x,e)[0]['objective']-unpaired.loss_grad(x,e)[0]['objective']),1e-3)

    def test_strict_maps_and_checkpoint_validation(self):
        a=VAE(hidden=1);before=a.to_dict()
        for bad in [None,[0],[0,0],[1,2],[True,1],['0',1],{0:0,1:1}]:
            with self.assertRaises(ValueError):relabel_latents(a,bad,[1,1])
        for bad in [None,[1],[1,0],[True,1],[1.,1],['1',1]]:
            with self.assertRaises(ValueError):relabel_latents(a,[0,1],bad)
        with self.assertRaises(ValueError):relabel_latents({},[0,1],[1,1])
        self.assertEqual(a.to_dict(),before)
        a.params[0]=float('nan')
        with self.assertRaises(ValueError):relabel_latents(a,[0,1],[1,1])
        a=VAE(hidden=64);b=relabel_latents(a,[1,0],[-1,1]);self.close(a.decode([.2,-.3]),b.decode([.3,.2]))
