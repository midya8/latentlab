import copy
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from latentlab import VAE
from latentlab.importance import (summarize_log_weights,log_diagonal_normal,gaussian_importance,
    audit_importance,linear_gaussian_reference,rare_atom_reference,run_importance_demo)


def normal_density(x,mean,var):
    return math.exp(-(x-mean)**2/(2*var))/math.sqrt(2*math.pi*var)


def simpson(f,lo=-10,hi=10,n=4000):
    h=(hi-lo)/n
    return h*(f(lo)+f(hi)+sum((4 if j%2 else 2)*f(lo+j*h) for j in range(1,n)))/3


class ImportanceTests(unittest.TestCase):
    def test_log_density_independent_product(self):
        for x,mu,lv in [([.2,-.7],[.1,.3],[.4,-.8]),([3,-4],[-1,2],[-2,1]),([0,0],[0,0],[0,0])]:
            p=math.prod(normal_density(x[i],mu[i],math.exp(lv[i])) for i in range(2))
            self.assertAlmostEqual(log_diagonal_normal(x,mu,lv),math.log(p),places=12)

    def test_stable_extreme_weights_and_underflow_report(self):
        r=summarize_log_weights([-10000,0,10000])
        self.assertAlmostEqual(r['log_evidence_estimate'],10000-math.log(3))
        self.assertEqual(r['normalized_weights'],[0,0,1])
        self.assertEqual(r['effective_sample_size'],1)
        self.assertEqual(r['numerically_zero_normalized_weights'],2)
        self.assertEqual(r['sample_mean_log_weight'],0)

    def test_translation_permutation_and_weight_normalization(self):
        a=[-10,2,4,.5];r=summarize_log_weights(a)
        s=summarize_log_weights([v+10000 for v in reversed(a)])
        self.assertAlmostEqual(s['log_evidence_estimate']-10000,r['log_evidence_estimate'],places=10)
        self.assertAlmostEqual(s['effective_sample_size'],r['effective_sample_size'],places=10)
        self.assertAlmostEqual(math.fsum(r['normalized_weights']),1)
        self.assertGreater(r['sample_jensen_gap'],0)
        self.assertAlmostEqual(r['effective_sample_size'],1/sum(p*p for p in r['normalized_weights']))

    def test_single_and_constant_weights(self):
        for k in [1,4,64,1024]:
            r=summarize_log_weights([-7.2]*k)
            self.assertAlmostEqual(r['log_evidence_estimate'],-7.2)
            self.assertEqual(r['sample_jensen_gap'],0)
            self.assertAlmostEqual(r['effective_sample_size'],k)

    def test_exact_small_iid_enumeration_expected_bound(self):
        q=[Fraction(2,3),Fraction(1,3)];joint=[Fraction(1,4),Fraction(3,4)]
        previous=-float('inf')
        for k in [1,2,4,8]:
            expected=0.0;mean_est=Fraction(0)
            for seq in itertools.product(range(2),repeat=k):
                probability=math.prod(q[i] for i in seq)
                estimate=sum((joint[i]/q[i] for i in seq),Fraction(0))/k
                mean_est+=probability*estimate
                r=summarize_log_weights([math.log(float(joint[i]/q[i])) for i in seq])
                expected+=float(probability)*r['log_evidence_estimate']
                self.assertGreaterEqual(r['sample_jensen_gap'],-1e-13)
            self.assertEqual(mean_est,1)
            self.assertLessEqual(expected,1e-12)
            self.assertGreater(expected,previous);previous=expected
        # A legal sample realization can overshoot true log evidence=0.
        self.assertGreater(summarize_log_weights([math.log(9/4)]*8)['log_evidence_estimate'],0)

    def test_rare_atom_exact_expectations_and_probability(self):
        previous=-float('inf')
        for k in [1,4,16,64]:
            r=rare_atom_reference(k)
            self.assertEqual(r['expected_evidence_estimate'],[1,1])
            self.assertEqual(r['binomial_probability_sum'],[1,1])
            self.assertEqual(Fraction(*r['all_common_probability']),Fraction(999,1000)**k)
            self.assertGreater(r['expected_log_estimate'],previous);previous=r['expected_log_estimate']
            self.assertLess(r['expected_log_estimate'],0)

    def test_high_ess_can_be_wrong_with_high_event_probability(self):
        r=rare_atom_reference(64);a=r['all_common_audit']
        self.assertEqual(a['effective_sample_size'],64)
        self.assertLess(a['log_evidence_estimate'],-13)
        self.assertEqual(a['sample_jensen_gap'],0)
        self.assertGreater(float(Fraction(*r['all_common_probability'])),.93)
        self.assertEqual(r['true_evidence'],[1,1])

    def test_linear_evidence_independent_simpson_integrals(self):
        for x,a,b,sigma in [([.8,-.3],[1.2,-.7],[.1,.2],.3),([0,.5],[0,1],[0,-.2],.4),
                            ([1.3,-.2],[-.5,.8],[.3,-.1],.2),([0,0],[0,0],[0,0],.2)]:
            exact=linear_gaussian_reference(x,a,b,sigma)
            p=math.prod(simpson(lambda z:normal_density(z,0,1)*normal_density(x[i],a[i]*z+b[i],sigma**2)) for i in range(2))
            self.assertAlmostEqual(exact['log_evidence'],math.log(p),places=10)
            for mu,lv in [([0,0],[0,0]),([.4,-.8],[-.7,.2])]:
                r=linear_gaussian_reference(x,a,b,sigma,mu,lv)
                self.assertGreaterEqual(r['kl_to_true_posterior'],-1e-13)
                self.assertAlmostEqual(r['gap_identity_residual'],0,places=10)
                self.assertGreaterEqual(r['expected_negative_elbo']+r['log_evidence'],-1e-12)

    def test_exact_posterior_constant_weights_arbitrary_epsilon(self):
        x,a,b,sigma=[.8,-.3],[1.2,-.7],[.1,.2],.3
        r=linear_gaussian_reference(x,a,b,sigma)
        def joint(z):
            return sum(math.log(normal_density(z[i],0,1))+math.log(normal_density(x[i],a[i]*z[i]+b[i],sigma**2)) for i in range(2))
        out=gaussian_importance(joint,r['posterior_mu'],r['posterior_logvar'],[[0,0],[1,-1],[-2,.5],[.3,2.1]])
        for v in out['log_weights']:self.assertAlmostEqual(v,r['log_evidence'],places=12)
        self.assertAlmostEqual(out['effective_sample_size'],4,places=12)
        post=linear_gaussian_reference(x,a,b,sigma,r['posterior_mu'],r['posterior_logvar'])
        self.assertAlmostEqual(post['kl_to_true_posterior'],0,places=12)
        self.assertAlmostEqual(post['expected_negative_elbo'],-r['log_evidence'],places=12)

    def test_vae_weights_joint_prior_q_and_analytic_kl_distinct(self):
        m=VAE();x=[1,0];eps=[[.4,-.7],[1.2,.1],[-1,-.8]]
        r=audit_importance(m,x,eps);mu,lv=m.encode(x)
        for j,z in enumerate(r['z']):
            y=m.decode(z)
            joint=math.prod(normal_density(z[i],0,1)*normal_density(x[i],y[i],m.sigma**2) for i in range(2))
            q=math.prod(normal_density(z[i],mu[i],math.exp(lv[i])) for i in range(2))
            self.assertAlmostEqual(r['log_weights'][j],math.log(joint/q),places=11)
        expected=sum(m.loss_grad(x,e)[0]['negative_elbo'] for e in eps)/len(eps)
        self.assertAlmostEqual(r['negative_elbo_analytic_kl_mc'],expected,places=12)
        self.assertGreater(abs(r['sample_mean_log_weight']+r['negative_elbo_analytic_kl_mc']),.01)

    def test_validation_and_no_callback_on_invalid_noise(self):
        for w in [[],[True],[float('inf')],[1e7],[0]*1025]:
            with self.assertRaises(ValueError):summarize_log_weights(w)
        for args in [([0,0],[41,0],[[0,0]]),([0,0],[0,0],[[0,0],[1,float('nan')]]),([0,0],[0,0],[])]:
            calls=[]
            with self.assertRaises(ValueError):gaussian_importance(lambda z:calls.append(z),*args)
            self.assertEqual(calls,[])
        for k in [0,True,257]:
            with self.assertRaises(ValueError):rare_atom_reference(k)
        with self.assertRaises(ValueError):audit_importance(None,[0,0],[[0,0]])
        with self.assertRaises(ValueError):linear_gaussian_reference([0,0],sigma=0)

    def test_no_mutation_or_global_rng_use(self):
        m=VAE();checkpoint=m.to_dict();eps=[[.4,.2],[-.7,1]];before=copy.deepcopy(eps)
        random.seed(91);state=random.getstate()
        a=audit_importance(m,[1,0],eps);b=audit_importance(m,[1,0],eps)
        self.assertEqual(a,b);self.assertEqual(eps,before);self.assertEqual(m.to_dict(),checkpoint)
        self.assertEqual(random.getstate(),state)

    def test_cli_checkpoint_audit_and_rejected_fields(self):
        payload={'version':1,'checkpoint':VAE(hidden=2).to_dict(),'x':[1,0],'epsilon':[[.3,-.2],[1,0]]}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.json';p.write_text(json.dumps(payload))
            proc=subprocess.run([sys.executable,'-m','latentlab','importance',str(p)],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertEqual(json.loads(proc.stdout),audit_importance(VAE.from_dict(payload['checkpoint']),payload['x'],payload['epsilon']))
            payload['version']=True;p.write_text(json.dumps(payload))
            proc=subprocess.run([sys.executable,'-m','latentlab','importance',str(p)],capture_output=True,text=True)
            self.assertEqual(proc.returncode,2);self.assertNotIn('Traceback',proc.stderr)

    def test_cli_missing_and_malformed_input(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.json'
            for text in [None,'{','{}']:
                if text is not None:p.write_text(text)
                proc=subprocess.run([sys.executable,'-m','latentlab','importance',str(p)],capture_output=True,text=True)
                self.assertEqual(proc.returncode,2);self.assertNotIn('Traceback',proc.stderr)

    def test_measured_neural_prefixes_are_not_monotone(self):
        r=run_importance_demo();self.assertTrue(all(r['gates'].values()))
        neural=r['neural'][2]['prefixes']
        self.assertEqual([p['count'] for p in neural],[1,4,16,64])
        self.assertGreater(neural[1]['log_evidence_estimate'],neural[0]['log_evidence_estimate'])
        self.assertLess(neural[-1]['log_evidence_estimate'],neural[1]['log_evidence_estimate'])
        for arm in r['neural']:
            for a in arm['prefixes']:
                self.assertGreaterEqual(a['sample_jensen_gap'],-1e-12)
                self.assertGreaterEqual(a['effective_sample_size'],1-1e-12)
                self.assertLessEqual(a['effective_sample_size'],a['count']+1e-10)
