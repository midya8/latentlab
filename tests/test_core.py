import copy
import json
import math
import random
import subprocess
import sys
import unittest
from latentlab import VAE, train, evaluate, sample, sliced_w2
from latentlab.demo import ring, run_demo


def independent_loss(params, h, sigma, x, eps, beta):
    """Separate flat cursor/row implementation, never production encode/loss."""
    cursor = 0
    def dense(x, n, activation):
        nonlocal cursor
        result = []
        for _ in range(n):
            row = params[cursor:cursor+len(x)]
            cursor += len(x)
            b = params[cursor]
            cursor += 1
            val = b + sum(a*w for a,w in zip(x,row))
            result.append(math.tanh(val) if activation else val)
        return result
    e = dense(dense(x,h,True),4,False)
    lv = [4*math.tanh(e[j+2]/4) for j in range(2)]
    z = [e[j]+math.exp(lv[j]/2)*eps[j] for j in range(2)]
    y = dense(dense(z,h,True),2,False)
    assert cursor == len(params)
    likelihood = sum((a-b)**2 for a,b in zip(x,y))/(2*sigma**2) + math.log(2*math.pi*sigma**2)
    divergence = sum((e[j]**2+math.exp(lv[j])-1-lv[j])/2 for j in range(2))
    return likelihood+beta*divergence


class GradientTests(unittest.TestCase):
    def test_every_parameter_independent_finite_difference(self):
        for hidden,seed,x,eps in [(2,0,[0.3,-0.4],[0.7,-1.2]),(3,8,[-0.8,0.2],[-0.3,0.9]),
                                   (12,7,[0.1,0.9],[1.5,-0.8])]:
            for beta in [0,1,4]:
                model=VAE(hidden=hidden,seed=seed)
                terms,g=model.loss_grad(x,eps,beta)
                p=list(model.params)
                self.assertAlmostEqual(terms['objective'],independent_loss(p,hidden,model.sigma,x,eps,beta),places=12)
                for j in range(len(p)):
                    a,b=p.copy(),p.copy()
                    a[j]+=1e-5; b[j]-=1e-5
                    fd=(independent_loss(a,hidden,model.sigma,x,eps,beta)-independent_loss(b,hidden,model.sigma,x,eps,beta))/2e-5
                    self.assertAlmostEqual(g[j],fd,delta=2e-6*(1+abs(fd)),msg=str((hidden,beta,j)))

    def test_fixed_noise_batch_gradient(self):
        m=VAE(hidden=3,seed=13)
        pairs=[([0.8,0.1],[0.5,-0.2]),([-0.1,-0.7],[-1,0.8])]
        grad=[sum(m.loss_grad(x,e,0.3)[1][j] for x,e in pairs)/2 for j in range(len(m.params))]
        for j in range(len(grad)):
            p,q=m.params.copy(),m.params.copy();p[j]+=1e-5;q[j]-=1e-5
            fd=sum((independent_loss(p,3,m.sigma,x,e,0.3)-independent_loss(q,3,m.sigma,x,e,0.3))/2e-5 for x,e in pairs)/2
            self.assertAlmostEqual(grad[j],fd,delta=2e-6*(1+abs(fd)))

    def test_zero_model_analytic_terms(self):
        m=VAE();m.params=[0.0]*len(m.params)
        t,g=m.loss_grad([0.3,0.4],[3,-2])
        self.assertEqual(t['kl'],0)
        self.assertAlmostEqual(t['nll'],0.25/(2*0.04)+math.log(2*math.pi*0.04))
        self.assertEqual(m.encode([0.3,0.4]),([0,0],[0,0]))

    def test_logvariance_saturation_chain_rule(self):
        m=VAE(hidden=2)
        off,ni,_=m.layers[1]
        m.params[off+2*(ni+1)+ni]=20
        m.params[off+3*(ni+1)+ni]=-20
        x,e=[0.4,0.1],[0.5,-0.7]
        _,g=m.loss_grad(x,e)
        for j in [off+2*(ni+1)+ni,off+3*(ni+1)+ni]:
            p,q=m.params.copy(),m.params.copy();p[j]+=1e-4;q[j]-=1e-4
            fd=(independent_loss(p,2,m.sigma,x,e,1)-independent_loss(q,2,m.sigma,x,e,1))/2e-4
            self.assertAlmostEqual(g[j],fd,delta=1e-6)
        self.assertTrue(all(-4<v<4 for v in m.encode(x)[1]))

    def test_beta_objective_is_not_elbo_except_one(self):
        m=VAE()
        t,_=m.loss_grad([0.7,0.2],[0.4,-0.8],0)
        self.assertEqual(t['objective'],t['nll'])
        self.assertAlmostEqual(t['negative_elbo'],t['nll']+t['kl'])
        self.assertGreater(t['kl'],0)

    def test_kl_nonnegative_and_zero_prior(self):
        m=VAE()
        for p in ring(16):
            t,_=m.loss_grad(p,[0,0]);self.assertGreaterEqual(t['kl'],-1e-14)
        off,ni,no=m.layers[1]
        m.params[off:off+no*(ni+1)]=[0.0]*(no*(ni+1))
        self.assertEqual(m.loss_grad([1,0],[1,-1])[0]['kl'],0)


class ContractTests(unittest.TestCase):
    def test_roundtrip_and_detached_storage(self):
        m=VAE();d=m.to_dict();n=VAE.from_dict(json.loads(json.dumps(d)))
        self.assertEqual(m.loss_grad([0.5,-0.4],[1,2]),n.loss_grad([0.5,-0.4],[1,2]))
        d['params'][0]+=1;self.assertNotEqual(d['params'][0],m.params[0])
        n.params[0]+=2;self.assertNotEqual(n.params,m.params)

    def test_checkpoint_strict_fields(self):
        for key,val in [('version',True),('version',2),('hidden',0),('hidden',True),('sigma',float('nan')),
                        ('params',[]),('params',[0]*149+[float('inf')])]:
            d=VAE().to_dict();d[key]=val
            with self.assertRaises(ValueError):VAE.from_dict(d)
        d=VAE().to_dict();d['extra']=1
        with self.assertRaises(ValueError):VAE.from_dict(d)

    def test_bad_shapes_and_nonfinite(self):
        m=VAE()
        for x in [[],[1],[1,2,3],[True,0],[float('inf'),0],['1',0],None]:
            with self.assertRaises(ValueError):m.encode(x)
            with self.assertRaises(ValueError):m.decode(x)
            with self.assertRaises(ValueError):m.loss_grad([0,1],x)
        for beta in [-1,True,float('nan'),101]:
            with self.assertRaises(ValueError):m.loss_grad([1,0],[0,1],beta)

    def test_model_config_validation(self):
        for args in [dict(hidden=0),dict(hidden=65),dict(hidden=2.5),dict(seed=-1),dict(sigma=0),dict(sigma=11)]:
            with self.assertRaises(ValueError):VAE(**args)

    def test_training_invalid_before_mutation(self):
        for kwargs in [dict(steps=True),dict(steps=-1),dict(batch_size=0),dict(lr=0),dict(beta=-1),dict(seed=True)]:
            m=VAE();before=m.to_dict()
            with self.assertRaises(ValueError):train(m,[[1,0]],**kwargs)
            self.assertEqual(before,m.to_dict())
        m=VAE();before=m.to_dict()
        with self.assertRaises(ValueError):train(m,[[1,0],[1,float('nan')]])
        self.assertEqual(before,m.to_dict())

    def test_rng_global_isolation(self):
        random.seed(83);before=random.getstate()
        m=VAE();ring(16);sample(m,8);evaluate(m,ring(8));train(m,ring(8),steps=2)
        self.assertEqual(before,random.getstate())

    def test_parameter_count_and_zero_steps(self):
        m=VAE();self.assertEqual(len(m.params),150)
        before=m.to_dict();self.assertEqual(train(m,ring(8),steps=0),[])
        self.assertEqual(before,m.to_dict())

    def test_training_determinism_and_input_immutable(self):
        data=ring(16);before=copy.deepcopy(data)
        a,b=VAE(),VAE()
        self.assertEqual(train(a,data,steps=5),train(b,data,steps=5))
        self.assertEqual(a.to_dict(),b.to_dict());self.assertEqual(data,before)

    def test_one_adam_step_independent(self):
        m=VAE(hidden=2);before=m.params.copy();r=random.Random(19);data=ring(8)
        x=data[r.randrange(len(data))];eps=[r.gauss(0,1),r.gauss(0,1)]
        grad=m.loss_grad(x,eps)[1]
        train(m,data,steps=1,batch_size=1)
        for a,b,g in zip(m.params,before,grad):
            self.assertAlmostEqual(a,b-0.006*g/(abs(g)+1e-8),places=13)


class AuditTests(unittest.TestCase):
    def test_sliced_transport_exact_tiny(self):
        self.assertEqual(sliced_w2([[0,0]],[[3,4]],directions=2),12.5)
        p=[[0,0],[1,1]];self.assertEqual(sliced_w2(p,list(reversed(p))),0)
        self.assertAlmostEqual(sliced_w2(p,[[2,0],[3,1]]),2)
        for a,b in [([],[]),([[1,0]],[]),([[1,0]],[[float('nan'),0]])]:
            with self.assertRaises(ValueError):sliced_w2(a,b)

    def test_reconstruction_is_not_prior_generation(self):
        m=VAE();data=ring(8);e=evaluate(m,data)
        self.assertEqual(e['reconstructed'],[m.decode(m.encode(x)[0]) for x in data])
        self.assertEqual(e['prior_samples'],sample(m,8))
        self.assertNotEqual(e['prior_samples'],e['reconstructed'])
        self.assertEqual(sum(e['prior_sample_mode_counts_within_0_3'])+round(e['prior_sample_outside_fraction']*8),8)

    def test_observation_noise_separate_stream(self):
        m=VAE();a=sample(m,8,observation_noise=False);b=sample(m,8)
        r=random.Random(32)
        for mean,draw in zip(a,b):
            for u,v in zip(mean,draw):self.assertEqual(v,u+m.sigma*r.gauss(0,1))

    def test_audit_validation(self):
        m=VAE()
        for kwargs in [dict(noise_draws=0),dict(noise_draws=True),dict(seed=-1)]:
            with self.assertRaises(ValueError):evaluate(m,[[1,0]],**kwargs)
        for n in [0,True,4097]:
            with self.assertRaises(ValueError):sample(m,n)
        with self.assertRaises(ValueError):sample(m,8,observation_noise=1)

    def test_cli_zero_steps_failure_is_valid_json(self):
        p=subprocess.run([sys.executable,'-m','latentlab','demo','--steps','0'],capture_output=True,text=True)
        self.assertEqual(p.returncode,1)
        self.assertFalse(json.loads(p.stdout)['gates']['trained_beta1_improves_fixed_mc_negative_elbo'])

    def test_cli_invalid_steps(self):
        p=subprocess.run([sys.executable,'-m','latentlab','demo','--steps','-1'],capture_output=True,text=True)
        self.assertEqual(p.returncode,2);self.assertNotIn('Traceback',p.stderr)

    def test_measured_fixture_and_collapsed_control(self):
        r=run_demo()
        self.assertTrue(all(r['gates'].values()))
        no,yes=[a['evaluation'] for a in r['arms']]
        self.assertLess(yes['negative_elbo_mc'],r['initial']['negative_elbo_mc'])
        self.assertLess(no['posterior_mean_mse_per_coordinate'],yes['posterior_mean_mse_per_coordinate'])
        self.assertGreater(no['kl_analytic'],yes['kl_analytic'])
        self.assertGreater(no['prior_sample_sliced_w2_squared'],yes['prior_sample_sliced_w2_squared'])
        # This particular trained model still misses much of the eight-mode support.
        self.assertGreater(yes['prior_sample_outside_fraction'],0.4)
        self.assertGreater(sliced_w2(r['heldout'],[[0,0]]*128),0.25)
        for a in r['arms']:
            n=VAE.from_dict(a['checkpoint'])
            self.assertEqual(a['evaluation'],evaluate(n,r['heldout']))


if __name__=='__main__':unittest.main()
