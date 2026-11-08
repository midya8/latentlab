"""Finite log-space importance audits, not an IWAE training implementation."""
from fractions import Fraction
import math
import random
from .core import VAE, integer, number, vector


def summarize_log_weights(log_weights):
    """Random log-mean-weight estimate and within-cloud diagnostics.

    E[mean(exp(log_weights))] is evidence for valid full-support IID proposals.
    E[logmeanexp] is a lower bound, not every finite realization.
    This function alone cannot check those sampling/model assumptions.
    """
    if not isinstance(log_weights,(list,tuple)) or not 1 <= len(log_weights) <= 1024:
        raise ValueError('need 1..1024 log weights')
    lw=[number(v,'log_weight') for v in log_weights]
    if any(abs(v)>1e6 for v in lw):
        raise ValueError('log weights outside supported [-1e6,1e6]')
    high=max(lw)
    raw=[math.exp(v-high) for v in lw]
    total=math.fsum(raw)
    weights=[v/total for v in raw]
    logmean=high+math.log(total/len(lw))
    meanlog=math.fsum(lw)/len(lw)
    return {'count':len(lw),'log_weights':lw,'normalized_weights':weights,
            'log_evidence_estimate':logmean,'sample_mean_log_weight':meanlog,
            'sample_jensen_gap':logmean-meanlog,
            'effective_sample_size':1/math.fsum(v*v for v in weights),
            'largest_normalized_weight':max(weights),
            'numerically_zero_normalized_weights':sum(v==0 for v in weights)}


def log_diagonal_normal(x,mu,logvar):
    x,mu,lv=vector(x,2,'x'),vector(mu,2,'mu'),vector(logvar,2,'logvar')
    if any(abs(v)>40 for v in lv):
        raise ValueError('logvar outside [-40,40]')
    result=-math.log(2*math.pi)-0.5*math.fsum(lv[i]+(x[i]-mu[i])**2*math.exp(-lv[i]) for i in range(2))
    return number(result,'log density')


def gaussian_importance(log_joint,mu,logvar,epsilons):
    """Use explicit proposal-noise draws, preserving every sampled witness.

    Gaussian proposals have full support. The caller supplies joint log p(x,z),
    not a posterior score with an unknown omitted normalization constant.
    Explicit epsilon values do not themselves prove IID Gaussian sampling.
    """
    mu,lv=vector(mu,2,'mu'),vector(logvar,2,'logvar')
    if any(abs(v)>40 for v in lv):
        raise ValueError('logvar outside [-40,40]')
    if not isinstance(epsilons,(list,tuple)) or not 1<=len(epsilons)<=1024:
        raise ValueError('need 1..1024 noise draws')
    eps=[vector(e,2,'epsilon') for e in epsilons]
    sd=[math.exp(v/2) for v in lv]
    zs=[[mu[i]+sd[i]*e[i] for i in range(2)] for e in eps]
    lq=[log_diagonal_normal(z,mu,lv) for z in zs]
    lj=[number(log_joint(z),'log joint') for z in zs]
    report=summarize_log_weights([j-q for j,q in zip(lj,lq)])
    report.update({'proposal_mu':mu,'proposal_logvar':lv,'epsilon':eps,'z':zs,
                   'log_joint':lj,'log_proposal':lq})
    return report


def audit_importance(model,x,epsilons):
    """Evidence audit for a trained/untrained VAE, without changing weights."""
    if not isinstance(model,VAE):
        raise ValueError('model must be VAE')
    x=vector(x,2,'x');mu,lv=model.encode(x)
    obs_lv=[2*math.log(model.sigma)]*2
    def log_joint(z):
        return log_diagonal_normal(z,[0,0],[0,0])+log_diagonal_normal(x,model.decode(z),obs_lv)
    report=gaussian_importance(log_joint,mu,lv,epsilons)
    # This analytic-KL estimator differs samplewise from mean(log weights).
    nll=math.fsum(-log_diagonal_normal(x,model.decode(z),obs_lv) for z in report['z'])/report['count']
    kl=0.5*math.fsum(mu[i]**2+math.exp(lv[i])-1-lv[i] for i in range(2))
    report.update({'x':x,'negative_elbo_analytic_kl_mc':nll+kl,'kl_analytic':kl})
    return report


def linear_gaussian_reference(x,scale=(1,1),bias=(0,0),sigma=0.2,proposal_mu=(0,0),proposal_logvar=(0,0)):
    """Exact diagonal linear-Gaussian marginal, posterior and variational gap.

    z~N(0,I); x|z~N(scale*z+bias,sigma² I). No neural decoder used.
    """
    x,a,b,mu,lv=[vector(v,2,n) for v,n in [(x,'x'),(scale,'scale'),(bias,'bias'),
                                           (proposal_mu,'proposal_mu'),(proposal_logvar,'proposal_logvar')]]
    sigma=number(sigma,'sigma')
    if not 0.01<=sigma<=10 or any(abs(v)>40 for v in lv):
        raise ValueError('unsupported sigma/logvar')
    marginal=[a[i]**2+sigma**2 for i in range(2)]
    postvar=[sigma**2/v for v in marginal]
    postmu=[a[i]*(x[i]-b[i])/marginal[i] for i in range(2)]
    logev=log_diagonal_normal(x,b,[math.log(v) for v in marginal])
    expected_nll=math.log(2*math.pi*sigma**2)+sum((x[i]-b[i]-a[i]*mu[i])**2+a[i]**2*math.exp(lv[i]) for i in range(2))/(2*sigma**2)
    kl_prior=0.5*sum(mu[i]**2+math.exp(lv[i])-1-lv[i] for i in range(2))
    kl_post=0.5*sum(math.log(postvar[i])-lv[i]+(math.exp(lv[i])+(mu[i]-postmu[i])**2)/postvar[i]-1 for i in range(2))
    return {'log_evidence':logev,'posterior_mu':postmu,'posterior_logvar':[math.log(v) for v in postvar],
            'expected_negative_elbo':expected_nll+kl_prior,'kl_to_true_posterior':kl_post,
            'gap_identity_residual':expected_nll+kl_prior+logev-kl_post}


def rare_atom_reference(k=64):
    """Exact probability/evidence arithmetic, finite binomial expected log bound.

    Joint masses sum to one. Proposal misses the large-mass rare atom with
    probability (999/1000)^k. All-common samples give ESS=k but wrong evidence.
    The expected log bound enumerates all k+1 possible rare-atom counts.
    """
    integer(k,1,256,'k')
    q=Fraction(1,1000)
    p=Fraction(999999,1000000)
    wcommon=(1-p)/(1-q);wrare=p/q
    expectation=0.0
    probability_sum=Fraction(0)
    expected_estimate=Fraction(0)
    for rare in range(k+1):
        probability=math.comb(k,rare)*q**rare*(1-q)**(k-rare)
        estimate=((k-rare)*wcommon+rare*wrare)/k
        probability_sum+=probability
        expected_estimate+=probability*estimate
        expectation+=float(probability)*math.log(float(estimate))
    audit=summarize_log_weights([math.log(float(wcommon))]*k)
    pair=lambda f:[f.numerator,f.denominator]
    return {'count':k,'joint_masses':[pair(1-p),pair(p)],'proposal_masses':[pair(1-q),pair(q)],
            'true_evidence':pair(Fraction(1)),'expected_evidence_estimate':pair(expected_estimate),
            'binomial_probability_sum':pair(probability_sum),
            'all_common_probability':pair((1-q)**k),'expected_log_estimate':expectation,
            'all_common_audit':audit}


def run_importance_demo():
    from .demo import run_demo
    original=run_demo()
    rng=random.Random(107)
    eps=[[rng.gauss(0,1),rng.gauss(0,1)] for _ in range(64)]
    models=[('initial',VAE())]+[(f"trained_beta_{a['beta']:g}",VAE.from_dict(a['checkpoint'])) for a in original['arms']]
    neural=[]
    for name,m in models:
        neural.append({'name':name,'prefixes':[audit_importance(m,[1,0],eps[:k]) for k in [1,4,16,64]]})
    x=[0.8,-0.3];a=[1.2,-0.7];b=[0.1,0.2];sigma=0.3
    exact=linear_gaussian_reference(x,a,b,sigma)
    def joint(z):
        return log_diagonal_normal(z,[0,0],[0,0])+log_diagonal_normal(x,[a[i]*z[i]+b[i] for i in range(2)],[2*math.log(sigma)]*2)
    exact_proposal=gaussian_importance(joint,exact['posterior_mu'],exact['posterior_logvar'],eps)
    return {'version':1,'config':{'noise_seed':107,'prefix_counts':[1,4,16,64],'anchor_x':[1,0]},
            'neural':neural,'linear_gaussian':{'reference':exact,'exact_posterior_audit':exact_proposal},
            'rare_atom':[rare_atom_reference(k) for k in [1,4,16,64]],
            'gates':{'all_sample_jensen_gaps_nonnegative':all(r['sample_jensen_gap']>=-1e-12 for m in neural for r in m['prefixes']),
                     'exact_posterior_matches_linear_evidence':abs(exact_proposal['log_evidence_estimate']-exact['log_evidence'])<1e-10,
                     'high_ess_rare_atom_estimate_is_wrong':rare_atom_reference(64)['all_common_audit']['log_evidence_estimate'] < -10}}
