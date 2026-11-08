"""Finite importance estimates and exact-reference failure controls."""
import argparse
import base64
from fractions import Fraction
import html
import json
from pathlib import Path
import zlib
ROOT=Path(__file__).resolve().parents[1]


def render(report):
    s=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="900" viewBox="0 0 1120 900" role="img" aria-labelledby="title desc">',
       '<title id="title">LatentLab: a bigger importance sample is not a realization-wise guarantee</title>',
       '<desc id="desc">Twelve neural importance-prefix rows distinguish log mean weight from mean log weight. An exact linear-Gaussian control recovers the evidence with its true posterior. A rare-atom control has ESS 64 while underestimating evidence almost one million-fold on an event with probability 93.80 percent.</desc>',
       '<metadata id="audit" data-encoding="zlib-base64">'+base64.b64encode(zlib.compress(json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),9)).decode()+'</metadata>',
       '<rect width="1120" height="900" fill="#f4efec"/><rect x="16" y="16" width="1088" height="868" rx="20" fill="white"/>',
       '<style>text{font-family:Arial,sans-serif;fill:#251f21}.small{font-size:13px;fill:#585254}.heading{font-size:18px;font-weight:600}.value{font-size:14px}.known{font-size:14px;fill:#568779;font-weight:600}.rule{stroke:#eae9ea;stroke-width:1}</style>']
    def text(x,y,v,cls='small',ident=None):
        idattr='' if ident is None else f' id="{ident}"'
        s.append(f'<text x="{x}" y="{y}" class="{cls}"{idattr}>{html.escape(v)}</text>')
    text(40,58,'LatentLab | importance evidence audit','heading')
    text(40,90,'A larger sample is not a realization-wise guarantee','heading')
    text(40,116,'Neural VAE anchor x = (1,0) | shared Gaussian-noise prefixes, seed 107 | natural log density units')
    text(40,140,'log mean w: a random log-evidence estimate. mean log w: sampled ELBO integrand, not analytic-KL MC.')
    names=['Before training','Trained, beta = 0','Trained, beta = 1']
    for col,(name,arm) in enumerate(zip(names,report['neural'])):
        left=40+col*354
        text(left,182,name,'heading')
        for x,h in zip([left+10,left+55,left+155,left+260],['K','log mean w','mean log w','ESS']):text(x,212,h)
        s.append(f'<path d="M {left} 224 h 324" class="rule"/>')
        for row,a in enumerate(arm['prefixes']):
            y=250+row*32
            vals=[str(a['count']),f"{a['log_evidence_estimate']:.4f}",f"{a['sample_mean_log_weight']:.4f}",f"{a['effective_sample_size']:.2f}"]
            for j,(x,v) in enumerate(zip([left+10,left+55,left+155,left+260],vals)):
                text(x,y,v,'known' if j==1 else 'value',f'neural-{col}-{row}-{j}')
    text(40,382,'Within each row: log mean w >= mean log w. Across K: realized log mean w can go up or down.')
    s.append('<path d="M 40 404 h 1040" class="rule"/>')
    ref=report['linear_gaussian']['reference'];good=report['linear_gaussian']['exact_posterior_audit']
    text(40,444,'Exact linear-Gaussian control','heading')
    text(40,478,f"True log evidence: {ref['log_evidence']:.6f}",'known','linear-true')
    text(40,504,f"True posterior proposal, log estimate: {good['log_evidence_estimate']:.6f}",'known','linear-estimate')
    text(40,530,f"ESS: {good['effective_sample_size']:.2f} / {good['count']} | constant weights for any proposal-noise draws",'value','linear-ess')
    text(40,556,'This separate control has an analytic answer; the nonlinear VAE does not.')
    rare=report['rare_atom'][-1];a=rare['all_common_audit'];prob=float(Fraction(*rare['all_common_probability']))
    text(604,444,'Rare-atom failure control','heading')
    text(604,478,'Joint mass: (0.000001, 0.999999). Total evidence: 1.','value')
    text(604,504,'Proposal: (0.999, 0.001). Full support, but poorly matched.','value')
    text(604,530,f"All 64 draws common: probability {prob*100:.2f}%",'value','rare-probability')
    text(604,556,f"ESS = {a['effective_sample_size']:.0f}, log estimate = {a['log_evidence_estimate']:.6f}; true log evidence = 0",'value','rare-failure')
    text(40,610,'Expected log estimate: finite binomial enumeration, not one random realization','heading')
    for col,r in enumerate(report['rare_atom']):
        left=40+col*265
        text(left,646,f"K = {r['count']}",'value',f'rare-k-{col}')
        text(left,678,f"{r['expected_log_estimate']:.6f}",'known',f'rare-expected-{col}')
    text(40,718,'Exact rational mean evidence estimate = 1 at every K. Expected log estimates stay below 0 and rise with K.')
    text(40,754,'High ESS only describes weight balance in the observed cloud. It cannot detect the missed rare atom.','heading')
    text(40,790,'No IWAE training is implemented. Gaussian-reference evidence is exact; neural evidence remains unknown.')
    text(40,818,'Expectation bounds require valid IID full-support proposals. Supplied noise values alone do not prove those assumptions.')
    text(40,854,'Standard importance/IWAE formulas credited to Burda, Grosse & Salakhutdinov. Original finite audit and fixtures.')
    return '\n'.join(s+['</svg>'])+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    report=json.loads((ROOT/'examples/importance-demo.json').read_text())
    svg=render(report);out=ROOT/'assets/importance-audit.svg'
    if a.check:
        if out.read_text()!=svg:raise SystemExit('importance SVG differs from report')
        print('Importance figure matches report byte-for-byte')
    else:out.write_text(svg)


if __name__=='__main__':main()
