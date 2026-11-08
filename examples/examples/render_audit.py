"""Render measured sample/reconstruction comparisons without plotting libraries."""
import argparse
import base64
import html
import json
import math
from pathlib import Path
import zlib

ROOT=Path(__file__).resolve().parents[1]


def render(report):
    rows=[report['initial']]+[a['evaluation'] for a in report['arms']]
    allpts=report['heldout']+sum([e['prior_samples']+e['reconstructed'] for e in rows],[])
    bound=math.ceil(max(abs(v) for p in allpts for v in p)*2)/2
    encoded=base64.b64encode(zlib.compress(json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),9)).decode()
    s=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="980" viewBox="0 0 1120 980" role="img" aria-labelledby="title desc">',
       '<title id="title">LatentLab: reconstruction is not generation</title>',
       '<desc id="desc">Measured initial, beta0, and beta1 Gaussian VAE results. Six equal-axis scatter plots distinguish noisy prior samples from posterior-mean reconstructions. The beta1 model still generates many points away from all eight data modes.</desc>',
       '<metadata id="audit" data-encoding="zlib-base64">'+encoded+'</metadata>',
       '<rect width="1120" height="980" fill="#f4efec"/><rect x="16" y="16" width="1088" height="948" rx="20" fill="white"/>',
       '<style>text{font-family:Arial,sans-serif;fill:#251f21}.small{font-size:13px;fill:#585254}.heading{font-size:18px;font-weight:600}.metric{font-size:14px}.teal{fill:#568779;font-weight:600}.grid{stroke:#eae9ea;stroke-width:1}</style>']
    def text(x,y,v,cls='small'):
        s.append(f'<text x="{x}" y="{y}" class="{cls}">{html.escape(v)}</text>')
    text(40,58,'LatentLab','heading')
    text(40,90,'Reconstruction is not generation','heading')
    text(40,115,'128 held-out points | 150 parameters | 800 Adam updates | identical initialization and training seed')
    text(40,140,'Gray: held-out data. Teal: model output. All six plots share axes. Prior samples include likelihood noise.')
    text(40,171,'Prior predictive samples','heading')
    names=['Before training','Trained, beta = 0','Trained, beta = 1']
    for col,(name,e) in enumerate(zip(names,rows)):
        left=40+col*354
        text(left,202,name,'heading')
        text(left,494,f"Negative ELBO (4-noise MC): {e['negative_elbo_mc']:.4f}",'metric')
        text(left,516,f"Analytic KL: {e['kl_analytic']:.4f} | sliced W2²: {e['prior_sample_sliced_w2_squared']:.4f}",'metric')
        text(left,538,f"Outside all radius-0.3 modes: {e['prior_sample_outside_fraction']*100:.2f}%",'metric')
        text(left,880,f"Mean-reconstruction MSE / coord: {e['posterior_mean_mse_per_coordinate']:.5f}",'metric')
        for row,key in enumerate(['prior_samples','reconstructed']):
            top=220 if row==0 else 600
            # 250x230 interior, equal x/y physical scale (230 square).
            ox,oy=left+45,top+8
            s.append(f'<g id="plot-{col}-{row}">')
            for tick in [-bound,0,bound]:
                px=ox+(tick+bound)/(2*bound)*230
                py=oy+(bound-tick)/(2*bound)*230
                s.append(f'<path d="M {px:.2f} {oy} v 230 M {ox} {py:.2f} h 230" class="grid"/>')
                text(f'{px-8:.2f}',oy+248,f'{tick:g}')
                text(ox-32,f'{py+4:.2f}',f'{tick:g}')
            for kind,points,color,radius in [('truth',report['heldout'],'#c0bebf',2),('model',e[key],'#568779',2.4)]:
                s.append(f'<g id="points-{col}-{row}-{kind}" fill="{color}" fill-opacity="0.7">')
                for x,y in points:
                    px=ox+(x+bound)/(2*bound)*230
                    py=oy+(bound-y)/(2*bound)*230
                    s.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius}"/>')
                s.append('</g>')
            s.append('</g>')
    text(40,578,'Posterior-mean reconstructions: decode(mu(x)), not samples from the generative model','heading')
    text(40,940,'Synthetic eight-mode fixture, one seed. Lower reconstruction error does not certify a matched prior or good generation.')
    s.append('</svg>')
    return '\n'.join(s)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    report=json.loads((ROOT/'examples/demo.json').read_text())
    output=render(report);path=ROOT/'assets/latent-audit.svg'
    if a.check:
        if path.read_text()!=output:raise SystemExit('SVG differs from measured report')
        print('Measured SVG matches report byte-for-byte')
    else:path.write_text(output)


if __name__=='__main__':main()
