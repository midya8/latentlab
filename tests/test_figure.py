import base64
import importlib.util
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import zlib

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('render_audit',ROOT/'examples/render_audit.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
NS={'s':'http://www.w3.org/2000/svg'}


class FigureTests(unittest.TestCase):
    def setUp(self):
        self.report=json.loads((ROOT/'examples/demo.json').read_text())
        self.svg=(ROOT/'assets/latent-audit.svg').read_text()
        self.root=ET.fromstring(self.svg)

    def test_byte_reproduction_and_complete_metadata(self):
        self.assertEqual(module.render(self.report),self.svg)
        encoded=self.root.find('s:metadata',NS).text
        self.assertEqual(json.loads(zlib.decompress(base64.b64decode(encoded))),self.report)
        self.report['arms'][1]['checkpoint']['params'][0]+=0.1
        self.assertNotEqual(module.render(self.report),self.svg)

    def test_all_metrics_displayed_from_report(self):
        texts=[e.text for e in self.root.findall('s:text',NS)]
        for e in [self.report['initial']]+[a['evaluation'] for a in self.report['arms']]:
            self.assertIn(f"Negative ELBO (4-noise MC): {e['negative_elbo_mc']:.4f}",texts)
            self.assertIn(f"Analytic KL: {e['kl_analytic']:.4f} | sliced W2²: {e['prior_sample_sliced_w2_squared']:.4f}",texts)
            self.assertIn(f"Outside all radius-0.3 modes: {e['prior_sample_outside_fraction']*100:.2f}%",texts)
            self.assertIn(f"Mean-reconstruction MSE / coord: {e['posterior_mean_mse_per_coordinate']:.5f}",texts)

    def test_all_scatter_coordinates_and_shared_axes(self):
        import math
        rows=[self.report['initial']]+[a['evaluation'] for a in self.report['arms']]
        allpts=self.report['heldout']+sum([e['prior_samples']+e['reconstructed'] for e in rows],[])
        bound=math.ceil(max(abs(v) for p in allpts for v in p)*2)/2
        for col,e in enumerate(rows):
            for row,key in enumerate(['prior_samples','reconstructed']):
                ox,oy=85+col*354,(228 if row==0 else 608)
                for kind,pts in [('truth',self.report['heldout']),('model',e[key])]:
                    group=self.root.find(f".//s:g[@id='points-{col}-{row}-{kind}']",NS)
                    dots=group.findall('s:circle',NS);self.assertEqual(len(dots),128)
                    for dot,(x,y) in zip(dots,pts):
                        self.assertEqual(dot.get('cx'),f'{ox+(x+bound)/(2*bound)*230:.2f}')
                        self.assertEqual(dot.get('cy'),f'{oy+(bound-y)/(2*bound)*230:.2f}')
                        self.assertTrue(ox<=float(dot.get('cx'))<=ox+230)
                        self.assertTrue(oy<=float(dot.get('cy'))<=oy+230)
