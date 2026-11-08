import base64
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import zlib
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('render_importance',ROOT/'examples/render_importance.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
NS={'s':'http://www.w3.org/2000/svg'}


class ImportanceFigureTests(unittest.TestCase):
    def setUp(self):
        self.r=json.loads((ROOT/'examples/importance-demo.json').read_text())
        self.svg=(ROOT/'assets/importance-audit.svg').read_text();self.root=ET.fromstring(self.svg)

    def test_reproduction_metadata_and_hidden_witness_tamper(self):
        self.assertEqual(m.render(self.r),self.svg)
        self.assertEqual(json.loads(zlib.decompress(base64.b64decode(self.root.find('s:metadata',NS).text))),self.r)
        self.r['neural'][0]['prefixes'][-1]['log_joint'][0]+=.1
        self.assertNotEqual(m.render(self.r),self.svg)

    def test_all48_neural_cells_and_positions(self):
        for col,arm in enumerate(self.r['neural']):
            for row,a in enumerate(arm['prefixes']):
                values=[str(a['count']),f"{a['log_evidence_estimate']:.4f}",f"{a['sample_mean_log_weight']:.4f}",f"{a['effective_sample_size']:.2f}"]
                for j,(x,v) in enumerate(zip([50+354*col,95+354*col,195+354*col,300+354*col],values)):
                    e=self.root.find(f"s:text[@id='neural-{col}-{row}-{j}']",NS)
                    self.assertEqual(e.text,v);self.assertEqual(e.get('x'),str(x));self.assertEqual(e.get('y'),str(250+row*32))

    def test_exact_control_numbers(self):
        for col,r in enumerate(self.r['rare_atom']):
            e=self.root.find(f"s:text[@id='rare-expected-{col}']",NS)
            self.assertEqual(e.text,f"{r['expected_log_estimate']:.6f}")
        ref=self.r['linear_gaussian']['reference'];rare=self.r['rare_atom'][-1]
        self.assertEqual(self.root.find("s:text[@id='linear-true']",NS).text,f"True log evidence: {ref['log_evidence']:.6f}")
        prob=float(Fraction(*rare['all_common_probability']))
        self.assertEqual(self.root.find("s:text[@id='rare-probability']",NS).text,f"All 64 draws common: probability {prob*100:.2f}%")
        self.assertIn(f"{rare['all_common_audit']['log_evidence_estimate']:.6f}",self.root.find("s:text[@id='rare-failure']",NS).text)
