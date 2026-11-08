import argparse
import json
from pathlib import Path
from .core import VAE
from .demo import run_demo
from .importance import audit_importance, run_importance_demo


def main():
    p=argparse.ArgumentParser(description='Offline Gaussian VAE training and evidence audits')
    sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('demo');d.add_argument('--steps',type=int,default=800)
    sub.add_parser('importance-demo')
    a=sub.add_parser('importance');a.add_argument('input',type=Path)
    args=p.parse_args()
    try:
        if args.command=='demo':report=run_demo(args.steps)
        elif args.command=='importance-demo':report=run_importance_demo()
        else:
            if args.input.stat().st_size>1000000:raise ValueError('input exceeds one MB')
            payload=json.loads(args.input.read_text())
            if not isinstance(payload,dict) or set(payload)!={'version','checkpoint','x','epsilon'}:
                raise ValueError('invalid importance input fields')
            if type(payload['version']) is not int or payload['version']!=1:
                raise ValueError('unsupported importance input version')
            report=audit_importance(VAE.from_dict(payload['checkpoint']),payload['x'],payload['epsilon'])
    except (ValueError,OverflowError,OSError) as e:p.error(str(e))
    print(json.dumps(report,indent=2,allow_nan=False,sort_keys=True))
    return 0 if all(report.get('gates',{}).values()) else 1


if __name__=='__main__':raise SystemExit(main())
