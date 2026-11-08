import argparse
import json
from .demo import run_demo


def main():
    p = argparse.ArgumentParser(description='Train an offline 2-D Gaussian VAE fixture')
    p.add_argument('command', choices=['demo'])
    p.add_argument('--steps', type=int, default=800)
    args = p.parse_args()
    try:
        report = run_demo(args.steps)
    except (ValueError, OverflowError) as e:
        p.error(str(e))
    print(json.dumps(report, indent=2, allow_nan=False, sort_keys=True))
    return 0 if all(report['gates'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
