import argparse
import subprocess
import sys
from agar.runtime import ROOT, prepare, preflight, MOVES, VIEWS

def main():
    p=argparse.ArgumentParser(description='Agar local inference')
    p.add_argument('--check',action='store_true')
    p.add_argument('--config',default='configs/infworld_config.yaml')
    p.add_argument('--prompt',default='A serene campus walkway lined with modern glass buildings and maple trees.')
    p.add_argument('--image',default='assets/example_case/0001.jpg')
    p.add_argument('--chunks',type=int,default=13)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=30)
    p.add_argument('--move',choices=MOVES,default='go forward')
    p.add_argument('--view',choices=VIEWS,default='no-op')
    p.add_argument('--budget',type=float)
    a=p.parse_args()
    if a.check:
        errors=preflight(a.config)
        print('\n'.join(errors) if errors else 'All checkpoint paths exist; GPU/model load still needs validation.')
        return bool(errors)
    run,env=prepare(a.prompt,a.image,a.chunks,a.move,a.view,a.config,a.seed,a.steps,a.budget)
    print(f'Outputs: {run}',flush=True)
    return subprocess.call([sys.executable,'-u','scripts/infworld_inference.py'],cwd=ROOT,env=env)
if __name__=='__main__': sys.exit(main())
