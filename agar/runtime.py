"""Shared CPU-only preflight and request preparation."""
import json
import os
from pathlib import Path
import sys
import uuid
import yaml

ROOT = Path(__file__).resolve().parents[1]
MOVES = ['no-op', 'go forward', 'go back', 'go left', 'go right',
         'go forward and go left', 'go forward and go right',
         'go back and go left', 'go back and go right', 'uncertain']
VIEWS = ['no-op', 'turn up', 'turn down', 'turn left', 'turn right',
         'turn up and turn left', 'turn up and turn right',
         'turn down and turn left', 'turn down and turn right', 'uncertain']

def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    os.replace(temp, path)

def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path

def preflight(config):
    config = resolve(config)
    data = yaml.safe_load(config.read_text())
    paths = [('DiT', data['checkpoint_path'], False),
             ('VAE', data['vae_cfg']['vae_pth'], False),
             ('T5', data['text_encoder_cfg']['checkpoint_path'], False),
             ('tokenizer', data['text_encoder_cfg']['tokenizer_path'], True)]
    errors = []
    for name, path, directory in paths:
        p = resolve(path)
        valid = p.is_dir() and any(p.iterdir()) if directory else p.is_file() and p.stat().st_size > 0
        if not valid: errors.append(f'{name}: missing or empty {p}')
    if data['validation_data']['num_frames'] != 81:
        errors.append('This adapter requires validation_data.num_frames=81')
    return errors

def prepare(prompt, image, chunks=13, move='go forward', view='no-op',
            config='configs/infworld_config.yaml', seed=42, steps=30, budget=None):
    if not prompt.strip(): raise ValueError('Enter a scene description')
    if not 1 <= int(chunks) <= 50: raise ValueError('Chunks must be 1–50')
    if not 1 <= int(steps) <= 100: raise ValueError('Steps must be 1–100')
    if move not in MOVES or view not in VIEWS: raise ValueError('Unknown action')
    if budget is not None:
        from agar.monitor import DriftBudget
        DriftBudget(float(budget))
    image = resolve(image)
    if not image.is_file(): raise ValueError(f'Start image not found: {image}')
    errors = preflight(config)
    if errors: raise ValueError('\n'.join(errors))
    run = ROOT / 'outputs' / uuid.uuid4().hex
    run.mkdir(parents=True)
    (run / 'prompts.yaml').write_text(yaml.safe_dump({'prompts': [[prompt, str(image), str(run / 'actions.json')]]}))
    atomic_json(run / 'actions.json', [{'move': move, 'view': view}] * (int(chunks)*80+1))
    atomic_json(run / 'control.json', {'move': move, 'view': view})
    request = dict(prompt=prompt, image=str(image), chunks=int(chunks), seed=int(seed),
                   steps=int(steps), config=str(resolve(config)), budget=budget,
                   target_frames=1+80*int(chunks), python=sys.version)
    atomic_json(run / 'request.json', request)
    # Snapshot the model configuration; never mutate the user's file.
    (run / 'model.yaml').write_bytes(resolve(config).read_bytes())
    env = os.environ.copy()
    env.update(AGAR_RUN=str(run), AGAR_CONFIG=str(run/'model.yaml'),
               AGAR_PROMPTS=str(run/'prompts.yaml'), AGAR_CHUNKS=str(int(chunks)),
               AGAR_SEED=str(int(seed)), AGAR_STEPS=str(int(steps)))
    if budget is not None: env['AGAR_BUDGET'] = str(float(budget))
    else: env.pop('AGAR_BUDGET', None)
    return run, env
