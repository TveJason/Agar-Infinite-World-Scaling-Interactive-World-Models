import json
from pathlib import Path
import numpy as np
import pytest
from agar.monitor import DriftBudget
from agar.runtime import preflight, atomic_json, MOVES, VIEWS, prepare

def test_static_frames_do_not_spend_budget():
    m=DriftBudget(); x=np.zeros((4,4,3))
    for _ in range(13): assert not m.update(x,x)['stop']
    assert m.total==0

def test_cumulative_stop():
    m=DriftBudget(1.5); a=np.zeros((4,4,3)); b=a+255
    assert not m.update(a,b)['stop']
    assert m.update(b,a)['stop']

@pytest.mark.parametrize('budget',[0,-1,float('nan'),float('inf')])
def test_invalid_budget(budget):
    with pytest.raises(ValueError): DriftBudget(budget)

def test_nonfinite_frames_fail_closed():
    with pytest.raises(ValueError): DriftBudget().update(np.zeros((2,2,3)),np.full((2,2,3),float('nan')))

def test_shape_validation():
    with pytest.raises(ValueError): DriftBudget().update(np.zeros((2,2)),np.zeros((2,2)))

def test_missing_checkpoints_reported(tmp_path):
    import yaml
    config = {'checkpoint_path': str(tmp_path/'dit'), 'vae_cfg': {'vae_pth': str(tmp_path/'vae')}, 'text_encoder_cfg': {'checkpoint_path': str(tmp_path/'t5'), 'tokenizer_path': str(tmp_path/'tokenizer')}, 'validation_data': {'num_frames': 81}}
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    assert len(preflight(str(path)))==4

def test_atomic_control(tmp_path):
    p=tmp_path/'control.json'; atomic_json(p,{'move':MOVES[1],'view':VIEWS[0]})
    assert json.loads(p.read_text())['move']=='go forward'
    assert not p.with_suffix('.tmp').exists()

def test_empty_prompt_rejected():
    with pytest.raises(ValueError,match='scene'): prepare('', 'missing.jpg')

def test_prepare_preserves_settings(tmp_path,monkeypatch):
    import agar.runtime as r
    import yaml
    monkeypatch.setattr(r,'ROOT',tmp_path)
    for name in ['dit','vae','t5','image.jpg']: (tmp_path/name).write_bytes(b'weights')
    (tmp_path/'tokenizer').mkdir(); (tmp_path/'tokenizer'/'tokenizer.json').write_text('{}')
    config={'checkpoint_path':'dit','vae_cfg':{'vae_pth':'vae'},'text_encoder_cfg':{'checkpoint_path':'t5','tokenizer_path':'tokenizer'},'validation_data':{'num_frames':81}}
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(config))
    run,env=r.prepare('campus','image.jpg',config='config.yaml')
    assert json.loads((run/'request.json').read_text())['target_frames']==1041
    assert len(json.loads((run/'actions.json').read_text()))==1041
    assert env['AGAR_CHUNKS']=='13'
    assert (run/'model.yaml').read_bytes()==(tmp_path/'config.yaml').read_bytes()
