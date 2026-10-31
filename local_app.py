"""Launch with python local_app.py. Model stays in an isolated subprocess."""
import json
import subprocess
import sys
import time
from pathlib import Path
from agar.runtime import ROOT, MOVES, VIEWS, prepare, atomic_json

def generate(prompt, image, chunks, move, view, config, seed, steps, monitor, budget, state):
    run, env = prepare(prompt, image or 'assets/example_case/0001.jpg', chunks, move, view,
                       config, seed, steps, budget if monitor else None)
    # Store before yielding so the control buttons can target this run.
    state['run'] = str(run)
    with (run/'console.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-u', 'scripts/infworld_inference.py'],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                previews = sorted(run.glob('chunk-*.mp4'))
                yield str(previews[-1]) if previews else None, (run/'console.log').read_text()[-12000:]
                time.sleep(2)
            previews = sorted(run.glob('chunk-*.mp4'))
            status = (run/'console.log').read_text()[-12000:]
            if process.returncode: status += f'\nGeneration failed (exit {process.returncode}). See console.log.'
            yield str(previews[-1]) if previews else None, status
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()

def control(move, view, state, stop=False):
    if not state.get('run'): return 'Start a run first.'
    run = Path(state['run'])
    if stop: (run/'STOP').touch(); return 'Stop requested after the current chunk.'
    atomic_json(run/'control.json', {'move': move, 'view': view})
    return f'Next chunk: {move}, {view}'

def main():
    import gradio as gr
    with gr.Blocks(title='Agar Infinite World') as app:
        gr.Markdown('# Agar Infinite World\nLocal world-model laboratory. Start image required; default campus example included.\nControls apply at chunk boundaries, not in real time. GPU checkpoints required.')
        state = gr.State({})
        prompt = gr.Textbox(value='A serene campus walkway lined with modern glass buildings and maple trees.', label='Scene description')
        image = gr.Image(type='filepath', label='Starting image (optional: uses bundled campus image)')
        with gr.Row():
            move = gr.Dropdown(MOVES, value='go forward', label='Movement')
            view = gr.Dropdown(VIEWS, value='no-op', label='Camera')
            chunks = gr.Slider(1, 50, value=13, step=1, label='Chunks (13 = 1,041 frames)')
        with gr.Accordion('Model and experiment settings', open=False):
            config = gr.Textbox(value='configs/infworld_config.yaml', label='Model config')
            seed = gr.Number(value=42, precision=0, label='Seed')
            steps = gr.Slider(1,100,value=30,step=1,label='Sampling steps')
            monitor = gr.Checkbox(value=False, label='Experimental Gilles–Dini inspired appearance budget')
            budget = gr.Number(value=1, label='Uncalibrated cumulative drift threshold')
        with gr.Row():
            start = gr.Button('Generate world', variant='primary')
            apply = gr.Button('Apply actions to next chunk')
            stop = gr.Button('Stop after current chunk')
        action_status = gr.Textbox(label='Control status')
        video = gr.Video(label='Latest generated chunk')
        logs = gr.Textbox(label='Progress', lines=12)
        start.click(generate, [prompt,image,chunks,move,view,config,seed,steps,monitor,budget,state], [video,logs], concurrency_limit=1)
        apply.click(control,[move,view,state],action_status,queue=False)
        stop.click(lambda m,v,s: control(m,v,s,True),[move,view,state],action_status,queue=False)
    app.queue().launch(server_name='127.0.0.1', share=False)

if __name__ == '__main__': main()
