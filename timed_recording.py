"""Display a recording script; --run times prompts only and operates no camera/glove.

The default is the accepted 270-second script (recording_script.json). ``--script PATH`` displays another
timeline, e.g. recording_script_v2_proposed.json (release, same-object bimanual and close-objects blocks,
proposed and awaiting decision; not a validated study protocol)."""
import argparse,json,time
from pathlib import Path


def load_script(path):
    script=json.loads(Path(path).read_text())
    previous=0
    for b in script['blocks']:
        if b['start_s']!=previous or b['end_s']<=previous:raise ValueError('invalid script timeline')
        previous=b['end_s']
    if previous!=script['duration_s']:raise ValueError('script duration mismatch')
    return script


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0]);ap.add_argument('--run',action='store_true')
    ap.add_argument('--script',type=Path,default=Path(__file__).with_name('recording_script.json'))
    a=ap.parse_args(argv)
    script=load_script(a.script)
    print(f"script: {a.script.name} ({script.get('status','')}; {script['duration_s']} s)",flush=True)
    origin=time.monotonic()
    if a.run:
        # Wall-clock start so the prompts can be aligned with the camera file afterwards.
        print(f"started at {time.strftime('%Y-%m-%dT%H:%M:%S%z')} (host wall clock)",flush=True)
    for block in script['blocks']:
        if a.run:
            while time.monotonic()-origin<block['start_s']:time.sleep(.1)
        print(f"{block['start_s']:3d}–{block['end_s']:3d} s: {block['instruction']}",flush=True)
    if a.run:
        while time.monotonic()-origin<script['duration_s']:time.sleep(.1)
        print('Script complete.',flush=True)
if __name__=='__main__':main()
