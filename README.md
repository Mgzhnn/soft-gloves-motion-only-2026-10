# Soft glove: motion-only perception and control pipeline

An offline Python pipeline for an assistive soft glove that decides what the wearer wants from motion alone. It reads RGB-D recordings of the hands and the objects they touch, turns hand and object motion into per-action cues, scores what each hand is asking for, passes accepted proposals through a simulated glove state machine, and scores the result against annotated timelines.

The glove is a cable-driven soft hand with actuator housings on the wrist. This repository is the software side. It runs in simulation and offline evaluation; nothing here commands hardware.

![Offline replay frame: boxes on both gloved hands, object outlines and per-hand decisions](docs/images/v3_replay_boxes.jpg)

An offline replay frame: green boxes on both gloved hands, object outlines, hand-to-object distance lines and the per-hand decision ("no new request").

## How it works

1. **Perception.** Hand detections and three functional points per hand (thumb MCP, index MCP, trapezium) are lifted to 3-D from registered depth. Objects come from a fitted table plane, from colour, or from segmentation masks.
2. **Scene cues.** A seeded RANSAC table plane gives camera-pose-independent quantities: heights above the table, vertical rates, in-plane distances, swept angles and shake reversals.
3. **Readouts.** Each tick, a readout scores the joint request of both hands over seven outcomes: no new request (NQ), grasp (G), index-finger press (FP), rotate while holding (AR), shake while holding (NT), hold over a target (HP) and release (R). Readouts range from transparent hand-written rules to small learned models (factorised per-hand models, a flat MLP, a set-pooling MLP and a linear mixture over candidate objects, and causal GRU, TCN and transformer sequence models), compared under matched parameter budgets.
4. **Supervisor.** A per-hand state machine (open, closing, closed, tightening, tight, extending index, releasing, cool-down, stopped) turns confirmed proposals into glove commands. It has confirmation counts, a cool-down after every command, a hold rule when the hand is lost, a veto that opens the glove at once and an external stop.
5. **Replay and scoring.** Stored packets replay deterministically. Every request is scored by its first proposal (compatible, wrong action, too early, too late or missed), and unwanted proposals are counted per labelled no-request hand-minute. Comparisons hold out one person at a time and choose thresholds only on development or calibration people.

Everything that runs at decision time is causal: a decision at time t uses only observations available by t, and annotation never enters the runtime path.

## Quick start

Python 3.12. CPU only; no GPU is needed for anything below.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-learning.txt --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q
```

The learning requirements add PyTorch for the sequence readouts. Without them, those tests are skipped and everything else runs.

Two end-to-end checks run on synthetic data and print a summary:

```bash
python run_motion_checks.py
```

```bash
python run_rgbd_checks.py
```

## Working with recordings

Recording data, labels and participant data are not part of this repository. With your own RGB-D recordings, the path is:

| Step | Tool |
|---|---|
| Convert a RealSense D435 recording or an image-sequence export to a recording manifest | `d435_recording_adapter.py`, `image_sequence_adapter.py`, `inspect_d435_recording.py` |
| Segment objects (optional, SAM 2) | `segment_objects.py` |
| Run hand and object perception, write packets, scene records and a trace | `run_perception.py` |
| Rebuild scene records for an older run | `build_scene.py` |
| Replay stored packets under another configuration, optionally through the supervisor | `replay_packets.py` |
| Turn a per-hand timeline CSV into an annotation ledger | `import_requests.py` |
| Score one session | `score_requests.py` |
| Compare readouts across people, holding one person out at a time | `run_comparison.py` |
| Fit and replay on one person as a smoke test | `fit_diagnostic.py` |
| Choose the rate window on a development clip | `choose_rate_window.py` |
| Label hand points by hand (window or browser) | `label_frames.py`, `label_frames_web.py` |
| Evaluate perception against labels, or compare object providers | `evaluate_perception.py`, `compare_object_providers.py` |
| Export hand labels for detector adaptation | `prepare_hand_training.py` |
| Time a recording session from a script | `timed_recording.py` |
| Zero-shot hand models and colour-segmentation diagnostics | `zero_shot_hands.py`, `bench_segments.py`, `run_d435_bench_observations.py` |

Each tool documents its inputs in its module docstring and in `--help`. The recording format is specified in [`rgbd_contract.md`](rgbd_contract.md).

Perception needs model code that is not installed by the requirements files: a WiLoR checkout (hand detection and landmarks, with Ultralytics), optionally HaMeR, SAM 2 for object segmentation, and `pyrealsense2` for native RealSense files. `run_perception.py` uses a CUDA GPU when PyTorch finds one and the CPU otherwise.

## Repository layout

| Path | Contents |
|---|---|
| `sg_offline/contracts.py`, `inputs.py` | Observation contracts and the packet reader |
| `sg_offline/rgbd.py`, `recordings.py`, `tableplane.py` | RGB-D geometry, recording manifests and table-plane fitting |
| `sg_offline/perception.py`, `objects.py`, `appearance.py` | Hand and object observations |
| `sg_offline/scene.py`, `state.py` | Scene cues and causal motion state |
| `sg_offline/inference.py`, `sequence.py`, `training.py` | Readouts, sequence models and training under person-level splits |
| `sg_offline/replay.py` | Deterministic replay |
| `sg_offline/fsm.py` | Glove supervisor state machine |
| `sg_offline/ledger.py`, `evaluation.py`, `stats.py`, `experiment.py` | Annotation ledgers, scoring, per-person accounting and comparisons |
| `sg_offline/contact.py` | Contact-sensor reference events (evaluation only) |
| `*.py` at the root | Command-line tools |
| `tests/` | Pytest suite on synthetic fixtures |
| `config.json` | Replay configuration used by `run_motion_checks.py` |
| `fsm_still_contact.json` | Example supervisor configuration |
| `recording_script*.json` | Recording session scripts for `timed_recording.py` |

## Limits

- Simulation and offline evaluation only. No hardware interface is called.
- The glove's closing and opening times, the flexion targets and the object-size adjustment are declared placeholders, not measurements.
- The tests use synthetic fixtures. Passing them shows the software behaves as specified, not that the pipeline works for real users.

## License

MIT. See [LICENSE](LICENSE).
