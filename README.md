# Soft glove motion-only pipeline

An offline Python pipeline for an assistive soft glove. It reads RGB-D recordings of a hand and the objects it touches, estimates a set of per-action cues, turns them into proposals, gates them with a supervisor, drives a simulated glove state machine, and scores the result against annotated ledgers.

The software runs in simulation and offline evaluation only. No hardware interface is called.

## Status

- This is a snapshot of an earlier architecture, last edited on 16 October 2026. The project's direction has since changed, so this code shows one design, not the current one.
- Recording data, labels and participant data are not included. Scripts that process recordings expect those inputs and will not run as they are.
- Most tests use synthetic fixtures. Some expect recording data, and the suite was not run for this snapshot.

## What it does

1. **Scene geometry.** Per-hand features from 3D hand landmarks and a RANSAC table plane: heights, rates and pairwise distances that do not depend on camera pose.
2. **Contact events.** Onset and offset detection from sensor traces, using a relative-change rule.
3. **Action readouts.** Learned and rule-based readouts for each action, compared under matched parameter budgets and scored per person.
4. **Glove supervisor.** A per-hand finite-state machine with confirmation counts, a cool-down after each command, hold-on-loss, and a veto input. It emits glove commands and a per-tick trace.
5. **Replay and scoring.** Recorded packets are replayed deterministically and scored against annotated ledgers.

## Layout

| Path | Contents |
|---|---|
| `sg_offline/` | Pipeline package: scene geometry, table plane, objects, contact events, readouts, state machine, replay, training, ledgers, evaluation |
| `*.py` at the root | Command-line tools for perception, recording, replay, scoring, comparison and labelling |
| `tests/` | Pytest suite of about 350 test functions |
| `recording_script*.json` | Recording timelines read by `timed_recording.py` |
| `fsm_still_contact.json` | Example supervisor configuration |
| `rgbd_contract.md` | Recording manifest contract |

## Requirements

- Python 3
- numpy, OpenCV (`cv2`), PyTorch and ultralytics (segmentation)
- pytest for the tests
- pyrealsense2, only for the RealSense D435 adapter

No versions are pinned in this snapshot.

## Usage

```bash
python -m pytest            # tests; some need recording data that is not included
python timed_recording.py   # display the default recording timeline
```
