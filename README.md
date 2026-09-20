# Digitized Natya Shastra — AI-Assisted Bharatanatyam Posture Analysis

A working local application that analyses Bharatanatyam posture from video and
from labelled still images:

```
VIDEO / IMAGE
  -> ONNX pose estimation (person detector + whole-body keypoints + hand refinement)
  -> common 67-landmark representation
  -> Bharatanatyam geometric / biomechanical parameters
  -> posture recognition (parameter templates)
  -> fixed parameter-range check
  -> Correct / Incorrect + specific error
```

The video readout updates frame by frame and changes as the dancer moves
between postures. It is not a static image classifier.

---

## 1. Setup

Requires Python 3.10+ (developed on 3.12, Windows).

```bash
python -m pip install -r requirements.txt
python scripts/download_models.py        # ~610 MB of ONNX models, one time
```

The dataset stays **outside** this repository. The default location is

```
C:\Users\kavya\Desktop\VIT\SEM 9 YEAR 5\dataset\DATA FROM COLLEGE
```

Override it with an environment variable if needed:

```bash
set NATYA_DATASET_ROOT=D:\path\to\DATA FROM COLLEGE
```

---

## 2. Models

Model binaries are **not** committed (they exceed normal GitHub limits). They are
downloaded into `models/onnx/` by `scripts/download_models.py` and that folder is
git-ignored.

| Role | File | Input | Keypoints | Size |
|---|---|---|---|---|
| Person detector | `yolox_m_humanart_640.onnx` | 640×640 | — | 101 MB |
| Person detector (fast) | `yolox_s_humanart_640.onnx` | 640×640 | — | 36 MB |
| Pose — primary | `rtmw-dw-x-l_cocktail14_256x192.onnx` | 192×256 | 133 whole-body | 229 MB |
| Pose — second whole-body | `dwpose-l_ucoco_384x288.onnx` | 288×384 | 133 whole-body | 134 MB |
| Pose — body-only baseline | `rtmpose-m_halpe26_256x192.onnx` | 192×256 | 26 body+feet | 56 MB |
| Hand refinement | `rtmpose-m_hand5_256x256.onnx` | 256×256 | 21 per hand | 55 MB |

All are RTMPose-family SimCC exports from the OpenMMLab model zoo. The
pre/post-processing (bbox → centre/scale → affine warp → SimCC argmax decode) is
reimplemented in `src/pose/`, so there is no runtime dependency on
mmpose/mmdeploy.

Three **selectable pose models** are registered in `src/pose/registry.py` and all
three run through the *same* downstream pipeline. If a model file is missing, the
scripts report exactly which file, rather than substituting another model.

### Why there is a hand-refinement stage

The whole-body models carry 21 hand keypoints per hand, but they infer them from a
192×256 crop of the **whole body**. At the college camera distance the dancer's
hand covers only a few pixels of that input, and the finger geometry collapses
into a blob — the mudra parameters computed from it were meaningless. The
whole-body model is therefore used to *locate* each hand, and a dedicated
21-keypoint hand model re-estimates the fingers from the original
full-resolution pixels (the hands are 230–400 px wide there). This is recorded
per image as `hand_refined_left` / `hand_refined_right`.

---

## 3. Running the image analysis (208 labelled images)

Two stages, so the analysis can be re-run without repeating ONNX inference:

```bash
python scripts/extract_landmarks.py          # stage 1: ONNX -> landmarks_67_<model>.csv (~6 min)
python scripts/run_image_analysis.py         # stage 2: parameters, recognition, rules, summaries (~10 s)
python scripts/run_image_analysis.py --full  # both stages in one pass
python scripts/calibrate_thresholds.py       # measured values per camera / variation
```

Useful options: `--models rtmw_l_wholebody`, `--cameras "Cam 5"`.

Outputs land in `outputs/image_results/` (nothing overwrites the previous
project's CSVs, which live outside this repository):

| File | Contents |
|---|---|
| `landmarks_67_<model>.csv` | x/y/score for all 67 common landmarks, per image |
| `image_results_<model>.csv` | ground truth, landmark availability, every parameter, detected posture, status, error |
| `model_processing_summary.csv` | images processed, detections, accuracy, timing per model |
| `landmark_availability_summary.csv` | body / feet / arm / hand landmark availability per model |
| `parameter_calculation_summary.csv` | completeness and statistics for every parameter |
| `posture_correctness_summary.csv` | per-variation posture / status / error results |
| `camera_view_summary.csv` | per-camera viewpoint and accuracy |
| `error_identification_summary.csv` | per-error-code detection rate |
| `confusion_posture_<model>.csv` | posture confusion matrix |
| `parameter_thresholds_used.csv` | every threshold with its provenance |
| `data_calibrated_reference.csv` | what the labelled correct samples actually measure |

## 4. Running the video analysis

```bash
python scripts/run_video_analysis.py "path/to/video.MP4" --stride 10
python scripts/run_video_analysis.py video.MP4 --model dwpose_l_wholebody --max-frames 200
```

Writes to `outputs/video_results/`: an annotated MP4, a per-frame CSV (raw and
smoothed values) and a `_segments.csv` posture timeline.

## 5. Running the user interface

```bash
streamlit run app/streamlit_app.py
```

Upload a video, pick the ONNX model, press **Start analysis**. The page shows the
annotated frames, the detected posture, CORRECT/INCORRECT, the specific issue,
the live parameter values with their expected ranges, and a running posture
timeline.

---

## 6. Common 67-landmark representation

`src/landmarks/common.py` defines the fixed ordering required by the project:

```
0-14   pose 0-14      head, shoulders, elbows
15-24  pose 23-32     hips, knees, ankles, heels, foot index
25-45  left hand 0-20
46-66  right hand 0-20
```

(33 pose + 21 + 21 − 8 duplicated hand-related pose points = 67.)

Each model has an adapter in `src/landmarks/adapters.py`. Landmarks a layout
cannot supply stay `NaN` and are reported as unsupported — nothing is fabricated.

| Layout | Models | Body | Feet | Hands | Common slots filled |
|---|---|---|---|---|---|
| COCO-WholeBody 133 | RTMW-l, DWPose-l | yes | yes | yes | 67 / 67 |
| Halpe26 | RTMPose-m | yes | yes | **no** | 19 / 67 |

See `docs/landmark_availability.md` for the full mapping.

---

## 7. Parameter convention

Stated once and used everywhere (image, video, UI, rules, CSVs):

* `*_joint_angle_deg` is the **raw joint angle**, e.g. hip → knee → ankle.
* `*_flexion_deg` is **flexion = 180° − raw joint angle**.

All Bharatanatyam ranges in this project are expressed as **flexion**, so the
rules consume `left_knee_flexion_deg`, `right_knee_flexion_deg`,
`mean_knee_flexion_deg`. A raw joint angle is never called "flexion".

Distances are normalised by shoulder width (the unit the specification uses).
Two recognition-only cues are normalised by **torso height** instead, because
shoulder width collapses in a side view: `hip_height_torso_norm` and
`*_wrist_lift_norm`.

---

## 8. Thresholds and provenance

`src/correctness/thresholds.py` records provenance for every band:

* **PROJECT_SPEC** — fixed by the specification, never edited to flatter the data.
* **ENGINEERING_DEFAULT** — the specification left the value open, so the
  implementation chose a defensible starting value.
* **DATA_CALIBRATED** — measured from the 208 labelled images. Reported for the
  write-up; never auto-applied over a PROJECT_SPEC value.

Three definitions had to be pinned down because the specification gave a band but
not a reference point. All three are documented in the code:

* Aramandi target foot separation = **0.60 shoulder widths** (measured on the
  labelled correct Aramandi samples). The ±5% / ±15% band widths stay PROJECT_SPEC.
* Samapadham foot separation is applied to the **gap between the feet**, i.e.
  ankle-to-ankle distance minus the **0.30 shoulder widths** that remain when the
  feet are actually touching. The spec's `≤ 0.05` band is unreachable for raw
  ankle-joint distance, because an ankle joint is inset from the edge of the foot.
* Muzhumandi heel configuration uses heel lift ≥ **0.04 shoulder widths**; the
  specification only said "check whether the heel configuration is maintained".

Swastikapadam keeps no invented universal crossing-angle threshold. The crossing
angle, crossing distance, ankle placement, knee flexion, foot orientation and
trunk inclination are all computed and reported, and the correct/incorrect
decision uses the crossing distance and the Aramandi depth range.

**Weight distribution is a proxy.** RGB landmarks cannot measure ground reaction
force. "More weight on one leg" is decided from `weight_shift_proxy` (lateral
offset of the shoulder/pelvis midline from the ankle midline) and `hip_tilt_deg`,
and is labelled as a proxy in the code, the CSVs and the UI.

When several rules fail at once, the reported issue is the **most severe**
violation — how far past its correct limit the value sits, measured in
borderline-band widths — rather than whichever rule happens to be listed first.

---

## 9. Camera viewpoint gating (important)

The four cameras see the dancer from different angles, and the pipeline detects
this from the landmarks themselves (`src/features/view.py`):

| Camera | Detected view | Facing |
|---|---|---|
| Cam 5 | coronal (frontal) | front |
| Cam 6 | sagittal (profile) | — |
| Cam 7 | sagittal (profile) | — |
| Cam 8 | coronal | rear |

A single RGB camera cannot measure every parameter from every angle:

* Frontal-plane parameters (foot separation, turnout, foot-line deviation, knee
  symmetry, pelvic tilt, elbow–shoulder alignment, the weight-shift proxy) need a
  coronal view. From a profile the legs overlap and shoulder width collapses, so
  these are **blanked** rather than reported as nonsense.
* Sagittal-plane parameters (knee flexion, trunk inclination, heel lift) are
  measurable from a profile, and are in fact best seen there.
* Mudra parameters need the palm towards the camera, so they are **not
  measurable from the rear view** at all. Those frames report
  "mudra presented but finger landmarks are not measurable from this viewpoint"
  instead of guessing a leg posture.

Foot turnout is additionally suppressed when the heel-to-toe segment is shorter
than 0.15 shoulder widths, which happens in a full Muzhumandi squat where the
foot points at the camera; a turnout angle from a ten-pixel vector would be an
artefact.

---

## 10. Results on the 208 labelled images

208 images = 52 JPGs × 4 core camera folders (26 variations × 2 images each).
Only `Cam 5`, `Cam 6`, `Cam 7`, `Cam 8` are used; `Cam 6 extra`, `Cam 7 extra`,
`param_calc` and all previous CSVs are ignored.

Person detection and landmark extraction succeeded on **208/208 images for all
three models**.

| Model | Landmark slots | Mean landmarks found | Either hand usable | Posture accuracy | Status accuracy | ms/image (CPU) |
|---|---|---|---|---|---|---|
| RTMW-l whole-body | 67 / 67 | 64.2 | 77.9 % | **63.0 %** | 54.8 % | 582 |
| DWPose-l whole-body | 67 / 67 | 60.5 | 72.1 % | 61.1 % | 55.3 % | 673 |
| RTMPose-m Halpe26 | 19 / 67 | 19.0 | 0 % | 50.0 % | 50.5 % | 213 |

Body, feet and arm landmarks were available on ~100 % of images for every model
(the 3840×2736 stills are shot at a distance but the dancer is well framed).

Per camera, with the primary model:

| Camera | View | Posture accuracy | Status accuracy | Mudra measurable |
|---|---|---|---|---|
| Cam 5 | coronal front | **84.6 %** | 75.0 % | 96 % |
| Cam 6 | sagittal | 61.5 % | 40.4 % | 100 % |
| Cam 7 | sagittal | 57.7 % | 44.2 % | 100 % |
| Cam 8 | coronal rear | 48.1 % | 59.6 % | 15 % |

The frontal camera is the reference view for a frontal-plane parameter set, and
that is where the system performs best. The body-only model loses ~13 points
purely because it cannot supply finger landmarks, so its mudra-measurable rate is
0 % — which is the intended, honest behaviour of the availability reporting rather
than a silent failure.

See `docs/findings.md` for which of the 26 error distinctions are separable from a
single camera and which are not.

---

## 11. Project layout

```
data/annotations/     dataset_manifest_208.csv (generated)
models/onnx/          ONNX binaries (git-ignored, fetched by script)
src/
  config.py           paths, thresholds, DATASET_ROOT
  labels.py           the fixed 26-variation taxonomy
  dataset.py          manifest builder
  pipeline.py         single analysis entry point
  pose/               ONNX detector, SimCC pose, hand refinement, registry
  landmarks/          common 67-landmark representation + per-model adapters
  features/           geometry, body parameters, mudra parameters, view gating
  recognition/        parameter-template posture recognition
  correctness/        fixed thresholds + rule engine
  video/              smoothing, overlay, frame-by-frame processor
  evaluation/         image runner, landmark cache, summary tables
app/streamlit_app.py  user interface
scripts/              download, extract, analyse, calibrate, preview
outputs/              image_results, video_results, logs
```

## 12. Error handling

The pipeline never aborts a video because of one bad frame. Missing person,
missing hand, missing individual landmarks, low-confidence landmarks, degenerate
vectors, division by zero, invalid angles and unsupported model output are all
handled: the parameter returns `NaN`, the affected rule reports `UNKNOWN`, the
frame is marked incomplete, the last stable state is retained (for up to 45
analysed frames) and processing continues.
