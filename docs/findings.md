# Findings and limitations

Measured with the primary model (RTMW-l whole-body + hand refinement) on the 208
labelled college images, unless stated otherwise. All numbers come from the CSVs
in `outputs/image_results/`.

## 1. What works

**Pose extraction is not the bottleneck.** Person detection and landmark
extraction succeeded on 208/208 images for all three ONNX models. Body, feet and
arm landmarks were available on ~100 % of images. Cross-checking the three models
on the same image produced shoulder/hip/knee/ankle positions within a few pixels
of each other, which is what validated the reimplemented SimCC pre/post-processing.

**The frontal camera supports the analysis well.** On Cam 5: 84.6 % posture
recognition and 75.0 % correct/incorrect agreement, with the mudra parameters
measurable on 96 % of images.

**The video pipeline reacts dynamically.** A 53 s, 3840×2160 clip from Cam 5
(267 analysed frames at stride 10, zero incomplete frames) produced 15 posture
transitions — Samapadham → Alapadma → Aramandi → Mushti → Pataka → Swastikapadam
— with the status flipping between CORRECT and INCORRECT as the dancer settled
into or drifted out of each posture. Temporal smoothing (9-frame majority vote
plus EMA on the displayed values) removed frame-to-frame flicker without hiding
real transitions.

## 2. Hand landmarks from a whole-body crop are unusable at this camera distance

This was the single most consequential finding. The whole-body models carry 21
hand keypoints per hand, so on paper the mudra requirement was satisfied. In
practice the models infer those points from a 192×256 crop of the **whole body**,
where the dancer's hand is a few pixels across, and the predicted finger points
collapsed into a blob that did not follow the fingers at all. The mudra
parameters computed from them showed almost no separation between the four
mudras — a "correct Alapadma" (a fully splayed lotus hand) measured an
inter-finger spread of 4.9°, and a closed Mushti fist measured a palm opening of
0.92 (it should be ~0.5).

The hands themselves are 230–400 px wide in the original 3840×2736 frames, so the
information was present; only the resolution reaching the network was the problem.
Adding a hand-refinement stage — locate the hand with the whole-body model, then
re-estimate the 21 points with a dedicated hand ONNX model on the
full-resolution crop — fixed it. After refinement the inter-finger spread
separates cleanly: Pataka 2–14°, Alapadma 35–67°.

A second detail mattered here: closed mudras legitimately hide the fingertips, so
the hand model reports low confidence for them. Applying the body confidence
threshold (0.30) deleted Mushti and Katakamukha entirely. Hand keypoints are now
held to a 0.10 floor, documented in `src/landmarks/common.py`.

## 3. A single RGB camera cannot measure every parameter from every angle

The four cameras are a frontal view (Cam 5), two profiles (Cam 6, Cam 7) and a
rear view (Cam 8). The pipeline detects this from the shoulder-width to
torso-height ratio and from the left/right shoulder ordering, and reports it per
image.

Consequences, all now handled explicitly rather than silently:

* **Profile views** collapse the shoulder span, so every shoulder-normalised
  frontal-plane parameter blows up. Before gating, `hip_height_norm` reached 22.7
  on a side view where its frontal value is ~2.2. Frontal-plane parameters are now
  blanked for sagittal views; knee flexion, trunk inclination and heel lift stay
  available and are in fact more reliable from the side.
* **The rear view** hides both palms, so no finger parameter is measurable. Only
  15 % of Cam 8 images yield a usable hand, and the eight mudra variations are
  reported as "mudra presented but finger landmarks are not measurable from this
  viewpoint" instead of being mislabelled as a leg posture. This caps Cam 8 posture
  accuracy at 48.1 % overall (51.0 % of the images it can classify at all).
* **Recognition cues** had to be re-normalised by torso height rather than
  shoulder width to survive a profile view. Doing this for the wrist-height cue
  alone moved overall posture accuracy from 44.2 % to 52.9 %, and correcting its
  threshold took it to 63.0 %.

## 4. Two specification bands needed a reference point before they could be applied

* **Samapadham foot separation ("≤ 0.05 shoulder widths").** With the feet
  actually touching, the two *ankle joints* still measure ~0.30 shoulder widths
  apart, because an ankle joint is inset from the inner edge of the foot. Applied
  to raw ankle-to-ankle distance the band is unreachable, and every Samapadham
  image — including the correct one — failed as "feet too far apart". The band is
  therefore applied to the **gap between the feet** (ankle distance minus the
  0.30 baseline). The 0.05 / 0.15 band widths are unchanged.
* **Aramandi target foot separation ("within ±5 % of target").** The
  specification gave the tolerance but not the target. Measured on the labelled
  correct Aramandi samples of the frontal camera it is 0.60 shoulder widths.

Both are recorded as `DATA_CALIBRATED` in `src/correctness/thresholds.py`
alongside the unchanged `PROJECT_SPEC` bands.

## 5. Which of the 26 error distinctions are separable

Per-error results (16 or 8 images each, across all four cameras; "flagged" means
the system called the posture INCORRECT, "exact" means it also named the right
error code):

| Labelled error | Flagged | Exact match | Assessment |
|---|---|---|---|
| `INSUFFICIENT_ARAMANDI` | 75 % | 56 % | works |
| `BACK_BENT_FORWARD` | 100 % | 50 % | works from the profile cameras |
| `FEET_TOO_FAR_APART` | 63 % | 50 % | works |
| `FEET_NOT_ALIGNED` | 75 % | 50 % | works |
| `KNEES_FACING_FORWARD` | 50 % | 38 % | partial |
| `FINGERS_BENT` | 63 % | 38 % | partial, front-facing only |
| `ELBOWS_NOT_ALIGNED` | 56 % | 25 % | partial |
| `FEET_APART_HEEL_DOWN` | 50 % | 25 % | partial |
| `FINGERS_TOO_FAR_APART` | 50 % | 25 % | partial, confused with elbow alignment |
| `INSUFFICIENT_KNEE_BEND` | 88 % | 13 % | detected, but attributed to Swastikapadam depth |
| `PINKY_DROPPED` | 38 % | 13 % | weak |
| `OVERCROSSING_SHINS` | 50 % | 0 % | **not separable** |
| `WEIGHT_ON_ONE_LEG` | 50 % | 0 % | **not separable** |
| `FEET_NOT_IN_LINE` | 63 % | 0 % | **not separable** from `FEET_NOT_ALIGNED` |

The three failures are informative rather than arbitrary:

* **`OVERCROSSING_SHINS`.** How far one shin crosses the other is largely a
  depth measurement. From the front the crossing legs occlude each other, and the
  signed ankle gap does not order the way the labels do: the two *correct*
  crossings measure +0.02 and −0.14 shoulder widths while the two *overcrossed*
  ones measure −0.04 and −0.49, so the correct and incorrect ranges interleave.
  The specification explicitly says no universal
  crossing-angle threshold exists; the geometry is computed and reported, but the
  labelled samples do not support a reliable binary cut from one RGB view.
* **`WEIGHT_ON_ONE_LEG`.** This is a ground-reaction-force distinction. The
  landmark proxy (lateral offset of the shoulder/pelvis midline from the ankle
  midline, plus pelvic tilt) does respond, but in this dataset the demonstration
  of "more weight on one leg" also widens the stance, and the foot-separation
  violation is numerically more severe. A force plate or pressure mat would be
  needed to measure this properly; the code and UI label it as a proxy throughout.
* **`FEET_NOT_IN_LINE` vs `FEET_NOT_ALIGNED`.** These are the Aramandi and
  Samapadham versions of "the feet form a V instead of a line". Both reduce to the
  same two measurements (heel-line deviation and turnout asymmetry), so the system
  reports the posture-appropriate one; when the posture itself is confused, the
  error code follows.

`INSUFFICIENT_KNEE_BEND` is the clearest case of an error that is *detected* but
*attributed* to a neighbouring posture: a shallow Aramandi looks geometrically
like Swastikapadam depth once the recogniser has to choose, so the verdict is
INCORRECT (88 %) with the right cause but the neighbouring label.

## 6. Cost

CPU-only, single-threaded ONNX Runtime on the development machine:

| Stage | Cost |
|---|---|
| YOLOX-m person detection, 640×640 | ~400 ms/frame |
| RTMW-l whole-body, 192×256 | ~95 ms |
| Hand refinement, 2 × 256×256 | ~80 ms |
| Parameters + recognition + rules | < 2 ms |

Detection dominates, which is why the video path reuses the previous person box
for several frames (`--detector-interval`) and the UI exposes a frame stride. With
stride 10 and a detector interval of 6, a 4K clip processed at ~1.6 analysed
frames per second. A GPU execution provider, or the lighter `yolox_s` detector,
would be the first thing to change for real-time use.

## 7. What was deliberately not done

* No deep-learning posture classifier was trained. Parameter templates were
  sufficient, are inspectable, and keep the correctness logic tied to measurable
  geometry.
* No new ONNX pose models were benchmarked beyond the three registered ones plus
  the hand-refinement model that the mudra requirement made necessary.
* The previous project's CSVs (`67_landmarks_all.csv`, `pose_parameters_all.csv`,
  `parameter_thresholds.csv`, …) were neither read nor overwritten. They remain
  untouched in the dataset folder.
