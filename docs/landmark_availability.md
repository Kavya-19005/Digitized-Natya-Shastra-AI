# Landmark availability per ONNX model

The downstream code never touches a model's raw output. Every model is mapped
into the common 67-landmark representation by an adapter in
`src/landmarks/adapters.py`, and any landmark a layout cannot supply stays `NaN`
with score `0.0`.

## Common 67-landmark ordering (fixed)

| Common index | Landmark |
|---|---|
| 0 | nose |
| 1–3 | left eye inner / left eye / left eye outer |
| 4–6 | right eye inner / right eye / right eye outer |
| 7–8 | left ear / right ear |
| 9–10 | mouth left / mouth right |
| 11–12 | left shoulder / right shoulder |
| 13–14 | left elbow / right elbow |
| 15–16 | left hip / right hip |
| 17–18 | left knee / right knee |
| 19–20 | left ankle / right ankle |
| 21–22 | left heel / right heel |
| 23–24 | left foot index (big toe) / right foot index |
| 25–45 | left hand 0–20 |
| 46–66 | right hand 0–20 |

Hand block offsets: 0 wrist, 1–4 thumb CMC/MCP/IP/tip, 5–8 index MCP/PIP/DIP/tip,
9–12 middle, 13–16 ring, 17–20 pinky.

## COCO-WholeBody 133 — RTMW-l, DWPose-l

Raw layout: 0–16 body (COCO), 17–22 feet, 23–90 face (68 points), 91–111 left
hand, 112–132 right hand.

| Common | Raw | Note |
|---|---|---|
| nose, left/right eye, left/right ear | 0, 1, 2, 3, 4 | direct |
| left/right shoulder, elbow | 5, 6, 7, 8 | direct |
| left/right hip, knee, ankle | 11–16 | direct |
| left foot index / left heel | 17 / 19 | big toe is used as "foot index" |
| right foot index / right heel | 20 / 22 | |
| eye inner/outer corners | face 36, 39, 42, 45 | from the 68-point face block |
| mouth left / right | face 54 / 48 | from the 68-point face block |
| left hand 0–20 | 91–111 | replaced by the hand-refinement stage |
| right hand 0–20 | 112–132 | replaced by the hand-refinement stage |

**All 67 common slots are filled.** Extras carried alongside (no common slot):
`left_small_toe` (18), `right_small_toe` (21).

## Halpe26 — RTMPose-m

Raw layout: 0–16 body (COCO), 17 head, 18 neck, 19 pelvis, 20/21 left/right big
toe, 22/23 left/right small toe, 24/25 left/right heel.

Maps the same 19 body + feet slots as above. **No hand block**: common indices
25–66 are unsupported, so every mudra parameter is `NaN` and
`mudra_measurable = 0` for every image. This model therefore supports the body
postures (Aramandi, Samapadham, Muzhumandi, Swastikapadam) but cannot support
Pataka, Alapadma, Mushti or Katakamukha.

`COCO17` is also registered in the adapter module for plain 17-keypoint body
models; it additionally lacks heels and toes, so foot orientation and heel
configuration would be unavailable. No model currently uses it.

## Hand refinement

For layouts with `has_hands = True`, `src/pose/hand_refiner.py` re-estimates both
hand blocks from the original full-resolution frame:

1. locate the hand from the whole-body model's 21 coarse hand points (falling
   back to the wrist, with the box size bounded below by the forearm length),
2. run `rtmpose-m_hand5_256x256.onnx` on that crop,
3. overwrite common indices 25–45 and 46–66, recording
   `hand_refined_left` / `hand_refined_right`.

Hand keypoints are held to a lower confidence floor than body keypoints
(`HAND_SCORE_FLOOR = 0.10` vs `0.30`) because closed mudras such as Mushti and
Katakamukha genuinely hide the fingertips behind the palm. Judging them at the
body threshold erased those mudras entirely.

## Measured availability (208 images)

| Model | Slots supported | Mean slots found | Body | Feet | Arms | Left hand | Right hand | Either hand |
|---|---|---|---|---|---|---|---|---|
| RTMW-l | 67 | 64.2 | 100 % | 100 % | 100 % | 68.8 % | 72.1 % | 77.9 % |
| DWPose-l | 67 | 60.5 | 99.5 % | 100 % | 99.5 % | 64.9 % | 56.2 % | 72.1 % |
| RTMPose-m Halpe26 | 19 | 19.0 | 100 % | 99.0 % | 100 % | 0 % | 0 % | 0 % |

The hand percentages are dominated by viewpoint rather than by model quality: the
rear camera (Cam 8) hides both palms, and a hand is only counted as usable when
the wrist plus at least 15 of the 21 points are present.
