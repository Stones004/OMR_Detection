\# 📄 OMR Detection System --- Full Architecture & Pipeline
Documentation

\-\--

\# 🧠 1. Overview

This project implements a \*\*hybrid OMR (Optical Mark Recognition)
system\*\* combining:

\* Deep Learning (\*\*YOLOv5\*\*) → detection \* Rule-based logic →
structure enforcement \* Synthetic data → robustness

\-\--

\## 🎯 Objective

Detect \*\*filled bubbles in structured OMR sheets\*\* with the
constraint:

\> \*\*Each band must have at most one filled bubble per column (A, B,
C)\*\*

\-\--

\# 🧩 2. High-Level Architecture

\`\`\` RAW IMAGE ↓ Preprocessing ↓ YOLO Inference ↓ Raw Predictions
(.txt) ↓ \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--
POST-PROCESSING PIPELINE
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-- ↓ 1.
Deduplication ↓ 2. Constraint Enforcement ↓
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-- Final
Structured Output
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-- \`\`\`

\-\--

\# 📁 3. Project Structure

\`\`\` project/ │ ├── raw_ans_scripts/ \# Original images ├──
processed_images/ \# Preprocessed images ├── synthetic_data/ \#
Generated synthetic dataset │ ├── images/ │ └── labels/ │ ├──
synthetic_data_processed/ \# Preprocessed synthetic data ├── runs/ \#
YOLO outputs │ ├── pre_processing_new.py \# Image preprocessing ├──
train_yolo.py \# Training script ├── model_check.py \# Inference script
├── image_generator.py \# Synthetic data generator │ ├──
post_processing/ │ ├── deduplicate.py │ ├── constraint_filter.py │ ├──
pipeline.py │ └── README.md \`\`\`

\-\--

\# ⚙️ 4. Module Breakdown

\-\--

\## 🧩 4.1 Preprocessing Module

\### File:

\`pre_processing_new.py\`

\### Purpose:

\* Normalize images \* Remove noise \* Improve contrast \* Align OMR
sheets

\### Input:

\`\`\` raw_ans_scripts/ \`\`\`

\### Output:

\`\`\` processed_images/ \`\`\`

\-\--

\## 🧩 4.2 Synthetic Data Generator

\### File:

\`image_generator.py\`

\### Purpose:

\* Generate training data with:

\* Different shading styles \* Noise variations \* Skew & blur

\### Key Design:

\* The \*\*OMR Margin\*\*, \*\*shaded bubbles and the horizontal bands
are labelled. \*\*  \* Simulates real-world marking variability

\### Output:

\`\`\` synthetic_data/images/ synthetic_data/labels/ \`\`\`

\-\--

\## 🧩 4.3 YOLO Training Module

\### File:

\`train_yolo.py\`

\### Purpose:

Train YOLOv5 model on:

\* real + synthetic data

\### Classes:

\`\`\` 0 → OMR region 1 → Band 2 → Filled bubble \`\`\`

\### Output:

\`\`\` runs/train/exp/weights/best.pt \`\`\`

\-\--

\## 🧩 4.4 Inference Module

\### File:

\`model_check.py\`

\### Purpose:

Run YOLO model on processed images

\### Key Parameters:

\`\`\`python conf = 0.6 iou = 0.5 \`\`\`

\### Output:

\`\`\` raw_inference/ ├── images/ └── labels/ \`\`\`

Each \`.txt\` file contains:

\`\`\` class x_center y_center width height confidence \`\`\`

\-\--

\# 🔧 5. Post-Processing Pipeline (CORE SYSTEM)

\-\--

\## 🧩 5.1 Deduplication Module

\### File:

\`deduplicate.py\`

\### Purpose:

Remove \*\*overlapping duplicate detections\*\*

\### Key Idea:

\* Keep highest confidence box \* Remove overlapping boxes using IoU

\-\--

\### Code:

\`\`\`python def deduplicate_boxes(boxes, iou_thresh=0.6): boxes =
sorted(boxes, key=lambda x: x\[4\], reverse=True) final = \[\]

while boxes: best = boxes.pop(0) final.append(best)

boxes = \[ b for b in boxes if compute_iou(best, b) \< iou_thresh \]

return final \`\`\`

\-\--

\## 🧩 5.2 Constraint Enforcement Module

\### File:

\`constraint_filter.py\`

\-\--

\### Purpose:

Enforce structural rule:

\`\`\` Each band → max 1 bubble per column \`\`\`

\-\--

\### Logic:

1\. Group bubbles inside each band 2. Divide band into 3 columns 3.
Select highest confidence bubble per column

\-\--

\### Code:

\`\`\`python def enforce_constraints(bands, bubbles): final = \[\]

for band in bands: bx, by, bw, bh, \_ = band

band_bubbles = \[ b for b in bubbles if by \< b\[1\] \< by + bh \]

col_w = bw / 3 cols = {0: \[\], 1: \[\], 2: \[\]}

for b in band_bubbles: x, y, w, h, conf = b col_idx = int((x - bx) /
col_w) col_idx = min(max(col_idx, 0), 2) cols\[col_idx\].append(b)

for col in cols.values(): if col: best = max(col, key=lambda x: x\[4\])
final.append(best)

return final \`\`\`

\-\--

\## 🧩 5.3 Pipeline Wrapper

\### File:

\`pipeline.py\`

\-\--

\### Purpose:

Combine all post-processing steps

\-\--

\### Code:

\`\`\`python def full_pipeline(preds): omr, bands, bubbles =
split_classes(preds)

bands = deduplicate_boxes(bands) bubbles = deduplicate_boxes(bubbles)

final_bubbles = enforce_constraints(bands, bubbles)

\# Extract coordinates for downstream text extraction bubble_coords = \[
{ \"x_center\": b\[0\], \"y_center\": b\[1\], \"width\": b\[2\],
\"height\": b\[3\], \"confidence\": b\[4\] } for b in final_bubbles \]

return { \"bands\": bands, \"raw_bubbles\": bubbles, \"final_bubbles\":
final_bubbles, \"bubble_coords\": bubble_coords } \`\`\`

\-\--

\# 🔄 6. Data Flow Across System

\-\--

\## 📌 Training Flow

\`\`\` raw_ans_scripts ↓ pre_processing_new.py ↓ processed_images ↓
image_generator.py (optional) ↓ synthetic_data ↓ train_yolo.py ↓ best.pt
\`\`\`

\-\--

\## 📌 Inference Flow

\`\`\` input image ↓ pre_processing_new.py ↓ model_check.py ↓ YOLO
predictions (.txt) ↓ deduplicate.py ↓ constraint_filter.py ↓ final
structured output \`\`\`

\-\--

\# 🧠 7. Key Design Principles

\-\--

\## 🔹 1. Separation of Concerns

\| Component \| Responsibility \| \| \-\-\-\-\-\-\-\-\-\-\-\-\-\-- \|
\-\-\-\-\-\-\-\-\-\-\-\-\-- \| \| YOLO \| Detection \| \|
Post-processing \| Logic \| \| Generator \| Data diversity \|

\-\--

\## 🔹 2. Detection vs Decision

\`\`\` Model → \"possible bubbles\" Logic → \"valid bubbles\" \`\`\`

\-\--

\## 🔹 3. Preserve Raw Data

Always keep:

\`\`\` raw_predictions/ \`\`\`

For:

\* retraining \* debugging \* analysis

\-\--

\## 🔹 4. Structured Problem Handling

OMR is not generic detection:

\`\`\` Layout → deterministic Detection → probabilistic \`\`\`

\-\--

\# ⚠️ 8. Known Challenges

\-\--

\### ❌ Duplicate detections

Solved via deduplication

\-\--

\### ❌ Multiple bubbles per column

Solved via constraint enforcement

\-\--

\### ❌ Synthetic bias

Mitigated by:

\* realistic generator \* post-processing

\-\--

\# 🚀 9. Future Improvements

\-\--

\## 🔹 1. Band alignment correction

Improve vertical consistency

\-\--

\## 🔹 2. Grid-based bubble inference

Replace detection with geometry

\-\--

\## 🔹 3. Fill classification model

Binary classifier for bubble fill

\-\--

\## 🔹 4. Active learning loop

\`\`\` Predict → Correct → Retrain \`\`\`

\-\--

\# 🏁 10. Final Summary

\-\--

\### Your system now:

✔ Uses YOLO for perception ✔ Uses rules for correctness ✔ Uses synthetic
data for robustness

\-\--

\### Core Innovation:

\`\`\` Detection + Structure = Reliable OMR System \`\`\`

\-\--

\### Final Insight:

\> \*\*Accuracy comes more from system design than model size\*\*

\-\--

\# 📌 End of Document
