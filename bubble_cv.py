import cv2
import numpy as np
import json
import os

ANNOTATION_FILE = r"G:/annotations/default.json"
IMAGE_DIR = r"G:/images/default"
OUTPUT_DIR = r"G:/OMR_Detection/output_final"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD JSON
# =========================
with open(ANNOTATION_FILE, "r") as f:
    data = json.load(f)

# =========================
# FILLED DETECTION
# =========================
def is_filled(gray, bbox):

    x, y, w, h = map(int, bbox)

    crop = gray[y:y+h, x:x+w]

    if crop.size == 0:
        return False

    # threshold inside bubble
    _, thresh = cv2.threshold(crop, 150, 255, cv2.THRESH_BINARY_INV)

    filled_pixels = cv2.countNonZero(thresh)
    total_pixels = crop.shape[0] * crop.shape[1]

    fill_ratio = filled_pixels / total_pixels

    return fill_ratio > 0.4

# =========================
# PROCESS IMAGE
# =========================
def process_item(item):

    img_path = os.path.join(IMAGE_DIR, item["image"]["path"])
    print(f"\n📥 Processing: {img_path}")

    img = cv2.imread(img_path)

    if img is None:
        print("❌ Failed to load")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    omr_bbox = None
    bubble_bboxes = []

    # extract annotations
    for ann in item["annotations"]:
        label_id = ann["label_id"]

        if label_id == 0:  # OMR
            omr_bbox = ann["bbox"]

        elif label_id == 2:  # bubbles
            bubble_bboxes.append(ann["bbox"])

    if omr_bbox is None:
        print("⚠️ No OMR found")
        return

    # crop OMR region
    x, y, w, h = map(int, omr_bbox)
    omr = img[y:y+h, x:x+w]
    gray_omr = gray[y:y+h, x:x+w]

    results = []

    for bbox in bubble_bboxes:

        bx, by, bw, bh = map(int, bbox)

        # adjust relative to OMR crop
        bx_rel = bx - x
        by_rel = by - y

        filled = is_filled(gray, bbox)

        results.append((bx_rel, by_rel, bw, bh, filled))

    # draw
    out = omr.copy()

    for (bx, by, bw, bh, filled) in results:
        color = (0,0,255) if filled else (0,255,0)
        cv2.rectangle(out, (bx, by), (bx+bw, by+bh), color, 2)

    name = item["image"]["path"].split(".")[0]
    out_path = os.path.join(OUTPUT_DIR, f"{name}_result.png")

    cv2.imwrite(out_path, out)

    print(f"✔ Bubbles processed: {len(results)}")

# =========================
# RUN
# =========================
for item in data["items"]:
    process_item(item)

print("\n✅ DONE")