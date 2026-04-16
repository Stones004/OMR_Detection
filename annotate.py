import cv2
import json
import os

# ✅ Correct paths
BASE_DIR = r"D:/img_process/annotated"
IMG_DIR = os.path.join(BASE_DIR, "images", "default")
ANN_FILE = r"D:/img_process/annotated/annotations/default.json"

# load json
with open(ANN_FILE) as f:
    data = json.load(f)

for item in data["items"]:

    # 🔥 correct filename from CVAT
    filename = item["image"]["path"].split("/")[-1]

    img_path = os.path.join(IMG_DIR, filename)

    print(f"🔍 Loading: {img_path}")

    if not os.path.exists(img_path):
        print(f"❌ File not found: {filename}")
        continue

    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠️ Failed to load: {img_path}")
        continue

    # draw boxes
    for ann in item["annotations"]:
        if ann["type"] == "bbox":
            x, y, w, h = ann["bbox"]

            cv2.rectangle(
                img,
                (int(x), int(y)),
                (int(x+w), int(y+h)),
                (0,255,0),
                2
            )

    # save output
    out_path = os.path.join(BASE_DIR, f"annotated_{filename}")
    cv2.imwrite(out_path, img)

    print(f"💾 Saved: {out_path}")