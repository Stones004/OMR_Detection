import cv2
import numpy as np
import os
import pytesseract

# =========================
# HARD-CODED OMR REGION
# =========================
X1, Y1 = 80, 150
X2, Y2 = 310, 3450

OUTPUT_DIR = "output_bubbles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# PARAMETERS (LOOSE DETECTION)
# =========================
MIN_AREA = 120
MAX_AREA = 2500
CIRCULARITY_THRESH = 0.5
SOLIDITY_THRESH = 0.4


# =========================
# PREPROCESS
# =========================
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 3
    )

    return gray, thresh


# =========================
# FIND BUBBLES (LOOSE)
# =========================
def find_bubbles(thresh):
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bubbles = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < MIN_AREA or area > MAX_AREA:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity < CIRCULARITY_THRESH:
            continue

        # solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            continue

        solidity = area / hull_area

        if solidity < SOLIDITY_THRESH:
            continue

        bubbles.append(cnt)

    return bubbles


# =========================
# GROUP INTO ROWS
# =========================
def group_rows(contours):
    centers = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w//2
        cy = y + h//2
        centers.append((cx, cy, cnt))

    centers.sort(key=lambda x: x[1])

    rows = []
    current_row = []
    threshold = 25

    for pt in centers:
        if not current_row:
            current_row.append(pt)
            continue

        if abs(pt[1] - current_row[0][1]) < threshold:
            current_row.append(pt)
        else:
            rows.append(current_row)
            current_row = [pt]

    if current_row:
        rows.append(current_row)

    return rows


# =========================
# SORT ROW LEFT → RIGHT
# =========================
def sort_row(row):
    return sorted(row, key=lambda x: x[0])


# =========================
# FILLED CHECK (ROBUST)
# =========================
def is_filled(gray, cnt):
    mask = np.zeros(gray.shape, dtype="uint8")
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    mean_val = cv2.mean(gray, mask=mask)[0]

    return mean_val < 160   # 🔥 key threshold




'''def is_filled(gray, cnt):
    mask = np.zeros(gray.shape, dtype="uint8")
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    roi = cv2.bitwise_and(gray, gray, mask=mask)

    dark_pixels = np.sum(roi < 120)
    total_pixels = np.sum(mask == 255)

    if total_pixels == 0:
        return False

    fill_ratio = dark_pixels / total_pixels

    return fill_ratio > 0.25'''


# =========================
# DRAW RESULTS
# =========================
def draw(img, results):
    out = img.copy()

    for (x, y, w, h, filled) in results:
        color = (0, 0, 255) if filled else (0, 255, 0)
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)

    return out

def filter_real_bubbles(contours):
    # get all areas
    areas = [cv2.contourArea(c) for c in contours]

    if len(areas) == 0:
        return []

    median_area = np.median(areas)

    filtered = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # keep only similar-sized blobs
        if not (0.6 * median_area < area < 1.4 * median_area):
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = w / float(h)

        # must be roughly square
        if aspect_ratio < 0.7 or aspect_ratio > 1.3:
            continue

        filtered.append(cnt)

    return filtered


# =========================
# MAIN PIPELINE
# =========================
def detect_bubbles(image_path):

    print("📥 Loading image...")
    img = cv2.imread(image_path)

    print("✂️ Cropping OMR region...")
    omr = img[Y1:Y2, X1:X2]

    # remove extreme top/bottom noise
    h = omr.shape[0]
    omr = omr[int(0.03*h):int(0.97*h), :]

    print("🔍 Preprocessing...")
    gray, thresh = preprocess(omr)

    print("🧠 Detecting contours...")
    contours = find_bubbles(thresh)
    contours = filter_real_bubbles(contours)
    print(f"✔ Raw contours: {len(contours)}")

    print("📊 Grouping into rows...")
    rows = group_rows(contours)
    print(f"✔ Rows detected: {len(rows)}")

    results = []

    print("🎯 Classifying bubbles...")
    for row in rows:
        row_sorted = sort_row(row)

        for (cx, cy, cnt) in row_sorted:
            filled = is_filled(gray, cnt)

            x, y, w, h = cv2.boundingRect(cnt)
            results.append((x, y, w, h, filled))

    filled_count = sum([1 for r in results if r[4]])
    print(f"✅ Filled bubbles: {filled_count}")

    print("🖊 Drawing results...")
    out = draw(omr, results)

    cv2.imwrite(f"{OUTPUT_DIR}/final_result.png", out)
    print("💾 Saved: output_bubbles/final_result.png")

    print("🔤 Extracting text near filled bubbles...")

    # sort top → bottom (important)
    results_sorted = sorted(results, key=lambda r: r[1])

    texts = extract_text_from_bubbles(omr, results_sorted)

    print("\n📊 FINAL OUTPUT:")
    for t in texts:
        print(f"📍 {t['bbox']} → {t['text']}")

    return results


def extract_text_from_bubbles(img, bubbles):
    results = []

    os.makedirs("debug", exist_ok=True)

    for (x, y, w, h, filled) in bubbles:
        if not filled:
            continue

        # 🔥 crop region to the RIGHT of bubble
        x1 = x + w + 10
        x2 = x1 + 100

        y1 = max(0, y - 5)
        y2 = y + h + 5

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # preprocess for OCR
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh_crop = cv2.threshold(gray_crop, 150, 255, cv2.THRESH_BINARY)

        # OCR
        text = pytesseract.image_to_string(
            thresh_crop,
            config="--psm 10 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz"
        )

        print(f"RAW: '{text}'")

        results.append({
            "text": text.strip(),
            "bbox": (x, y, w, h)
        })

    return results

# =========================
# RUN
# =========================
if __name__ == "__main__":
    detect_bubbles(r"D:/img_process/annotated/annotated_page_5_final.png")