import cv2
import numpy as np
import os
import random
from glob import glob


def draw_detected_lines(img, vertical, pair=None, all_lines=None):
    debug = img.copy()

    # If working on cropped image, overlay should match that
    if vertical.shape[:2] != img.shape[:2]:
        h, w = img.shape[:2]
        debug = img[:, :vertical.shape[1]].copy()

    # Draw all detected candidate lines
    if all_lines is not None:
        for (x, y, cw, ch) in all_lines:
            cv2.rectangle(debug, (x, y), (x+cw, y+ch), (255, 0, 0), 2)  # BLUE

    # Draw selected pair (final margin)
    if pair is not None:
        l1, l2 = pair

        cv2.rectangle(debug, (l1[0], l1[1]),
                      (l1[0]+l1[2], l1[1]+l1[3]),
                      (0, 255, 0), 3)

        cv2.rectangle(debug, (l2[0], l2[1]),
                      (l2[0]+l2[2], l2[1]+l2[3]),
                      (0, 255, 0), 3)

    return debug



# =========================
# 1. PCA DESKEW
# =========================
import cv2
import numpy as np

def deskew_horizontal_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur to suppress handwriting noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Hough Lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=150,
        minLineLength=img.shape[1]//2,
        maxLineGap=50
    )

    if lines is None:
        return img

    angles = []

    for x1, y1, x2, y2 in lines[:,0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Only near-horizontal lines
        if -15 < angle < 15:
            angles.append(angle)

    if len(angles) < 5:
        return img

    median_angle = np.median(angles)

    # 🔥 Clamp to avoid over-rotation
    median_angle = np.clip(median_angle, -5, 5)

    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1.0)

    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


# =========================
# 2. VERTICAL LINE ENHANCEMENT
# =========================
def enhance_vertical_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 5
    )

    h, w = th.shape

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, h//20))

    vertical = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    vertical = cv2.dilate(vertical, kernel, iterations=2)

    return vertical


# =========================
# 3. DETECT MARGIN LINES
# =========================
def detect_margin_pair(vertical):
    contours, _ = cv2.findContours(
        vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = vertical.shape
    lines = []

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)

        # Strong filtering
        if ch > h * 0.6 and cw < 40:

            # ❌ remove page borders
            if x < 20 or x > w - 20:
                continue

            lines.append((x, y, cw, ch))

    if len(lines) < 2:
        return None, lines

    lines = sorted(lines, key=lambda x: x[0])

    # Pick closest pair
    best_pair = None
    min_dist = float('inf')

    for i in range(len(lines)-1):
        d = lines[i+1][0] - lines[i][0]

        if 10 < d < min_dist:
            min_dist = d
            best_pair = (lines[i], lines[i+1])

    return best_pair, lines


# =========================
# 4. FULL EXTRACTION
# =========================
'''def extract_margin(img):
    # Step 1: Deskew
    img = deskew_horizontal_lines(img)

    # Step 2: Vertical lines
    vertical = enhance_vertical_lines(img)

    # Step 3: Detect margin
    pair, all_lines = detect_margin_pair(vertical)

    if pair is None:
        debug = draw_detected_lines(img_left, vertical, None, all_lines)
        return None, vertical, debug

    l1, l2 = pair

    x1 = l1[0]
    x2 = l2[0] + l2[2]

    margin = img[:, x1:x2]

    debug = draw_detected_lines(img_left, vertical, pair, all_lines)
    return margin, vertical, debug'''




def extract_margin(img):

    # 1. Deskew
    img = deskew_horizontal_lines(img)

    # 2. Crop LEFT region
    h, w = img.shape[:2]
    img_left = img[:, :int(w * 0.35)]

    # 3. Vertical extraction
    vertical = enhance_vertical_lines(img_left)

    # 4. Detect margin lines
    pair, all_lines = detect_margin_pair(vertical)

    # Debug image always created
    debug = draw_detected_lines(img_left, vertical, pair, all_lines)

    if pair is None:
        return None, vertical, debug

    l1, l2 = pair

    x1 = l1[0]
    x2 = l2[0] + l2[2]

    margin = img_left[:, x1:x2]

    return margin, vertical, debug


# =========================
# 5. BATCH PROCESSING
# =========================
def process_folder(input_folder, output_folder, n_samples=10):
    os.makedirs(output_folder, exist_ok=True)

    out_margin = os.path.join(output_folder, "margin")
    out_vertical = os.path.join(output_folder, "vertical")
    out_debug = os.path.join(output_folder, "debug")

    os.makedirs(out_margin, exist_ok=True)
    os.makedirs(out_vertical, exist_ok=True)
    os.makedirs(out_debug, exist_ok=True)

    image_paths = glob(os.path.join(input_folder, "*.png")) + \
                  glob(os.path.join(input_folder, "*.jpg"))

    if len(image_paths) == 0:
        print("No images found")
        return

    # Random sample
    image_paths = random.sample(
        image_paths, min(n_samples, len(image_paths))
    )

    for path in image_paths:
        img = cv2.imread(path)

        if img is None:
            continue

        margin, vertical, debug = extract_margin(img)

        name = os.path.basename(path)

        # Save outputs
        cv2.imwrite(os.path.join(out_vertical, name), vertical)
        #cv2.imwrite(os.path.join(out_debug, name), deskewed)
        cv2.imwrite(os.path.join(out_debug, name), debug)

        if margin is not None:
            cv2.imwrite(os.path.join(out_margin, name), margin)
        else:
            print(f"[WARN] Failed: {name}")




# =========================
# RUN
# =========================
if __name__ == "__main__":
    input_folder = "G:/OMR_Detection/processed_imgs"
    output_folder = "G:/OMR_Detection/test2_images"

    process_folder(input_folder, output_folder, n_samples=50)