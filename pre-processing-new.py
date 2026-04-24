# -----------------------------
# IMPORTS + CONFIG
# -----------------------------
import fitz
import cv2
import numpy as np
import os

# -----------------------------
# PATHS
# -----------------------------
#INPUT_DIR = r"G:/raw_ans_scripts"
INPUT_DIR = r"G:\OMR_Detection\synthetic_data"
#OUTPUT_DIR = r"G:/OMR_Detection/processed_imgs"
OUTPUT_DIR = r"G:\OMR_Detection\synthetic_data_processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# GLOBAL COUNTER (persistent)
# -----------------------------
COUNTER_FILE = os.path.join(OUTPUT_DIR, "counter.txt")

def load_counter():
    if os.path.exists(COUNTER_FILE):
        return int(open(COUNTER_FILE).read())
    return 0

def save_counter(counter):
    with open(COUNTER_FILE, "w") as f:
        f.write(str(counter))

GLOBAL_COUNTER = load_counter()

def get_next_filename():
    global GLOBAL_COUNTER
    name = f"img_{GLOBAL_COUNTER:06d}.png"
    GLOBAL_COUNTER += 1
    return name

# -----------------------------
# STAGE 1: PDF → IMAGE
# -----------------------------
DPI = 300

def load_pdf_pages(path):
    doc = fitz.open(path)

    zoom = DPI / 72
    mat = fitz.Matrix(zoom, zoom)

    pages = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)

        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.h, pix.w)

        pages.append((i, img))
    print(f"Loaded page {i}")

    return pages

# -----------------------------
# STAGE 2: PREPROCESS
# -----------------------------
def enhance_contrast(img):
    return cv2.createCLAHE(2.0, (8,8)).apply(img)

def denoise(img):
    return cv2.GaussianBlur(img, (5,5), 0)

def sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

# -----------------------------
# ROBUST DESKEW (FIXED)
# -----------------------------
def deskew(img):
    edges = cv2.Canny(img, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=500,
        maxLineGap=50
    )

    if lines is None:
        print("⚠️ No lines → skip deskew")
        return img

    angles = []

    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # keep near-horizontal lines only
        if -30 < angle < 30:
            angles.append(angle)

    if len(angles) == 0:
        print("No valid angles")
        return img

    median_angle = np.median(angles)

    # clamp extreme errors
    if abs(median_angle) > 10:
        print(f"Ignoring extreme angle: {median_angle:.2f}")
        return img

    print(f"Skew angle: {median_angle:.2f}")

    h, w = img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1)

    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


# -----------------------------
# OMR DESKEW (Updated needed to check)
# -----------------------------
'''def deskew_omr(img):
    h, w = img.shape

    # -----------------------------
    # STEP 1: Focus on LEFT STRIP
    # -----------------------------
    left_region = img[:, :int(w * 0.25)]  # left 25%

    # -----------------------------
    # STEP 2: Edge detection
    # -----------------------------
    edges = cv2.Canny(left_region, 50, 150)

    # -----------------------------
    # STEP 3: Detect lines
    # -----------------------------
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=int(h * 0.4),  # long vertical lines
        maxLineGap=50
    )

    if lines is None:
        print("⚠️ No lines → skip deskew")
        return img

    # -----------------------------
    # STEP 4: Extract vertical angles
    # -----------------------------
    angles = []

    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # keep near-vertical lines
        if abs(angle) > 60:
            angles.append(angle)

    if len(angles) == 0:
        print("⚠️ No vertical angles")
        return img

    median_angle = np.median(angles)

    # -----------------------------
    # STEP 5: Convert to skew angle
    # -----------------------------
    # vertical should be 90°, so:
    if median_angle > 0:
        skew_angle = median_angle - 90
    else:
        skew_angle = median_angle + 90

    print(f"📐 OMR Skew angle: {skew_angle:.2f}")

    # -----------------------------
    # STEP 6: Clamp extreme values
    # -----------------------------
    if abs(skew_angle) > 10:
        print("⚠️ Extreme skew ignored")
        return img

    # -----------------------------
    # STEP 7: Rotate
    # -----------------------------
    M = cv2.getRotationMatrix2D((w // 2, h // 2), skew_angle, 1)

    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated '''


# -----------------------------
# CORE PIPELINE
# -----------------------------
def run_pipeline(img):
    """Applies the preprocessing steps: contrast, denoise, sharpen, deskew."""
    c = enhance_contrast(img)
    d = denoise(c)
    s = sharpen(d)
    final = deskew(s)
    return final

def save_result(img):
    """Generates next filename and saves the image."""
    filename = get_next_filename()
    save_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(save_path, img)
    print(f"Saved: {save_path}")

# -----------------------------
# PROCESS INDIVIDUAL FILES
# -----------------------------
def process_image_file(path):
    print(f"\nProcessing Image: {os.path.basename(path)}")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read image: {path}")
        return

    final = run_pipeline(img)
    save_result(final)

def process_pdf(path):
    print(f"\nProcessing PDF: {os.path.basename(path)}")
    pages = load_pdf_pages(path)

    for page_num, img in pages:
        print(f"Processing Page {page_num}")
        final = run_pipeline(img)
        save_result(final)

# -----------------------------
# MAIN DISPATCHER
# -----------------------------
def process_all(input_path):
    # Support both directory and single file
    if os.path.isfile(input_path):
        target_files = [os.path.basename(input_path)]
        base_dir = os.path.dirname(input_path)
    elif os.path.isdir(input_path):
        target_files = os.listdir(input_path)
        base_dir = input_path
    else:
        print(f"Path not found: {input_path}")
        return

    # Filter supported extensions
    valid_extensions = ('.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    to_process = [f for f in target_files if f.lower().endswith(valid_extensions)]

    if not to_process:
        print(f"No supported files found in: {input_path}")
        return

    print(f"Found {len(to_process)} files to process")

    for filename in to_process:
        full_path = os.path.join(base_dir, filename)
        ext = os.path.splitext(filename)[1].lower()

        print("\n" + "="*50)
        print(f"File: {filename}")
        print("="*50)

        try:
            if ext == '.pdf':
                process_pdf(full_path)
            else:
                process_image_file(full_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # save counter after all processing
    save_counter(GLOBAL_COUNTER)

# -----------------------------
# RUN
# -----------------------------
process_all(INPUT_DIR)

print("\nALL PROCESSING COMPLETED")