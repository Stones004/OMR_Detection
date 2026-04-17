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
INPUT_DIR = r"G:/Ans_Scripts"
OUTPUT_DIR = r"G:/OMR_Detection/processed_imgs"

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
        print(f"📄 Loaded page {i}")

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
        print("⚠️ No valid angles")
        return img

    median_angle = np.median(angles)

    # clamp extreme errors
    if abs(median_angle) > 10:
        print(f"⚠️ Ignoring extreme angle: {median_angle:.2f}")
        return img

    print(f"📐 Skew angle: {median_angle:.2f}")

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
# PROCESS SINGLE PDF
# -----------------------------
def process_pdf(path):

    pages = load_pdf_pages(path)

    for page_num, img in pages:

        print(f"\n🚀 Processing Page {page_num}")

        # preprocessing pipeline
        c = enhance_contrast(img)
        d = denoise(c)
        s = sharpen(d)
        final = deskew(s)

        filename = get_next_filename()
        save_path = os.path.join(OUTPUT_DIR, filename)

        cv2.imwrite(save_path, final)
        print(f"💾 Saved: {save_path}")

# -----------------------------
# PROCESS ALL PDFs
# -----------------------------
def process_all_pdfs(input_dir):

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("❌ No PDFs found")
        return

    print(f"📂 Found {len(pdf_files)} PDFs")

    for pdf_name in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_name)

        print("\n" + "="*50)
        print(f"📘 Processing: {pdf_name}")
        print("="*50)

        try:
            process_pdf(pdf_path)
        except Exception as e:
            print(f"❌ Error: {e}")

    # save counter after all processing
    save_counter(GLOBAL_COUNTER)

# -----------------------------
# RUN
# -----------------------------
process_all_pdfs(INPUT_DIR)

print("\n✅ ALL PDFs PROCESSED")