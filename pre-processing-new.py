# -----------------------------
# IMPORTS + CONFIG
# -----------------------------
import fitz
import cv2
import numpy as np
import os

DPI = 300
OUTPUT_DIR = "output_stage1_2_3"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(name, img):
    path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(path, img)
    print(f"💾 Saved: {path}")


# -----------------------------
# STAGE 1: PDF → IMAGE
# -----------------------------
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


def deskew(img):
    edges = cv2.Canny(img, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    if lines is None:
        print("⚠️ No lines → skip deskew")
        return img

    angles = [(theta - np.pi/2) for rho, theta in lines[:,0]]
    median_angle = np.median(angles)

    angle_deg = np.degrees(median_angle)
    print(f"📐 Skew angle: {angle_deg:.2f}")

    h, w = img.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle_deg, 1)

    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# -----------------------------
# STAGE 3.3.1: PROJECTION SPLIT
# -----------------------------
def projection_split(img):
    # horizontal projection
    proj = np.sum(img, axis=1)

    proj = proj / np.max(proj)

    # smooth
    proj_smooth = cv2.GaussianBlur(proj.reshape(-1,1), (51,1), 0).flatten()

    # find valleys
    threshold = 0.5
    valleys = np.where(proj_smooth < threshold)[0]

    if len(valleys) < 3:
        return None, 0.0

    # pick 3 best split points (approx)
    splits = np.linspace(0, len(valleys)-1, 4, dtype=int)[1:-1]
    split_points = valleys[splits]

    confidence = 1 - np.mean(proj_smooth[valleys])

    return split_points, confidence


# -----------------------------
# STAGE 3.3.2: SPLIT FROM PEAKS
# -----------------------------
def split_from_peaks(img, peaks):
    h = img.shape[0]

    points = [0] + list(peaks) + [h]

    bands = []

    for i in range(len(points)-1):
        y1, y2 = points[i], points[i+1]
        band = img[y1:y2]
        bands.append((i, band))

    return bands


# -----------------------------
# STAGE 3.3.3: ORCHESTRATOR
# -----------------------------
def band_split_orchestrator(img):

    peaks, conf = projection_split(img)

    if peaks is not None and conf >= 0.65:
        print(f"✅ OpenCV band split (conf={conf:.2f})")
        return split_from_peaks(img, peaks)

    else:
        print("⚠️ Fallback to equal split")

        h = img.shape[0]
        band_h = h // 4

        bands = []
        for i in range(4):
            band = img[i*band_h:(i+1)*band_h]
            bands.append((i, band))

        return bands


# -----------------------------
# PIPELINE (1 + 2 + 3)
# -----------------------------
def process_pdf(path):

    pages = load_pdf_pages(path)

    for page_num, img in pages:

        if page_num == 5 :

            print(f"\n🚀 Processing Page {page_num}")

            # Stage 2
            c = enhance_contrast(img)
            d = denoise(c)
            s = sharpen(d)
            final = deskew(s)

            save(f"page_{page_num}_final.png", final)

            # Stage 3
            bands = band_split_orchestrator(final)

            for band_idx, band in bands:
                save(f"page_{page_num}_band_{band_idx}.png", band)


# -----------------------------
# RUN
# -----------------------------
pdf_path = r"C:/Users/Acer/Downloads/2220278184522039.pdf"

process_pdf(pdf_path)

print("\n✅ STAGE 1 + 2 + 3 COMPLETE")