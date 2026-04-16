'''# -----------------------------
# GLOBAL CONFIGURATION
# -----------------------------
DPI = 450

# Scaling base (important!)
BASE_DPI = 300
SCALE = DPI / BASE_DPI

# Canny Edge Detection
CANNY_LOW = int(50 * SCALE)
CANNY_HIGH = int(150 * SCALE)

# Hough Transform
HOUGH_THRESHOLD = int(100 * SCALE)
MIN_LINE_LENGTH = int(100 * SCALE)
MAX_LINE_GAP = int(10 * SCALE)

# CLAHE
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

# Blur
BLUR_KERNEL = (5, 5)

# Sharpen kernel strength
SHARP_CENTER = 5   # increase → sharper, but noisy

# Morphology scaling factor (for adaptive kernels)
KERNEL_SCALE_DIVISOR = 40  # lower → larger kernels

# Display
MAX_WIDTH = 800


import fitz
import cv2
import numpy as np


import os

OUTPUT_DIR = "output"

def save_image(name, img):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(path, img)
    print(f"💾 Saved: {path}")

DEBUG = True
def log(msg):
    if DEBUG:
        print(msg)

# -----------------------------
# STEP 1: PDF → IMAGE
# -----------------------------
def preprocess_pdf_page(pdf_path, page_index=0):
    log(f"\n📄 Opening: {pdf_path}")

    doc = fitz.open(pdf_path)
    log(f"📊 Total pages: {len(doc)}")

    if len(doc) == 0:
        raise ValueError("PDF has no pages")

    if page_index >= len(doc):
        raise IndexError("Page index out of range")

    page = doc[page_index]

    zoom = DPI / 72
    log(f"🔍 DPI: {DPI} | Zoom: {zoom:.2f}")

    mat = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)

    log(f"🖼️ Image shape: {img.shape}")
    return img


# -----------------------------
# STEP 2: DESKEW
# -----------------------------
def deskew_and_correct(img):
    log("\n🔍 Checking skew...")

    edges = cv2.Canny(img, CANNY_LOW, CANNY_HIGH)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP
    )

    angles = []

    if lines is not None:
        log(f"📏 Lines detected: {len(lines)}")

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
    else:
        log("⚠️ No lines detected")

    median_angle = np.median(angles) if angles else 0
    log(f"📐 Skew angle: {median_angle:.2f}")

    if abs(median_angle) < 1:
        log("🟢 Skew too small — skipping")
        return img

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    log("✅ Deskew applied")
    return rotated


# -----------------------------
# STEP 3: CONTRAST
# -----------------------------
def enhance_contrast(img):
    log("\n✨ Enhancing contrast...")
    log(f"CLAHE → clip={CLAHE_CLIP}, tile={CLAHE_TILE}")

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP,
        tileGridSize=CLAHE_TILE
    )
    return clahe.apply(img)


# -----------------------------
# STEP 4: DENOISE
# -----------------------------
def denoise(img):
    log("\n🧹 Denoising...")
    log(f"Blur kernel: {BLUR_KERNEL}")
    return cv2.GaussianBlur(img, BLUR_KERNEL, 0)


# -----------------------------
# STEP 5: SHARPEN
# -----------------------------
def sharpen_image(img):
    log("\n🔪 Sharpening...")
    log(f"Sharpen strength: {SHARP_CENTER}")

    kernel = np.array([
        [0, -1, 0],
        [-1, SHARP_CENTER, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(img, -1, kernel)


# -----------------------------
# STEP 6: EDGE ENHANCEMENT
# -----------------------------
def enhance_edges(img):
    log("\n🧱 Enhancing edges...")

    edges = cv2.Canny(img, CANNY_LOW, CANNY_HIGH)

    h, w = img.shape
    k = max(2, int(min(h, w) / 500))

    log(f"Kernel size (adaptive): {k}")

    kernel_small = np.ones((k, k), np.uint8)
    kernel_large = np.ones((k + 1, k + 1), np.uint8)

    edges = cv2.dilate(edges, kernel_small, iterations=1)
    edges = cv2.dilate(edges, kernel_large, iterations=1)

    return edges

# -----------------------------
# STEP 7: LINE EXTRACTION
# -----------------------------
def extract_lines(img):
    log("\n📏 Extracting lines...")

    h, w = img.shape

    horizontal_size = max(10, w // KERNEL_SCALE_DIVISOR)
    vertical_size   = max(10, h // KERNEL_SCALE_DIVISOR)

    log(f"Horizontal kernel: {horizontal_size}")
    log(f"Vertical kernel: {vertical_size}")

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_size, 5)
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (5, vertical_size)
    )

    horizontal = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel)
    vertical   = cv2.morphologyEx(img, cv2.MORPH_OPEN, vertical_kernel)

    lines = cv2.add(horizontal, vertical)
    return lines

# -----------------------------
# STEP 8: CLEAN NOISE
# -----------------------------
def remove_noise(img):
    log("\n🧼 Cleaning noise...")

    h, w = img.shape
    k = max(2, int(min(h, w) / 500))

    log(f"Closing kernel: {k}")

    kernel = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# -----------------------------
# STEP 9a: CELL EXTRACTION
# -----------------------------
def extract_cells_from_lines(lines, original):
    log("\n🧩 Extracting cells (grid method)...")

    h, w = lines.shape

    # Separate horizontal & vertical
    horizontal = cv2.morphologyEx(
        lines,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (w // 20, 1))
    )

    vertical = cv2.morphologyEx(
        lines,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 20))
    )

    # Intersections
    intersections = cv2.bitwise_and(horizontal, vertical)

    contours, _ = cv2.findContours(intersections, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cells = []
    vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w > 20 and h > 20:
            cells.append((x, y, w, h))
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

    log(f"🟩 Cells detected: {len(cells)}")

    return vis, cells


def extract_table_corners(lines, original):
    log("\n✳️ Extracting table corners...")

    h, w = lines.shape

    # -----------------------------
    # STEP 1: Thin the lines
    # -----------------------------
    thin = cv2.erode(lines, np.ones((3,3), np.uint8), iterations=1)

    # -----------------------------
    # STEP 2: Separate horizontal & vertical
    # -----------------------------
    horizontal = cv2.morphologyEx(
        thin,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (w // 20, 1))
    )

    vertical = cv2.morphologyEx(
        thin,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 20))
    )

    # -----------------------------
    # STEP 3: Get intersections (corners)
    # -----------------------------
    intersections = cv2.bitwise_and(horizontal, vertical)

    # Optional: clean tiny noise
    intersections = cv2.dilate(intersections, np.ones((3,3), np.uint8), iterations=1)

    # -----------------------------
    # STEP 4: Extract points
    # -----------------------------
    contours, _ = cv2.findContours(intersections, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    points = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        cx = x + w // 2
        cy = y + h // 2

        points.append((cx, cy))

        # BIG visible marker
        cv2.circle(vis, (cx, cy), 12, (0, 0, 255), -1)

        # optional bounding box
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)

    log(f"✳️ Corners detected: {len(points)}")

    return vis, points

# -----------------------------
# STEP 9b: CELL EXTRACTION
# -----------------------------
def extract_table_regions(edges, original):
    log("\n📦 Extracting STRICT table regions...")

    h, w = edges.shape

    # -----------------------------
    # STEP 1: Erode to break weak connections
    # -----------------------------
    kernel = np.ones((3,3), np.uint8)
    separated = cv2.erode(edges, kernel, iterations=1)

    # -----------------------------
    # STEP 2: Find contours
    # -----------------------------
    contours, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    regions = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 3000:   # 🔥 increase this to ignore small tables
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        aspect_ratio = bw / float(bh + 1e-5)

        # -----------------------------
        # STEP 3: Shape filtering
        # -----------------------------
        if not (0.5 < aspect_ratio < 4):
            continue

        # -----------------------------
        # STEP 4: Density check (VERY IMPORTANT)
        # -----------------------------
        roi = edges[y:y+bh, x:x+bw]
        density = np.sum(roi > 0) / (bw * bh)

        if density < 0.05:   # too sparse → probably noise
            continue

        regions.append((x, y, bw, bh))
        cv2.rectangle(vis, (x, y), (x+bw, y+bh), (255, 0, 0), 2)

    log(f"📦 Regions detected: {len(regions)}")

    return vis, regions




def extract_main_table_region(lines, original):
    log("\n📦 Extracting main table (robust)...")

    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_score = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 200 or h < 200:
            continue

        roi = lines[y:y+h, x:x+w]

        # score = number of white pixels (line density)
        score = np.sum(roi > 0)

        aspect_ratio = w / float(h + 1e-5)

        # table is wide + dense
        if score > best_score and aspect_ratio > 1.5:
            best_score = score
            best_box = (x, y, w, h)

    if best_box is None:
        log("❌ No table found")
        return original, None

    x, y, w, h = best_box

    log(f"✅ Table found: {x,y,w,h}")

    # 🔥 shrink slightly to remove borders/noise
    pad = 10
    x1 = max(0, x + pad)
    y1 = max(0, y + pad)
    x2 = min(original.shape[1], x + w - pad)
    y2 = min(original.shape[0], y + h - pad)

    table_crop = original[y1:y2, x1:x2]

    return table_crop, (x1, y1, x2-x1, y2-y1)


def group_points_into_rows(points, threshold=20):
    rows = []

    # sort by y first, then x
    points = sorted(points, key=lambda p: (p[1], p[0]))

    for pt in points:
        placed = False

        for row in rows:
            if abs(row[0][1] - pt[1]) < threshold:
                row.append(pt)
                placed = True
                break

        if not placed:
            rows.append([pt])

    # sort each row left → right
    for row in rows:
        row.sort(key=lambda p: p[0])

    return rows


def build_cells(rows):
    cells = []

    for i in range(len(rows) - 1):
        row1 = rows[i]
        row2 = rows[i+1]

        # take only common columns
        cols = min(len(row1), len(row2))

        for j in range(cols - 1):
            x1, y1 = row1[j]
            x2, y2 = row2[j+1]

            w = x2 - x1
            h = y2 - y1

            if w > 10 and h > 10:
                cells.append((x1, y1, w, h))

    return cells


def extract_cells_from_image(image, cells):
    cropped = []

    for (x, y, w, h) in cells:
        pad = 2

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)

        crop = image[y1:y2, x1:x2]
        cropped.append(crop)

    return cropped


def save_cells(cells, prefix="cell"):
    os.makedirs("output/cells", exist_ok=True)

    for i, cell in enumerate(cells):
        path = f"output/cells/{prefix}_{i}.png"
        cv2.imwrite(path, cell)

# -----------------------------
# STEP 10: RESIZE
# -----------------------------
def resize_for_display(img):
    h, w = img.shape[:2]
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


# -----------------------------
# MAIN PIPELINE
# -----------------------------
pdf_path = r"C:/Users/Acer/Downloads/2220278184522039.pdf"

raw = preprocess_pdf_page(pdf_path)

deskewed = deskew_and_correct(raw)

contrast = enhance_contrast(deskewed)

denoised = denoise(contrast)

sharp = sharpen_image(denoised)

# YOLO INPUT
model_input = sharp

# Layout
edges = enhance_edges(sharp)

lines = extract_lines(edges)

clean_layout = remove_noise(lines)


# -----------------------------
# EXTRACTION METHODS
# -----------------------------
cell_vis, cells = extract_cells_from_lines(lines, sharp)

region_vis, regions = extract_table_regions(edges, sharp)

corner_vis, corners = extract_table_corners(lines, sharp)



# -----------------------------
# SAVE OUTPUTS
# -----------------------------
save_image("1_edges.png", edges)
save_image("2_lines.png", lines)
save_image("3_cells.png", cell_vis)
save_image("4_regions.png", region_vis)
save_image("5_corners.png", corner_vis)


table_img, table_box = extract_main_table_region(lines, sharp)
save_image("table_only.png", table_img)


edges_table = enhance_edges(table_img)
lines_table = extract_lines(edges_table)
corner_vis, corners = extract_table_corners(lines_table, table_img)
save_image("table_corners.png", corner_vis)

h, w = table_img.shape
corners = [p for p in corners if p[0] < w * 0.8]

rows = group_points_into_rows(corners)
cells = build_cells(rows)
cropped_cells = extract_cells_from_image(table_img, cells)

save_cells(cropped_cells, prefix="table_cell")

print("\n✅ All outputs saved in /output folder")'''





# -----------------------------
# IMPORTS + CONFIG
# -----------------------------
import fitz
import cv2
import numpy as np
import os

DPI = 450
BASE_DPI = 300
SCALE = DPI / BASE_DPI

CANNY_LOW = int(50 * SCALE)
CANNY_HIGH = int(150 * SCALE)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_image(name, img):
    path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(path, img)
    print(f"💾 Saved: {path}")


# -----------------------------
# PDF → ALL PAGES
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
# PREPROCESS
# -----------------------------
def preprocess(img):
    img = cv2.createCLAHE(2.0, (8, 8)).apply(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)

    return img


# -----------------------------
# EDGE + LINE
# -----------------------------
def get_edges(img):
    edges = cv2.Canny(img, CANNY_LOW, CANNY_HIGH)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
    return edges


def get_lines(edges):
    h, w = edges.shape

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w//40, 5))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, h//40))

    h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)

    return cv2.add(h_lines, v_lines)


# -----------------------------
# MULTI TABLE DETECTION
# -----------------------------
def detect_tables(lines, img):
    separated = cv2.erode(lines, np.ones((3,3), np.uint8), 1)

    contours, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tables = []
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 150 or h < 150:
            continue

        roi = lines[y:y+h, x:x+w]
        density = np.sum(roi > 0) / (w*h)

        if density > 0.05:
            tables.append((x,y,w,h))
            cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)

    return tables, vis


# -----------------------------
# CORNER DETECTION
# -----------------------------
def get_corners(lines, img):
    thin = cv2.erode(lines, np.ones((3,3), np.uint8), 1)

    h, w = lines.shape

    hor = cv2.morphologyEx(thin, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (w//20,1)))

    ver = cv2.morphologyEx(thin, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1,h//20)))

    inter = cv2.bitwise_and(hor, ver)

    contours, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pts = []
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        cx, cy = x+w//2, y+h//2

        pts.append((cx, cy))
        cv2.circle(vis, (cx,cy), 6, (0,0,255), -1)

    return vis, pts


# -----------------------------
# GRID + CELLS
# -----------------------------
def group_rows(points, thresh=20):
    points = sorted(points, key=lambda p:(p[1],p[0]))
    rows = []

    for p in points:
        placed = False
        for r in rows:
            if abs(r[0][1] - p[1]) < thresh:
                r.append(p)
                placed = True
                break

        if not placed:
            rows.append([p])

    for r in rows:
        r.sort(key=lambda p:p[0])

    return rows


def build_cells(rows):
    cells = []

    for i in range(len(rows)-1):
        r1, r2 = rows[i], rows[i+1]

        cols = min(len(r1), len(r2))

        for j in range(cols-1):
            x1,y1 = r1[j]
            x2,y2 = r2[j+1]

            if x2-x1 > 10 and y2-y1 > 10:
                cells.append((x1,y1,x2-x1,y2-y1))

    return cells


def save_cells(img, cells, prefix):
    cell_dir = os.path.join(OUTPUT_DIR, "cells")
    os.makedirs(cell_dir, exist_ok=True)

    for i,(x,y,w,h) in enumerate(cells):
        crop = img[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(cell_dir, f"{prefix}_{i}.png"), crop)


# -----------------------------
# MAIN PIPELINE
# -----------------------------
pdf_path = r"C:/Users/Acer/Downloads/2220278184522039.pdf"

pages = load_pdf_pages(pdf_path)

for page_num, img in pages:

    if page_num == 2 :

        print(f"\n🚀 Processing page {page_num}")

        img = preprocess(img)

        edges = get_edges(img)
        lines = get_lines(edges)

        save_image(f"page_{page_num}_edges.png", edges)
        save_image(f"page_{page_num}_lines.png", lines)

        tables, vis = detect_tables(lines, img)
        save_image(f"page_{page_num}_tables.png", vis)

        for i,(x,y,w,h) in enumerate(tables):

            print(f"📊 Table {i}")

            table = img[y:y+h, x:x+w]
            save_image(f"page_{page_num}_table_{i}.png", table)

            edges_t = get_edges(table)
            lines_t = get_lines(edges_t)

            vis_c, corners = get_corners(lines_t, table)
            save_image(f"page_{page_num}_table_{i}_corners.png", vis_c)

            if len(corners) < 10:
                print("⚠️ Skipping (few corners)")
                continue

            rows = group_rows(corners)

            # normalize rows
            min_cols = min(len(r) for r in rows if len(r)>0)
            rows = [r[:min_cols] for r in rows]

            cells = build_cells(rows)

            if len(cells) == 0:
                print("⚠️ No cells")
                continue

            save_cells(table, cells, f"page{page_num}_table{i}")

print("\n✅ DONE")