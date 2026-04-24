import cv2
import numpy as np
import random
import os

# ══════════════════════════════════════════════════════
# CANVAS  —  A4 @ 150 dpi
# ══════════════════════════════════════════════════════
W          = 1240
H          = 1820
TOP_MARGIN = 80
BG         = 245
DARK       = 20
MED        = 85

# ──────────────────────────────────────────────────────
# HUMAN ERROR TYPES (applied per bubble randomly)
# ──────────────────────────────────────────────────────
# NORMAL_FILL   – clean dark pencil fill (most common)
# LIGHT_FILL    – faint, under-pressured shading
# PATCHY_FILL   – uneven strokes, gaps inside
# OVERPRESS     – very dark, slightly over-sized
# DOUBLE_BUBBLE – two bubbles in same row marked
# SCRATCH_ERASE – bubble crossed/scratched out
# SMUDGE        – fill bleeds slightly outside circle
# HALF_FILL     – only half the bubble shaded
# TICK_MARK     – tick drawn inside instead of fill
# EMPTY_STRAY   – tiny pencil dot near an empty bubble

ERROR_TYPES = [
    "normal",       # weight 50
    "light",        # weight 15
    "patchy",       # weight 10
    "overpress",    # weight 8
    "smudge",       # weight 7
    "half",         # weight 5
    "tick",         # weight 3
    "scratch",      # weight 2  (applied on top of a fill)
]
ERROR_WEIGHTS = [50, 15, 10, 8, 7, 5, 3, 2]


# ──────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────

def draw_qr(img, x, y, size):
    cv2.rectangle(img, (x, y), (x+size, y+size), (DARK,), 2)
    cv2.rectangle(img, (x+2, y+2), (x+size-2, y+size-2), (BG,), -1)

    def finder(fx, fy, s=16):
        cv2.rectangle(img, (fx,   fy  ), (fx+s,   fy+s  ), (DARK,), -1)
        cv2.rectangle(img, (fx+2, fy+2), (fx+s-2, fy+s-2), (BG,  ), -1)
        cv2.rectangle(img, (fx+4, fy+4), (fx+s-4, fy+s-4), (DARK,), -1)

    finder(x+3, y+3)
    finder(x+size-19, y+3)
    finder(x+3, y+size-19)

    rng = np.random.default_rng(seed=abs(x*31+y))
    for _ in range(140):
        mx = x + int(rng.integers(4, size-6))
        my = y + int(rng.integers(4, size-6))
        if rng.random() < 0.45:
            cv2.rectangle(img, (mx, my), (mx+3, my+3), (DARK,), -1)


def draw_logo(img, x, y, w, h):
    cx, cy = x + w//2, y + h//2
    r = min(w, h)//2 - 2
    cv2.circle(img, (cx, cy), r,   (DARK,), 2)
    cv2.circle(img, (cx, cy), r-6, (DARK,), 1)
    for dy, txt in [(-16, "CHRIST"), (-4, "(Deemed to be"), (8, "University)"), (20, "BANGALORE")]:
        sz = 0.28 if dy != -16 else 0.38
        tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sz, 1)[0][0]
        cv2.putText(img, txt, (cx - tw//2, cy + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, sz, (DARK,), 1)


# ──────────────────────────────────────────────────────
# BUBBLE DRAWING WITH HUMAN ERRORS
# ──────────────────────────────────────────────────────

def bubble_normal(img, cx, cy, r):
    """Standard dark pencil fill with slight unevenness."""
    base = random.randint(18, 45)
    cv2.circle(img, (cx, cy), r, (base,), -1)
    # slight highlight off-centre
    hi = min(base + random.randint(15, 30), 85)
    ox, oy = random.randint(-2, 2), random.randint(-2, 2)
    cv2.circle(img, (cx+ox, cy+oy), max(1, r-4), (hi,), -1)
    cv2.circle(img, (cx, cy), r, (DARK,), 2)


def bubble_light(img, cx, cy, r):
    """Faint / under-pressured shading — student pressed too lightly."""
    base = random.randint(140, 185)
    cv2.circle(img, (cx, cy), r, (base,), -1)
    # streaky lighter bands
    for _ in range(4):
        angle = random.uniform(0, np.pi)
        x1 = int(cx + (r-2)*np.cos(angle))
        y1 = int(cy + (r-2)*np.sin(angle))
        x2 = int(cx - (r-2)*np.cos(angle))
        y2 = int(cy - (r-2)*np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), (random.randint(160, 200),), 1)
    cv2.circle(img, (cx, cy), r, (DARK,), 2)


def bubble_patchy(img, cx, cy, r):
    """Uneven strokes — some areas dark, some left pale."""
    cv2.circle(img, (cx, cy), r, (BG,), -1)   # start clean
    cv2.circle(img, (cx, cy), r, (DARK,), 2)
    # Draw random dark arcs/strokes inside
    for _ in range(random.randint(6, 12)):
        angle = random.uniform(0, 2*np.pi)
        length = random.randint(r//2, r-1)
        shade = random.randint(20, 70)
        x1 = cx + int(random.uniform(-r+2, r-2))
        y1 = cy + int(random.uniform(-r+2, r-2))
        x2 = x1 + int(length * np.cos(angle))
        y2 = y1 + int(length * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), (shade,), random.randint(1, 2))


def bubble_overpress(img, cx, cy, r):
    """Heavy press — very dark, slightly bleeds over outline."""
    bleed = random.randint(1, 3)
    cv2.circle(img, (cx, cy), r+bleed, (random.randint(10, 25),), -1)
    cv2.circle(img, (cx, cy), r, (DARK,), 2)


def bubble_smudge(img, cx, cy, r):
    """Smudged — fill spreads unevenly in one direction."""
    base = random.randint(25, 55)
    cv2.circle(img, (cx, cy), r, (base,), -1)
    # smudge tail in a random direction
    direction = random.uniform(0, 2*np.pi)
    for i in range(1, random.randint(5, 10)):
        ox = int(i * 2 * np.cos(direction))
        oy = int(i * 2 * np.sin(direction))
        fade = min(base + i*20, 210)
        rad  = max(1, r - i*2)
        cv2.circle(img, (cx+ox, cy+oy), rad, (fade,), -1)
    cv2.circle(img, (cx, cy), r, (DARK,), 2)


def bubble_half(img, cx, cy, r):
    """Only half the bubble is shaded — student gave up halfway."""
    # draw full outline
    cv2.circle(img, (cx, cy), r, (DARK,), 2)
    # fill an ellipse covering roughly one half
    side = random.choice([-1, 1])
    pts = []
    for angle_deg in range(0, 181):
        a = np.radians(angle_deg)
        px = int(cx + r * np.cos(a))
        py = int(cy + side * r * np.sin(a))
        pts.append([px, py])
    pts.append([cx, cy])
    pts = np.array(pts, dtype=np.int32)
    shade = random.randint(25, 60)
    cv2.fillPoly(img, [pts], (shade,))
    cv2.circle(img, (cx, cy), r, (DARK,), 2)


def bubble_tick(img, cx, cy, r):
    """Tick mark drawn inside the bubble instead of fill."""
    cv2.circle(img, (cx, cy), r, (DARK,), 2)
    # tick: short left-down stroke then longer right-up stroke
    shade = random.randint(20, 50)
    t = max(1, r//3)
    p1 = (cx - t, cy)
    p2 = (cx - t//2, cy + t)
    p3 = (cx + t, cy - t)
    cv2.line(img, p1, p2, (shade,), 2)
    cv2.line(img, p2, p3, (shade,), 2)


def scratch_over(img, cx, cy, r):
    """Cross/scratch lines drawn over a bubble (erase attempt)."""
    shade = random.randint(15, 35)
    # 2–4 jagged lines crossing the bubble
    for _ in range(random.randint(2, 4)):
        angle = random.uniform(0, np.pi)
        jitter = random.randint(-3, 3)
        x1 = int(cx - r * np.cos(angle)) + jitter
        y1 = int(cy - r * np.sin(angle)) + jitter
        x2 = int(cx + r * np.cos(angle)) + jitter
        y2 = int(cy + r * np.sin(angle)) + jitter
        cv2.line(img, (x1, y1), (x2, y2), (shade,),
                 random.randint(1, 2))


def stray_dot(img, cx, cy, r):
    """Tiny accidental pencil mark near an empty bubble."""
    ox = random.randint(-r-6, r+6)
    oy = random.randint(-r-6, r+6)
    dot_r = random.randint(1, 3)
    cv2.circle(img, (cx+ox, cy+oy), dot_r,
               (random.randint(60, 120),), -1)


def draw_filled_bubble(img, cx, cy, r):
    """
    Draw a filled bubble with a randomly chosen human error style.
    Returns the error type string for logging.
    """
    etype = random.choices(ERROR_TYPES, weights=ERROR_WEIGHTS, k=1)[0]

    if   etype == "normal":   bubble_normal(img, cx, cy, r)
    elif etype == "light":    bubble_light(img, cx, cy, r)
    elif etype == "patchy":   bubble_patchy(img, cx, cy, r)
    elif etype == "overpress":bubble_overpress(img, cx, cy, r)
    elif etype == "smudge":   bubble_smudge(img, cx, cy, r)
    elif etype == "half":     bubble_half(img, cx, cy, r)
    elif etype == "tick":     bubble_tick(img, cx, cy, r)
    elif etype == "scratch":
        bubble_normal(img, cx, cy, r)   # fill first, then scratch
        scratch_over(img, cx, cy, r)

    # ~12% chance of an additional scratch on top of any filled bubble
    if etype != "scratch" and random.random() < 0.12:
        scratch_over(img, cx, cy, r)

    return etype


def draw_empty_bubble(img, cx, cy, r):
    """Draw an empty bubble, occasionally with a stray dot nearby."""
    cv2.circle(img, (cx, cy), r, (DARK,), 2)
    if random.random() < 0.08:      # 8% chance of stray pencil mark
        stray_dot(img, cx, cy, r)


# ══════════════════════════════════════════════════════
# MAIN GENERATOR
# ══════════════════════════════════════════════════════

def generate_omr_sheet(image_save_path, label_save_path,
                       page_number=7, set_label="A"):

    img    = np.full((H, W), BG, dtype=np.uint8)
    labels = []

    # ── Page border (= OMR left/top/bottom wall) ──────
    PB  = TOP_MARGIN
    PL  = 22
    PR  = W - 22
    PBT = H - 22
    cv2.rectangle(img, (PL, PB), (PR, PBT), (DARK,), 3)

    # ── Header elements OUTSIDE border ────────────────
    QR_SIZE = 58
    QR_Y    = PB - QR_SIZE - 5

    cv2.putText(img, set_label, (PL + 2, QR_Y + QR_SIZE - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (DARK,), 3)
    draw_qr(img, PL + 38, QR_Y, QR_SIZE)

    pg_str = str(page_number)
    pw = cv2.getTextSize(pg_str, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 2)[0][0]
    cv2.putText(img, pg_str,
                (W//2 - pw//2 - QR_SIZE//2 - 8, QR_Y + QR_SIZE - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (DARK,), 2)
    draw_qr(img, W//2 - QR_SIZE//2 + 20, QR_Y, QR_SIZE)

    draw_logo(img, PR - 150, PB - 72, 150, 68)

    # ── Layout ────────────────────────────────────────
    OMR_X1 = PL
    OMR_X2 = PL + 195
    OMR_Y1 = PB
    OMR_Y2 = PBT

    cv2.line(img, (OMR_X2, OMR_Y1), (OMR_X2, OMR_Y2), (DARK,), 3)

    # ── Writing lines ─────────────────────────────────
    WX1, WX2 = OMR_X2 + 12, PR - 8
    for ly in range(OMR_Y1 + 45, OMR_Y2 - 5, 40):
        cv2.line(img, (WX1, ly), (WX2, ly), (MED,), 1)

    # ── OMR bands ─────────────────────────────────────
    NUM_BANDS   = 4
    NUM_ROWS    = 10
    NUM_COLS    = 3
    COL_LETTERS = ['a', 'b', 'c']
    BUBBLE_R    = 11
    FONT        = cv2.FONT_HERSHEY_SIMPLEX
    FS_QBOX     = 0.42
    FT          = 1
    QBOX_H      = 38

    total_h = OMR_Y2 - OMR_Y1
    BAND_H  = total_h // NUM_BANDS

    strip_w      = OMR_X2 - OMR_X1
    LEFT_PAD     = 28
    RIGHT_PAD    = 8
    bubble_zone  = strip_w - LEFT_PAD - RIGHT_PAD
    col_step     = bubble_zone // NUM_COLS
    COL_XS       = [OMR_X1 + LEFT_PAD + col_step*c + col_step//2
                    for c in range(NUM_COLS)]

    for band in range(NUM_BANDS):
        y_start = OMR_Y1 + band * BAND_H
        y_end   = y_start + BAND_H

        cv2.rectangle(img, (OMR_X1, y_start), (OMR_X2, y_end), (DARK,), 2)

        # Q.No header box
        cv2.rectangle(img, (OMR_X1, y_start),
                      (OMR_X2, y_start + QBOX_H), (DARK,), 2)
        cv2.putText(img, "Q.", (OMR_X1+3, y_start+16),
                    FONT, FS_QBOX, (DARK,), FT)
        cv2.putText(img, "No", (OMR_X1+3, y_start+32),
                    FONT, FS_QBOX, (DARK,), FT)
        for c, letter in enumerate(COL_LETTERS):
            tw = cv2.getTextSize(letter, FONT, FS_QBOX, FT)[0][0]
            cv2.putText(img, letter,
                        (COL_XS[c] - tw//2, y_start + QBOX_H - 8),
                        FONT, FS_QBOX, (DARK,), FT)

        # Left timing marker
        MK_Y = y_start + BAND_H // 2
        cv2.rectangle(img,
                      (OMR_X1-18, MK_Y-13), (OMR_X1-3, MK_Y+13),
                      (DARK,), -1)

        # Dashed separator on writing side
        if band > 0:
            dx = WX1
            while dx < WX2:
                cv2.line(img, (dx, y_start),
                         (min(dx+16, WX2), y_start), (DARK,), 1)
                dx += 28

        # Row y positions
        avail_h  = BAND_H - QBOX_H - 6
        row_step = avail_h // NUM_ROWS
        ROW_YS   = [y_start + QBOX_H + row_step//2 + r*row_step
                    for r in range(NUM_ROWS)]

        # ── Decide fills: ~3 filled per band ──────────
        # Pick 2–4 rows to answer (centred around 3)
        n_filled = random.choices([2, 3, 3, 3, 4], weights=[10,40,40,40,10])[0]
        answered_rows = random.sample(range(NUM_ROWS), n_filled)

        # Primary fill: one column per answered row
        fill_map = {r: random.randint(0, NUM_COLS-1) for r in answered_rows}

        # Double-bubble: ~15% chance one answered row also has a second col marked
        for r in list(fill_map.keys()):
            if random.random() < 0.15:
                extra_c = random.choice([c for c in range(NUM_COLS)
                                         if c != fill_map[r]])
                fill_map[f"{r}_extra"] = (r, extra_c)

        # Build a set of (row, col) that are filled
        filled_set = set()
        for key, val in fill_map.items():
            if isinstance(key, int):
                filled_set.add((key, val))
            else:
                filled_set.add(val)

        # ── Draw rows ─────────────────────────────────
        for r in range(NUM_ROWS):
            cy_b = ROW_YS[r]

            # Row number label
            rn = str(r)
            tw = cv2.getTextSize(rn, FONT, FS_QBOX, FT)[0][0]
            cv2.putText(img, rn,
                        (OMR_X1 + LEFT_PAD - tw - 4, cy_b + 5),
                        FONT, FS_QBOX, (DARK,), FT)

            for c in range(NUM_COLS):
                cx_b = COL_XS[c]

                if (r, c) in filled_set:
                    draw_filled_bubble(img, cx_b, cy_b, BUBBLE_R)
                    # YOLO label
                    xc = cx_b / W;  yc = cy_b / H
                    bw = (2*BUBBLE_R)/W;  bh = (2*BUBBLE_R)/H
                    labels.append(
                        f"2 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                else:
                    draw_empty_bubble(img, cx_b, cy_b, BUBBLE_R)

    # ── Watermark ─────────────────────────────────────
    ov = img.copy()
    cv2.circle(ov, (W//2, H//2), 360, (175,), 2)
    img = cv2.addWeighted(ov, 0.07, img, 0.93, 0)

    # ── Paper texture ─────────────────────────────────
    noise = np.random.normal(0, 6, img.shape)
    img   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img   = cv2.GaussianBlur(img, (3, 3), 0)

    # ── Scan skew ─────────────────────────────────────
    angle = random.uniform(-0.7, 0.7)
    M     = cv2.getRotationMatrix2D((W//2, H//2), angle, 1)
    img   = cv2.warpAffine(img, M, (W, H), borderValue=BG)

    # ── Contrast boost ────────────────────────────────
    img = cv2.convertScaleAbs(img, alpha=1.18, beta=-12)

    # ── Save ──────────────────────────────────────────
    cv2.imwrite(image_save_path, img)
    with open(label_save_path, "w") as f:
        f.write("\n".join(labels))

    print(f"[{os.path.basename(image_save_path)}]  "
          f"filled={len(labels)}  page={page_number}  set={set_label}")
    return img


# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs("synthetic_data", exist_ok=True)
    os.makedirs("synthetic_labels", exist_ok=True)

    for i in range(5):
        generate_omr_sheet(
            image_save_path=f"synthetic_data/omr_{i:04d}.png",
            label_save_path=f"synthetic_labels/omr_{i:04d}.txt",
            page_number=random.randint(1, 20),
            set_label=random.choice(["A", "B", "C", "D"]),
        )

    print("\nDone — synthetic_dataset/ and synthetic_labels/")