import os
import cv2
import numpy as np

BRAIN = "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p_devbox/ff1be343-f26a-4dda-8b06-9a95d3c4f1c1"
PIPE = "/home/antigravity/DioramaStickerApp/assets_pipeline"


def fit_poly2_surface(img, sample_mask, target_mask):
    H, W = img.shape[:2]
    ys, xs = np.where(sample_mask > 0)
    xn = (xs - W / 2.0) / (W / 2.0)
    yn = (ys - H / 2.0) / (H / 2.0)
    A = np.column_stack([np.ones_like(xn), xn, yn, xn**2, xn * yn, yn**2])

    ty, tx = np.where(target_mask > 0)
    txn = (tx - W / 2.0) / (W / 2.0)
    tyn = (ty - H / 2.0) / (H / 2.0)
    At = np.column_stack([np.ones_like(txn), txn, tyn, txn**2, txn * tyn, tyn**2])

    out = img.copy().astype(np.float32)
    for c in range(3):
        vals = img[ys, xs, c].astype(np.float32)
        coef, _, _, _ = np.linalg.lstsq(A, vals, rcond=None)
        pred = np.clip(At @ coef, 0, 255)
        out[ty, tx, c] = pred
    return np.clip(out, 0, 255).astype(np.uint8)


# ==============================================================================
# 1. HOTPOT FEAST — 100% FLAWLESS BARE ROUND WOODEN TABLE + PURE TATAMI MAT!
# ==============================================================================
def build_hotpot_empty_bg():
    orig = cv2.imread(os.path.join(BRAIN, "hotpot_feast_scene_1795514917338.jpg"))
    H, W = orig.shape[:2]
    cx, cy = 448.5, 593.5
    rx, ry = 395.0, 386.0

    Y, X = np.mgrid[:H, :W].astype(np.float32)
    E = np.sqrt(((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2)

    b, g, r = orig[:, :, 0].astype(np.int16), orig[:, :, 1].astype(np.int16), orig[:, :, 2].astype(np.int16)
    is_wood = (
        (E < 0.96)
        & (r > 180) & (r < 236)
        & (g > 125) & (g < 185)
        & (b > 72) & (b < 130)
        & ((r - g) > 35) & ((r - g) < 68)
        & ((g - b) > 35) & ((g - b) < 68)
    ).astype(np.uint8)
    is_wood_clean = cv2.erode(is_wood, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    table_disk = (E <= 1.005).astype(np.uint8)
    fitted = fit_poly2_surface(orig, is_wood_clean, table_disk)

    grain = np.zeros((H, W, 3), dtype=np.float32)
    rng = np.random.default_rng(42)
    for py in [268, 342, 416, 490, 564, 638, 712, 786, 860, 932]:
        cv2.line(grain, (0, py), (W, py), (-13, -17, -21), 2, cv2.LINE_AA)
        cv2.line(grain, (0, py + 2), (W, py + 2), (4, 5, 6), 1, cv2.LINE_AA)
    for _ in range(190):
        sy = int(rng.integers(212, 976))
        sx0 = int(rng.integers(55, 720))
        slen = int(rng.integers(45, 220))
        delta = float(rng.uniform(-8.5, 4.5))
        cv2.line(grain, (sx0, sy), (min(W - 1, sx0 + slen), sy), (delta * 0.7, delta * 0.85, delta), 1, cv2.LINE_AA)
    grain = cv2.GaussianBlur(grain, (0, 0), 0.7)
    wood_full = np.clip(fitted.astype(np.float32) + grain, 0, 255).astype(np.uint8)

    # PURE woven tatami tile sampled from bottom-left unobstructed tatami (x=94..132, y=878..942) — ZERO table edge!
    tatami_tile = orig[878:942, 94:132].copy()
    th, tw = tatami_tile.shape[:2]
    tatami_full = orig.copy()
    for yy in range(234, 968, th):
        for xx in range(90, 806, tw):
            hh = min(th, 968 - yy)
            ww = min(tw, 806 - xx)
            tatami_full[yy:yy+hh, xx:xx+ww] = tatami_tile[:hh, :ww]

    bg = orig.copy()

    # Restore top dark tatami border where boba & plum straws crossed
    bg[214:238, 120:176] = orig[214:238, 200:256]
    bg[214:238, 728:792] = orig[214:238, 632:696]
    # Restore right vertical tatami border where veggie basket slightly overlapped
    bg[355:445, 804:824] = orig[245:335, 804:824]
    bg[445:515, 804:824] = orig[245:315, 804:824]

    corner_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(corner_mask, (94, 236), (195, 354), 255, -1)   # boba cup
    cv2.rectangle(corner_mask, (705, 236), (805, 354), 255, -1)  # plum tea
    cv2.rectangle(corner_mask, (740, 348), (805, 510), 255, -1)  # veggie overhang on tatami
    cv2.rectangle(corner_mask, (722, 888), (805, 974), 255, -1)  # calico cat (full chin!)
    corner_mask[E <= 1.005] = 0
    m_c = (cv2.GaussianBlur(corner_mask.astype(np.float32), (5, 5), 0.9) / 255.0)[:, :, None]
    bg = (tatami_full.astype(np.float32) * m_c + bg.astype(np.float32) * (1.0 - m_c)).astype(np.uint8)

    # Fill entire circular wooden tabletop interior
    m_t = np.clip((0.999 - E) / 0.008, 0.0, 1.0)[:, :, None]
    bg = (wood_full.astype(np.float32) * m_t + bg.astype(np.float32) * (1.0 - m_t)).astype(np.uint8)

    # Redraw crisp hand-drawn dark-brown circular tabletop outline right along E=1.00!
    cv2.ellipse(bg, (int(round(cx)), int(round(cy))), (int(round(rx)), int(round(ry))),
                0, 0, 360, (32, 42, 62), 3, cv2.LINE_AA)

    # Clean right-edge tatami outer border line cleanly (x=820..826, y=355..510)
    bg[355:445, 804:824] = orig[245:335, 804:824]
    bg[445:515, 804:824] = orig[245:315, 804:824]
    # Re-apply tabletop & rim on top of right tatami border where E <= 1.004
    m_t2 = np.clip((0.999 - E) / 0.008, 0.0, 1.0)[:, :, None]
    bg = (wood_full.astype(np.float32) * m_t2 + bg.astype(np.float32) * (1.0 - m_t2)).astype(np.uint8)
    cv2.ellipse(bg, (int(round(cx)), int(round(cy))), (int(round(rx)), int(round(ry))),
                0, 0, 360, (32, 42, 62), 3, cv2.LINE_AA)

    cv2.imwrite(os.path.join(PIPE, "empty_bg_hotpot_feast.png"), bg)
    print("Saved flawless empty_bg_hotpot_feast.png")


# ==============================================================================
# 2. DESERT CAMP — 100% CLEAN GOLDEN SAND DIAMOND + OASIS + DUNES + ROCKS!
# ==============================================================================
def build_desert_empty_bg():
    orig = cv2.imread(os.path.join(BRAIN, "desert_camp_scene_1795514833166.jpg"))
    H, W = orig.shape[:2]

    # Exact outer diamond sand floor traced RIGHT UP to the bottom bevel edge (no stray feet/tails!)
    sand_poly = np.array([
        [54, 622], [180, 502], [335, 508], [392, 526], [448, 536],
        [520, 528], [585, 514], [638, 508], [768, 594], [842, 622],
        [448, 969]
    ], dtype=np.int32)
    sand_region = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(sand_region, [sand_poly], 1)

    b, g, r = orig[:, :, 0].astype(np.int16), orig[:, :, 1].astype(np.int16), orig[:, :, 2].astype(np.int16)
    is_sand = (
        (sand_region > 0)
        & (r > 224) & (g > 186) & (g < 236) & (b > 106) & (b < 176)
        & ((r - b) > 65) & ((r - g) < 52)
    ).astype(np.uint8)
    is_sand_clean = cv2.erode(is_sand, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    fitted = fit_poly2_surface(orig, is_sand_clean, sand_region)

    rng = np.random.default_rng(108)
    sand_tex = fitted.copy()
    for _ in range(110):
        sy = int(rng.integers(525, 952))
        sx = int(rng.integers(75, 820))
        if sand_region[sy, sx] > 0:
            cv2.circle(sand_tex, (sx, sy), int(rng.choice([1, 1, 2])), (104, 150, 200), -1, cv2.LINE_AA)
    for _ in range(30):
        sy = int(rng.integers(540, 938))
        sx = int(rng.integers(110, 765))
        if sand_region[sy, sx] > 0:
            slen = int(rng.integers(20, 46))
            pts = np.array([[sx, sy], [sx + slen // 2, sy + 2], [sx + slen, sy]], dtype=np.int32)
            cv2.polylines(sand_tex, [pts], False, (124, 170, 218), 1, cv2.LINE_AA)

    bg = orig.copy()

    # 1) Left sand dune behind cream bell tent
    dune_poly = np.array([
        [130, 540], [250, 422], [295, 412], [335, 456], [338, 514], [130, 540]
    ], dtype=np.int32)
    dune_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(dune_mask, [dune_poly], 1)
    sample_dune = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(sample_dune, (370, 415), (460, 452), 1, -1)
    cv2.rectangle(sample_dune, (185, 415), (225, 465), 1, -1)
    sample_dune = sample_dune & ((r > 200) & (g > 140) & (b < 140)).astype(np.uint8)
    dune_fitted = fit_poly2_surface(orig, sample_dune, dune_mask)

    tent_peak = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(tent_peak, [np.array([[135, 535], [256, 412], [362, 520], [135, 535]], dtype=np.int32)], 1)
    tent_peak[:476, 314:355] = 0
    m_dp = (cv2.GaussianBlur((tent_peak * 255).astype(np.float32), (5, 5), 1.0) / 255.0)[:, :, None]
    bg = (dune_fitted.astype(np.float32) * m_dp + bg.astype(np.float32) * (1.0 - m_dp)).astype(np.uint8)
    ridge_pts = np.array([[182, 462], [238, 446], [286, 466], [322, 492]], dtype=np.int32)
    cv2.polylines(bg, [ridge_pts], False, (48, 66, 98), 2, cv2.LINE_AA)

    # 2) Oasis water fixes
    water_zone = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(water_zone, [np.array([[345, 505], [395, 528], [452, 538], [520, 530], [585, 514], [585, 480], [345, 480]], dtype=np.int32)], 1)
    is_water = ((water_zone > 0) & (b > 175) & (g > 185) & (r < 170)).astype(np.uint8)
    is_water_clean = cv2.erode(is_water, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    water_fitted = fit_poly2_surface(orig, is_water_clean, water_zone)
    water_patch = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(water_patch, (403, 515), (27, 25), 0, 0, 360, 1, -1)
    cv2.ellipse(water_patch, (505, 526), (34, 18), 0, 0, 360, 1, -1)
    water_patch = water_patch & water_zone
    m_wp = (cv2.GaussianBlur((water_patch * 255).astype(np.float32), (5, 5), 1.0) / 255.0)[:, :, None]
    bg = (water_fitted.astype(np.float32) * m_wp + bg.astype(np.float32) * (1.0 - m_wp)).astype(np.uint8)

    # 3) Terracotta canyon rock base behind teal tent
    rock_zone = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(rock_zone, [np.array([[615, 515], [730, 495], [782, 595], [620, 560]], dtype=np.int32)], 1)
    sample_rock = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(sample_rock, (665, 455), (755, 510), 1, -1)
    sample_rock = sample_rock & ((r > 185) & (g > 100) & (g < 165) & (b > 65) & (b < 120)).astype(np.uint8)
    rock_fitted = fit_poly2_surface(orig, sample_rock, rock_zone)
    tent2_above = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(tent2_above, [np.array([[616, 520], [724, 496], [774, 595], [625, 560]], dtype=np.int32)], 1)
    tent2_above = tent2_above & (1 - sand_region)
    m_rk = (cv2.GaussianBlur((tent2_above * 255).astype(np.float32), (5, 5), 1.0) / 255.0)[:, :, None]
    bg = (rock_fitted.astype(np.float32) * m_rk + bg.astype(np.float32) * (1.0 - m_rk)).astype(np.uint8)

    # 4) Fill sand floor cleanly up to the exact diamond bevel edge!
    m_sf = (cv2.GaussianBlur((sand_region * 255).astype(np.float32), (5, 5), 1.0) / 255.0)[:, :, None]
    bg = (sand_tex.astype(np.float32) * m_sf + bg.astype(np.float32) * (1.0 - m_sf)).astype(np.uint8)

    # 5) Clean up left camel cleanly along the exact straight upper-left diamond edge!
    navy_col = tuple(int(v) for v in orig[50, 50])
    for y in range(472, 622):
        for x in range(38, 195):
            d_in = ((y - 452.0) + 1.018 * (x - 195.0)) / np.sqrt(1.0 + 1.018**2)
            if d_in < -14.5:
                bg[y, x] = navy_col
            elif d_in < -1.5:
                bg[y, x] = (255, 255, 255)
            elif d_in < 1.8:
                bg[y, x] = (32, 40, 59)
            else:
                if sand_region[y, x] > 0:
                    bg[y, x] = sand_tex[y, x]
                else:
                    t = np.clip((y - 472.0) / 130.0, 0.0, 1.0)
                    bg[y, x] = (int(120 + 28 * t), int(178 + 32 * t), int(238 + 14 * t))

    shore_pts = np.array([[345, 508], [392, 526], [448, 536], [518, 528], [585, 514]], dtype=np.int32)
    cv2.polylines(bg, [shore_pts], False, (48, 62, 86), 2, cv2.LINE_AA)
    rock_base_pts = np.array([[624, 514], [692, 548], [768, 594]], dtype=np.int32)
    cv2.polylines(bg, [rock_base_pts], False, (42, 54, 78), 2, cv2.LINE_AA)
    cv2.polylines(bg, [np.array([[54, 622], [448, 969], [842, 622]], dtype=np.int32)], False, (42, 54, 78), 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(PIPE, "empty_bg_desert_camp.png"), bg)
    print("Saved flawless empty_bg_desert_camp.png")


# ==============================================================================
# 3. TAIWAN NIGHT MARKET — 100% SPOTLESS STREET + 4 BARE EMPTY WOODEN STALLS!
# ==============================================================================
def build_taiwan_empty_bg():
    orig = cv2.imread(os.path.join(BRAIN, "taiwan_night_market_scene_1795514903307.jpg"))
    H, W = orig.shape[:2]
    bg = orig.copy()

    street_poly = np.array([
        [80, 768], [192, 704], [452, 558], [762, 744],
        [818, 774], [448, 1006]
    ], dtype=np.int32)
    street_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(street_mask, [street_poly], 1)

    b, g, r = orig[:, :, 0].astype(np.int16), orig[:, :, 1].astype(np.int16), orig[:, :, 2].astype(np.int16)
    is_stone = (
        (street_mask > 0)
        & (r > 176) & (r < 224)
        & (g > 158) & (g < 206)
        & (b > 148) & (b < 196)
        & (np.abs(r - g) < 28) & (np.abs(g - b) < 22)
    ).astype(np.uint8)
    is_stone_clean = cv2.erode(is_stone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    stone_base = fit_poly2_surface(orig, is_stone_clean, street_mask)

    stone_tiled = stone_base.copy()
    grout_col = (128, 134, 146)
    rng = np.random.default_rng(77)
    dx, dy = 52.0, 30.0
    for i in range(-15, 16):
        for j in range(-15, 16):
            cx_t = 448 + (i - j) * (dx / 2.0)
            cy_t = 755 + (i + j) * (dy / 2.0)
            pts = np.array([
                [cx_t, cy_t - dy / 2.0],
                [cx_t + dx / 2.0, cy_t],
                [cx_t, cy_t + dy / 2.0],
                [cx_t - dx / 2.0, cy_t]
            ], dtype=np.int32)
            delta = int(rng.integers(-7, 8))
            tile_col = np.clip(stone_base[min(H-1, max(0, int(cy_t))), min(W-1, max(0, int(cx_t)))].astype(np.int16) + delta, 0, 255)
            cv2.fillPoly(stone_tiled, [pts], tuple(int(v) for v in tile_col))
            cv2.polylines(stone_tiled, [pts], True, grout_col, 1, cv2.LINE_AA)
    stone_tiled = cv2.GaussianBlur(stone_tiled, (3, 3), 0.55)

    street_fill = street_mask.copy()
    cv2.rectangle(street_fill, (45, 725), (100, 782), 0, -1)
    m_st = (cv2.GaussianBlur((street_fill * 255).astype(np.float32), (5, 5), 1.0) / 255.0)[:, :, None]
    bg = (stone_tiled.astype(np.float32) * m_st + bg.astype(np.float32) * (1.0 - m_st)).astype(np.uint8)

    # Erase ENTIRE tapioca jar against left wall & stone floor (x=96..158, y=668..768)!
    wall_col_left = (146, 152, 164)
    cv2.fillPoly(bg, [np.array([[96, 668], [156, 668], [156, 738], [96, 768]], dtype=np.int32)], wall_col_left)
    jar_floor = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(jar_floor, [np.array([[94, 756], [162, 718], [180, 742], [94, 774]], dtype=np.int32)], 1)
    bg[jar_floor > 0] = stone_tiled[jar_floor > 0]

    # Erase right crate + calico cat cleanly
    cv2.fillPoly(bg, [np.array([[778, 690], [830, 690], [830, 766], [778, 766]], dtype=np.int32)], (146, 166, 192))
    crate_floor = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(crate_floor, [np.array([[744, 755], [818, 710], [838, 766], [786, 804], [744, 774]], dtype=np.int32)], 1)
    crate_floor = crate_floor & street_mask
    bg[crate_floor > 0] = stone_tiled[crate_floor > 0]

    shiba_gap = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(shiba_gap, (300, 600), (344, 660), 1, -1)
    bg[shiba_gap > 0] = stone_tiled[shiba_gap > 0]

    # Colors sampled directly from the illustration's wooden stalls
    ink = (36, 46, 68)
    wall_bg = (165, 188, 212)
    wood_top = (114, 168, 224)
    wood_fr = (94, 146, 202)
    wood_sd = (78, 126, 180)
    cream = (205, 232, 248)

    def draw_slats(img, p_tl, p_tr, p_br, p_bl, n=7, fill_col=wood_fr):
        pts = np.array([p_tl, p_tr, p_br, p_bl], dtype=np.int32)
        cv2.fillPoly(img, [pts], fill_col)
        for k in range(1, n):
            t = k / float(n)
            x1 = int(round(p_tl[0] + t * (p_tr[0] - p_tl[0])))
            y1 = int(round(p_tl[1] + t * (p_tr[1] - p_tl[1])))
            x2 = int(round(p_bl[0] + t * (p_br[0] - p_bl[0])))
            y2 = int(round(p_bl[1] + t * (p_br[1] - p_bl[1])))
            cv2.line(img, (x1, y1), (x2, y2), (58, 92, 138), 1, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, ink, 2, cv2.LINE_AA)

    def draw_quad(img, pts, fill_col):
        arr = np.array(pts, dtype=np.int32)
        cv2.fillPoly(img, [arr], fill_col)
        cv2.polylines(img, [arr], True, ink, 2, cv2.LINE_AA)

    # ---- STALL 1 (珍珠奶茶) — Keep top signboard wooden frame intact! Fill strictly below signboard ----
    cv2.fillPoly(bg, [np.array([[128, 558], [292, 466], [292, 595], [128, 650]], dtype=np.int32)], wall_bg)
    draw_slats(bg, (124, 646), (176, 678), (176, 740), (124, 708), n=4, fill_col=wood_sd)
    draw_slats(bg, (176, 678), (304, 602), (304, 664), (176, 740), n=8, fill_col=wood_fr)
    draw_quad(bg, [(118, 640), (244, 566), (308, 602), (176, 678)], wood_top)
    cv2.line(bg, (124, 520), (124, 640), ink, 3, cv2.LINE_AA)
    cv2.line(bg, (176, 548), (176, 678), ink, 3, cv2.LINE_AA)
    cv2.line(bg, (292, 476), (292, 602), ink, 3, cv2.LINE_AA)
    cv2.fillPoly(bg, [np.array([[178, 494], [285, 434], [285, 478], [178, 538]], dtype=np.int32)], cream)

    # ---- STALL 2 (盐酥鸡) — Fill strictly below signboard frame ----
    cv2.fillPoly(bg, [np.array([[304, 462], [452, 378], [452, 502], [304, 560]], dtype=np.int32)], wall_bg)
    draw_slats(bg, (292, 540), (344, 574), (344, 632), (292, 598), n=4, fill_col=wood_sd)
    draw_slats(bg, (344, 574), (462, 504), (462, 562), (344, 632), n=8, fill_col=wood_fr)
    draw_quad(bg, [(288, 534), (404, 466), (466, 502), (344, 574)], wood_top)
    cv2.line(bg, (344, 456), (344, 574), ink, 3, cv2.LINE_AA)
    cv2.line(bg, (452, 384), (452, 502), ink, 3, cv2.LINE_AA)
    cv2.fillPoly(bg, [np.array([[336, 399], [444, 337], [444, 382], [336, 444]], dtype=np.int32)], cream)

    # ---- STALL 3 (蚵仔煎 / 芒果冰) — Fill strictly below signboard frame ----
    cv2.fillPoly(bg, [np.array([[486, 424], [616, 496], [616, 595], [486, 525]], dtype=np.int32)], wall_bg)
    draw_slats(bg, (466, 536), (580, 604), (580, 662), (466, 594), n=8, fill_col=wood_fr)
    draw_slats(bg, (580, 604), (636, 570), (636, 628), (580, 662), n=4, fill_col=wood_sd)
    draw_quad(bg, [(462, 532), (526, 494), (640, 562), (580, 604)], wood_top)
    cv2.line(bg, (486, 412), (486, 526), ink, 3, cv2.LINE_AA)
    cv2.line(bg, (580, 468), (580, 604), ink, 3, cv2.LINE_AA)
    cv2.line(bg, (614, 482), (614, 576), ink, 3, cv2.LINE_AA)
    cv2.fillPoly(bg, [np.array([[493, 369], [603, 432], [605, 477], [496, 414]], dtype=np.int32)], cream)

    # ---- STALL 4 (套圈圈) — Fill strictly below signboard & ALSO clear right plushie bear! ----
    cv2.fillPoly(bg, [np.array([[636, 514], [798, 540], [798, 680], [636, 628]], dtype=np.int32)], wall_bg)
    # Restore Stall 4 upper right roof wooden beam cleanly
    cv2.line(bg, (748, 568), (798, 538), ink, 3, cv2.LINE_AA)
    draw_quad(bg, [(652, 580), (682, 562), (796, 630), (766, 648)], (102, 152, 206))
    draw_slats(bg, (634, 632), (748, 702), (748, 764), (634, 694), n=8, fill_col=wood_fr)
    draw_slats(bg, (748, 702), (806, 666), (806, 728), (748, 764), n=4, fill_col=wood_sd)
    draw_quad(bg, [(630, 628), (688, 592), (808, 664), (748, 702)], wood_top)
    cv2.line(bg, (636, 502), (636, 628), ink, 3, cv2.LINE_AA)
    cv2.line(bg, (748, 568), (748, 702), ink, 3, cv2.LINE_AA)
    cv2.line(bg, (798, 540), (798, 664), ink, 3, cv2.LINE_AA)
    cv2.fillPoly(bg, [np.array([[644, 457], [744, 515], [744, 558], [644, 500]], dtype=np.int32)], cream)

    # Redraw crisp bottom diamond curb outline
    curb_pts = np.array([[80, 768], [448, 1006], [818, 774]], dtype=np.int32)
    cv2.polylines(bg, [curb_pts], False, ink, 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(PIPE, "empty_bg_taiwan_night_market.png"), bg)
    print("Saved flawless empty_bg_taiwan_night_market.png")


if __name__ == "__main__":
    build_hotpot_empty_bg()
    build_desert_empty_bg()
    build_taiwan_empty_bg()
