import json
import os
import cv2
import numpy as np
from scipy import ndimage

ORIG_PATH = "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p_devbox/ff1be343-f26a-4dda-8b06-9a95d3c4f1c1/campfire_diorama_scene_1795488810869.jpg"
BG_PATH = "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p_devbox/ff1be343-f26a-4dda-8b06-9a95d3c4f1c1/campfire_diorama_empty_bg_1795490181683.jpg"
TENTS_PATH = "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p_devbox/ff1be343-f26a-4dda-8b06-9a95d3c4f1c1/campfire_diorama_tents_layer_1795492283529.jpg"

OUT_DIR = "/home/antigravity/DioramaStickerApp/assets_pipeline/extracted"
os.makedirs(OUT_DIR, exist_ok=True)

orig = cv2.imread(ORIG_PATH)
bg = cv2.imread(BG_PATH)
tents = cv2.imread(TENTS_PATH)
H, W = orig.shape[:2]

diff_orig_bg = np.linalg.norm(orig.astype(np.float32) - bg.astype(np.float32), axis=2)
diff_tents_bg = np.linalg.norm(tents.astype(np.float32) - bg.astype(np.float32), axis=2)

def get_outline_mask(img, r_max=118, g_max=98, b_max=88):
    b, g, r = cv2.split(img)
    return ((r < r_max) & (g < g_max) & (b < b_max) & (r.astype(np.int16) >= b.astype(np.int16) - 12)).astype(np.uint8)

outline_orig = get_outline_mask(orig)
# Slightly broader outline detector for enclosed flood-fill (catches warm brown outlines around bright flames too)
outline_orig_strong = get_outline_mask(orig, r_max=132, g_max=108, b_max=92)
outline_tents = get_outline_mask(tents)

def smooth_alpha_from_binary(bin_mask, blur_sigma=0.55):
    alpha = bin_mask.astype(np.float32) * 255.0
    if blur_sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), blur_sigma)
        eroded = cv2.erode(bin_mask.astype(np.uint8), np.ones((3, 3), np.uint8))
        alpha = np.where(eroded > 0, 255.0, alpha)
    return np.clip(alpha, 0, 255).astype(np.uint8)

def save_rgba_from_mask(img, alpha_full, out_path, pad=2):
    ys, xs = np.where(alpha_full > 10)
    if len(xs) == 0:
        raise ValueError(f"Empty mask for {out_path}")
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(img.shape[1], int(xs.max()) + 1 + pad)
    y1 = min(img.shape[0], int(ys.max()) + 1 + pad)
    crop_bgr = img[y0:y1, x0:x1]
    crop_a = alpha_full[y0:y1, x0:x1]
    rgba = np.dstack([crop_bgr, crop_a])
    cv2.imwrite(out_path, rgba)
    return x0, y0, x1 - x0, y1 - y0

def extract_outline_enclosed(img, roi, bg_seeds=None, exclude_polys=None, keep_largest=True,
                             min_area=120, close_outline=3, dilate_out=2, strong_outline=False):
    """
    Ultra-clean cartoon sticker extractor!
    Flood-fills background from ROI borders (+ optional extra bg_seeds) stopped by the
    dark-brown cartoon outlines. Everything enclosed inside the dark-brown outline +
    the outline itself becomes a razor-sharp sticker!
    """
    x0, y0, x1, y1 = roi
    sub_out = (outline_orig_strong if strong_outline else outline_orig)[y0:y1, x0:x1].copy()
    sh, sw = sub_out.shape

    if exclude_polys:
        for poly in exclude_polys:
            pts = np.array([[px - x0, py - y0] for px, py in poly], dtype=np.int32)
            cv2.fillPoly(sub_out, [pts], 0)

    # Close tiny 1-2px gaps in the outline so flood-fill doesn't leak inside
    if close_outline > 1:
        wall = cv2.morphologyEx(sub_out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_outline, close_outline)))
    else:
        wall = sub_out.copy()

    # Flood-fill exterior from a 1px padded border
    padded = np.zeros((sh + 2, sw + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = wall
    ff_mask = np.zeros((sh + 4, sw + 4), dtype=np.uint8)
    cv2.floodFill(padded, ff_mask, (0, 0), 2)
    if bg_seeds:
        for (sx, sy) in bg_seeds:
            lx, ly = sx - x0 + 1, sy - y0 + 1
            if 0 <= lx < sw + 2 and 0 <= ly < sh + 2 and padded[ly, lx] == 0:
                cv2.floodFill(padded, ff_mask, (lx, ly), 2)

    # Foreground is everything NOT marked as exterior (2)
    fg = (padded[1:-1, 1:-1] != 2).astype(np.uint8)

    if exclude_polys:
        for poly in exclude_polys:
            pts = np.array([[px - x0, py - y0] for px, py in poly], dtype=np.int32)
            cv2.fillPoly(fg, [pts], 0)

    # Clean stray lines sticking outside the main silhouette via slight open then recover outlines
    fg_core = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(fg_core)
    cleaned = np.zeros_like(fg_core)
    if num_l > 1:
        if keep_largest:
            best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            cleaned[labels == best] = 1
        else:
            for i in range(1, num_l):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    cleaned[labels == i] = 1

    # Recover full outer edge + dark outline pixels belonging to the chosen object
    dil_big = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cleaned = np.where((cleaned > 0) | ((dil_big > 0) & (fg > 0)), 1, 0).astype(np.uint8)
    if dilate_out > 0:
        d_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_out + 1, 2 * dilate_out + 1))
        dil = cv2.dilate(cleaned, d_k)
        cleaned = np.where((cleaned > 0) | ((dil > 0) & (sub_out > 0)), 1, 0).astype(np.uint8)

    cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
    if exclude_polys:
        for poly in exclude_polys:
            pts = np.array([[px - x0, py - y0] for px, py in poly], dtype=np.int32)
            cv2.fillPoly(cleaned, [pts], 0)

    alpha_full = np.zeros((H, W), dtype=np.uint8)
    alpha_full[y0:y1, x0:x1] = smooth_alpha_from_binary(cleaned, blur_sigma=0.55)
    return alpha_full

def extract_diff_object(img, diff_map, roi, thresh=28, keep_largest=True, min_area=120,
                        exclude_polys=None, include_outline=True, soft_glow=False, close_k=3):
    x0, y0, x1, y1 = roi
    sub_diff = diff_map[y0:y1, x0:x1].copy()
    sub_mask = (sub_diff > thresh).astype(np.uint8)

    if exclude_polys:
        for poly in exclude_polys:
            pts = np.array([[px - x0, py - y0] for px, py in poly], dtype=np.int32)
            cv2.fillPoly(sub_mask, [pts], 0)

    if soft_glow:
        alpha_sub = np.clip((sub_diff - 10.0) / 48.0, 0.0, 1.0)
        if exclude_polys:
            ex = np.ones_like(alpha_sub)
            for poly in exclude_polys:
                pts = np.array([[px - x0, py - y0] for px, py in poly], dtype=np.int32)
                cv2.fillPoly(ex, [pts], 0)
            alpha_sub *= ex
        core = (sub_diff > thresh).astype(np.uint8)
        if exclude_polys:
            for poly in exclude_polys:
                pts = np.array([[px - x0, py - y0] for px, py in poly], dtype=np.int32)
                cv2.fillPoly(core, [pts], 0)
        core_dil = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
        alpha_sub = np.where(core_dil > 0, alpha_sub, 0.0)
        alpha_full = np.zeros((H, W), dtype=np.uint8)
        alpha_full[y0:y1, x0:x1] = np.clip(alpha_sub * 255.0, 0, 255).astype(np.uint8)
        return alpha_full

    sub_mask = cv2.morphologyEx(sub_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if close_k > 1:
        sub_mask = cv2.morphologyEx(sub_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))

    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(sub_mask)
    cleaned = np.zeros_like(sub_mask)
    if num_l > 1:
        if keep_largest:
            best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            cleaned[labels == best] = 1
        else:
            for i in range(1, num_l):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    cleaned[labels == i] = 1

    cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
    if include_outline:
        sub_out = outline_orig[y0:y1, x0:x1]
        dil = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cleaned = np.where((cleaned > 0) | ((dil > 0) & (sub_out > 0)), 1, 0).astype(np.uint8)
        cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)

    alpha_full = np.zeros((H, W), dtype=np.uint8)
    alpha_full[y0:y1, x0:x1] = smooth_alpha_from_binary(cleaned, blur_sigma=0.55)
    return alpha_full

def extract_grabcut_object(img, roi, fg_rects=None, fg_pts=None, bg_pts=None, bg_polys=None,
                           exclude_polys=None, use_diff_bg_hint=True, iter_count=7, grass_bg_hint=False):
    x0, y0, x1, y1 = roi
    sub = img[y0:y1, x0:x1]
    sh, sw = sub.shape[:2]
    gc_mask = np.full((sh, sw), cv2.GC_PR_BGD, dtype=np.uint8)

    gc_mask[0:3, :] = cv2.GC_BGD
    gc_mask[-3:, :] = cv2.GC_BGD
    gc_mask[:, 0:3] = cv2.GC_BGD
    gc_mask[:, -3:] = cv2.GC_BGD

    if use_diff_bg_hint:
        sub_diff = diff_orig_bg[y0:y1, x0:x1]
        gc_mask[sub_diff < 18] = cv2.GC_BGD

    if grass_bg_hint:
        # Mark warm glowing yellow-green grass (G > R - 20 and G > B + 35 and not dark outline) as probable bg
        b, g, r = cv2.split(sub)
        is_grass = (g.astype(np.int16) > b.astype(np.int16) + 32) & (g.astype(np.int16) > r.astype(np.int16) - 22) & (r > 95) & (r < 215) & (g > 115)
        gc_mask[is_grass] = cv2.GC_BGD

    if fg_rects:
        for (rx0, ry0, rx1, ry1) in fg_rects:
            gc_mask[max(0, ry0-y0):min(sh, ry1-y0), max(0, rx0-x0):min(sw, rx1-x0)] = cv2.GC_PR_FGD

    if fg_pts:
        for (px, py, rad) in fg_pts:
            cv2.circle(gc_mask, (px - x0, py - y0), rad, int(cv2.GC_FGD), -1)

    if bg_pts:
        for (px, py, rad) in bg_pts:
            cv2.circle(gc_mask, (px - x0, py - y0), rad, int(cv2.GC_BGD), -1)

    if bg_polys:
        for poly in bg_polys:
            pts = np.array([[px - x0, py - y0] for px, py in poly], dtype=np.int32)
            cv2.fillPoly(gc_mask, [pts], int(cv2.GC_BGD))

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(sub, gc_mask, None, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_MASK)

    bin_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    if exclude_polys:
        for poly in exclude_polys:
            pts = np.array([[px - x0, py - y0] for px, py in poly], dtype=np.int32)
            cv2.fillPoly(bin_mask, [pts], 0)

    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask)
    cleaned = np.zeros_like(bin_mask)
    if num_l > 1:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cleaned[labels == best] = 1

    cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
    sub_out = outline_orig[y0:y1, x0:x1]
    dil = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cleaned = np.where((cleaned > 0) | ((dil > 0) & (sub_out > 0)), 1, 0).astype(np.uint8)
    cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
    if exclude_polys:
        for poly in exclude_polys:
            pts = np.array([[px - x0, py - y0] for px, py in poly], dtype=np.int32)
            cv2.fillPoly(cleaned, [pts], 0)

    alpha_full = np.zeros((H, W), dtype=np.uint8)
    alpha_full[y0:y1, x0:x1] = smooth_alpha_from_binary(cleaned, blur_sigma=0.55)
    return alpha_full

stickers_meta = []

def register_sticker(sid, name_zh, name_en, category, z_index, img, alpha_full,
                     target_box_override=None, pad=2):
    out_file = f"{sid}.png"
    out_path = os.path.join(OUT_DIR, out_file)
    x, y, w, h = save_rgba_from_mask(img, alpha_full, out_path, pad=pad)
    if target_box_override is not None:
        tx, ty, tw, th = target_box_override
        rgba = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
        rgba_resized = cv2.resize(rgba, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_path, rgba_resized)
        x, y, w, h = tx, ty, tw, th
    stickers_meta.append({
        "id": sid,
        "file": out_file,
        "name_zh": name_zh,
        "name_en": name_en,
        "category": category,
        "zIndex": z_index,
        "x": int(x),
        "y": int(y),
        "width": int(w),
        "height": int(h),
        "normX": round((x + w / 2.0) / W, 5),
        "normY": round((y + h / 2.0) / H, 5),
        "normW": round(w / W, 5),
        "normH": round(h / H, 5),
    })
    print(f"Extracted {sid:22s} -> ({x:3d}, {y:3d}, {w:3d}x{h:3d}) z={z_index}")

# ==============================================================================
# LAYER 1 & 2: TENTS & BLANKETS (from unoccluded tents_layer)
# ==============================================================================
mask_tb = (diff_tents_bg > 26).astype(np.uint8)
mask_tb = cv2.morphologyEx(mask_tb, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
mask_tb = cv2.morphologyEx(mask_tb, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

left_tent_poly = np.array([[135, 445], [445, 445], [445, 645], [302, 695], [255, 708], [140, 640]], dtype=np.int32)
left_tent_region = np.zeros((H, W), dtype=np.uint8)
cv2.fillPoly(left_tent_region, [left_tent_poly], 1)
m_tent_l = ndimage.binary_fill_holes(mask_tb & left_tent_region).astype(np.uint8)
dil = cv2.dilate(m_tent_l, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
m_tent_l = np.where((m_tent_l > 0) | ((dil > 0) & (outline_tents > 0) & (left_tent_region > 0)), 1, 0).astype(np.uint8)

left_blanket_poly = np.array([[105, 640], [255, 702], [302, 690], [440, 680], [440, 865], [105, 865]], dtype=np.int32)
left_blanket_region = np.zeros((H, W), dtype=np.uint8)
cv2.fillPoly(left_blanket_region, [left_blanket_poly], 1)
m_blanket_l = ndimage.binary_fill_holes(mask_tb & left_blanket_region).astype(np.uint8)
nl, lb, st, _ = cv2.connectedComponentsWithStats(m_blanket_l)
m_blanket_l = (lb == (1 + np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)

right_tent_poly = np.array([[470, 480], [810, 480], [810, 690], [678, 756], [625, 738], [495, 685]], dtype=np.int32)
right_tent_region = np.zeros((H, W), dtype=np.uint8)
cv2.fillPoly(right_tent_region, [right_tent_poly], 1)
m_tent_r = ndimage.binary_fill_holes(mask_tb & right_tent_region).astype(np.uint8)
dil = cv2.dilate(m_tent_r, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
m_tent_r = np.where((m_tent_r > 0) | ((dil > 0) & (outline_tents > 0) & (right_tent_region > 0)), 1, 0).astype(np.uint8)

right_blanket_poly = np.array([[465, 690], [625, 732], [678, 748], [810, 700], [810, 890], [465, 890]], dtype=np.int32)
right_blanket_region = np.zeros((H, W), dtype=np.uint8)
cv2.fillPoly(right_blanket_region, [right_blanket_poly], 1)
m_blanket_r = ndimage.binary_fill_holes(mask_tb & right_blanket_region).astype(np.uint8)
nl, lb, st, _ = cv2.connectedComponentsWithStats(m_blanket_r)
m_blanket_r = (lb == (1 + np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)

register_sticker("blanket_left", "蓝格野餐垫", "Blue Picnic Blanket", "营地设施", 10,
                 tents, smooth_alpha_from_binary(m_blanket_l), target_box_override=(55, 608, 275, 165))
register_sticker("blanket_right", "粉格野餐垫", "Pink Picnic Blanket", "营地设施", 10,
                 tents, smooth_alpha_from_binary(m_blanket_r), target_box_override=(495, 578, 290, 175))
register_sticker("tent_left", "紫条纹帐篷", "Striped Tent", "营地设施", 12,
                 tents, smooth_alpha_from_binary(m_tent_l), target_box_override=(162, 405, 232, 215))
register_sticker("tent_right", "粉色几何帐篷", "Pink Tent", "营地设施", 12,
                 tents, smooth_alpha_from_binary(m_tent_r), target_box_override=(522, 408, 240, 222))

# ==============================================================================
# LAYER 2: CAMPFIRE & PROPS (outline-enclosed = ZERO yellow grass halo!)
# ==============================================================================
# 5. Campfire: outline-enclosed! (Note flames have warm orange-brown outline, use strong_outline + GrabCut fallback)
a_campfire = extract_grabcut_object(
    orig, (370, 588, 526, 766),
    fg_rects=[(380, 600, 518, 756)],
    fg_pts=[(448, 665, 26), (448, 725, 18), (402, 718, 14), (492, 718, 14), (448, 622, 12)],
    bg_pts=[(378, 598, 8), (516, 598, 8), (378, 756, 8), (516, 756, 8), (382, 660, 8), (514, 660, 8), (412, 608, 8), (488, 608, 8)],
    bg_polys=[[(370, 588), (412, 588), (412, 640), (370, 640)],
              [(495, 588), (526, 588), (526, 635), (495, 635)]],
    use_diff_bg_hint=False, grass_bg_hint=True, iter_count=8
)
register_sticker("campfire", "温暖篝火", "Cozy Campfire", "露营道具", 25, orig, a_campfire)

# 6. Cooking pot on stones: outline-enclosed!
a_pot = extract_outline_enclosed(
    orig, (268, 676, 355, 768),
    bg_seeds=[(272, 680), (350, 680), (272, 762), (350, 762)],
    exclude_polys=[[(268, 676), (310, 676), (268, 705)]],
    close_outline=3, strong_outline=True
)
register_sticker("cooking_pot", "篝火炖锅", "Cooking Pot", "露营道具", 24, orig, a_pot)

# 7. Left camping lantern: outline-enclosed!
a_lantern_l = extract_outline_enclosed(
    orig, (366, 560, 416, 636),
    bg_seeds=[(370, 565), (412, 565), (370, 632), (412, 632)],
    exclude_polys=[[(366, 560), (378, 560), (375, 625), (366, 625)],
                   [(398, 560), (416, 560), (416, 585), (398, 585)]],
    close_outline=3, strong_outline=True
)
register_sticker("lantern_left", "帐篷马灯", "Tent Lantern", "露营道具", 26, orig, a_lantern_l)

# 8. Right camping lantern: outline-enclosed!
a_lantern_r = extract_outline_enclosed(
    orig, (534, 678, 585, 758),
    bg_seeds=[(538, 682), (580, 682), (538, 754), (580, 754)],
    close_outline=3, strong_outline=True
)
register_sticker("lantern_right", "篝火马灯", "Campfire Lantern", "露营道具", 26, orig, a_lantern_r)

# 9. Wooden log with two cocoa mugs (clean diff + exclude girl's shoe)
a_log = extract_diff_object(
    orig, diff_orig_bg, (252, 885, 398, 1015), thresh=32, keep_largest=True,
    exclude_polys=[[(350, 885), (398, 885), (398, 932), (355, 932)]]
)
register_sticker("log_mugs", "热可可圆木", "Cocoa & Log", "露营道具", 45, orig, a_log)

# 10. Signpost with two red mushrooms
a_sign = extract_diff_object(orig, diff_orig_bg, (395, 938, 502, 1055), thresh=28, keep_largest=False, min_area=180)
register_sticker("signpost_mushrooms", "路牌与蘑菇", "Signpost & Mushrooms", "露营道具", 46, orig, a_sign)

# ==============================================================================
# LAYER 3: 11 CARTOON CHARACTERS (clean GrabCut / outline-enclosed!)
# ==============================================================================
# 11. Boy holding firefly jar (clean bottom edge from tent/lantern!)
a_jar_boy = extract_grabcut_object(
    orig, (346, 392, 466, 592),
    fg_rects=[(354, 400, 458, 586)],
    fg_pts=[(395, 435, 18), (398, 478, 16), (398, 520, 14), (402, 562, 10), (440, 514, 14)],
    bg_pts=[(352, 400, 6), (460, 400, 6), (354, 578, 8), (458, 578, 8), (366, 558, 8)],
    bg_polys=[[(346, 515), (382, 515), (382, 592), (346, 592)],
              [(436, 545), (466, 545), (466, 592), (436, 592)]],
    exclude_polys=[[(346, 522), (376, 522), (376, 592), (346, 592)]]
)
register_sticker("char_jar_boy", "萤火虫罐子男孩", "Firefly Jar Boy", "卡通人物", 22, orig, a_jar_boy)

# 12. Girl in pink beanie clapping (keep beanie pom-pom intact!)
a_clap_girl = extract_grabcut_object(
    orig, (466, 392, 565, 584),
    fg_rects=[(474, 396, 556, 576)],
    fg_pts=[(540, 404, 8), (516, 434, 18), (514, 474, 16), (512, 516, 14), (512, 556, 10)],
    bg_pts=[(472, 400, 6), (560, 400, 4), (472, 576, 6), (556, 574, 8), (548, 555, 8)],
    bg_polys=[[(536, 520), (565, 520), (565, 584), (536, 584)],
              [(466, 550), (484, 550), (484, 584), (466, 584)]],
    exclude_polys=[[(540, 535), (565, 535), (565, 584), (540, 584)]]
)
register_sticker("char_clap_girl", "拍手女孩", "Clapping Girl", "卡通人物", 22, orig, a_clap_girl)

# 13. Girl in yellow beanie sitting on left blue blanket
a_cocoa_girl = extract_grabcut_object(
    orig, (132, 526, 254, 704),
    fg_rects=[(152, 540, 232, 686)],
    fg_pts=[(180, 565, 16), (186, 610, 14), (190, 650, 14), (205, 675, 12), (234, 596, 5)],
    bg_pts=[(140, 540, 6), (244, 540, 8), (142, 650, 5), (242, 665, 8), (150, 695, 6), (235, 695, 8)],
    bg_polys=[[(132, 615), (152, 615), (152, 704), (132, 704)],
              [(228, 635), (254, 635), (254, 704), (228, 704)]],
)
register_sticker("char_cocoa_girl", "热可可女孩", "Hot Cocoa Girl", "卡通人物", 30, orig, a_cocoa_girl)

# 14. Girl eating cookie at left tent entrance
a_cookie_girl = extract_grabcut_object(
    orig, (285, 508, 378, 658),
    fg_rects=[(298, 522, 368, 646)],
    fg_pts=[(330, 545, 14), (332, 580, 14), (332, 615, 12), (348, 638, 10)],
    bg_pts=[(292, 518, 6), (370, 518, 6), (292, 645, 6), (372, 575, 5), (370, 645, 6)],
    bg_polys=[[(366, 560), (378, 560), (378, 645), (366, 645)]]
)
register_sticker("char_cookie_girl", "饼干女孩", "Cookie Girl", "卡通人物", 30, orig, a_cookie_girl)

# 15. Camera kid inside right tent (trim tent flap on right edge cleanly!)
a_camera_kid = extract_grabcut_object(
    orig, (540, 502, 632, 642),
    fg_rects=[(548, 510, 618, 632)],
    fg_pts=[(588, 542, 14), (584, 575, 14), (586, 608, 12), (566, 618, 10)],
    bg_pts=[(544, 512, 6), (624, 512, 6), (622, 545, 6), (626, 595, 6), (546, 560, 6), (546, 632, 6), (626, 630, 6)],
    bg_polys=[[(540, 585), (550, 585), (550, 642), (540, 642)],
              [(616, 502), (632, 502), (632, 642), (622, 642)]],
)
register_sticker("char_camera_kid", "小摄影师", "Camera Kid", "卡通人物", 30, orig, a_camera_kid)

# 16. Sparkler kid on right blanket
a_sparkler_kid = extract_grabcut_object(
    orig, (660, 534, 780, 702),
    fg_rects=[(686, 550, 768, 688)],
    fg_pts=[(730, 574, 16), (726, 612, 14), (722, 650, 14), (722, 678, 12), (674, 588, 8)],
    bg_pts=[(666, 544, 6), (772, 544, 6), (666, 645, 8), (675, 690, 8), (772, 635, 6), (772, 690, 6)],
    bg_polys=[[(765, 620), (780, 620), (780, 702), (765, 702)]]
)
register_sticker("char_sparkler_kid", "仙女棒小朋友", "Sparkler Kid", "卡通人物", 30, orig, a_sparkler_kid)

# 17. Guitar boy on left stump
a_guitar = extract_grabcut_object(
    orig, (150, 722, 306, 922),
    fg_rects=[(158, 728, 298, 914)],
    fg_pts=[(202, 762, 18), (206, 800, 16), (218, 842, 18), (274, 816, 10), (232, 892, 14), (178, 882, 10)],
    bg_pts=[(156, 728, 6), (298, 728, 8), (275, 752, 10), (298, 868, 10), (156, 845, 6), (158, 914, 8), (298, 914, 8)],
    bg_polys=[[(230, 722), (306, 722), (306, 788), (248, 788)],
              [(150, 722), (174, 722), (150, 760)],
              [(268, 838), (306, 838), (306, 922), (268, 922)]],
    exclude_polys=[[(254, 722), (306, 722), (306, 774), (254, 774)]],
    use_diff_bg_hint=True, grass_bg_hint=True, iter_count=8
)
register_sticker("char_guitar_boy", "吉他男孩", "Guitar Boy", "卡通人物", 35, orig, a_guitar)

# 18. Marshmallow roasting girl (KEEP both white marshmallows on roasting stick!)
a_marsh = extract_grabcut_object(
    orig, (292, 748, 444, 944),
    fg_rects=[(298, 756, 438, 938)],
    fg_pts=[(382, 782, 18), (382, 822, 16), (380, 866, 16), (380, 914, 14),
            (310, 832, 8), (322, 845, 8), (336, 858, 5), (348, 868, 5)],
    bg_pts=[(302, 756, 8), (340, 756, 8), (436, 756, 8), (302, 920, 10), (436, 920, 10), (332, 795, 8)],
    bg_polys=[[(292, 748), (350, 748), (350, 792), (292, 792)],
              [(292, 880), (342, 880), (342, 944), (292, 944)]],
    exclude_polys=[[(292, 748), (355, 748), (355, 780), (292, 780)]],
    use_diff_bg_hint=True, grass_bg_hint=True, iter_count=8
)
register_sticker("char_marshmallow_girl", "烤棉花糖女孩", "Marshmallow Girl", "卡通人物", 36, orig, a_marsh)

# 19. Dancing boy with tambourine
a_tamb = extract_grabcut_object(
    orig, (456, 756, 608, 938),
    fg_rects=[(466, 764, 602, 932)],
    fg_pts=[(516, 792, 18), (528, 824, 16), (528, 864, 16), (528, 906, 12), (586, 816, 14), (484, 870, 8)],
    bg_pts=[(464, 764, 8), (566, 764, 10), (602, 764, 6), (464, 928, 10), (598, 928, 10), (578, 875, 10)],
    bg_polys=[[(602, 756), (608, 756), (608, 938), (602, 938)]],
    exclude_polys=[[(602, 756), (608, 756), (608, 938), (602, 938)]],
    use_diff_bg_hint=True, grass_bg_hint=True, iter_count=8
)
register_sticker("char_tambourine_boy", "手鼓男孩", "Tambourine Boy", "卡通人物", 36, orig, a_tamb)

# 20. Dancing girl in yellow coat
a_dance = extract_diff_object(orig, diff_orig_bg, (600, 750, 710, 938), thresh=28, keep_largest=True,
                              exclude_polys=[[(600, 750), (606, 750), (606, 845), (600, 845)]])
register_sticker("char_dance_girl", "跳舞女孩", "Dancing Girl", "卡通人物", 36, orig, a_dance)

# 21. Telescope girl on right stump
a_tele = extract_grabcut_object(
    orig, (692, 692, 842, 874),
    fg_rects=[(698, 700, 836, 868)],
    fg_pts=[(788, 730, 18), (788, 768, 16), (786, 812, 16), (784, 848, 14), (732, 756, 10), (722, 816, 10), (748, 822, 6)],
    bg_pts=[(700, 700, 10), (738, 702, 12), (702, 725, 10), (700, 864, 8), (836, 700, 8), (836, 864, 8), (702, 800, 8)],
    bg_polys=[[(692, 692), (762, 692), (762, 734), (692, 734)]],
    exclude_polys=[[(692, 692), (760, 692), (742, 738), (692, 734)]],
    use_diff_bg_hint=True, grass_bg_hint=True, iter_count=8
)
register_sticker("char_telescope_girl", "望远镜女孩", "Telescope Girl", "卡通人物", 35, orig, a_tele)

# ==============================================================================
# LAYER 4: 6 SMALL ANIMALS
# ==============================================================================
a_bear = extract_grabcut_object(
    orig, (54, 662, 114, 752),
    fg_rects=[(60, 670, 108, 745)],
    fg_pts=[(84, 692, 12), (84, 725, 10)],
    bg_pts=[(56, 668, 4), (108, 668, 4), (56, 708, 4), (56, 745, 4), (108, 745, 4)],
    exclude_polys=[[(54, 690), (62, 690), (62, 728), (54, 728)]]
)
register_sticker("animal_bear", "呆萌小熊", "Little Bear", "可爱动物", 32, orig, a_bear)

a_bunny = extract_grabcut_object(
    orig, (98, 616, 164, 722),
    fg_rects=[(106, 624, 158, 714)],
    fg_pts=[(118, 642, 6), (138, 638, 6), (134, 672, 12), (136, 700, 10)],
    bg_pts=[(102, 622, 4), (158, 622, 4), (102, 716, 4), (158, 716, 4)]
)
register_sticker("animal_bunny", "小白兔", "Fluffy Bunny", "可爱动物", 32, orig, a_bunny)

a_fox = extract_diff_object(orig, diff_orig_bg, (54, 766, 160, 868), thresh=28, keep_largest=True)
register_sticker("animal_fox", "小狐狸", "Little Fox", "可爱动物", 34, orig, a_fox)

a_hedge = extract_diff_object(orig, diff_orig_bg, (500, 930, 598, 1018), thresh=28, keep_largest=True)
register_sticker("animal_hedgehog", "小刺猬", "Hedgehog", "可爱动物", 46, orig, a_hedge)

# 26. Squirrel (full body cleanly extracted via diff + tiny pink blanket corner exclusion!)
a_squir = extract_diff_object(
    orig, diff_orig_bg, (760, 615, 845, 704), thresh=30, keep_largest=True,
    exclude_polys=[[(760, 615), (768, 615), (768, 704), (760, 704)],
                   [(760, 686), (784, 686), (784, 704), (760, 704)]]
)
register_sticker("animal_squirrel", "小松鼠", "Squirrel", "可爱动物", 32, orig, a_squir)

# 27. Owl (clean diff extraction, zero pine leaves!)
a_owl = extract_diff_object(orig, diff_orig_bg, (674, 294, 756, 376), thresh=34, keep_largest=True)
register_sticker("animal_owl", "树梢猫头鹰", "Tree Owl", "可爱动物", 40, orig, a_owl)

# ==============================================================================
# LAYER 5: NIGHT SKY & ATMOSPHERE
# ==============================================================================
# 28. Smiling crescent moon (outline-enclosed so ZERO square navy glow box!)
a_moon = extract_outline_enclosed(
    orig, (226, 146, 330, 254),
    bg_seeds=[(230, 150), (325, 150), (230, 250), (325, 250), (308, 185)],
    close_outline=3, strong_outline=True
)
register_sticker("sky_moon", "微笑弯月", "Crescent Moon", "星空光效", 50, orig, a_moon)

# 29. Night sky stars cluster
a_stars = extract_diff_object(
    orig, diff_orig_bg, (150, 105, 775, 345), thresh=45, soft_glow=True,
    exclude_polys=[
        [(222, 142), (334, 142), (334, 258), (222, 258)],
        [(670, 290), (760, 290), (760, 345), (670, 345)],
        [(258, 285), (305, 285), (305, 330), (258, 330)],
        [(618, 205), (665, 205), (665, 250), (618, 250)],
        [(350, 220), (580, 220), (580, 345), (350, 345)],
    ]
)
register_sticker("sky_stars", "夜空星辰", "Twinkling Stars", "星空光效", 49, orig, a_stars)

# 30. Glowing fireflies
firefly_spots = [
    (280, 305, 22), (640, 225, 22), (150, 468, 22), (80, 535, 22),
    (748, 464, 22), (814, 460, 22), (466, 558, 18), (268, 782, 18),
    (282, 862, 18), (460, 908, 18), (626, 738, 18), (734, 882, 20),
]
a_fireflies = np.zeros((H, W), dtype=np.uint8)
for (fx, fy, fr) in firefly_spots:
    y0, y1 = max(0, fy - fr), min(H, fy + fr)
    x0, x1 = max(0, fx - fr), min(W, fx + fr)
    sub_d = diff_orig_bg[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - fx)**2 + (yy - fy)**2)
    radial = np.clip(1.0 - np.maximum(0.0, dist - fr * 0.55) / (fr * 0.45), 0.0, 1.0)
    alpha_sub = np.clip((sub_d - 12.0) / 42.0, 0.0, 1.0) * radial
    a_fireflies[y0:y1, x0:x1] = np.maximum(a_fireflies[y0:y1, x0:x1], (alpha_sub * 255).astype(np.uint8))

register_sticker("fireflies", "森林萤火虫", "Forest Fireflies", "星空光效", 52, orig, a_fireflies)

# Save clean background PNG
cv2.imwrite(os.path.join(OUT_DIR, "background.png"), bg)

# Save level JSON metadata
level_data = {
    "levelId": "campfire_party",
    "title_zh": "森林篝火晚会",
    "title_en": "Forest Campfire Party",
    "canvasWidth": W,
    "canvasHeight": H,
    "backgroundFile": "background.png",
    "stickers": sorted(stickers_meta, key=lambda s: (s["zIndex"], s["id"]))
}
with open(os.path.join(OUT_DIR, "level_campfire.json"), "w", encoding="utf-8") as f:
    json.dump(level_data, f, ensure_ascii=False, indent=2)

# Generate contact sheet
cols, rows = 5, 6
cell_w, cell_h = 220, 230
sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
for r in range(0, rows * cell_h, 16):
    for c in range(0, cols * cell_w, 16):
        col = (68, 56, 50) if ((r // 16 + c // 16) % 2 == 0) else (84, 70, 62)
        sheet[r:r+16, c:c+16] = col

for idx, s in enumerate(stickers_meta):
    r, c = divmod(idx, cols)
    rgba = cv2.imread(os.path.join(OUT_DIR, s["file"]), cv2.IMREAD_UNCHANGED)
    sh, sw = rgba.shape[:2]
    max_w, max_h = cell_w - 24, cell_h - 46
    scale = min(max_w / sw, max_h / sh, 1.4)
    nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
    res = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)
    cx = c * cell_w + (cell_w - nw) // 2
    cy = r * cell_h + 12 + (max_h - nh) // 2
    alpha = res[:, :, 3:4].astype(np.float32) / 255.0
    roi = sheet[cy:cy+nh, cx:cx+nw].astype(np.float32)
    blended = res[:, :, :3].astype(np.float32) * alpha + roi * (1.0 - alpha)
    sheet[cy:cy+nh, cx:cx+nw] = blended.astype(np.uint8)
    cv2.rectangle(sheet, (c*cell_w, r*cell_h), ((c+1)*cell_w-1, (r+1)*cell_h-1), (110, 95, 85), 1)
    cv2.putText(sheet, f"{idx+1}. {s['id']}", (c*cell_w + 8, (r+1)*cell_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 240, 255), 1, cv2.LINE_AA)

cv2.imwrite("/home/antigravity/DioramaStickerApp/assets_pipeline/sticker_sheet_preview.png", sheet)

# Composite verification
comp = bg.copy().astype(np.float32)
for s in level_data["stickers"]:
    rgba = cv2.imread(os.path.join(OUT_DIR, s["file"]), cv2.IMREAD_UNCHANGED)
    x, y, w, h = s["x"], s["y"], s["width"], s["height"]
    if rgba.shape[1] != w or rgba.shape[0] != h:
        rgba = cv2.resize(rgba, (w, h), interpolation=cv2.INTER_AREA)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    comp[y:y+h, x:x+w] = rgba[:, :, :3].astype(np.float32) * alpha + comp[y:y+h, x:x+w] * (1.0 - alpha)

comp_u8 = np.clip(comp, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(OUT_DIR, "reference_complete.png"), comp_u8)

side = np.hstack([
    cv2.resize(bg, (448, 600)),
    cv2.resize(comp_u8, (448, 600)),
    cv2.resize(orig, (448, 600)),
])
cv2.imwrite("/home/antigravity/DioramaStickerApp/assets_pipeline/composite_verify.png", side)
print(f"\nDONE! Re-extracted {len(stickers_meta)} clean stickers.")
