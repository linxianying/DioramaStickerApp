import json
import os
import shutil
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image

from synth_3_empty_bgs import (
    build_hotpot_empty_bg,
    build_desert_empty_bg,
    build_taiwan_empty_bg,
    fit_poly2_surface,
)
from build_4_new_levels import LevelBuilder, get_outline_mask, smooth_alpha_from_binary, save_rgba_from_mask

BRAIN = "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p_devbox/ff1be343-f26a-4dda-8b06-9a95d3c4f1c1"
PIPE = "/home/antigravity/DioramaStickerApp/assets_pipeline"
IOS_LEVELS = "/home/antigravity/DioramaStickerApp/iOS/DioramaSticker/DioramaSticker/Resources/Levels"
DOCS_DIR = "/home/antigravity/DioramaStickerApp/docs"


def build_vertical_gradient(img, sample_mask, y_min, y_max):
    """Build a rock-solid 1D vertical color gradient by interpolating per-row medians (never overshoots in x!)."""
    H, W = img.shape[:2]
    ys_valid, cols_valid = [], []
    for y in range(y_min, y_max):
        xs = np.where(sample_mask[y] > 0)[0]
        if len(xs) >= 8:
            ys_valid.append(y)
            cols_valid.append(np.median(img[y, xs], axis=0))
    ys_valid = np.array(ys_valid, dtype=np.float32)
    cols_valid = np.array(cols_valid, dtype=np.float32)
    grad = np.zeros((H, W, 3), dtype=np.uint8)
    all_y = np.arange(H, dtype=np.float32)
    for c in range(3):
        vals = np.interp(all_y, ys_valid, cols_valid[:, c])
        grad[:, :, c] = np.clip(vals[:, None], 0, 255).astype(np.uint8)
    return grad


# ==============================================================================
# 1. WINTER CABIN — EXTRACT ALL 4 BACKGROUND SNOW-PINE TREES WITH 1D NIGHT SKY!
# ==============================================================================
def upgrade_winter_cabin():
    ios_dir = os.path.join(IOS_LEVELS, "winter_cabin")
    bg_orig = cv2.imread(os.path.join(BRAIN, "winter_cabin_empty_bg_1795496259560.jpg"))
    H, W = bg_orig.shape[:2]
    b, g, r = bg_orig[:, :, 0].astype(np.int16), bg_orig[:, :, 1].astype(np.int16), bg_orig[:, :, 2].astype(np.int16)

    navy_outer = tuple(int(v) for v in bg_orig[50, 50])
    ink = (36, 46, 68)

    # 1D vertical starry night sky gradient sampled from pure blue night sky (no stars, no trees)
    sky_sample = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(sky_sample, (80, 320), (220, 460), 1, -1)
    cv2.rectangle(sky_sample, (730, 320), (816, 460), 1, -1)
    sky_sample = sky_sample & ((b > r + 22) & (r < 125) & (g < 145)).astype(np.uint8)
    sky_grad = build_vertical_gradient(bg_orig, sky_sample, 280, 520)

    # Smooth flat snow color sampled from clean snow floor
    snow_sample = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(snow_sample, (180, 740), (340, 840), 1, -1)
    cv2.rectangle(snow_sample, (580, 740), (740, 840), 1, -1)
    snow_sample = snow_sample & ((r > 220) & (g > 225) & (b > 230)).astype(np.uint8)
    snow_grad = build_vertical_gradient(bg_orig, snow_sample, 600, 860)

    # First construct the 100% CLEAN BARE CABIN BACKGROUND (no 4 pine trees!) directly:
    bare = bg_orig.copy()
    # Left tree clearing zone: x=18..214 (above snow) & x=18..252 (below snow horizon y=600..776)
    for y in range(378, 778):
        for x in range(18, 250):
            # Preserve left cabin log wall (x >= 214 and above snow floor)
            if x >= 214 and y < 600:
                continue
            # Preserve the clean bottom-left white/snow curved corner below y=745 if inside border
            if x < 54:
                if y < 735:
                    bare[y, x] = navy_outer
                continue
            elif x < 70:
                if y < 735:
                    bare[y, x] = (255, 255, 255)
                continue
            elif x < 73:
                if y < 742:
                    bare[y, x] = ink
                continue
            else:
                hy = 662.0 - (x - 72.0) * (64.0 / 144.0)
                if y < hy:
                    bare[y, x] = sky_grad[y, x]
                elif y < 774:
                    bare[y, x] = snow_grad[y, x]

    # Right tree clearing zone: x=722..878
    for y in range(412, 778):
        for x in range(718, 878):
            if x <= 722 and y < 604:
                continue
            if x > 842:
                if y < 735:
                    bare[y, x] = navy_outer
                continue
            elif x > 826:
                if y < 735:
                    bare[y, x] = (255, 255, 255)
                continue
            elif x > 823:
                if y < 742:
                    bare[y, x] = ink
                continue
            else:
                hy = 602.0 + (x - 722.0) * (60.0 / 102.0)
                if y < hy:
                    bare[y, x] = sky_grad[y, x]
                elif y < 774:
                    bare[y, x] = snow_grad[y, x]

    # Restore exact rounded left/right snow corner rim (x=24..148, y=734..780 & x=748..872, y=734..780) cleanly
    # Actually keep original snow floor rim below y=768 untouched; fill only tree trunks at y=734..768 with snow_grad!
    for sx, sy in [(112, 445), (156, 415), (132, 520), (182, 490), (105, 580),
                   (762, 455), (795, 505), (754, 540), (788, 575)]:
        cv2.circle(bare, (sx, sy), 2, (235, 245, 255), -1, cv2.LINE_AA)
    cv2.line(bare, (72, 662), (214, 598), ink, 2, cv2.LINE_AA)
    cv2.line(bare, (722, 602), (824, 662), ink, 2, cv2.LINE_AA)

    # NOW extract ALL 4 snow-capped pine trees cleanly by diffing bg_orig vs bare!!
    diff_wc = np.linalg.norm(bg_orig.astype(np.float32) - bare.astype(np.float32), axis=2)
    out_wc = get_outline_mask(bg_orig)

    def extract_tree_diff(roi, exclude_polys=None, thresh=22):
        x0, y0, x1, y1 = roi
        sub = (diff_wc[y0:y1, x0:x1] > thresh).astype(np.uint8)
        if exclude_polys:
            for poly in exclude_polys:
                pts = np.array([[px - x0, py - y0] for px, py in poly], dtype=np.int32)
                cv2.fillPoly(sub, [pts], 0)
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        nl, lbl, st, _ = cv2.connectedComponentsWithStats(sub)
        cleaned = np.zeros_like(sub)
        if nl > 1:
            best = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            cleaned[lbl == best] = 1
        cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
        dil = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cleaned = np.where((cleaned > 0) | ((dil > 0) & (out_wc[y0:y1, x0:x1] > 0)), 1, 0).astype(np.uint8)
        cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
        if exclude_polys:
            for poly in exclude_polys:
                pts = np.array([[px - x0, py - y0] for px, py in poly], dtype=np.int32)
                cv2.fillPoly(cleaned, [pts], 0)
        alpha = np.zeros((H, W), dtype=np.uint8)
        alpha[y0:y1, x0:x1] = smooth_alpha_from_binary(cleaned, blur_sigma=0.55)
        return alpha

    a_l_small = extract_tree_diff((20, 560, 152, 774))
    a_l_big = extract_tree_diff((68, 382, 248, 704),
                                exclude_polys=[[(20, 564), (144, 564), (144, 774), (20, 774)],
                                               [(214, 382), (248, 382), (248, 594), (214, 594)]])
    a_r_small = extract_tree_diff((744, 560, 876, 774))
    a_r_big = extract_tree_diff((710, 418, 834, 698),
                                exclude_polys=[[(750, 564), (876, 564), (876, 774), (750, 774)],
                                               [(710, 418), (722, 418), (722, 604), (710, 604)]])

    json_path = os.path.join(ios_dir, "level_winter_cabin.json")
    with open(json_path, "r", encoding="utf-8") as f:
        ldata = json.load(f)
    ldata["stickers"] = [s for s in ldata["stickers"] if not s["id"].startswith("bg_pine_")]

    new_trees = [
        ("bg_pine_left_big", "左侧大雪松树", "Tall Snow Pine Left", "营地设施", 6, a_l_big),
        ("bg_pine_left_small", "左侧小雪松树", "Small Snow Pine Left", "营地设施", 8, a_l_small),
        ("bg_pine_right_big", "右侧大雪松树", "Tall Snow Pine Right", "营地设施", 6, a_r_big),
        ("bg_pine_right_small", "右侧小雪松树", "Small Snow Pine Right", "营地设施", 8, a_r_small),
    ]
    for sid, nzh, nen, cat, z, alpha in new_trees:
        fn = f"{sid}.png"
        x, y, w, h = save_rgba_from_mask(bg_orig, alpha, os.path.join(ios_dir, fn), pad=2)
        ldata["stickers"].append({
            "id": sid, "file": fn, "name_zh": nzh, "name_en": nen,
            "category": cat, "zIndex": z,
            "x": int(x), "y": int(y), "width": int(w), "height": int(h),
            "normX": round((x + w / 2.0) / W, 5),
            "normY": round((y + h / 2.0) / H, 5),
            "normW": round(w / W, 5),
            "normH": round(h / H, 5),
        })

    ldata["stickers"] = sorted(ldata["stickers"], key=lambda s: (s["zIndex"], s["id"]))
    cv2.imwrite(os.path.join(ios_dir, "background.png"), bare)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ldata, f, ensure_ascii=False, indent=2)

    comp = bare.copy().astype(np.float32)
    for s in ldata["stickers"]:
        rgba = cv2.imread(os.path.join(ios_dir, s["file"]), cv2.IMREAD_UNCHANGED)
        x, y, w, h = s["x"], s["y"], s["width"], s["height"]
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        comp[y:y+h, x:x+w] = rgba[:, :, :3].astype(np.float32) * alpha + comp[y:y+h, x:x+w] * (1.0 - alpha)
    comp_u8 = np.clip(comp, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(ios_dir, "reference_complete.png"), comp_u8)
    orig_wc = cv2.imread(os.path.join(BRAIN, "winter_cabin_scene_1795496213198.jpg"))
    side = np.hstack([cv2.resize(bare, (448, 600)), cv2.resize(comp_u8, (448, 600)), cv2.resize(orig_wc, (448, 600))])
    cv2.imwrite(os.path.join(PIPE, "verify_winter_cabin.png"), side)
    shutil.copy2(os.path.join(PIPE, "verify_winter_cabin.png"), os.path.join(BRAIN, "verify_winter_cabin.png"))
    print(f"  => Upgraded winter_cabin: {len(ldata['stickers'])} stickers (ALL 4 stickable pine trees!)")


# ==============================================================================
# 2. CHRISTMAS FOREST — FINE LAYERED SPLIT (29 STICKERS, SOLID SNOWMAN!)
# ==============================================================================
def upgrade_christmas_forest():
    cf = LevelBuilder(
        "christmas_forest", "冬季圣诞森林", "Winter Christmas Forest", "极光森林与驯鹿雪橇",
        os.path.join(BRAIN, "christmas_forest_scene_1795514837808.jpg"),
        os.path.join(BRAIN, "christmas_forest_empty_bg_1795515008300.jpg"),
    )
    cf.add("wooden_signpost", "森林雪地路牌", "Forest Signpost", "露营道具", 12,
           cf.extract_diff((480, 326, 556, 444), thresh=26, keep_largest=True))
    cf.add("santa_claus", "圣诞老人与礼物袋", "Santa Claus", "卡通人物", 16,
           cf.extract_diff((266, 358, 412, 534), thresh=26, keep_largest=True))
    cf.add("sledding_kids", "滑雪橇双人组", "Sledding Kids", "卡通人物", 18,
           cf.extract_diff((530, 404, 672, 582), thresh=26, keep_largest=True))
    cf.add("elf_standing", "绑丝带小精灵", "Ribbon Elf", "卡通人物", 20,
           cf.extract_diff((204, 448, 326, 596), thresh=26, keep_largest=True,
                           exclude_polys=[[(274, 448), (326, 448), (326, 520), (274, 520)]]))
    cf.add("elf_left", "左包礼物精灵", "Gift Wrapping Elf Left", "卡通人物", 22,
           cf.extract_diff((64, 514, 166, 678), thresh=26, keep_largest=True,
                           exclude_polys=[[(136, 514), (166, 514), (166, 594), (136, 594)],
                                          [(148, 596), (166, 596), (166, 678), (148, 678)]]))
    cf.add("elf_mid", "中包礼物精灵", "Gift Wrapping Elf Mid", "卡通人物", 21,
           cf.extract_diff((134, 490, 224, 610), thresh=26, keep_largest=True,
                           exclude_polys=[[(204, 490), (224, 490), (224, 575), (204, 575)]]))
    cf.add("robin_left", "知更小鸟", "Little Robin", "可爱动物", 22,
           cf.extract_diff((330, 566, 382, 622), thresh=26, keep_largest=True))
    cf.add("robin_right", "雪地知更鸟", "Snow Robin", "可爱动物", 22,
           cf.extract_diff((578, 578, 632, 632), thresh=26, keep_largest=True))
    cf.add("cocoa_cart", "热可可木推车", "Hot Chocolate Cart", "营地设施", 23,
           cf.extract_diff((672, 480, 836, 732), thresh=24, keep_largest=True,
                           exclude_polys=[[(750, 700), (836, 700), (836, 732), (750, 732)],
                                          [(672, 570), (712, 570), (712, 732), (672, 732)]]))
    cf.add("cocoa_customer", "买热可可男孩", "Hot Cocoa Boy", "卡通人物", 25,
           cf.extract_diff((644, 566, 738, 728), thresh=26, keep_largest=True,
                           exclude_polys=[[(714, 566), (738, 566), (738, 632), (714, 632)],
                                          [(712, 668), (738, 668), (738, 728), (712, 728)]]))
    cf.add("cocoa_board", "热可可菜单牌", "Cocoa Menu Board", "露营道具", 26,
           cf.extract_diff((618, 626, 668, 700), thresh=26, keep_largest=True))
    cf.add("christmas_tree_body", "星光大圣诞树", "Glowing Christmas Tree", "营地设施", 25,
           cf.extract_diff((360, 462, 536, 736), thresh=24, keep_largest=True,
                           exclude_polys=[[(494, 598), (536, 598), (536, 736), (494, 736)],
                                          [(360, 462), (404, 462), (404, 536), (360, 536)],
                                          [(360, 566), (382, 566), (382, 622), (360, 622)],
                                          [(360, 696), (516, 696), (516, 736), (360, 736)]]))
    cf.add("tree_gifts", "树下圣诞礼物堆", "Christmas Tree Gifts", "露营道具", 27,
           cf.extract_diff((360, 690, 514, 764), thresh=24, keep_largest=False, min_area=120))
    cf.add("girl_decorating", "挂星星女孩", "Star Decorating Girl", "卡通人物", 28,
           cf.extract_diff((490, 596, 588, 752), thresh=26, keep_largest=True))

    # 100% SOLID WHITE SNOWMAN (hat/head/scarf + full solid lower snowball!)
    snowball_mask = np.zeros((cf.H, cf.W), dtype=np.uint8)
    cv2.ellipse(snowball_mask, (251, 712), (42, 41), 0, 0, 360, 255, -1)
    snowball_mask = cv2.GaussianBlur(snowball_mask.astype(np.float32), (3, 3), 0.6).astype(np.uint8)
    a_snowman = np.maximum(
        np.maximum(
            cf.extract_outline_enclosed(
                (202, 586, 304, 760),
                bg_seeds=[(205, 589), (301, 589), (205, 756), (301, 756)],
                exclude_polys=[[(202, 610), (218, 610), (218, 748), (202, 748)],
                               [(284, 608), (304, 608), (304, 748), (284, 748)]],
                close_outline=5,
            ),
            cf.extract_diff((202, 586, 304, 760), thresh=24, keep_largest=False, min_area=80,
                            exclude_polys=[[(202, 610), (218, 610), (218, 748), (202, 748)],
                                           [(284, 608), (304, 608), (304, 748), (284, 748)]])
        ),
        snowball_mask
    )
    cf.add("snowman", "围巾大雪人", "Scarf Snowman", "营地设施", 27, a_snowman)
    cf.add("snowman_boy", "堆雪人男孩", "Snowman Boy", "卡通人物", 28,
           cf.extract_diff((152, 610, 224, 750), thresh=26, keep_largest=True))
    cf.add("snowman_girl", "堆雪人女孩", "Snowman Girl", "卡通人物", 28,
           cf.extract_diff((276, 602, 356, 750), thresh=26, keep_largest=True))

    cf.add("mini_tree_right", "小圣诞树", "Mini Pine Tree", "露营道具", 28,
           cf.extract_diff((578, 684, 638, 774), thresh=26, keep_largest=True))
    cf.add("animal_fox", "森林小红狐", "Red Forest Fox", "可爱动物", 30,
           cf.extract_diff((68, 676, 160, 780), thresh=26, keep_largest=True))
    cf.add("animal_squirrel", "抱松果小松鼠", "Pinecone Squirrel", "可爱动物", 32,
           cf.extract_diff((748, 704, 836, 792), thresh=26, keep_largest=True))
    cf.add("reindeer", "森林大驯鹿", "Christmas Reindeer", "可爱动物", 34,
           cf.extract_diff((494, 750, 624, 932), thresh=24, keep_largest=True))
    cf.add("gift_sleigh", "满载礼物雪橇", "Gift Wooden Sleigh", "营地设施", 33,
           cf.extract_diff((604, 726, 748, 886), thresh=24, keep_largest=True,
                           exclude_polys=[[(604, 822), (624, 822), (624, 886), (604, 886)]]))
    cf.add("mini_tree_center", "雪地迷你树", "Tiny Snow Tree", "露营道具", 32,
           cf.extract_diff((438, 772, 488, 852), thresh=26, keep_largest=True))
    cf.add("animal_bunny", "雪地小白兔", "Snow Bunny", "可爱动物", 34,
           cf.extract_diff((150, 746, 218, 834), thresh=24, keep_largest=True))
    cf.add("animal_fawn", "森林小梅花鹿", "Baby Forest Fawn", "可爱动物", 36,
           cf.extract_diff((218, 770, 294, 878), thresh=26, keep_largest=True))
    cf.add("carol_girl", "唱颂歌女孩", "Caroling Girl", "卡通人物", 36,
           cf.extract_diff((294, 750, 366, 912), thresh=24, keep_largest=False, min_area=35))
    cf.add("carol_boy", "唱颂歌男孩", "Caroling Boy", "卡通人物", 36,
           cf.extract_diff((360, 740, 434, 912), thresh=24, keep_largest=False, min_area=35))
    cf.add("owl_stump", "树桩雪鸮", "Snowy Owl on Stump", "可爱动物", 40,
           cf.extract_diff((392, 868, 522, 1004), thresh=24, keep_largest=True))
    cf.add("robin_bottom", "树桩小知更鸟", "Stump Robin", "可爱动物", 42,
           cf.extract_diff((364, 918, 412, 964), thresh=26, keep_largest=True))
    cf.finalize()


# ==============================================================================
# 3. DESERT CAMP — 1D SUNSET SKY/DUNE + 2 STICKABLE PALM TREES (27 STICKERS!)
# ==============================================================================
def upgrade_desert_camp():
    build_desert_empty_bg()
    bg = cv2.imread(os.path.join(PIPE, "empty_bg_desert_camp.png"))
    H, W = bg.shape[:2]
    out_bg = get_outline_mask(bg, r_max=120, g_max=100, b_max=88)
    b, g, r = bg[:, :, 0].astype(np.int16), bg[:, :, 1].astype(np.int16), bg[:, :, 2].astype(np.int16)

    def extract_palm(roi):
        x0, y0, x1, y1 = roi
        cb, cg, cr = b[y0:y1, x0:x1], g[y0:y1, x0:x1], r[y0:y1, x0:x1]
        is_green = ((cg > cb + 24) & (cg > cr - 18) & (cr < 185)).astype(np.uint8)
        is_trunk = ((cr > 120) & (cr < 210) & (cg > 75) & (cg < 158) & (cb > 40) & (cb < 105) & ((cr - cg) > 25)).astype(np.uint8)
        sub = ((is_green > 0) | (is_trunk > 0)).astype(np.uint8)
        sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        nl, lbl, st, _ = cv2.connectedComponentsWithStats(sub)
        cleaned = np.zeros_like(sub)
        for i in range(1, nl):
            if st[i, cv2.CC_STAT_AREA] >= 120:
                cleaned[lbl == i] = 1
        dil = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cleaned = np.where((cleaned > 0) | ((dil > 0) & (out_bg[y0:y1, x0:x1] > 0)), 1, 0).astype(np.uint8)
        cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
        alpha = np.zeros((H, W), dtype=np.uint8)
        alpha[y0:y1, x0:x1] = smooth_alpha_from_binary(cleaned, blur_sigma=0.55)
        return alpha

    a_palm_l = extract_palm((284, 336, 404, 502))
    a_palm_r = extract_palm((544, 322, 676, 514))

    # 1D vertical sunset sky + golden dune gradients (ZERO red overshoot!)
    sky_sample = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(sky_sample, (405, 300), (520, 396), 1, -1)
    sky_grad = build_vertical_gradient(bg, sky_sample, 290, 400)

    dune_sample = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(dune_sample, (400, 412), (525, 456), 1, -1)
    dune_sample = dune_sample & ((r > 210) & (g > 155) & (b < 145)).astype(np.uint8)
    dune_grad = build_vertical_gradient(bg, dune_sample, 400, 465)

    palm_zone = cv2.dilate(((a_palm_l > 16) | (a_palm_r > 16)).astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    navy_col = tuple(int(v) for v in bg[50, 50])
    bg_clean = bg.copy()
    for y in range(300, 516):
        for x in range(280, 682):
            # Always clean the upper-right palm frond outer border bump outside the straight diamond edge!
            d_in_r = ((y - 198.0) - (x - 448.0)) / np.sqrt(2.0)
            if x > 560 and y < 420 and d_in_r < -14.5:
                bg_clean[y, x] = navy_col
                continue
            if palm_zone[y, x] == 0:
                continue
            # Keep turquoise oasis water intact
            if 352 <= x <= 575 and y >= 474:
                continue
            if x > 540 and d_in_r < -1.5:
                bg_clean[y, x] = (255, 255, 255)
                continue
            elif x > 540 and d_in_r < 1.8:
                bg_clean[y, x] = (36, 46, 68)
                continue
            hy = 402 + int(5 * np.sin((x - 300) / 45.0))
            if y < hy:
                bg_clean[y, x] = sky_grad[y, x]
            else:
                bg_clean[y, x] = dune_grad[y, x]

    cv2.polylines(bg_clean, [np.array([[284, 404], [338, 398], [392, 408]], dtype=np.int32)], False, (76, 102, 146), 2, cv2.LINE_AA)
    cv2.polylines(bg_clean, [np.array([[552, 404], [608, 396], [656, 412]], dtype=np.int32)], False, (76, 102, 146), 2, cv2.LINE_AA)

    bg_no_palms_path = os.path.join(PIPE, "empty_bg_desert_camp_nopalm.png")
    cv2.imwrite(bg_no_palms_path, bg_clean)

    dc = LevelBuilder(
        "desert_camp", "沙漠帐篷探险", "Desert Oasis Expedition", "绿洲星夜与波西米亚帐篷",
        os.path.join(BRAIN, "desert_camp_scene_1795514833166.jpg"),
        bg_no_palms_path,
    )
    dc.add("bg_palm_left", "左侧绿洲棕榈树", "Oasis Palm Tree Left", "营地设施", 8, a_palm_l)
    dc.add("bg_palm_right", "右侧双株棕榈树", "Twin Oasis Palms Right", "营地设施", 8, a_palm_r)
    dc.add("bell_tent_rug", "波西米亚编织地毯", "Boho Woven Rug", "营地设施", 14,
           dc.extract_poly_diff([(234, 622), (344, 578), (402, 612), (298, 666)], thresh=26, min_area=80, fill_holes=True))
    dc.add("bell_tent_body", "波西米亚大帐篷", "Boho Bell Tent", "营地设施", 16,
           dc.extract_diff((146, 412, 390, 628), thresh=26, keep_largest=True,
                           exclude_polys=[[(146, 584), (214, 584), (214, 628), (146, 628)],
                                          [(146, 496), (192, 496), (192, 586), (146, 586)],
                                          [(374, 540), (390, 540), (390, 600), (374, 600)],
                                          [(234, 622), (344, 578), (402, 612), (298, 666)]]))
    dc.add("teal_explorer_tent", "青绿探险帐篷", "Teal Explorer Tent", "营地设施", 18,
           dc.extract_diff((542, 494, 804, 698), thresh=26, keep_largest=True,
                           exclude_polys=[[(542, 494), (618, 494), (618, 598), (542, 598)],
                                          [(562, 594), (654, 594), (654, 698), (562, 698)],
                                          [(694, 620), (746, 620), (746, 698), (694, 698)]]))
    dc.add("cactus_oasis", "绿洲盆栽仙人掌", "Oasis Potted Cactus", "露营道具", 18,
           dc.extract_diff((376, 498, 434, 594), thresh=26, keep_largest=True))
    dc.add("astronomer_boy", "观星望远镜少年", "Desert Astronomer", "卡通人物", 20,
           dc.extract_diff((428, 520, 540, 646), thresh=26, keep_largest=True,
                           exclude_polys=[[(428, 590), (470, 590), (470, 646), (428, 646)]]))
    dc.add("animal_fennec_top", "绿洲耳廓狐", "Oasis Fennec Fox", "可爱动物", 20,
           dc.extract_diff((534, 520, 616, 598), thresh=26, keep_largest=True))
    dc.add("animal_camel_left", "驼包骆驼", "Caravan Camel", "可爱动物", 22,
           dc.extract_diff((52, 492, 198, 650), thresh=26, keep_largest=True,
                           exclude_polys=[[(132, 586), (198, 586), (198, 650), (132, 650)]]))
    dc.add("archaeologist", "考古小女孩", "Relic Archaeologist", "卡通人物", 26,
           dc.extract_diff((130, 584, 206, 712), thresh=26, keep_largest=True))
    dc.add("ancient_tablet", "古代沙漠石碑", "Ancient Sandstone Relic", "露营道具", 25,
           dc.extract_diff((194, 642, 246, 704), thresh=26, keep_largest=True))
    dc.add("desert_kettle", "沙漠铁茶壶", "Campfire Kettle", "露营道具", 23,
           dc.extract_diff((374, 610, 424, 660), thresh=26, keep_largest=True))
    dc.add("desert_campfire", "暖夜沙漠篝火", "Desert Campfire", "营地设施", 24,
           dc.extract_diff((416, 590, 492, 680), thresh=26, keep_largest=True))
    dc.add("guitar_explorer", "弹吉他探险家", "Campfire Guitarist", "卡通人物", 26,
           dc.extract_diff((562, 600, 654, 716), thresh=26, keep_largest=True))
    dc.add("cactus_right", "帐篷仙人掌盆", "Tent Potted Cactus", "露营道具", 28,
           dc.extract_diff((694, 620, 746, 716), thresh=26, keep_largest=True))
    dc.add("explorer_boy", "探险小领队", "Navigator Explorer", "卡通人物", 29,
           dc.extract_diff((248, 652, 326, 774), thresh=26, keep_largest=True,
                           exclude_polys=[[(270, 728), (326, 728), (326, 774), (270, 774)]]))
    dc.add("map_crate", "藏宝图木箱", "Treasure Map Crate", "营地设施", 31,
           dc.extract_diff((266, 714, 354, 810), thresh=26, keep_largest=True))
    dc.add("animal_meerkat_left", "站哨小狐獴", "Lookout Meerkat", "可爱动物", 32,
           dc.extract_diff((200, 716, 250, 778), thresh=26, keep_largest=True))
    dc.add("marshmallow_girl", "烤棉花糖女孩", "Marshmallow Girl", "卡通人物", 30,
           dc.extract_diff((358, 660, 456, 776), thresh=26, keep_largest=True,
                           exclude_polys=[[(430, 706), (456, 706), (456, 776), (430, 776)]]))
    dc.add("marshmallow_fire", "小火盆编织坐垫", "Mini Fire Pouf", "营地设施", 29,
           dc.extract_diff((428, 700, 488, 782), thresh=26, keep_largest=True))
    dc.add("marshmallow_boy", "烤棉花糖男孩", "Marshmallow Boy", "卡通人物", 30,
           dc.extract_diff((460, 652, 558, 776), thresh=26, keep_largest=True,
                           exclude_polys=[[(460, 706), (486, 706), (486, 776), (460, 776)]]))
    dc.add("animal_meerkats_right", "狐獴兄弟", "Meerkat Pair", "可爱动物", 30,
           dc.extract_diff((642, 696, 704, 754), thresh=26, keep_largest=False, min_area=60))
    dc.add("animal_gecko", "沙漠小壁虎", "Desert Gecko", "可爱动物", 32,
           dc.extract_diff((610, 734, 668, 780), thresh=26, keep_largest=True))
    dc.add("animal_fennec_low", "坐姿耳廓狐", "Sitting Fennec Fox", "可爱动物", 34,
           dc.extract_diff((558, 760, 634, 836), thresh=26, keep_largest=True))
    dc.add("tea_boy", "泡茶探险家", "Mint Tea Camper", "卡通人物", 36,
           dc.extract_diff((356, 790, 444, 912), thresh=26, keep_largest=True,
                           exclude_polys=[[(416, 858), (444, 858), (444, 912), (416, 912)]]))
    dc.add("tea_barrel", "薄荷茶木桶", "Mint Tea Barrel", "露营道具", 35,
           dc.extract_diff((412, 854, 456, 912), thresh=26, keep_largest=True))
    dc.add("animal_camel_low", "暖阳小骆驼", "Friendly Camel", "可爱动物", 36,
           dc.extract_diff((426, 780, 560, 914), thresh=26, keep_largest=True,
                           exclude_polys=[[(426, 840), (456, 840), (456, 914), (426, 914)]]))
    dc.finalize()


# ==============================================================================
# 4. HOTPOT FEAST — RAZOR-SHARP 22 STICKERS (CLEAN LADLES & STEAM!)
# ==============================================================================
def upgrade_hotpot_feast():
    build_hotpot_empty_bg()
    hp = LevelBuilder(
        "hotpot_feast", "火锅食材盛宴", "Overhead Hotpot Feast", "天花板俯视角鸳鸯火锅",
        os.path.join(BRAIN, "hotpot_feast_scene_1795514917338.jpg"),
        os.path.join(PIPE, "empty_bg_hotpot_feast.png"),
    )
    hp.add("yin_yang_hotpot", "鸳鸯麻辣骨汤锅", "Yin-Yang Hotpot", "营地设施", 20,
           hp.extract_diff((244, 412, 652, 770), thresh=34, keep_largest=True))
    hp.add("steam_left", "热气小蒸气(左)", "Hotpot Steam Left", "星空光效", 25,
           hp.extract_diff((260, 392, 386, 496), thresh=32, keep_largest=False, min_area=120,
                           exclude_polys=[[(305, 422), (386, 422), (386, 496), (305, 496)],
                                          [(260, 392), (350, 392), (350, 444), (260, 444)]]))
    hp.add("steam_right", "热气小蒸气(右)", "Hotpot Steam Right", "星空光效", 25,
           hp.extract_diff((550, 410, 636, 524), thresh=32, keep_largest=False, min_area=120,
                           exclude_polys=[[(550, 466), (605, 466), (605, 524), (550, 524)]]))

    hp.add("beef_rolls_boat", "木船雪花肥牛卷", "Marbled Beef Boat", "露营道具", 22,
           hp.extract_diff((88, 276, 368, 488), thresh=34, keep_largest=True,
                           exclude_polys=[[(88, 276), (190, 276), (190, 345), (88, 345)]]))
    hp.add("shrimp_platter", "鲜虾与竹筒虾滑", "Prawns & Shrimp Paste", "露营道具", 22,
           hp.extract_diff((352, 224, 536, 404), thresh=34, keep_largest=True))
    hp.add("lamb_rolls", "鲜切草原肥羊卷", "Sliced Lamb Rolls", "露营道具", 22,
           hp.extract_diff((528, 260, 676, 408), thresh=34, keep_largest=True))
    hp.add("veggie_basket", "时蔬玉米竹篮", "Fresh Veggies & Corn", "露营道具", 22,
           hp.extract_diff((632, 340, 818, 514), thresh=34, keep_largest=True))
    hp.add("mushroom_platter", "鲜嫩菌菇拼盘", "Assorted Mushrooms", "露营道具", 22,
           hp.extract_diff((658, 514, 836, 676), thresh=34, keep_largest=True))
    hp.add("lotus_tofu_platter", "莲藕土豆冻豆腐", "Lotus Root & Tofu", "露营道具", 22,
           hp.extract_diff((622, 668, 790, 826), thresh=34, keep_largest=True))
    hp.add("meatball_bowl", "鱼丸蟹棒鹌鹑蛋", "Fish Balls & Quail Eggs", "露营道具", 22,
           hp.extract_diff((546, 796, 698, 944), thresh=34, keep_largest=True))
    hp.add("noodle_bowl", "乌冬与水晶宽粉", "Udon & Glass Noodles", "露营道具", 22,
           hp.extract_diff((210, 798, 356, 944), thresh=34, keep_largest=True))
    hp.add("tofu_skin_rolls", "黄金炸响铃卷", "Crispy Tofu Skin Rolls", "露营道具", 22,
           hp.extract_diff((110, 662, 272, 824), thresh=34, keep_largest=True))

    # Clean diagonal split between soup_ladle and skimmer_spoon!
    hp.add("soup_ladle", "带碟汤勺", "Soup Ladle & Rest", "露营道具", 22,
           hp.extract_poly_diff([(56, 480), (250, 480), (250, 594), (112, 616), (56, 620)], thresh=34, min_area=180, fill_holes=True))
    hp.add("skimmer_spoon", "不锈钢漏勺", "Slotted Skimmer", "露营道具", 22,
           hp.extract_poly_diff([(62, 622), (120, 618), (250, 596), (250, 664), (62, 664)], thresh=34, min_area=180, fill_holes=True))

    hp.add("sauce_sesame", "麻酱香油蘸料碟", "Sesame Oil Dip", "露营道具", 24,
           hp.extract_diff((332, 766, 408, 844), thresh=34, keep_largest=True))
    hp.add("sauce_scallion", "葱花蒜泥蘸料碟", "Scallion Garlic Dip", "露营道具", 24,
           hp.extract_diff((412, 770, 488, 848), thresh=34, keep_largest=True))
    hp.add("sauce_chili", "秘制红油辣椒碟", "Spicy Chili Dip", "露营道具", 24,
           hp.extract_diff((488, 766, 564, 844), thresh=34, keep_largest=True))
    hp.add("chopsticks_cat", "三花猫筷架竹筷", "Calico Cat Chopsticks", "可爱动物", 26,
           hp.extract_diff((356, 850, 544, 916), thresh=34, keep_largest=True))
    hp.add("chopsticks_panda", "小熊猫筷架竹筷", "Panda Chopsticks", "可爱动物", 26,
           hp.extract_diff((356, 912, 544, 976), thresh=34, keep_largest=True))
    hp.add("boba_milk_tea", "黑糖珍珠奶茶", "Brown Sugar Boba", "露营道具", 26,
           hp.extract_diff((102, 214, 184, 340), thresh=34, keep_largest=True))
    hp.add("iced_plum_tea", "冰镇酸梅汤", "Iced Plum Tea", "露营道具", 26,
           hp.extract_diff((716, 218, 796, 340), thresh=34, keep_largest=True))
    hp.add("table_calico_cat", "桌角馋嘴三花猫", "Hungry Calico Cat", "可爱动物", 28,
           hp.extract_diff((726, 892, 810, 978), thresh=34, keep_largest=True))
    hp.finalize()


# ==============================================================================
# 5. TAIWAN NIGHT MARKET — ULTRA-FINE 29 STICKERS (CLEAN DINERS & KID!)
# ==============================================================================
def upgrade_taiwan_night_market():
    build_taiwan_empty_bg()
    tw = LevelBuilder(
        "taiwan_night_market", "台湾夜市街", "Taiwan Night Market", "珍珠奶茶、盐酥鸡与套圈圈",
        os.path.join(BRAIN, "taiwan_night_market_scene_1795514903307.jpg"),
        os.path.join(PIPE, "empty_bg_taiwan_night_market.png"),
    )
    tw.add("sign_boba", "珍珠奶茶招牌", "Bubble Tea Sign", "营地设施", 14,
           tw.extract_poly_mask([(176, 493), (287, 432), (287, 479), (176, 540)]))
    tw.add("sign_chicken", "盐酥鸡招牌", "Fried Chicken Sign", "营地设施", 14,
           tw.extract_poly_mask([(334, 398), (446, 335), (446, 383), (334, 446)]))
    tw.add("sign_oyster", "蚵仔煎芒果冰招牌", "Oyster & Mango Sign", "营地设施", 14,
           tw.extract_poly_mask([(491, 367), (605, 431), (606, 479), (494, 415)]))
    tw.add("sign_ringtoss", "套圈圈招牌", "Ring-Toss Sign", "营地设施", 14,
           tw.extract_poly_mask([(643, 456), (746, 514), (746, 560), (643, 501)]))

    tw.add("vendor_boba", "奶茶老板娘", "Bubble Tea Vendor", "卡通人物", 16,
           tw.extract_poly_diff([(130, 560), (218, 512), (218, 632), (130, 646)],
                                thresh=24, min_area=80, fill_holes=True,
                                exclude_polys=[[(170, 594), (218, 594), (218, 650), (170, 650)]]))
    tw.add("boba_cups_set", "台面珍珠奶茶杯组", "Counter Boba Cups", "露营道具", 19,
           tw.extract_poly_diff([(170, 594), (288, 524), (294, 616), (174, 672)],
                                thresh=24, min_area=80, fill_holes=True))
    tw.add("vendor_chicken", "盐酥鸡老板", "Fried Chicken Vendor", "卡通人物", 16,
           tw.extract_poly_diff([(308, 456), (398, 412), (398, 510), (308, 530)],
                                thresh=24, min_area=80, fill_holes=True))
    tw.add("chicken_grill_set", "炸鸡篮与烤串架", "Fried Chicken & Skewers", "露营道具", 19,
           tw.extract_poly_diff([(304, 510), (402, 466), (444, 392), (456, 512), (344, 578), (304, 548)],
                                thresh=24, min_area=80, fill_holes=True))
    tw.add("vendor_oyster", "蚵仔煎老板", "Oyster Omelet Chef", "卡通人物", 16,
           tw.extract_poly_diff([(534, 444), (620, 492), (620, 544), (534, 514)],
                                thresh=24, min_area=80, fill_holes=True))
    tw.add("oyster_griddle", "铁板蚵仔煎锅", "Sizzling Oyster Griddle", "露营道具", 19,
           tw.extract_poly_diff([(476, 494), (576, 502), (566, 574), (476, 544)],
                                thresh=24, min_area=80, fill_holes=True))
    tw.add("mango_ice_bowl", "大碗绵绵芒果冰", "Giant Mango Shaved Ice", "露营道具", 20,
           tw.extract_poly_diff([(548, 534), (642, 544), (642, 600), (576, 610), (548, 586)],
                                thresh=24, min_area=80, fill_holes=True))
    tw.add("ringtoss_plushies", "套圈圈玩偶奖品", "Ring-Toss Plush Prizes", "露营道具", 17,
           tw.extract_poly_diff([(660, 520), (796, 544), (796, 612), (660, 578)],
                                thresh=24, min_area=60, fill_holes=True))
    tw.add("ringtoss_bottles", "套圈圈彩瓶阵", "Ring-Toss Glass Bottles", "露营道具", 19,
           tw.extract_poly_diff([(632, 576), (788, 608), (796, 668), (748, 704), (632, 634)],
                                thresh=24, min_area=60, fill_holes=True))

    tw.add("tapioca_jar", "大罐黑糖珍珠", "Giant Tapioca Jar", "露营道具", 24,
           tw.extract_diff((94, 670, 154, 760), thresh=34, keep_largest=True))
    tw.add("animal_tuxedo_cat", "夜市奶牛猫", "Tuxedo Alley Cat", "可爱动物", 28,
           tw.extract_diff((146, 714, 224, 796), thresh=36, keep_largest=True))
    tw.add("boba_boy", "买奶茶男孩", "Bubble Tea Boy", "卡通人物", 27,
           tw.extract_diff((218, 618, 286, 756), thresh=36, keep_largest=True))
    tw.add("boba_girl", "喝奶茶女孩", "Bubble Tea Girl", "卡通人物", 25,
           tw.extract_diff((276, 590, 344, 728), thresh=36, keep_largest=True,
                           exclude_polys=[[(326, 600), (344, 600), (344, 700), (326, 700)],
                                          [(276, 622), (288, 622), (288, 728), (276, 728)]]))
    tw.add("animal_shiba", "盯烤串小柴犬", "Skewer Shiba Inu", "可爱动物", 24,
           tw.extract_diff((328, 606, 398, 698), thresh=36, keep_largest=True))
    tw.add("snack_boy", "吸珍珠奶茶男孩", "Boba Sipping Boy", "卡通人物", 28,
           tw.extract_diff((392, 608, 468, 752), thresh=36, keep_largest=True))
    tw.add("snack_girl", "捧盐酥鸡女孩", "Fried Chicken Girl", "卡通人物", 28,
           tw.extract_diff((462, 614, 534, 748), thresh=36, keep_largest=True))
    tw.add("kid_ringtoss", "套圈圈小朋友", "Ring-Toss Kid", "卡通人物", 28,
           tw.extract_diff((580, 630, 696, 762), thresh=32, keep_largest=False, min_area=80,
                           exclude_polys=[[(668, 630), (696, 630), (696, 668), (668, 668)]]))
    tw.add("wood_crate", "夜市木箱", "Wooden Market Crate", "营地设施", 24,
           tw.extract_diff((742, 722, 838, 800), thresh=32, keep_largest=True,
                           exclude_polys=[[(746, 690), (824, 690), (824, 738), (746, 738)]]))
    tw.add("calico_cat_nap", "箱上睡觉三花猫", "Napping Calico Cat", "可爱动物", 27,
           tw.extract_diff((746, 690, 826, 742), thresh=30, keep_largest=True))
    tw.add("mint_scooter", "薄荷复古小绵羊", "Mint Retro Scooter", "营地设施", 34,
           tw.extract_diff((204, 748, 364, 916), thresh=36, keep_largest=True))
    tw.add("girl_skewers", "撸烤串女孩", "Skewer Girl", "卡通人物", 34,
           tw.extract_diff((342, 746, 434, 888), thresh=36, keep_largest=True,
                           exclude_polys=[[(342, 820), (366, 820), (366, 888), (342, 888)]]))

    # Clean 3-layer split: mango_diner_right (whole boy, z=36) -> mango_table_dish (table+bowl, z=37) -> mango_diner_left (front diner, z=39)!
    tw.add("mango_diner_right", "吃芒果冰男孩", "Mango Ice Boy", "卡通人物", 36,
           tw.extract_diff((504, 756, 586, 922), thresh=36, keep_largest=True,
                           exclude_polys=[[(504, 864), (542, 864), (542, 922), (504, 922)]]))
    tw.add("mango_table_dish", "红木夜市桌与芒果冰", "Red Street Table & Mango Ice", "营地设施", 37,
           tw.extract_diff((460, 822, 562, 962), thresh=36, keep_largest=True,
                           exclude_polys=[[(460, 822), (488, 822), (488, 962), (460, 962)],
                                          [(528, 822), (562, 822), (562, 864), (528, 864)]]))
    tw.add("mango_diner_left", "红凳吃冰食客", "Red Stool Diner", "卡通人物", 39,
           tw.extract_diff((412, 818, 496, 982), thresh=36, keep_largest=True))
    tw.add("girl_camera", "夜市打卡拍照女孩", "Night Market Photographer", "卡通人物", 36,
           tw.extract_diff((588, 740, 676, 886), thresh=36, keep_largest=True))
    tw.finalize()


def sync_all_12_to_docs():
    with open(os.path.join(IOS_LEVELS, "levels_manifest.json"), "r", encoding="utf-8") as f:
        ios_m = json.load(f)

    upgraded_ids = {"winter_cabin", "christmas_forest", "desert_camp", "hotpot_feast", "taiwan_night_market"}
    docs_levels = []

    for item in ios_m["levels"]:
        lid = item["id"]
        src_dir = os.path.join(IOS_LEVELS, item["folder"])
        with open(os.path.join(src_dir, item["jsonFile"] + ".json"), "r", encoding="utf-8") as f:
            jdata = json.load(f)
        cnt = len(jdata["stickers"])

        if lid in upgraded_ids:
            dst_dir = os.path.join(DOCS_DIR, "levels", lid)
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            os.makedirs(dst_dir, exist_ok=True)
            Image.open(os.path.join(src_dir, "background.png")).convert("RGB").save(
                os.path.join(dst_dir, "background.webp"), "WEBP", quality=86, method=5
            )
            ref_img = Image.open(os.path.join(src_dir, "reference_complete.png")).convert("RGB")
            ref_img.resize((360, 482), Image.LANCZOS).save(
                os.path.join(dst_dir, "thumb.webp"), "WEBP", quality=84, method=5
            )
            web_stickers = []
            for s in jdata["stickers"]:
                webp_name = s["file"].replace(".png", ".webp")
                Image.open(os.path.join(src_dir, s["file"])).convert("RGBA").save(
                    os.path.join(dst_dir, webp_name), "WEBP", quality=88, method=5
                )
                ws = dict(s)
                ws["file"] = webp_name
                web_stickers.append(ws)
            web_level = dict(jdata)
            web_level["backgroundFile"] = "background.webp"
            web_level["stickers"] = web_stickers
            with open(os.path.join(dst_dir, "level.json"), "w", encoding="utf-8") as f:
                json.dump(web_level, f, ensure_ascii=False, separators=(",", ":"))

        docs_levels.append({
            "id": lid,
            "title_zh": item["title_zh"],
            "title_en": item["title_en"],
            "subtitle_zh": item["subtitle_zh"],
            "emoji": item["emoji"],
            "accentHex": item["accentHex"],
            "stickerCount": cnt,
            "thumb": f"levels/{lid}/thumb.webp",
            "dataUrl": f"levels/{lid}/level.json",
        })

    with open(os.path.join(DOCS_DIR, "levels_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"levels": docs_levels}, f, ensure_ascii=False, indent=2)

    tw, th = 360, 482
    grid = np.full((th * 3, tw * 4, 3), (58, 46, 38), dtype=np.uint8)
    for idx, item in enumerate(docs_levels):
        lid = item["id"]
        bg_p = os.path.join(IOS_LEVELS, lid, "background.png")
        img = cv2.resize(cv2.imread(bg_p), (tw, th))
        r, c = divmod(idx, 4)
        grid[r*th:(r+1)*th, c*tw:(c+1)*tw] = img
        cv2.putText(grid, f"{idx+1}. {lid} ({item['stickerCount']})", (c*tw + 10, r*th + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (90, 240, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(PIPE, "audit_all_12_bgs.png"), grid)
    shutil.copy2(os.path.join(PIPE, "audit_all_12_bgs.png"), os.path.join(BRAIN, "audit_all_12_bgs.png"))

    grand_total = sum(x["stickerCount"] for x in docs_levels)
    print(f"\nALL 12 LEVELS UPGRADED & SYNCED! GRAND TOTAL: {grand_total} STICKERS!")
    for x in docs_levels:
        print(f"  - {x['emoji']} {x['title_zh']} ({x['id']}): {x['stickerCount']} stickers")


if __name__ == "__main__":
    upgrade_winter_cabin()
    upgrade_christmas_forest()
    upgrade_desert_camp()
    upgrade_hotpot_feast()
    upgrade_taiwan_night_market()
    sync_all_12_to_docs()
