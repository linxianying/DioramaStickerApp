import json
import os
import shutil
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image

from build_4_new_levels import LevelBuilder, get_outline_mask, smooth_alpha_from_binary, save_rgba_from_mask
from upgrade_classic_3_levels import clip_alpha_to_poly, exclude_poly_from_alpha, render_contact_sheet

BRAIN = "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p_devbox/ff1be343-f26a-4dda-8b06-9a95d3c4f1c1"
PIPE = "/home/antigravity/DioramaStickerApp/assets_pipeline"
IOS_LEVELS = "/home/antigravity/DioramaStickerApp/iOS/DioramaSticker/DioramaSticker/Resources/Levels"
DOCS_DIR = "/home/antigravity/DioramaStickerApp/docs"


def strip_outer_white_halo(img_bgr, alpha, out_mask, white_thresh=232):
    """Remove exterior white sticker-cutout halo around characters/objects while preserving interior whites (eyes, bowls, clothes)."""
    fg = (alpha > 16).astype(np.uint8)
    if fg.sum() == 0:
        return alpha
    ys, xs = np.where(fg > 0)
    y0, y1 = max(0, int(ys.min()) - 2), min(img_bgr.shape[0], int(ys.max()) + 3)
    x0, x1 = max(0, int(xs.min()) - 2), min(img_bgr.shape[1], int(xs.max()) + 3)
    sub_bgr = img_bgr[y0:y1, x0:x1]
    sub_fg = fg[y0:y1, x0:x1]
    sub_out = out_mask[y0:y1, x0:x1]

    # Barrier = dark hand-drawn outline (dilated 1px) OR any non-white foreground pixel
    b, g, r = sub_bgr[:, :, 0], sub_bgr[:, :, 1], sub_bgr[:, :, 2]
    is_white = ((r > white_thresh) & (g > white_thresh) & (b > white_thresh)).astype(np.uint8)
    is_bg_or_halo = ((sub_fg == 0) | ((is_white > 0) & (sub_out == 0))).astype(np.uint8)

    # Flood fill from padded outside border through is_bg_or_halo -> marks exterior + white halo as 2
    sh, sw = is_bg_or_halo.shape
    padded = np.zeros((sh + 2, sw + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = np.where(is_bg_or_halo > 0, 0, 1).astype(np.uint8)  # 0=passable (bg or white halo), 1=solid object
    ff = np.zeros((sh + 4, sw + 4), dtype=np.uint8)
    cv2.floodFill(padded, ff, (0, 0), 2)
    keep = (padded[1:-1, 1:-1] != 2).astype(np.uint8)
    # Remove any thin 1-2px detached halo ring outside the main body
    keep = cv2.morphologyEx(keep, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    nl, lbl, st, _ = cv2.connectedComponentsWithStats(keep)
    if nl > 1:
        max_a = int(np.max(st[1:, cv2.CC_STAT_AREA]))
        cleaned = np.zeros_like(keep)
        for i in range(1, nl):
            if st[i, cv2.CC_STAT_AREA] >= max(60, int(max_a * 0.04)):
                cleaned[lbl == i] = 1
        keep = cleaned
    # Restore 1px outline edge cleanly
    dil = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    keep = np.where((keep > 0) | ((dil > 0) & (sub_out > 0)), 1, 0).astype(np.uint8)
    keep = ndimage.binary_fill_holes(keep).astype(np.uint8)

    out_a = np.zeros_like(alpha)
    out_a[y0:y1, x0:x1] = smooth_alpha_from_binary(keep, blur_sigma=0.55)
    return out_a


# ==============================================================================
# 1. UPGRADE hotpot_feast -> GRAND LONG-TABLE BANQUET (12 CHARACTERS + 44 STICKERS!)
# ==============================================================================
def build_hotpot_banquet():
    orig_path = os.path.join(BRAIN, "hotpot_banquet_scene_1795541180183.jpg")
    bg_path = os.path.join(BRAIN, "hotpot_banquet_empty_bg_1795541347018.jpg")
    hp = LevelBuilder("hotpot_feast", "火锅聚会盛宴", "Grand Hotpot Banquet", "12人超长桌俯视火锅派对", orig_path, bg_path)
    out_hp = get_outline_mask(hp.orig, r_max=115, g_max=96, b_max=88)

    # Clean up old hotpot_feast PNGs before writing the upgraded set
    for fn in os.listdir(hp.out_dir):
        if fn.endswith(".png") and fn not in ("background.png", "reference_complete.png"):
            os.remove(os.path.join(hp.out_dir, fn))

    def ext_clean(roi, thresh=22, keep_largest=True, min_area=85, exclude_polys=None, strip_halo=False, keep_poly=None, fill_holes=True):
        a = hp.extract_diff(roi, thresh=thresh, keep_largest=keep_largest, min_area=min_area,
                            exclude_polys=exclude_polys, fill_holes=fill_holes)
        if keep_poly is not None:
            a = clip_alpha_to_poly(a, keep_poly)
        if strip_halo:
            a = strip_outer_white_halo(hp.orig, a, out_hp, white_thresh=228)
        return a

    # --- A. 5 Wooden Round Stools around the room (z=12, behind characters) ---
    hp.add("stool_left_top", "左上原木圆凳", "Top-Left Stool", "营地设施", 12,
           ext_clean((74, 312, 148, 420), thresh=22, keep_largest=True))
    hp.add("stool_left_bot", "左下原木圆凳", "Bottom-Left Stool", "营地设施", 12,
           ext_clean((74, 818, 148, 926), thresh=22, keep_largest=True))
    hp.add("stool_right_top", "右上原木圆凳", "Top-Right Stool", "营地设施", 12,
           ext_clean((754, 300, 824, 410), thresh=22, keep_largest=True))
    hp.add("stool_right_mid", "右中原木圆凳", "Mid-Right Stool", "营地设施", 12,
           ext_clean((768, 714, 824, 822), thresh=22, keep_largest=True))
    hp.add("stool_right_bot", "右下原木圆凳", "Bottom-Right Stool", "营地设施", 12,
           ext_clean((612, 854, 680, 944), thresh=22, keep_largest=True,
                     exclude_polys=[[(656, 854), (680, 854), (680, 944), (656, 944)],
                                    [(612, 922), (680, 922), (680, 944), (612, 944)]], strip_halo=True))

    # --- B. Side Table Condiments & Plates (2 stickers, z=14) ---
    hp.add("side_table_left_items", "左边桌餐具与调味罐", "Left Side Table Condiments", "露营道具", 14,
           ext_clean((76, 480, 140, 682), thresh=22, keep_largest=False, min_area=90))
    hp.add("side_table_right_items", "右边桌餐盘与胡椒瓶", "Right Side Table Condiments", "露营道具", 14,
           ext_clean((764, 474, 822, 656), thresh=22, keep_largest=False, min_area=90))

    # --- C. 2 Big Yin-Yang Hotpots (z=18) ---
    hp.add("hotpot_top", "麻辣鲜香红白鸳鸯锅", "Spicy Yin-Yang Hotpot", "营地设施", 18,
           ext_clean((366, 360, 544, 526), thresh=20, keep_largest=True))
    hp.add("hotpot_bottom", "滋补菌菇双拼鸳鸯锅", "Mushroom Yin-Yang Hotpot", "营地设施", 18,
           ext_clean((366, 654, 544, 818), thresh=20, keep_largest=True))

    # --- D. 12 Center Food Plates, Ladles & Stacked Dishes (z=20) ---
    hp.add("plate_meatballs", "手打鲜肉丸拼盘", "Meatballs Plate", "露营道具", 20,
           ext_clean((378, 300, 450, 368), thresh=22, keep_largest=True))
    hp.add("plate_mushrooms", "菌菇鲜蔬大拼盘", "Mushroom & Veggie Platter", "露营道具", 20,
           ext_clean((450, 290, 536, 368), thresh=22, keep_largest=True))
    hp.add("plate_beef_rolls", "现切雪花肥牛卷", "Sliced Beef Rolls", "露营道具", 20,
           ext_clean((370, 528, 456, 588), thresh=22, keep_largest=True))
    hp.add("plate_shrimp", "鲜红大明虾盘", "Fresh Shrimp Plate", "露营道具", 20,
           ext_clean((458, 528, 538, 588), thresh=22, keep_largest=True))
    hp.add("plate_fishballs", "弹牙爆浆鱼丸", "Fish Balls Plate", "露营道具", 20,
           ext_clean((374, 588, 450, 656), thresh=22, keep_largest=True))
    hp.add("plate_bokchoy", "翠绿小白菜盘", "Fresh Bok Choy", "露营道具", 20,
           ext_clean((458, 590, 542, 654), thresh=22, keep_largest=True))
    hp.add("plate_corn", "香甜玉米段", "Sweet Corn Plate", "露营道具", 20,
           ext_clean((376, 818, 442, 882), thresh=22, keep_largest=True))
    hp.add("plate_dumplings", "手工鲜肉水饺", "Dumplings Plate", "露营道具", 20,
           ext_clean((446, 818, 518, 882), thresh=22, keep_largest=True))
    hp.add("plate_noodles", "劲道手工拉面", "Hotpot Noodles", "露营道具", 20,
           ext_clean((522, 760, 594, 826), thresh=22, keep_largest=True))
    hp.add("ladle_top", "顶部不锈钢汤勺", "Top Soup Ladle", "露营道具", 20,
           ext_clean((492, 208, 558, 290), thresh=22, keep_largest=True,
                     exclude_polys=[[(492, 208), (510, 208), (510, 256), (492, 256)]], strip_halo=True))
    hp.add("ladles_bottom", "底部双漏勺", "Bottom Skimmer Ladles", "露营道具", 20,
           ext_clean((522, 824, 604, 868), thresh=22, keep_largest=False, min_area=90))
    hp.add("stacked_plates", "一摞备用白餐盘", "Stacked Clean Plates", "露营道具", 20,
           ext_clean((548, 874, 602, 932), thresh=22, keep_largest=True))

    # --- E. 8 Fine Table Drinks & Dipping Sauce Sets along both sides (z=21) ---
    hp.add("sauce_top_center", "顶部冰茶与蘸碟", "Top Tea & Sauce Bowl", "露营道具", 21,
           ext_clean((370, 240, 446, 302), thresh=22, keep_largest=False, min_area=120))
    hp.add("sauce_left_1", "左上葱花蘸碟与鲜奶", "Top-Left Dip & Milk", "露营道具", 21,
           ext_clean((306, 274, 378, 398), thresh=22, keep_largest=False, min_area=120,
                     exclude_polys=[[(306, 310), (354, 310), (354, 360), (306, 360)]]))
    hp.add("sauce_left_2", "左中蒜泥蘸料与冷饮", "Mid-Left Dip & Tea", "露营道具", 21,
           ext_clean((306, 404, 376, 534), thresh=22, keep_largest=False, min_area=120,
                     exclude_polys=[[(306, 444), (352, 444), (352, 502), (306, 502)]]))
    hp.add("sauce_left_3", "左三红油碟与绿茶", "Lower-Mid-Left Dip & Tea", "露营道具", 21,
           ext_clean((306, 540, 376, 666), thresh=22, keep_largest=False, min_area=120,
                     exclude_polys=[[(306, 576), (352, 576), (352, 626), (306, 626)]]))
    hp.add("sauce_left_4", "左下香油碟与饮品", "Bottom-Left Dip & Drinks", "露营道具", 21,
           ext_clean((306, 672, 378, 924), thresh=22, keep_largest=False, min_area=120,
                     exclude_polys=[[(306, 708), (352, 708), (352, 764), (306, 764)],
                                    [(306, 834), (352, 834), (352, 886), (306, 886)]]))
    hp.add("sauce_right_1", "右上冰饮与红油碟", "Top-Right Tea & Chili Dip", "露营道具", 21,
           ext_clean((526, 266, 596, 454), thresh=22, keep_largest=False, min_area=120,
                     exclude_polys=[[(554, 306), (596, 306), (596, 362), (554, 362)]]))
    hp.add("sauce_right_2", "右中果汁与双蘸碟", "Mid-Right Juice & Dips", "露营道具", 21,
           ext_clean((530, 462, 594, 664), thresh=22, keep_largest=False, min_area=120,
                     exclude_polys=[[(558, 446), (594, 446), (594, 502), (558, 502)],
                                    [(558, 580), (594, 580), (594, 636), (558, 636)]]))
    hp.add("sauce_right_3", "底部蘸料碟与冰饮", "Bottom Drinks & Dips", "露营道具", 21,
           ext_clean((388, 886, 548, 926), thresh=22, keep_largest=False, min_area=120,
                     exclude_polys=[[(430, 886), (484, 886), (484, 926), (430, 926)]]))

    # --- F. 12 Cute Chibi Diners & Waiter + 1 Calico Cat (13 stickers, z=24..26, clean outer halo stripped!) ---
    hp.add("diner_top", "顶部黄衣丸子头女孩", "Top Bun Hair Girl", "卡通人物", 25,
           ext_clean((388, 64, 514, 268), thresh=22, keep_largest=True, strip_halo=True))

    hp.add("diner_left_1", "左上红毛衣女孩", "Left-1 Red Sweater Girl", "卡通人物", 24,
           ext_clean((194, 218, 356, 382), thresh=22, keep_largest=True,
                     exclude_polys=[[(336, 274), (356, 274), (356, 314), (336, 314)],
                                    [(344, 348), (356, 348), (356, 382), (344, 382)]], strip_halo=True))
    hp.add("diner_left_2", "左二粉毛衣丸子头", "Left-2 Pink Sweater Girl", "卡通人物", 24,
           ext_clean((196, 372, 354, 520), thresh=22, keep_largest=True,
                     exclude_polys=[[(324, 404), (354, 404), (354, 444), (324, 444)],
                                    [(348, 480), (354, 480), (354, 520), (348, 520)]], strip_halo=True))
    hp.add("diner_left_3", "左三绿毛衣女孩", "Left-3 Green Sweater Girl", "卡通人物", 24,
           ext_clean((182, 506, 354, 654), thresh=22, keep_largest=True,
                     exclude_polys=[[(328, 538), (354, 538), (354, 574), (328, 574)],
                                    [(344, 622), (354, 622), (354, 654), (344, 654)]], strip_halo=True))
    hp.add("diner_left_4", "左四红贝雷帽男孩", "Left-4 Beret Boy", "卡通人物", 25,
           ext_clean((194, 634, 352, 780), thresh=22, keep_largest=True,
                     exclude_polys=[[(334, 672), (352, 672), (352, 710), (334, 710)],
                                    [(344, 748), (352, 748), (352, 780), (344, 780)]], strip_halo=True))
    hp.add("diner_left_5", "左下毛衣女孩", "Left-5 Knit Sweater Girl", "卡通人物", 25,
           ext_clean((186, 756, 354, 968), thresh=22, keep_largest=True,
                     exclude_polys=[[(328, 798), (354, 798), (354, 834), (328, 834)],
                                    [(344, 874), (354, 874), (354, 930), (344, 930)]], strip_halo=True))

    hp.add("diner_right_1", "右上蓝帽男孩", "Right-1 Beanie Boy", "卡通人物", 24,
           ext_clean((552, 224, 706, 382), thresh=22, keep_largest=True,
                     exclude_polys=[[(552, 264), (576, 264), (576, 310), (552, 310)],
                                    [(552, 354), (576, 354), (576, 382), (552, 382)]], strip_halo=True))
    hp.add("diner_right_2", "右二红贝雷帽女孩", "Right-2 Red Beret Girl", "卡通人物", 24,
           ext_clean((554, 366, 714, 526), thresh=22, keep_largest=True,
                     exclude_polys=[[(554, 404), (578, 404), (578, 446), (554, 446)],
                                    [(554, 492), (576, 492), (576, 526), (554, 526)]], strip_halo=True))
    hp.add("diner_right_3", "右三粉毛衣发夹女孩", "Right-3 Hairpin Girl", "卡通人物", 24,
           ext_clean((554, 512, 714, 658), thresh=22, keep_largest=True,
                     exclude_polys=[[(554, 534), (582, 534), (582, 582), (554, 582)],
                                    [(554, 632), (582, 632), (582, 658), (554, 658)]], strip_halo=True))
    hp.add("diner_right_4", "右四紫毛衣波波头", "Right-4 Purple Sweater Girl", "卡通人物", 25,
           ext_clean((556, 646, 714, 850), thresh=22, keep_largest=True,
                     exclude_polys=[[(556, 672), (580, 672), (580, 716), (556, 716)],
                                    [(556, 760), (592, 760), (592, 826), (556, 826)]], strip_halo=True))

    hp.add("diner_bottom", "底部背影红衣食客", "Bottom Red Sweater Diner", "卡通人物", 26,
           ext_clean((390, 886, 524, 1086), thresh=22, keep_largest=True, strip_halo=True))
    hp.add("waiter_boy", "端肥牛盘服务员", "Hotpot Waiter Boy", "卡通人物", 26,
           ext_clean((612, 850, 804, 1068), thresh=22, keep_largest=True,
                     exclude_polys=[[(612, 850), (664, 850), (664, 932), (612, 932)]], strip_halo=True))
    hp.add("calico_cat", "馋嘴小三花猫", "Calico Kitty", "可爱动物", 26,
           ext_clean((190, 932, 294, 1036), thresh=22, keep_largest=True, strip_halo=True))

    hp.add("drink_right_4", "右四冰红茶杯", "Iced Black Tea", "露营道具", 21,
           ext_clean((542, 670, 578, 728), thresh=22, keep_largest=True))
    hp.add("drink_bottom_right", "右下橙汁杯", "Orange Juice Cup", "露营道具", 21,
           ext_clean((512, 872, 546, 922), thresh=22, keep_largest=True))

    hp.finalize()
    render_contact_sheet(hp.out_dir, hp.stickers, "sheet_hotpot_feast.png")


# ==============================================================================
# 2. NEW LEVEL 13: pearl_earring_studio (《戴珍珠耳环的少女》荷兰画室 — 23 STICKERS!)
# ==============================================================================
def build_pearl_earring_studio():
    orig_path = os.path.join(BRAIN, "pearl_earring_studio_scene_1795541241189.jpg")
    bg_path = os.path.join(BRAIN, "pearl_earring_studio_empty_bg_1795541296349.jpg")
    ps = LevelBuilder("pearl_earring_studio", "珍珠耳环画室", "Pearl Earring Studio", "维米尔的暖光荷兰画室", orig_path, bg_path)
    H, W = ps.H, ps.W

    # --- A. 5 Framed Wall Paintings (z=12) ---
    ps.add("frame_pearl_mini", "墙上珍珠少女小画", "Mini Pearl Portrait", "露营道具", 12,
           ps.extract_diff((342, 260, 424, 394), thresh=22, keep_largest=True))
    ps.add("frame_dutch_1", "右墙荷兰人物画", "Dutch Portrait Frame 1", "露营道具", 12,
           ps.extract_diff((642, 340, 704, 448), thresh=22, keep_largest=True))
    ps.add("frame_dutch_2", "右墙风俗油画", "Dutch Genre Frame 2", "露营道具", 12,
           ps.extract_diff((706, 370, 784, 504), thresh=22, keep_largest=True))
    ps.add("frame_dutch_3", "右墙古典小像画", "Dutch Miniature Frame 3", "露营道具", 12,
           ps.extract_diff((636, 428, 704, 538), thresh=22, keep_largest=True))
    ps.add("frame_dutch_4", "右墙代尔夫特风景画", "Delft Landscape Frame 4", "露营道具", 12,
           ps.extract_diff((708, 476, 774, 570), thresh=22, keep_largest=True))

    # --- B. Persian Rug (z=14 — clean solid vintage Persian rug, strictly inside rug_outer!) ---
    rug_outer = np.array([[454, 650], [616, 744], [492, 856], [326, 758]], dtype=np.int32)
    rug_inner = np.array([[454, 662], [598, 744], [492, 840], [344, 758]], dtype=np.int32)
    rug_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.fillPoly(rug_canvas, [rug_outer], (62, 66, 152))
    cv2.fillPoly(rug_canvas, [rug_inner], (54, 58, 136))
    cv2.polylines(rug_canvas, [rug_inner], True, (116, 178, 224), 3, cv2.LINE_AA)
    cv2.polylines(rug_canvas, [rug_outer], True, (46, 52, 74), 3, cv2.LINE_AA)
    rug_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(rug_mask, [rug_outer], 255)
    # Copy real visible rug texture pixels strictly INSIDE rug_outer where characters/cat/easel don't cover it
    vis_mask = rug_mask.copy()
    for ep in [
        [(392, 646), (546, 646), (546, 754), (392, 754)],
        [(428, 732), (534, 732), (534, 818), (428, 818)],
        [(516, 646), (628, 646), (628, 868), (516, 868)],
        [(294, 660), (350, 660), (350, 868), (294, 868)],
        [(332, 700), (434, 700), (434, 868), (332, 868)],
    ]:
        cv2.fillPoly(vis_mask, [np.array(ep, dtype=np.int32)], 0)
    vis = (vis_mask > 0)
    rug_canvas[vis] = ps.orig[vis]
    a_rug_final = smooth_alpha_from_binary((rug_mask > 0).astype(np.uint8), blur_sigma=0.55)
    fn_rug = "persian_rug.png"
    rx, ry, rw, rh = save_rgba_from_mask(rug_canvas, a_rug_final, os.path.join(ps.out_dir, fn_rug), pad=2)
    ps.stickers.append({
        "id": "persian_rug", "file": fn_rug, "name_zh": "波斯复古花地毯", "name_en": "Vintage Persian Rug",
        "category": "营地设施", "zIndex": 14,
        "x": int(rx), "y": int(ry), "width": int(rw), "height": int(rh),
        "normX": round((rx + rw / 2.0) / W, 5), "normY": round((ry + rh / 2.0) / H, 5),
        "normW": round(rw / W, 5), "normH": round(rh / H, 5),
    })

    # --- C. Leaning Paintings, Lute, Fern Plant, Canvas Basket (z=16..18) ---
    ps.add("leaning_paintings", "靠墙双幅油画框", "Leaning Framed Paintings", "营地设施", 16,
           ps.extract_diff((334, 428, 448, 656), thresh=22, keep_largest=True,
                           exclude_polys=[[(406, 468), (448, 468), (448, 656), (406, 656)]]))
    ps.add("dutch_lute", "古典木质鲁特琴", "Classical Dutch Lute", "露营道具", 16,
           ps.extract_diff((532, 468, 624, 666), thresh=22, keep_largest=True,
                           exclude_polys=[[(592, 540), (624, 540), (624, 666), (592, 666)]]))
    ps.add("potted_fern", "窗边落地蕨类盆栽", "Potted Fern Plant", "露营道具", 18,
           ps.extract_diff((84, 626, 228, 794), thresh=22, keep_largest=True,
                           exclude_polys=[[(182, 626), (228, 626), (228, 682), (182, 682)]]))
    ps.add("canvas_scroll_basket", "画卷收纳竹篮", "Canvas Scrolls Basket", "露营道具", 18,
           ps.extract_diff((712, 610, 810, 794), thresh=22, keep_largest=True,
                           exclude_polys=[[(712, 610), (736, 610), (736, 660), (712, 660)]]))

    # --- D. Window Side Table + 4 Tabletop Treasures (crisp outline-enclosed cutouts!) ---
    a_table_full = ps.extract_diff((174, 536, 368, 728), thresh=22, keep_largest=True,
                                   exclude_polys=[[(220, 662), (346, 662), (346, 728), (220, 728)]])
    ps.add("window_side_table", "窗边复古木边桌", "Window Wooden Table", "营地设施", 16, a_table_full)
    ps.add("brass_candlestick", "黄铜点亮烛台", "Brass Candlestick", "露营道具", 19,
           ps.extract_outline_enclosed((198, 544, 246, 636),
                                       bg_seeds=[(200, 546), (244, 546), (200, 634), (244, 634)]))
    ps.add("pearl_jewelry_box", "珍珠首饰宝盒", "Pearl Jewelry Box", "露营道具", 20,
           ps.extract_outline_enclosed((242, 574, 304, 648),
                                       bg_seeds=[(244, 576), (302, 576), (244, 646), (302, 646)]))
    ps.add("delft_tulip_vase", "代尔夫特青花郁金香", "Delft Tulip Vase", "露营道具", 19,
           ps.extract_outline_enclosed((264, 466, 334, 598),
                                       bg_seeds=[(266, 468), (332, 468), (266, 596), (332, 596), (266, 540), (332, 540)]))
    ps.add("fruit_bowl", "彩绘陶瓷果盘", "Dutch Fruit Bowl", "露营道具", 20,
           ps.extract_outline_enclosed((302, 554, 356, 610),
                                       bg_seeds=[(304, 556), (354, 556), (304, 608), (354, 608)]))

    # --- E. Foreground Art Stool + Paint Brushes & Tubes (clean outline-enclosed!) ---
    a_stool_full = ps.extract_diff((328, 770, 434, 906), thresh=22, keep_largest=True)
    ps.add("art_stool", "四脚实木画具圆凳", "Painter's Wood Stool", "营地设施", 26, a_stool_full)
    ps.add("brushes_paints_set", "画笔罐与油画颜料管", "Brush Jar & Paint Tubes", "露营道具", 28,
           ps.extract_outline_enclosed((344, 700, 430, 842),
                                       bg_seeds=[(346, 702), (428, 702), (346, 840), (428, 840), (346, 760), (428, 760)]))

    # --- F. Iconic Characters, Orange Cat & Easel (ZERO square rug background!) ---
    ps.add("girl_pearl_earring", "戴珍珠耳环的少女", "Girl with a Pearl Earring", "卡通人物", 22,
           ps.extract_outline_enclosed((392, 464, 544, 754),
                                       bg_seeds=[(394, 466), (542, 466), (394, 752), (542, 752), (394, 640), (542, 640)]))
    ps.add("orange_studio_cat", "画室瞌睡小橘猫", "Studio Orange Tabby", "可爱动物", 24,
           ps.extract_outline_enclosed((428, 732, 534, 818),
                                       bg_seeds=[(430, 734), (532, 734), (430, 816), (532, 816)]))
    ps.add("vermeer_painter", "贝雷帽荷兰画家", "Dutch Master Painter", "卡通人物", 22,
           ps.extract_diff((592, 540, 734, 822), thresh=22, keep_largest=True,
                           exclude_polys=[[(592, 668), (626, 668), (626, 822), (592, 822)]]))
    ps.add("wooden_easel", "木质画架与肖像画布", "Wooden Portrait Easel", "营地设施", 26,
           ps.extract_outline_enclosed((518, 626, 636, 894),
                                       bg_seeds=[(520, 628), (634, 628), (520, 892), (634, 892), (520, 760), (634, 760)]))
    ps.add("apprentice_tea_girl", "端茶盘荷兰小助手", "Tea Tray Apprentice Girl", "卡通人物", 27,
           ps.extract_outline_enclosed((216, 662, 348, 876),
                                       bg_seeds=[(218, 664), (346, 664), (218, 874), (346, 874), (218, 770), (346, 770)]))

    ps.finalize()
    render_contact_sheet(ps.out_dir, ps.stickers, "sheet_pearl_earring_studio.png")


def register_and_sync_all_13():
    # Ensure pearl_earring_studio is in iOS levels_manifest.json
    ios_m_path = os.path.join(IOS_LEVELS, "levels_manifest.json")
    with open(ios_m_path, "r", encoding="utf-8") as f:
        ios_m = json.load(f)

    # Update hotpot_feast subtitle in iOS manifest
    for it in ios_m["levels"]:
        if it["id"] == "hotpot_feast":
            it["title_zh"] = "火锅聚会盛宴"
            it["subtitle_zh"] = "12人超长桌俯视火锅派对"

    if not any(x["id"] == "pearl_earring_studio" for x in ios_m["levels"]):
        ios_m["levels"].append({
            "id": "pearl_earring_studio",
            "folder": "pearl_earring_studio",
            "jsonFile": "level_pearl_earring_studio",
            "title_zh": "珍珠耳环画室",
            "title_en": "Pearl Earring Studio",
            "subtitle_zh": "维米尔的暖光荷兰画室",
            "emoji": "🎨",
            "accentHex": "#D4A359"
        })
        with open(ios_m_path, "w", encoding="utf-8") as f:
            json.dump(ios_m, f, ensure_ascii=False, indent=2)
    else:
        with open(ios_m_path, "w", encoding="utf-8") as f:
            json.dump(ios_m, f, ensure_ascii=False, indent=2)

    sync_ids = {"hotpot_feast", "pearl_earring_studio"}
    docs_levels = []

    for item in ios_m["levels"]:
        lid = item["id"]
        src_dir = os.path.join(IOS_LEVELS, item["folder"])
        with open(os.path.join(src_dir, item["jsonFile"] + ".json"), "r", encoding="utf-8") as f:
            jdata = json.load(f)
        cnt = len(jdata["stickers"])

        if lid in sync_ids:
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

    grand_total = sum(x["stickerCount"] for x in docs_levels)
    print(f"\nALL {len(docs_levels)} LEVELS SYNCED! GRAND TOTAL: {grand_total} STICKERS!")
    for x in docs_levels:
        print(f"  - {x['emoji']} {x['title_zh']} ({x['id']}): {x['stickerCount']} stickers")


if __name__ == "__main__":
    build_hotpot_banquet()
    build_pearl_earring_studio()
    register_and_sync_all_13()
