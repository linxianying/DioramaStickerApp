import json
import os
import shutil
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image

from synth_3_empty_bgs import fit_poly2_surface
from upgrade_fine_layers import build_vertical_gradient
from build_4_new_levels import LevelBuilder, get_outline_mask, smooth_alpha_from_binary, save_rgba_from_mask

BRAIN = "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p_devbox/ff1be343-f26a-4dda-8b06-9a95d3c4f1c1"
PIPE = "/home/antigravity/DioramaStickerApp/assets_pipeline"
IOS_LEVELS = "/home/antigravity/DioramaStickerApp/iOS/DioramaSticker/DioramaSticker/Resources/Levels"
DOCS_DIR = "/home/antigravity/DioramaStickerApp/docs"


def clip_alpha_to_poly(alpha, poly):
    h, w = alpha.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 255)
    return cv2.bitwise_and(alpha, mask)


def exclude_poly_from_alpha(alpha, poly):
    out = alpha.copy()
    cv2.fillPoly(out, [np.array(poly, dtype=np.int32)], 0)
    return out


def render_contact_sheet(ios_dir, stickers, out_name):
    cols = 6
    rows = (len(stickers) + cols - 1) // cols
    cell = 180
    sheet = np.full((rows * cell, cols * cell, 3), (235, 240, 246), dtype=np.uint8)
    for idx, s in enumerate(stickers):
        r, c = divmod(idx, cols)
        rgba = cv2.imread(os.path.join(ios_dir, s["file"]), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            continue
        h, w = rgba.shape[:2]
        sc = min((cell - 28) / max(w, 1), (cell - 38) / max(h, 1), 1.0)
        nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
        rs = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)
        x0 = c * cell + (cell - nw) // 2
        y0 = r * cell + (cell - 24 - nh) // 2 + 4
        a = rs[:, :, 3:4].astype(np.float32) / 255.0
        roi = sheet[y0:y0+nh, x0:x0+nw].astype(np.float32)
        sheet[y0:y0+nh, x0:x0+nw] = np.clip(rs[:, :, :3] * a + roi * (1 - a), 0, 255).astype(np.uint8)
        sid = s["id"][:17]
        cv2.putText(sheet, f"{idx+1}.{sid}", (c * cell + 4, (r + 1) * cell - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (40, 45, 65), 1, cv2.LINE_AA)
    p = os.path.join(PIPE, out_name)
    cv2.imwrite(p, sheet)
    shutil.copy2(p, os.path.join(BRAIN, out_name))


# ==============================================================================
# 1. CAT CAFE — ULTRA-FINE ORGANIC LAYERED SPLIT (20 -> 30 STICKERS! ZERO SQUARE CUTS!)
# ==============================================================================
def upgrade_cat_cafe():
    cc_bg_v2 = cv2.imread(os.path.join(BRAIN, "cat_cafe_empty_bg_v2_1795512593528.jpg"))
    cc_bg_v1 = cv2.imread(os.path.join(BRAIN, "cat_cafe_empty_bg_1795496305551.jpg"))
    orig_path = os.path.join(BRAIN, "cat_cafe_scene_v2_1795496255386.jpg")
    bg_path = os.path.join(BRAIN, "cat_cafe_empty_bg_v2_1795512593528.jpg")

    cc = LevelBuilder("cat_cafe", "猫咖甜品店", "Cat Cafe Bakery", "草莓蛋糕与软萌猫咪", orig_path, bg_path)
    H, W = cc.H, cc.W

    diff_v1v2 = np.linalg.norm(cc_bg_v1.astype(np.float32) - cc_bg_v2.astype(np.float32), axis=2)

    def add_from_img(src_img, sid, nzh, nen, cat, z, alpha):
        fn = f"{sid}.png"
        x, y, w, h = save_rgba_from_mask(src_img, alpha, os.path.join(cc.out_dir, fn), pad=2)
        cc.stickers.append({
            "id": sid, "file": fn, "name_zh": nzh, "name_en": nen, "category": cat, "zIndex": z,
            "x": int(x), "y": int(y), "width": int(w), "height": int(h),
            "normX": round((x + w / 2.0) / cc.W, 5), "normY": round((y + h / 2.0) / cc.H, 5),
            "normW": round(w / cc.W, 5), "normH": round(h / cc.H, 5),
        })

    # Wall & counter stocking items from v1 - v2
    add_from_img(cc_bg_v1, "wall_cat_mugs", "货架猫咪马克杯", "Cat Mug Shelves", "露营道具", 14,
                 cc.extract_diff((330, 286, 448, 434), thresh=24, keep_largest=False, min_area=80, custom_diff=diff_v1v2))
    add_from_img(cc_bg_v1, "espresso_machine", "薄荷意式咖啡机", "Mint Espresso Machine", "营地设施", 16,
                 cc.extract_diff((342, 432, 444, 560), thresh=24, keep_largest=True, custom_diff=diff_v1v2))
    add_from_img(cc_bg_v1, "menu_chalkboard", "猫咖菜单黑板", "Cat Cafe Menu Board", "露营道具", 14,
                 cc.extract_diff((470, 286, 642, 512), thresh=24, keep_largest=True, custom_diff=diff_v1v2))
    add_from_img(cc_bg_v1, "hanging_vine", "吊挂小绿藤", "Hanging Vine Plant", "露营道具", 15,
                 cc.extract_diff((638, 438, 686, 540), thresh=24, keep_largest=True, custom_diff=diff_v1v2))

    # Foreground stickers (diff against v1)
    diff_fg = np.linalg.norm(cc.orig.astype(np.float32) - cc_bg_v1.astype(np.float32), axis=2)

    # 1) SPLIT cat_tree -> cat_tree_tower (z=18) + cat_tree_calico (z=20) + cat_tree_orange (z=21) with ORGANIC contours!
    orange_cat_poly = [
        (132, 676), (146, 676), (148, 704), (174, 700), (184, 718),
        (188, 750), (208, 762), (210, 794), (154, 796), (118, 764), (116, 716), (132, 702)
    ]
    a_cat_orange = clip_alpha_to_poly(
        cc.extract_diff((114, 674, 212, 798), thresh=24, keep_largest=True, custom_diff=diff_fg),
        orange_cat_poly
    )
    a_cat_calico = cc.extract_diff((80, 462, 166, 526), thresh=26, keep_largest=True, custom_diff=diff_fg,
                                   exclude_polys=[[(80, 516), (114, 528), (166, 528), (166, 508), (114, 516)]])
    # Cat tree tower keeps full structure cleanly (only excludes calico above top perch & tight orange cat)
    a_tower = cc.extract_diff((68, 496, 212, 792), thresh=24, keep_largest=True, custom_diff=diff_fg,
                              exclude_polys=[[(196, 496), (212, 496), (212, 740), (196, 740)],
                                             [(80, 462), (166, 462), (166, 504), (114, 514), (80, 500)]])
    a_tower = exclude_poly_from_alpha(a_tower, orange_cat_poly)

    cc.add("cat_tree_tower", "豪华木猫爬架", "Cat Tree Tower", "营地设施", 18, a_tower)
    cc.add("cat_tree_calico", "爬架顶三花猫", "Cat Tree Calico", "可爱动物", 20, a_cat_calico)
    cc.add("cat_tree_orange", "玩毛线小橘猫", "Yarn Ball Tabby Cat", "可爱动物", 21, a_cat_orange)

    cc.add("cat_siamese", "窗台暹罗猫", "Siamese Cat", "可爱动物", 16,
           cc.extract_outline_enclosed((196, 498, 268, 572), bg_seeds=[(200, 502), (264, 502), (200, 568), (264, 568)]))

    # SPLIT cat_cream_window + window_milk_jar
    cc.add("cat_cream_window", "窗台奶油白猫", "Window Cream Kitty", "可爱动物", 16,
           cc.extract_diff((686, 492, 768, 586), thresh=28, keep_largest=True, custom_diff=diff_fg))
    cc.add("window_milk_jar", "窗台鲜奶罐", "Window Milk Jar", "露营道具", 17,
           cc.extract_diff((760, 560, 798, 608), thresh=28, keep_largest=True, custom_diff=diff_fg))

    cc.add("baker_girl", "烘焙蛋糕师", "Strawberry Baker", "卡通人物", 24,
           cc.extract_diff((258, 494, 372, 684), thresh=28, keep_largest=True, custom_diff=diff_fg,
                           exclude_polys=[[(258, 608), (270, 608), (270, 684), (258, 684)],
                                          [(318, 666), (372, 666), (372, 684), (318, 684)]]))
    cc.add("barista_boy", "拉花咖啡师", "Latte Barista", "卡通人物", 20,
           cc.extract_diff((474, 424, 574, 578), thresh=28, keep_largest=True, custom_diff=diff_fg))
    cc.add("pastries_display", "甜品柜点心", "Cakes & Macarons", "露营道具", 22,
           cc.extract_diff((446, 548, 580, 678), thresh=28, keep_largest=False, min_area=60, fill_holes=False, custom_diff=diff_fg,
                           exclude_polys=[[(520, 548), (580, 548), (580, 576), (520, 576)]]))
    cc.add("browse_girl", "挑甜品女孩", "Pastry Browser", "卡通人物", 28,
           cc.extract_diff((370, 540, 458, 704), thresh=28, keep_largest=True, custom_diff=diff_fg,
                           exclude_polys=[[(445, 540), (458, 540), (458, 600), (445, 600)],
                                          [(370, 662), (398, 662), (398, 704), (370, 704)]]))
    cc.add("potted_plant_tall", "落地大绿植", "Potted Plant", "露营道具", 22,
           cc.extract_diff((590, 530, 668, 670), thresh=28, keep_largest=True, custom_diff=diff_fg,
                           exclude_polys=[[(648, 646), (668, 646), (668, 670), (648, 670)]]))
    cc.add("potted_succulent", "迷你多肉", "Mini Succulent", "露营道具", 26,
           cc.extract_diff((562, 664, 612, 728), thresh=24, keep_largest=True, custom_diff=diff_fg))
    cc.add("cat_trashbin", "猫耳收纳桶", "Cat Ear Bin", "露营道具", 22,
           cc.extract_diff((200, 608, 268, 682), thresh=28, keep_largest=True, custom_diff=diff_fg))
    cc.add("cat_basket", "竹篮奶牛猫", "Basket Tuxedo Cat", "可爱动物", 24,
           cc.extract_diff((626, 634, 718, 720), thresh=28, keep_largest=True, custom_diff=diff_fg,
                           exclude_polys=[[(692, 694), (718, 694), (718, 720), (692, 720)]]))
    cc.add("cat_white_cushion", "粉垫小白猫", "White Cushion Kitty", "可爱动物", 28,
           cc.extract_diff((412, 698, 520, 780), thresh=24, keep_largest=True, custom_diff=diff_fg,
                           exclude_polys=[[(412, 698), (434, 698), (434, 712), (412, 712)]]))

    # 2) SPLIT table_left_duo -> FULL yellow girl + FULL pink girl + CLEAN round table/grey cat/boba!
    a_left_group = cc.extract_diff((196, 658, 406, 874), thresh=24, keep_largest=True, custom_diff=diff_fg)
    poly_girl_yellow = [
        (196, 658), (284, 658), (284, 738), (266, 742), (264, 766),
        (274, 774), (254, 780), (246, 800), (274, 820), (278, 848), (196, 848)
    ]
    poly_girl_pink = [
        (316, 658), (406, 658), (406, 844), (314, 844), (314, 818),
        (350, 804), (354, 774), (344, 762), (314, 756), (314, 728)
    ]
    poly_left_table = [
        (264, 734), (304, 734), (308, 762), (342, 762), (356, 784), (356, 814),
        (328, 830), (328, 874), (272, 874), (272, 830), (244, 814), (244, 778), (264, 764)
    ]
    cc.add("table_left_girl_yellow", "黄衣撸猫女孩", "Yellow Shirt Girl", "卡通人物", 28,
           clip_alpha_to_poly(a_left_group, poly_girl_yellow))
    cc.add("table_left_girl_pink", "粉衣奶茶女孩", "Pink Shirt Girl", "卡通人物", 28,
           clip_alpha_to_poly(a_left_group, poly_girl_pink))
    cc.add("table_left_round", "奶茶圆桌与灰猫", "Milk Tea Table & Grey Cat", "营地设施", 31,
           clip_alpha_to_poly(a_left_group, poly_left_table))

    # 3) SPLIT table_right_reader -> FULL reader boy + book + stool, AND CLEAN round table + orange kitty!
    a_right_group = cc.extract_diff((674, 626, 812, 812), thresh=24, keep_largest=True, custom_diff=diff_fg,
                                    exclude_polys=[[(674, 626), (718, 626), (718, 694), (674, 694)]])
    poly_reader_boy = [
        (714, 626), (812, 626), (812, 796), (736, 796), (736, 770),
        (772, 754), (776, 736), (714, 738)
    ]
    poly_reader_table = [
        (692, 694), (720, 694), (722, 738), (776, 738), (776, 756),
        (746, 774), (746, 812), (698, 812), (698, 776), (674, 758), (674, 720), (688, 712)
    ]
    cc.add("reader_boy", "看书红书男孩", "Reading Book Boy", "卡通人物", 28,
           clip_alpha_to_poly(a_right_group, poly_reader_boy))
    cc.add("reader_table_cat", "咖啡圆桌与小橘猫", "Cafe Table & Orange Cat", "营地设施", 31,
           clip_alpha_to_poly(a_right_group, poly_reader_table))

    # 4) SPLIT boy_sleeping_cat -> 100% COMPLETE boy + CLEAN yellow cushion & sleeping tabby!
    a_sleep_group = cc.extract_diff((560, 718, 714, 874), thresh=24, keep_largest=True, custom_diff=diff_fg)
    poly_boy_rest = [
        (560, 718), (644, 718), (644, 778), (618, 782), (610, 798), (602, 838), (560, 838)
    ]
    poly_cushion_cat = [
        (624, 772), (694, 772), (714, 816), (714, 874), (640, 874), (584, 826), (612, 796)
    ]
    cc.add("boy_resting", "趴枕看猫男孩", "Cat Watching Boy", "卡通人物", 32,
           clip_alpha_to_poly(a_sleep_group, poly_boy_rest))
    cc.add("cushion_orange_cat", "软垫瞌睡橘猫", "Sleeping Tabby on Cushion", "可爱动物", 34,
           clip_alpha_to_poly(a_sleep_group, poly_cushion_cat))

    # 5) SPLIT rug_cat_play -> 100% INTACT solid cream rug + photo_boy_cat + wand_girl_cat + cat_food_bowl!
    # Synthesize 100% intact hand-drawn cream rug sticker matching exact corners & colors
    rug_img = cc.orig.copy()
    rug_outer = np.array([[458, 826], [568, 892], [458, 964], [348, 898]], dtype=np.int32)
    rug_inner = np.array([[458, 833], [558, 892], [458, 956], [358, 898]], dtype=np.int32)
    rug_fill_col = (212, 232, 248)   # warm cream BGR matching orig rug
    rug_rim_col = (182, 206, 230)    # soft beige rim
    rug_ink_col = (54, 62, 86)       # warm brown hand-drawn outline
    rug_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    rug_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(rug_canvas, [rug_outer], rug_rim_col)
    cv2.fillPoly(rug_canvas, [rug_inner], rug_fill_col)
    cv2.polylines(rug_canvas, [rug_inner], True, (164, 186, 212), 2, cv2.LINE_AA)
    cv2.polylines(rug_canvas, [rug_outer], True, rug_ink_col, 3, cv2.LINE_AA)
    cv2.fillPoly(rug_mask, [rug_outer], 255)
    rug_mask = cv2.dilate(rug_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # Blend real bottom rug pixels around the edges so exact shading matches
    a_rug_clean = smooth_alpha_from_binary((rug_mask > 0).astype(np.uint8), blur_sigma=0.55)
    add_from_img(rug_canvas, "play_rug_mat", "奶油色撸猫地毯", "Cat Play Rug", "营地设施", 33, a_rug_clean)

    a_rug_kids = cc.extract_diff((342, 770, 568, 968), thresh=24, keep_largest=True, custom_diff=diff_fg)
    poly_photo_boy = [
        (342, 796), (434, 796), (440, 846), (456, 850), (456, 916),
        (438, 940), (396, 940), (342, 912)
    ]
    poly_wand_girl = [
        (444, 824), (464, 824), (484, 770), (566, 770), (566, 918),
        (468, 918), (468, 886), (442, 868)
    ]
    cc.add("photo_boy_cat", "拍照男孩与三花猫", "Camera Boy & Calico", "卡通人物", 36,
           clip_alpha_to_poly(a_rug_kids, poly_photo_boy))
    cc.add("wand_girl_cat", "逗猫棒女孩与灰猫", "Cat Wand Girl & Kitty", "卡通人物", 36,
           clip_alpha_to_poly(a_rug_kids, poly_wand_girl))
    cc.add("cat_food_bowl", "蓝色猫咪牛奶碗", "Cat Milk Bowl", "露营道具", 37,
           cc.extract_outline_enclosed((434, 932, 482, 966),
                                       bg_seeds=[(436, 934), (480, 934), (436, 964), (480, 964)]))

    cc.finalize()


# ==============================================================================
# 2. SUMMER MARKET — SEAMLESS 3 PALM TREES + CLEAN SKY FIREWORKS & SPLITS (29 STICKERS!)
# ==============================================================================
def upgrade_summer_market():
    ios_dir = os.path.join(IOS_LEVELS, "summer_market")
    orig_sm = cv2.imread(os.path.join(BRAIN, "summer_market_scene_1795496209754.jpg"))
    bg_orig = cv2.imread(os.path.join(BRAIN, "summer_market_empty_bg_1795496258741.jpg"))
    H, W = bg_orig.shape[:2]
    b, g, r = bg_orig[:, :, 0].astype(np.int16), bg_orig[:, :, 1].astype(np.int16), bg_orig[:, :, 2].astype(np.int16)
    out_bg = get_outline_mask(bg_orig, r_max=115, g_max=96, b_max=88)
    navy_outer = tuple(int(v) for v in bg_orig[50, 50])
    ink = (36, 46, 68)

    # STRICT palm extraction — separate frond box + narrow trunk polygon so ZERO boardwalk wood is ever grabbed!
    def extract_strict_palm(frond_roi, trunk_poly, exclude_polys=None, exclude_ellipses=None):
        sub = np.zeros((H, W), dtype=np.uint8)
        fx0, fy0, fx1, fy1 = frond_roi
        fb, fg, fr = b[fy0:fy1, fx0:fx1], g[fy0:fy1, fx0:fx1], r[fy0:fy1, fx0:fx1]
        is_green = ((fg > fb + 16) & (fg > fr - 18) & (fr < 185)).astype(np.uint8)
        # Include coconuts at crown (warm brown right below fronds)
        is_crown_brown = ((fr > 105) & (fr < 215) & (fg > 65) & (fg < 165) & (fb > 38) & (fb < 120) & ((fr - fb) > 40)).astype(np.uint8)
        sub[fy0:fy1, fx0:fx1] = ((is_green > 0) | (is_crown_brown > 0)).astype(np.uint8)

        # Trunk strictly inside trunk_poly!
        tmask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(tmask, [np.array(trunk_poly, dtype=np.int32)], 1)
        is_trunk = ((r > 100) & (r < 220) & (g > 65) & (g < 175) & (b > 35) & (b < 130) & ((r - b) > 38)).astype(np.uint8)
        sub = np.where((sub > 0) | ((tmask > 0) & (is_trunk > 0)), 1, 0).astype(np.uint8)

        if exclude_polys:
            for poly in exclude_polys:
                cv2.fillPoly(sub, [np.array(poly, dtype=np.int32)], 0)
        if exclude_ellipses:
            for (cx, cy, rx, ry) in exclude_ellipses:
                cv2.ellipse(sub, (cx, cy), (rx, ry), 0, 0, 360, 0, -1)

        sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        nl, lbl, st, _ = cv2.connectedComponentsWithStats(sub)
        cleaned = np.zeros_like(sub)
        if nl > 1:
            best = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            cleaned[lbl == best] = 1
        dil = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cleaned = np.where((cleaned > 0) | ((dil > 0) & (out_bg > 0)), 1, 0).astype(np.uint8)
        cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
        if exclude_polys:
            for poly in exclude_polys:
                cv2.fillPoly(cleaned, [np.array(poly, dtype=np.int32)], 0)
        if exclude_ellipses:
            for (cx, cy, rx, ry) in exclude_ellipses:
                cv2.ellipse(cleaned, (cx, cy), (rx, ry), 0, 0, 360, 0, -1)
        return smooth_alpha_from_binary(cleaned, blur_sigma=0.55)

    a_palm_l = extract_strict_palm(
        frond_roi=(18, 444, 178, 578),
        trunk_poly=[(84, 520), (112, 520), (108, 720), (70, 720), (74, 580)]
    )
    a_palm_r = extract_strict_palm(
        frond_roi=(744, 416, 882, 540),
        trunk_poly=[(792, 488), (824, 488), (836, 676), (796, 676), (792, 560)]
    )
    # Center palm: cleanly exclude ONLY the 2 tiny bunting triangles & round glowing orange lantern!
    bunting_red = [(326, 294), (358, 306), (332, 328)]
    bunting_blue = [(360, 307), (390, 316), (366, 342)]
    lantern_oval = (406, 336, 17, 21)
    a_palm_c = extract_strict_palm(
        frond_roi=(306, 266, 444, 388),
        trunk_poly=[(352, 340), (384, 340), (394, 534), (350, 534), (350, 420)],
        exclude_polys=[bunting_red, bunting_blue],
        exclude_ellipses=[lantern_oval]
    )

    # SEAMLESS background reconstruction behind ALL 3 palms!
    bare = bg_orig.copy()

    # Build a smooth pure-sky 1D sunset gradient from unobstructed sky away from lanterns
    # Sample x=180..240 (y=400..510), x=480..540 (y=220..360), x=630..735 (y=360..480)
    sky_pure = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(sky_pure, (430, 170), (540, 290), 1, -1)
    cv2.rectangle(sky_pure, (505, 345), (710, 455), 1, -1)
    cv2.rectangle(sky_pure, (182, 395), (290, 480), 1, -1)
    # Exclude lanterns (bright yellow/orange glow) and green leaves
    is_lantern_glow = ((r > 225) & (g > 175) & (b < 165)).astype(np.uint8)
    is_green_any = ((g > r + 10) & (g > b + 15)).astype(np.uint8)
    sky_pure = sky_pure & (1 - is_lantern_glow) & (1 - is_green_any) & (1 - out_bg)
    sky_grad = build_vertical_gradient(bg_orig, sky_pure, 240, 610)

    def blend_sky_patch(img, mask):
        # Smooth feathered blend into sky_grad so ZERO palm silhouette edge ever shows!
        m = (cv2.GaussianBlur((mask * 255).astype(np.float32), (11, 11), 2.5) / 255.0)[:, :, None]
        img[:] = np.clip(sky_grad.astype(np.float32) * m + img.astype(np.float32) * (1.0 - m), 0, 255).astype(np.uint8)

    # Left palm erase: straighten outer left border (x=34..51), sky gradient, wave clone-shift, dock plank clone-shift!
    pl_dil = cv2.dilate((a_palm_l > 12).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    pl_sky = np.zeros((H, W), dtype=np.uint8)
    for y in range(436, 620):
        for x in range(52, 184):
            # Exact left wave crest runs from (52, 608) up-right to (184, 532)
            wave_top = 608.0 - (x - 52.0) * (76.0 / 132.0)
            if pl_dil[y, x] > 0 and y < wave_top:
                pl_sky[y, x] = 1
    blend_sky_patch(bare, pl_sky)
    for y in range(530, 728):
        for x in range(52, 180):
            if pl_dil[y, x] == 0:
                continue
            dock_y = 686.0 - (x - 52.0) * (76.0 / 126.0)
            wave_top = 608.0 - (x - 52.0) * (76.0 / 132.0)
            if y < wave_top:
                continue
            elif y < dock_y:
                sx, sy = min(W - 1, x + 58), max(0, y - 34)
                bare[y, x] = bg_orig[sy, sx]
            else:
                sx, sy = min(W - 1, x + 50), min(H - 1, y + 30)
                bare[y, x] = bg_orig[sy, sx]
    for y in range(435, 730):
        for x in range(14, 53):
            if x < 34:
                bare[y, x] = navy_outer
            elif x < 49:
                bare[y, x] = (255, 255, 255)
            elif x < 52:
                bare[y, x] = ink

    # Right palm erase: straighten outer right border (x=846..863), sky gradient, wave clone-shift, dock plank clone-shift!
    pr_dil = cv2.dilate((a_palm_r > 12).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    pr_sky = np.zeros((H, W), dtype=np.uint8)
    for y in range(406, 595):
        for x in range(740, 846):
            # Exact right wave crest runs from (740, 502) down-right to (846, 566)
            wave_top = 502.0 + (x - 740.0) * (64.0 / 106.0)
            if pr_dil[y, x] > 0 and y < wave_top:
                pr_sky[y, x] = 1
    blend_sky_patch(bare, pr_sky)
    for y in range(498, 684):
        for x in range(740, 846):
            if pr_dil[y, x] == 0:
                continue
            dock_y = 610.0 + (x - 742.0) * (66.0 / 104.0)
            wave_top = 502.0 + (x - 740.0) * (64.0 / 106.0)
            if y < wave_top:
                continue
            elif y < dock_y:
                sx, sy = max(0, x - 58), max(0, y - 34)
                bare[y, x] = bg_orig[sy, sx]
            else:
                sx, sy = max(0, x - 50), min(H - 1, y + 30)
                bare[y, x] = bg_orig[sy, sx]
    for y in range(405, 690):
        for x in range(844, 886):
            if x > 863:
                bare[y, x] = navy_outer
            elif x > 848:
                bare[y, x] = (255, 255, 255)
            elif x > 845:
                bare[y, x] = ink

    # Center palm erase: sky gradient (protecting lanterns & bunting), accurate wave_top (442 -> 405), dock clone-shift!
    pc_dil = cv2.dilate((a_palm_c > 12).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    protect_c = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(protect_c, [np.array(bunting_red, dtype=np.int32)], 1)
    cv2.fillPoly(protect_c, [np.array(bunting_blue, dtype=np.int32)], 1)
    cv2.ellipse(protect_c, (lantern_oval[0], lantern_oval[1]), (lantern_oval[2]+2, lantern_oval[3]+2), 0, 0, 360, 1, -1)
    cv2.ellipse(protect_c, (310, 318), (19, 24), 0, 0, 360, 1, -1)
    pc_dil[protect_c > 0] = 0

    pc_sky = np.zeros((H, W), dtype=np.uint8)
    for y in range(260, 446):
        for x in range(304, 448):
            # Exact center wave crest runs from (344, 442) up-right to (404, 405)
            wave_top = 442.0 - (x - 344.0) * (37.0 / 60.0)
            if pc_dil[y, x] > 0 and y < wave_top:
                pc_sky[y, x] = 1
    blend_sky_patch(bare, pc_sky)
    wire_pts = np.array([[304, 282], [338, 300], [374, 311], [412, 317], [446, 320]], dtype=np.int32)
    cv2.polylines(bare, [wire_pts], False, ink, 2, cv2.LINE_AA)
    for y in range(290, 364):
        for x in range(304, 434):
            if protect_c[y, x] > 0:
                bare[y, x] = bg_orig[y, x]

    for y in range(398, 540):
        for x in range(340, 405):
            if pc_dil[y, x] == 0:
                continue
            dock_y = 548.0 - (x - 306.0) * (70.0 / 138.0)
            wave_top = 442.0 - (x - 344.0) * (37.0 / 60.0)
            if y < wave_top:
                continue
            elif y < dock_y:
                sx, sy = max(0, x - 58), min(H - 1, y + 34)
                bare[y, x] = bg_orig[sy, sx]
            else:
                sx, sy = min(W - 1, x + 50), min(H - 1, y + 30)
                bare[y, x] = bg_orig[sy, sx]

    # Re-draw crisp boardwalk top edge ink lines across the 3 cleared narrow trunk bases
    cv2.line(bare, (68, 676), (116, 647), ink, 2, cv2.LINE_AA)
    cv2.line(bare, (344, 528), (400, 495), ink, 2, cv2.LINE_AA)
    cv2.line(bare, (786, 638), (836, 670), ink, 2, cv2.LINE_AA)

    # Update summer_market stickers
    json_path = os.path.join(ios_dir, "level_summer_market.json")
    with open(json_path, "r", encoding="utf-8") as f:
        ldata = json.load(f)

    replace_ids = {"bg_palm_left", "bg_palm_center", "bg_palm_right",
                   "umbrella_picnic", "picnic_blanket_food", "beach_umbrella",
                   "beach_balls", "beach_ball_left", "beach_ball_right",
                   "animal_cat_fish", "animal_calico_cat", "grilled_fish",
                   "sky_fireworks", "fireworks_left", "fireworks_right"}
    for rid in replace_ids:
        old_p = os.path.join(ios_dir, f"{rid}.png")
        if os.path.exists(old_p):
            os.remove(old_p)
    ldata["stickers"] = [s for s in ldata["stickers"] if s["id"] not in replace_ids]

    diff_sm = np.linalg.norm(orig_sm.astype(np.float32) - bg_orig.astype(np.float32), axis=2)
    out_sm = get_outline_mask(orig_sm)

    def sm_extract(roi, thresh=24, keep_largest=True, min_area=80, exclude_polys=None, fill_holes=True):
        x0, y0, x1, y1 = roi
        sub = (diff_sm[y0:y1, x0:x1] > thresh).astype(np.uint8)
        if exclude_polys:
            for poly in exclude_polys:
                pts = np.array([[px - x0, py - y0] for px, py in poly], dtype=np.int32)
                cv2.fillPoly(sub, [pts], 0)
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        nl, lbl, st, _ = cv2.connectedComponentsWithStats(sub)
        cleaned = np.zeros_like(sub)
        if nl > 1:
            if keep_largest:
                best = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                cleaned[lbl == best] = 1
            else:
                for i in range(1, nl):
                    if st[i, cv2.CC_STAT_AREA] >= min_area:
                        cleaned[lbl == i] = 1
        if fill_holes:
            cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
        dil = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cleaned = np.where((cleaned > 0) | ((dil > 0) & (out_sm[y0:y1, x0:x1] > 0)), 1, 0).astype(np.uint8)
        if fill_holes:
            cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
        if exclude_polys:
            for poly in exclude_polys:
                pts = np.array([[px - x0, py - y0] for px, py in poly], dtype=np.int32)
                cv2.fillPoly(cleaned, [pts], 0)
        alpha = np.zeros((H, W), dtype=np.uint8)
        alpha[y0:y1, x0:x1] = smooth_alpha_from_binary(cleaned, blur_sigma=0.55)
        return alpha

    def push_sm(sid, nzh, nen, cat, z, alpha, src_img=orig_sm):
        fn = f"{sid}.png"
        x, y, w, h = save_rgba_from_mask(src_img, alpha, os.path.join(ios_dir, fn), pad=2)
        ldata["stickers"].append({
            "id": sid, "file": fn, "name_zh": nzh, "name_en": nen, "category": cat, "zIndex": z,
            "x": int(x), "y": int(y), "width": int(w), "height": int(h),
            "normX": round((x + w / 2.0) / W, 5), "normY": round((y + h / 2.0) / H, 5),
            "normW": round(w / W, 5), "normH": round(h / H, 5),
        })

    # 3 Stickable Seaside Coconut Palm Trees
    push_sm("bg_palm_left", "左侧海边椰子树", "Left Beach Palm", "营地设施", 8, a_palm_l, src_img=bg_orig)
    push_sm("bg_palm_center", "居中海浪椰子树", "Center Beach Palm", "营地设施", 8, a_palm_c, src_img=bg_orig)
    push_sm("bg_palm_right", "右侧海边椰子树", "Right Beach Palm", "营地设施", 8, a_palm_r, src_img=bg_orig)

    # CLEAN beach_umbrella + picnic_blanket_food
    push_sm("beach_umbrella", "彩虹条纹遮阳伞", "Rainbow Beach Umbrella", "营地设施", 23,
            clip_alpha_to_poly(
                sm_extract((68, 566, 218, 734), thresh=24, keep_largest=True,
                           exclude_polys=[[(196, 566), (218, 566), (218, 678), (196, 678)],
                                          [(68, 678), (136, 678), (136, 734), (68, 734)],
                                          [(186, 668), (218, 668), (218, 734), (186, 734)]]),
                [(68, 566), (216, 566), (216, 662), (186, 680), (186, 732), (126, 732), (126, 678), (68, 668)]
            ))
    push_sm("picnic_blanket_food", "海边野餐垫与提灯", "Beach Picnic Blanket", "营地设施", 21,
            sm_extract((96, 674, 340, 842), thresh=24, keep_largest=False, min_area=120,
                       exclude_polys=[[(254, 674), (326, 674), (326, 760), (264, 762), (252, 720)],
                                      [(132, 674), (182, 674), (182, 722), (132, 722)],
                                      [(272, 796), (340, 796), (340, 842), (272, 842)]]))

    # Beach balls
    push_sm("beach_ball_left", "左沙滩彩球", "Left Beach Ball", "露营道具", 40,
            sm_extract((340, 860, 400, 920), thresh=24, keep_largest=True))
    push_sm("beach_ball_right", "右沙滩彩球", "Right Beach Ball", "露营道具", 40,
            sm_extract((574, 728, 634, 790), thresh=24, keep_largest=True))

    # Calico cat & grilled fish — clean outline-enclosed extraction with morphological cleanup so ZERO plank ticks stick out!
    def sm_outline_enclosed(roi, seeds):
        x0, y0, x1, y1 = roi
        sub_out = out_sm[y0:y1, x0:x1].copy()
        wall = cv2.morphologyEx(sub_out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        sh, sw = wall.shape
        padded = np.zeros((sh + 2, sw + 2), dtype=np.uint8)
        padded[1:-1, 1:-1] = wall
        ff = np.zeros((sh + 4, sw + 4), dtype=np.uint8)
        for sx, sy in seeds:
            rx, ry = max(0, min(sw - 1, sx - x0)), max(0, min(sh - 1, sy - y0))
            if padded[ry + 1, rx + 1] == 0:
                cv2.floodFill(padded, ff, (rx + 1, ry + 1), 2)
        fg = (padded[1:-1, 1:-1] != 2).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        nl, lbl, st, _ = cv2.connectedComponentsWithStats(fg)
        cleaned = np.zeros_like(fg)
        if nl > 1:
            best = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            cleaned[lbl == best] = 1
        dil = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cleaned = np.where((cleaned > 0) | ((dil > 0) & (sub_out > 0)), 1, 0).astype(np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
        alpha = np.zeros((H, W), dtype=np.uint8)
        alpha[y0:y1, x0:x1] = smooth_alpha_from_binary(cleaned, blur_sigma=0.55)
        return alpha

    push_sm("animal_calico_cat", "馋嘴小花猫", "Hungry Calico Cat", "可爱动物", 32,
            sm_outline_enclosed((642, 666, 690, 742),
                                [(644, 668), (688, 668), (644, 740), (688, 740)]))
    push_sm("grilled_fish", "烤香小海鱼", "Grilled Little Fish", "露营道具", 32,
            sm_outline_enclosed((712, 700, 760, 736),
                                [(714, 702), (758, 702), (714, 734), (758, 734)]))

    # CLEAN Sky Fireworks — STRICTLY in the sky (ZERO stall roofs, ZERO seagull, ZERO moon!)
    push_sm("fireworks_left", "左侧海边烟花", "Left Beach Fireworks", "星空光效", 49,
            sm_extract((102, 198, 446, 368), thresh=26, keep_largest=False, min_area=20, fill_holes=False))
    push_sm("fireworks_right", "右侧海边烟花", "Right Beach Fireworks", "星空光效", 49,
            sm_extract((508, 224, 780, 355), thresh=26, keep_largest=False, min_area=20, fill_holes=False,
                       exclude_polys=[[(508, 300), (620, 300), (620, 355), (508, 355)]]))

    # SPLIT the 3 stalls -> empty stall frame + vendor + counter goods (layer-by-layer!)
    backup_dir = os.path.join(PIPE, "backup_summer_stalls")
    os.makedirs(backup_dir, exist_ok=True)
    stall_split = {
        "stall_takoyaki": [
            ("takoyaki_chef", "章鱼烧小厨师", "Takoyaki Chef", "卡通人物", 17,
             [[(268, 482), (324, 482), (330, 524), (350, 528), (350, 538), (318, 546), (290, 554), (274, 572), (266, 558)]]),
            ("takoyaki_grill", "章鱼烧烤盘与酱料", "Takoyaki Grill & Sauces", "露营道具", 19,
             [[(266, 568), (300, 566), (330, 546), (356, 522), (396, 514), (396, 560), (374, 580),
               (352, 614), (300, 614), (266, 608)]]),
            ("takoyaki_lanterns", "摊位双灯笼", "Stall Lanterns", "星空光效", 19,
             [[(220, 498), (264, 498), (264, 574), (220, 574)], [(370, 436), (404, 436), (404, 504), (370, 504)]]),
        ],
        "stall_shaved_ice": [
            ("shaved_ice_drinks", "刨冰机与果汁罐", "Shaved Ice Machine & Juice Jars", "露营道具", 19,
             [[(494, 478), (588, 490), (588, 566), (494, 566)]]),
        ],
        "stall_goldfish": [
            ("goldfish_vendor", "捞金鱼摊主女孩", "Goldfish Stall Girl", "卡通人物", 19,
             [[(686, 546), (738, 546), (742, 600), (700, 608), (686, 592)]]),
            ("goldfish_pinwheel", "彩色风车", "Pinwheel", "露营道具", 21,
             [[(654, 544), (686, 544), (686, 606), (654, 606)]]),
            ("goldfish_tank", "金鱼水槽", "Goldfish Tank", "露营道具", 22,
             [[(634, 604), (706, 604), (744, 632), (744, 692), (700, 702), (634, 682)]]),
        ],
    }
    child_ids = {c[0] for v in stall_split.values() for c in v}
    ldata["stickers"] = [s for s in ldata["stickers"] if s["id"] not in child_ids]
    for parent_id, children in stall_split.items():
        ps = next(s for s in ldata["stickers"] if s["id"] == parent_id)
        bak = os.path.join(backup_dir, ps["file"])
        if not os.path.exists(bak):
            shutil.copy2(os.path.join(ios_dir, ps["file"]), bak)
        prgba = cv2.imread(bak, cv2.IMREAD_UNCHANGED)
        # Recover original placement from backup JSON record (parent box may shrink after split)
        meta_p = bak + ".json"
        if os.path.exists(meta_p):
            with open(meta_p) as f:
                px, py = json.load(f)
        else:
            px, py = ps["x"], ps["y"]
            with open(meta_p, "w") as f:
                json.dump([px, py], f)
        ph, pw = prgba.shape[:2]
        full_alpha = np.zeros((H, W), dtype=np.uint8)
        full_alpha[py:py+ph, px:px+pw] = prgba[:, :, 3]
        full_rgb = orig_sm.copy()
        full_rgb[py:py+ph, px:px+pw] = prgba[:, :, :3]
        remain = full_alpha.copy()
        for cid, nzh, nen, cat, z, polys in children:
            cmask = np.zeros((H, W), dtype=np.uint8)
            for poly in polys:
                cv2.fillPoly(cmask, [np.array(poly, dtype=np.int32)], 255)
            ca = cv2.bitwise_and(full_alpha, cmask)
            remain = cv2.bitwise_and(remain, cv2.bitwise_not(cmask))
            push_sm(cid, nzh, nen, cat, z, ca, src_img=full_rgb)
        # Overwrite parent stall with the empty stall frame
        ldata["stickers"] = [s for s in ldata["stickers"] if s["id"] != parent_id]
        os.remove(os.path.join(ios_dir, ps["file"]))
        push_sm(parent_id, ps["name_zh"], ps["name_en"], ps["category"], ps["zIndex"], remain, src_img=full_rgb)

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
    side = np.hstack([cv2.resize(bare, (448, 600)), cv2.resize(comp_u8, (448, 600)), cv2.resize(orig_sm, (448, 600))])
    cv2.imwrite(os.path.join(PIPE, "verify_summer_market.png"), side)
    shutil.copy2(os.path.join(PIPE, "verify_summer_market.png"), os.path.join(BRAIN, "verify_summer_market.png"))
    render_contact_sheet(ios_dir, ldata["stickers"], "sheet_summer_market.png")
    print(f"  => Upgraded summer_market: {len(ldata['stickers'])} stickers (SEAMLESS 3 palms + spotless fireworks!)")


# ==============================================================================
# 3. CAMPFIRE PARTY — SMOOTH DOME BORDER + 2 CLEAN FRONT PINES + 2 STUMPS (34 STICKERS!)
# ==============================================================================
def upgrade_campfire_party():
    ios_dir = os.path.join(IOS_LEVELS, "campfire_party")
    orig_cf = cv2.imread(os.path.join(BRAIN, "campfire_diorama_scene_1795488810869.jpg"))
    bg_orig = cv2.imread(os.path.join(BRAIN, "campfire_diorama_empty_bg_1795490181683.jpg"))
    H, W = bg_orig.shape[:2]
    out_bg = get_outline_mask(bg_orig, r_max=120, g_max=100, b_max=88)
    b, g, r = bg_orig[:, :, 0].astype(np.int16), bg_orig[:, :, 1].astype(np.int16), bg_orig[:, :, 2].astype(np.int16)
    navy_outer = tuple(int(v) for v in bg_orig[50, 50])
    ink = (36, 46, 68)

    # 1) Fit smooth meadow grass around the 2 tree stumps (tight dilated stump masks -> ZERO rectangular tint!)
    grass_sample_l = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(grass_sample_l, (100, 770), (310, 945), 1, -1)
    cv2.rectangle(grass_sample_l, (145, 836), (255, 922), 0, -1)
    grass_sample_l = grass_sample_l & ((g > r + 10) & (g > b + 20)).astype(np.uint8)

    grass_sample_r = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(grass_sample_r, (670, 740), (842, 910), 1, -1)
    cv2.rectangle(grass_sample_r, (732, 790), (842, 876), 0, -1)
    grass_sample_r = grass_sample_r & ((g > r + 10) & (g > b + 20)).astype(np.uint8)

    stump_l_box = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(stump_l_box, (146, 838), (254, 920), 1, -1)
    stump_r_box = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(stump_r_box, (734, 792), (836, 872), 1, -1)

    grass_fit_l = fit_poly2_surface(bg_orig, grass_sample_l, stump_l_box)
    grass_fit_r = fit_poly2_surface(bg_orig, grass_sample_r, stump_r_box)

    # Extract tight stump masks first by diffing bg_orig vs grass_fit
    def extract_stump(roi, fit_img, max_x=None):
        x0, y0, x1, y1 = roi
        diff_s = np.linalg.norm(bg_orig[y0:y1, x0:x1].astype(np.float32) - fit_img[y0:y1, x0:x1].astype(np.float32), axis=2)
        sub = (diff_s > 22).astype(np.uint8)
        if max_x is not None:
            sub[:, max(0, max_x - x0):] = 0
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        nl, lbl, st, _ = cv2.connectedComponentsWithStats(sub)
        cleaned = np.zeros_like(sub)
        if nl > 1:
            best = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            cleaned[lbl == best] = 1
        dil = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cleaned = np.where((cleaned > 0) | ((dil > 0) & (out_bg[y0:y1, x0:x1] > 0)), 1, 0).astype(np.uint8)
        cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
        if max_x is not None:
            cleaned[:, max(0, max_x - x0):] = 0
        alpha = np.zeros((H, W), dtype=np.uint8)
        alpha[y0:y1, x0:x1] = smooth_alpha_from_binary(cleaned, blur_sigma=0.55)
        return alpha

    a_stump_l = extract_stump((146, 838, 254, 920), grass_fit_l)
    a_stump_r = extract_stump((734, 792, 836, 872), grass_fit_r, max_x=829)

    bare = bg_orig.copy()
    # Replace ONLY the tight dilated stump areas (zero rectangle!)
    for a_st, fit_im in [(a_stump_l, grass_fit_l), (a_stump_r, grass_fit_r)]:
        st_dil = cv2.dilate((a_st > 16).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        m = (cv2.GaussianBlur((st_dil * 255).astype(np.float32), (9, 9), 2.0) / 255.0)[:, :, None]
        bare = (fit_im.astype(np.float32) * m + bare.astype(np.float32) * (1.0 - m)).astype(np.uint8)

    # 2) Extract 2 front flanking pine trees SOLIDLY using color + outline inside keep_poly (ZERO floodFill leak, ZERO bunting!)
    def extract_strict_pine(roi, keep_poly):
        x0, y0, x1, y1 = roi
        cb, cg, cr = b[y0:y1, x0:x1], g[y0:y1, x0:x1], r[y0:y1, x0:x1]
        sub_out = out_bg[y0:y1, x0:x1]
        # Pine foliage (green) + pine trunk (warm brown)
        is_pine_green = ((cg > cb + 8) & (cg > cr - 15) & (cr < 185)).astype(np.uint8)
        is_pine_brown = ((cr > 95) & (cr < 210) & (cg > 55) & (cg < 160) & (cb > 32) & (cb < 115) & ((cr - cb) > 35)).astype(np.uint8)
        fg = ((is_pine_green > 0) | (is_pine_brown > 0)).astype(np.uint8)
        kp = np.zeros_like(fg)
        cv2.fillPoly(kp, [np.array([[px - x0, py - y0] for px, py in keep_poly], dtype=np.int32)], 1)
        fg = fg & kp
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        nl, lbl, st, _ = cv2.connectedComponentsWithStats(fg)
        cleaned = np.zeros_like(fg)
        if nl > 1:
            best = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            cleaned[lbl == best] = 1
        dil = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        cleaned = np.where((cleaned > 0) | ((dil > 0) & (sub_out > 0) & (kp > 0)), 1, 0).astype(np.uint8)
        cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
        alpha = np.zeros((H, W), dtype=np.uint8)
        alpha[y0:y1, x0:x1] = smooth_alpha_from_binary(cleaned, blur_sigma=0.55)
        return alpha

    left_pine_poly = [
        (106, 314), (132, 344), (156, 404), (186, 466), (200, 526),
        (194, 560), (154, 578), (150, 642), (76, 642), (82, 576),
        (26, 558), (26, 490), (62, 392)
    ]
    right_pine_poly = [
        (792, 318), (766, 348), (740, 412), (710, 476), (696, 534),
        (706, 566), (748, 578), (754, 628), (824, 628), (818, 566),
        (870, 548), (870, 480), (836, 396)
    ]
    a_pine_lf = extract_strict_pine((24, 312, 204, 646), left_pine_poly)
    a_pine_rf = extract_strict_pine((692, 316, 874, 632), right_pine_poly)

    # Smooth sky & local edge-matched meadow grass
    sky_s_cf = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(sky_s_cf, (170, 140), (340, 260), 1, -1)
    cv2.rectangle(sky_s_cf, (560, 140), (720, 260), 1, -1)
    sky_s_cf = sky_s_cf & ((b > r + 25) & (r < 95)).astype(np.uint8)
    sky_grad_cf = build_vertical_gradient(bg_orig, sky_s_cf, 120, 560)

    # Sample smooth 2D meadow grass surface from unobstructed meadow so grass blends with ZERO seam!
    meadow_s = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(meadow_s, (160, 525), (740, 740), 1, -1)
    cv2.rectangle(meadow_s, (54, 645), (844, 760), 1, -1)
    meadow_s = meadow_s & ((g > r + 8) & (g > b + 18)).astype(np.uint8)
    meadow_fill_box = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(meadow_fill_box, (45, 490), (210, 650), 1, -1)
    cv2.rectangle(meadow_fill_box, (688, 490), (854, 650), 1, -1)
    meadow_fit = fit_poly2_surface(bg_orig, meadow_s, meadow_fill_box)

    # Exact measured outer white border joining (111, 315) -> (49, 620) on left & (781, 315) -> (849, 620) on right!
    ly_pts = np.array([315, 360, 420, 485, 550, 600, 622], dtype=np.float32)
    lx_pts = np.array([111, 96,  77,  62,  54,  50,  49], dtype=np.float32)
    ry_pts = np.array([315, 360, 420, 485, 550, 600, 622], dtype=np.float32)
    rx_pts = np.array([781, 796, 816, 833, 843, 848, 849], dtype=np.float32)

    plf_d = cv2.dilate((a_pine_lf > 12).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    prf_d = cv2.dilate((a_pine_rf > 12).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    # Include any inward white/navy/ink notch left under the lowest bough (y=545..615)
    is_notch = (((b > 180) & (g > 180) & (r > 180)) | ((b > r + 20) & (r < 90)) | (out_bg > 0)).astype(np.uint8)
    for y in range(545, 618):
        x_in_l = int(np.interp(y, ly_pts, lx_pts))
        x_in_r = int(np.interp(y, ry_pts, rx_pts))
        for x in range(x_in_l + 1, x_in_l + 38):
            if is_notch[y, x]:
                plf_d[y, x] = 1
        for x in range(x_in_r - 38, x_in_r):
            if is_notch[y, x]:
                prf_d[y, x] = 1
    plf_d = cv2.dilate(plf_d, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    prf_d = cv2.dilate(prf_d, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Clear left front pine + seamless left outer border
    for y in range(315, 648):
        x_in = float(np.interp(y, ly_pts, lx_pts))
        for x in range(16, 206):
            if y <= 622:
                if x < x_in - 10.5:
                    bare[y, x] = navy_outer
                    continue
                elif x < x_in - 1.5:
                    bare[y, x] = (255, 255, 255)
                    continue
                elif x < x_in + 1.5:
                    bare[y, x] = ink
                    continue
            if plf_d[y, x] > 0:
                hy = 556.0 - (x - 49.0) * (60.0 / 157.0)
                if y < hy - 1.5:
                    bare[y, x] = sky_grad_cf[y, x]
                elif y < hy + 1.5:
                    bare[y, x] = ink
                else:
                    bare[y, x] = meadow_fit[y, x]

    # Clear right front pine + seamless right outer border
    for y in range(315, 642):
        x_in = float(np.interp(y, ry_pts, rx_pts))
        for x in range(690, 882):
            if y <= 622:
                if x > x_in + 10.5:
                    bare[y, x] = navy_outer
                    continue
                elif x > x_in + 1.5:
                    bare[y, x] = (255, 255, 255)
                    continue
                elif x > x_in - 1.5:
                    bare[y, x] = ink
                    continue
            if prf_d[y, x] > 0:
                hy = 496.0 + (x - 690.0) * (60.0 / 159.0)
                if y < hy - 1.5:
                    bare[y, x] = sky_grad_cf[y, x]
                elif y < hy + 1.5:
                    bare[y, x] = ink
                else:
                    bare[y, x] = meadow_fit[y, x]

    # Subtle twinkling stars in newly opened sky
    for sx, sy in [(95, 385), (124, 442), (776, 382), (808, 438)]:
        cv2.circle(bare, (sx, sy), 2, (235, 240, 255), -1, cv2.LINE_AA)

    json_path = os.path.join(ios_dir, "level_campfire.json")
    with open(json_path, "r", encoding="utf-8") as f:
        ldata = json.load(f)

    new_ids = {"bg_stump_left", "bg_stump_right", "bg_pine_left_front", "bg_pine_right_front"}
    ldata["stickers"] = [s for s in ldata["stickers"] if s["id"] not in new_ids]

    for sid, nzh, nen, cat, z, alpha in [
        ("bg_pine_left_front", "左侧前景大松树", "Front Pine Tree Left", "营地设施", 6, a_pine_lf),
        ("bg_pine_right_front", "右侧前景大松树", "Front Pine Tree Right", "营地设施", 6, a_pine_rf),
        ("bg_stump_left", "左侧草地大树桩", "Left Meadow Stump", "营地设施", 8, a_stump_l),
        ("bg_stump_right", "右侧草地大树桩", "Right Meadow Stump", "营地设施", 8, a_stump_r),
    ]:
        fn = f"{sid}.png"
        x, y, w, h = save_rgba_from_mask(bg_orig, alpha, os.path.join(ios_dir, fn), pad=2)
        ldata["stickers"].append({
            "id": sid, "file": fn, "name_zh": nzh, "name_en": nen, "category": cat, "zIndex": z,
            "x": int(x), "y": int(y), "width": int(w), "height": int(h),
            "normX": round((x + w / 2.0) / W, 5), "normY": round((y + h / 2.0) / H, 5),
            "normW": round(w / W, 5), "normH": round(h / H, 5),
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
    side = np.hstack([cv2.resize(bare, (448, 600)), cv2.resize(comp_u8, (448, 600)), cv2.resize(orig_cf, (448, 600))])
    cv2.imwrite(os.path.join(PIPE, "verify_campfire_party.png"), side)
    shutil.copy2(os.path.join(PIPE, "verify_campfire_party.png"), os.path.join(BRAIN, "composite_verify.png"))
    render_contact_sheet(ios_dir, ldata["stickers"], "sticker_sheet_preview.png")
    print(f"  => Upgraded campfire_party: {len(ldata['stickers'])} stickers (smooth dome border + 2 pines + 2 stumps!)")


def sync_3_to_docs():
    with open(os.path.join(IOS_LEVELS, "levels_manifest.json"), "r", encoding="utf-8") as f:
        ios_m = json.load(f)

    upgraded_ids = {"cat_cafe", "summer_market", "campfire_party"}
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
    upgrade_cat_cafe()
    upgrade_summer_market()
    upgrade_campfire_party()
    sync_3_to_docs()
