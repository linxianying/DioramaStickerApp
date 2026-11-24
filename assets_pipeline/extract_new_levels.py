import json
import os
import cv2
import numpy as np
from scipy import ndimage

BRAIN = "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p_devbox/ff1be343-f26a-4dda-8b06-9a95d3c4f1c1"
PIPE = "/home/antigravity/DioramaStickerApp/assets_pipeline"
IOS_LEVELS = "/home/antigravity/DioramaStickerApp/iOS/DioramaSticker/DioramaSticker/Resources/Levels"


def get_outline_mask(img, r_max=115, g_max=95, b_max=85):
    b, g, r = cv2.split(img)
    return ((r < r_max) & (g < g_max) & (b < b_max) & (r.astype(np.int16) >= b.astype(np.int16) - 12)).astype(np.uint8)


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


class LevelExtractor:
    def __init__(self, level_id, title_zh, title_en, subtitle_zh, orig_file, bg_file):
        self.level_id = level_id
        self.title_zh = title_zh
        self.title_en = title_en
        self.subtitle_zh = subtitle_zh
        self.orig = cv2.imread(os.path.join(BRAIN, orig_file))
        self.bg = cv2.imread(os.path.join(BRAIN, bg_file))
        if self.bg.shape != self.orig.shape:
            self.bg = cv2.resize(self.bg, (self.orig.shape[1], self.orig.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        self.H, self.W = self.orig.shape[:2]
        self.diff = np.linalg.norm(self.orig.astype(np.float32) - self.bg.astype(np.float32), axis=2)
        self.outline = get_outline_mask(self.orig)
        self.outline_strong = get_outline_mask(self.orig, r_max=130, g_max=108, b_max=92)
        self.out_dir = os.path.join(PIPE, f"extracted_{level_id}")
        os.makedirs(self.out_dir, exist_ok=True)
        self.stickers = []

    def extract_outline_enclosed(self, roi, bg_seeds=None, exclude_polys=None, close_outline=3, dilate_out=2):
        x0, y0, x1, y1 = roi
        sub_out = self.outline_strong[y0:y1, x0:x1].copy()
        sh, sw = sub_out.shape
        if exclude_polys:
            for poly in exclude_polys:
                pts = np.array([[px - x0, py - y0] for px, py in poly], dtype=np.int32)
                cv2.fillPoly(sub_out, [pts], 0)
        wall = cv2.morphologyEx(sub_out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_outline, close_outline))) if close_outline > 1 else sub_out.copy()
        padded = np.zeros((sh + 2, sw + 2), dtype=np.uint8)
        padded[1:-1, 1:-1] = wall
        ff_mask = np.zeros((sh + 4, sw + 4), dtype=np.uint8)
        cv2.floodFill(padded, ff_mask, (0, 0), 2)
        if bg_seeds:
            for (sx, sy) in bg_seeds:
                lx, ly = sx - x0 + 1, sy - y0 + 1
                if 0 <= lx < sw + 2 and 0 <= ly < sh + 2 and padded[ly, lx] == 0:
                    cv2.floodFill(padded, ff_mask, (lx, ly), 2)
        fg = (padded[1:-1, 1:-1] != 2).astype(np.uint8)
        if exclude_polys:
            for poly in exclude_polys:
                pts = np.array([[px - x0, py - y0] for px, py in poly], dtype=np.int32)
                cv2.fillPoly(fg, [pts], 0)
        fg_core = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        num_l, labels, stats, _ = cv2.connectedComponentsWithStats(fg_core)
        cleaned = np.zeros_like(fg_core)
        if num_l > 1:
            best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            cleaned[labels == best] = 1
        dil_big = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        cleaned = np.where((cleaned > 0) | ((dil_big > 0) & (fg > 0)), 1, 0).astype(np.uint8)
        if dilate_out > 0:
            dil = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_out + 1, 2 * dilate_out + 1)))
            cleaned = np.where((cleaned > 0) | ((dil > 0) & (sub_out > 0)), 1, 0).astype(np.uint8)
        cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
        if exclude_polys:
            for poly in exclude_polys:
                pts = np.array([[px - x0, py - y0] for px, py in poly], dtype=np.int32)
                cv2.fillPoly(cleaned, [pts], 0)
        alpha_full = np.zeros((self.H, self.W), dtype=np.uint8)
        alpha_full[y0:y1, x0:x1] = smooth_alpha_from_binary(cleaned, blur_sigma=0.55)
        return alpha_full

    def extract_diff(self, roi, thresh=28, keep_largest=True, min_area=120,
                     exclude_polys=None, include_outline=True, soft_glow=False, close_k=3, fill_holes=True):
        x0, y0, x1, y1 = roi
        sub_diff = self.diff[y0:y1, x0:x1].copy()

        if soft_glow:
            alpha_sub = np.clip((sub_diff - 12.0) / 48.0, 0.0, 1.0)
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
            core_dil = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
            alpha_sub = np.where(core_dil > 0, alpha_sub, 0.0)
            alpha_full = np.zeros((self.H, self.W), dtype=np.uint8)
            alpha_full[y0:y1, x0:x1] = np.clip(alpha_sub * 255.0, 0, 255).astype(np.uint8)
            return alpha_full

        sub_mask = (sub_diff > thresh).astype(np.uint8)
        if exclude_polys:
            for poly in exclude_polys:
                pts = np.array([[px - x0, py - y0] for px, py in poly], dtype=np.int32)
                cv2.fillPoly(sub_mask, [pts], 0)

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

        if fill_holes:
            cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)

        if include_outline:
            sub_out = self.outline[y0:y1, x0:x1]
            dil = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            cleaned = np.where((cleaned > 0) | ((dil > 0) & (sub_out > 0)), 1, 0).astype(np.uint8)
            if fill_holes:
                cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)

        if exclude_polys:
            for poly in exclude_polys:
                pts = np.array([[px - x0, py - y0] for px, py in poly], dtype=np.int32)
                cv2.fillPoly(cleaned, [pts], 0)

        alpha_full = np.zeros((self.H, self.W), dtype=np.uint8)
        alpha_full[y0:y1, x0:x1] = smooth_alpha_from_binary(cleaned, blur_sigma=0.55)
        return alpha_full

    def extract_multi_rois(self, rois, thresh=28, min_area=100, soft_glow=False, exclude_polys=None, fill_holes=True):
        alpha_full = np.zeros((self.H, self.W), dtype=np.uint8)
        for r in rois:
            a = self.extract_diff(r, thresh=thresh, keep_largest=False, min_area=min_area,
                                  soft_glow=soft_glow, exclude_polys=exclude_polys, fill_holes=fill_holes)
            alpha_full = np.maximum(alpha_full, a)
        return alpha_full

    def add(self, sid, name_zh, name_en, category, z_index, alpha_full):
        out_file = f"{sid}.png"
        out_path = os.path.join(self.out_dir, out_file)
        x, y, w, h = save_rgba_from_mask(self.orig, alpha_full, out_path, pad=2)
        self.stickers.append({
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
            "normX": round((x + w / 2.0) / self.W, 5),
            "normY": round((y + h / 2.0) / self.H, 5),
            "normW": round(w / self.W, 5),
            "normH": round(h / self.H, 5),
        })

    def finalize(self):
        cv2.imwrite(os.path.join(self.out_dir, "background.png"), self.bg)
        ordered = sorted(self.stickers, key=lambda s: (s["zIndex"], s["id"]))
        level_data = {
            "levelId": self.level_id,
            "title_zh": self.title_zh,
            "title_en": self.title_en,
            "subtitle_zh": self.subtitle_zh,
            "canvasWidth": self.W,
            "canvasHeight": self.H,
            "backgroundFile": "background.png",
            "stickers": ordered
        }
        json_name = f"level_{self.level_id}.json"
        with open(os.path.join(self.out_dir, json_name), "w", encoding="utf-8") as f:
            json.dump(level_data, f, ensure_ascii=False, indent=2)

        comp = self.bg.copy().astype(np.float32)
        for s in ordered:
            rgba = cv2.imread(os.path.join(self.out_dir, s["file"]), cv2.IMREAD_UNCHANGED)
            x, y, w, h = s["x"], s["y"], s["width"], s["height"]
            alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
            comp[y:y+h, x:x+w] = rgba[:, :, :3].astype(np.float32) * alpha + comp[y:y+h, x:x+w] * (1.0 - alpha)
        comp_u8 = np.clip(comp, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(self.out_dir, "reference_complete.png"), comp_u8)

        n = len(self.stickers)
        cols = 5
        rows = (n + cols - 1) // cols
        cell_w, cell_h = 210, 220
        sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        for r in range(0, rows * cell_h, 16):
            for c in range(0, cols * cell_w, 16):
                sheet[r:r+16, c:c+16] = (68, 56, 50) if ((r // 16 + c // 16) % 2 == 0) else (84, 70, 62)

        for idx, s in enumerate(self.stickers):
            r, c = divmod(idx, cols)
            rgba = cv2.imread(os.path.join(self.out_dir, s["file"]), cv2.IMREAD_UNCHANGED)
            sh, sw = rgba.shape[:2]
            max_w, max_h = cell_w - 22, cell_h - 44
            scale = min(max_w / sw, max_h / sh, 1.35)
            nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
            res = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)
            cx = c * cell_w + (cell_w - nw) // 2
            cy = r * cell_h + 10 + (max_h - nh) // 2
            alpha = res[:, :, 3:4].astype(np.float32) / 255.0
            roi = sheet[cy:cy+nh, cx:cx+nw].astype(np.float32)
            sheet[cy:cy+nh, cx:cx+nw] = (res[:, :, :3].astype(np.float32) * alpha + roi * (1.0 - alpha)).astype(np.uint8)
            cv2.rectangle(sheet, (c*cell_w, r*cell_h), ((c+1)*cell_w-1, (r+1)*cell_h-1), (110, 95, 85), 1)
            cv2.putText(sheet, f"{idx+1}. {s['id']}", (c*cell_w + 8, (r+1)*cell_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 240, 255), 1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(PIPE, f"sheet_{self.level_id}.png"), sheet)
        side = np.hstack([
            cv2.resize(self.bg, (448, 600)),
            cv2.resize(comp_u8, (448, 600)),
            cv2.resize(self.orig, (448, 600)),
        ])
        cv2.imwrite(os.path.join(PIPE, f"verify_{self.level_id}.png"), side)

        ios_dest = os.path.join(IOS_LEVELS, self.level_id)
        os.makedirs(ios_dest, exist_ok=True)
        for fn in os.listdir(self.out_dir):
            src_p = os.path.join(self.out_dir, fn)
            dst_p = os.path.join(ios_dest, fn)
            with open(src_p, "rb") as rf, open(dst_p, "wb") as wf:
                wf.write(rf.read())
        print(f"Polished & synced {self.level_id}: {len(self.stickers)} stickers!")


# ==============================================================================
# LEVEL 2: CAT CAFE BAKERY (polished)
# ==============================================================================
cc = LevelExtractor(
    "cat_cafe", "猫咖甜品店", "Cat Cafe Bakery", "草莓蛋糕与软萌猫咪",
    "cat_cafe_scene_v2_1795496255386.jpg", "cat_cafe_empty_bg_1795496305551.jpg"
)
cc.add("cat_tree", "猫爬架与猫咪", "Cat Tree Tower", "营地设施", 18,
       cc.extract_diff((68, 462, 214, 792), thresh=26, keep_largest=True,
                       exclude_polys=[[(196, 500), (214, 500), (214, 685), (196, 685)],
                                      [(198, 685), (214, 685), (214, 735), (198, 735)]]))
cc.add("cat_siamese", "窗台暹罗猫", "Siamese Cat", "可爱动物", 16,
       cc.extract_outline_enclosed((196, 498, 268, 572), bg_seeds=[(200, 502), (264, 502), (200, 568), (264, 568)]))
cc.add("cat_cream_milk", "窗台白猫奶罐", "Window Kitty & Milk", "可爱动物", 16,
       cc.extract_multi_rois([(686, 492, 768, 586), (760, 560, 798, 608)], thresh=28, min_area=120))
cc.add("baker_girl", "烘焙蛋糕师", "Strawberry Baker", "卡通人物", 24,
       cc.extract_diff((258, 494, 372, 684), thresh=28, keep_largest=True,
                       exclude_polys=[[(258, 608), (270, 608), (270, 684), (258, 684)],
                                      [(318, 666), (372, 666), (372, 684), (318, 684)]]))
cc.add("barista_boy", "拉花咖啡师", "Latte Barista", "卡通人物", 20,
       cc.extract_diff((474, 424, 574, 578), thresh=28, keep_largest=True))
cc.add("pastries_display", "甜品柜点心", "Cakes & Macarons", "露营道具", 22,
       cc.extract_diff((446, 548, 580, 678), thresh=28, keep_largest=False, min_area=60, fill_holes=False,
                       exclude_polys=[[(520, 548), (580, 548), (580, 576), (520, 576)]]))
cc.add("browse_girl", "挑甜品女孩", "Pastry Browser", "卡通人物", 28,
       cc.extract_diff((370, 540, 458, 704), thresh=28, keep_largest=True,
                       exclude_polys=[[(445, 540), (458, 540), (458, 600), (445, 600)],
                                      [(370, 662), (398, 662), (398, 704), (370, 704)]]))
cc.add("potted_plant_tall", "落地大绿植", "Potted Plant", "露营道具", 22,
       cc.extract_diff((590, 530, 668, 670), thresh=28, keep_largest=True,
                       exclude_polys=[[(648, 646), (668, 646), (668, 670), (648, 670)]]))
cc.add("potted_succulent", "迷你多肉", "Mini Succulent", "露营道具", 26,
       cc.extract_diff((562, 664, 612, 728), thresh=24, keep_largest=True))
cc.add("cat_trashbin", "猫耳收纳桶", "Cat Ear Bin", "露营道具", 22,
       cc.extract_diff((200, 608, 268, 682), thresh=28, keep_largest=True))
cc.add("cat_basket", "猫窝奶牛猫", "Basket Tuxedo Cat", "可爱动物", 26,
       cc.extract_diff((620, 632, 718, 715), thresh=26, keep_largest=True,
                       exclude_polys=[[(620, 632), (640, 632), (640, 650), (620, 650)],
                                      [(698, 694), (718, 694), (718, 715), (698, 715)]]))
cc.add("cat_white_cushion", "粉垫小白猫", "Fluffy White Cat", "可爱动物", 28,
       cc.extract_diff((412, 694, 522, 774), thresh=28, keep_largest=True,
                       exclude_polys=[[(412, 694), (452, 694), (452, 712), (412, 712)]]))
cc.add("table_left_duo", "奶茶撸猫桌", "Boba & Kitty Table", "卡通人物", 34,
       cc.extract_diff((198, 660, 404, 870), thresh=34, keep_largest=True,
                       exclude_polys=[[(344, 795), (404, 795), (404, 870), (344, 870)],
                                      [(198, 745), (214, 745), (214, 790), (198, 790)]]))
cc.add("table_right_reader", "阅读撸猫桌", "Reading Cat Table", "卡通人物", 34,
       cc.extract_diff((676, 628, 808, 806), thresh=32, keep_largest=True,
                       exclude_polys=[[(676, 628), (716, 628), (716, 702), (676, 702)]]))
cc.add("boy_sleeping_cat", "睡猫与男孩", "Boy & Sleepy Tabby", "卡通人物", 35,
       cc.extract_diff((562, 716, 712, 868), thresh=28, keep_largest=True,
                       exclude_polys=[[(562, 716), (610, 716), (610, 730), (562, 730)]]))
cc.add("rug_cat_play", "逗猫拍照地毯", "Cat Play Rug", "卡通人物", 38,
       cc.extract_diff((340, 774, 572, 968), thresh=26, keep_largest=True,
                       exclude_polys=[[(340, 774), (404, 774), (404, 802), (340, 802)]]))
cc.finalize()

# ==============================================================================
# LEVEL 3: SEASIDE SUMMER NIGHT MARKET (polished)
# ==============================================================================
sm = LevelExtractor(
    "summer_market", "海边夏日夜市", "Seaside Night Market", "烟花海风与美食摊位",
    "summer_market_scene_1795496209754.jpg", "summer_market_empty_bg_1795496258741.jpg"
)
sm.add("stall_takoyaki", "章鱼烧烤串摊", "Takoyaki Stall", "营地设施", 18,
       sm.extract_diff((190, 366, 408, 660), thresh=28, keep_largest=True,
                       exclude_polys=[[(190, 598), (266, 598), (266, 660), (190, 660)],
                                      [(328, 590), (408, 590), (408, 660), (328, 660)]]))
sm.add("stall_shaved_ice", "刨冰果汁摊", "Shaved Ice Stall", "营地设施", 18,
       sm.extract_diff((488, 366, 660, 628), thresh=28, keep_largest=True,
                       exclude_polys=[[(542, 354), (620, 354), (620, 412), (542, 412)],
                                      [(580, 562), (660, 562), (660, 628), (580, 628)],
                                      [(488, 495), (504, 495), (504, 570), (488, 570)]]))
sm.add("stall_goldfish", "捞金鱼风车摊", "Goldfish Stall", "营地设施", 20,
       sm.extract_diff((626, 440, 806, 718), thresh=28, keep_largest=True,
                       exclude_polys=[[(762, 550), (806, 550), (806, 696), (762, 696)],
                                      [(626, 562), (654, 562), (654, 678), (626, 678)],
                                      [(640, 666), (682, 666), (682, 718), (640, 718)]]))
sm.add("animal_seagull", "屋顶海鸥", "Roof Seagull", "可爱动物", 25,
       sm.extract_outline_enclosed((540, 354, 624, 428), bg_seeds=[(544, 358), (620, 358), (544, 424), (620, 424)]))
sm.add("surfboards", "彩色冲浪板", "Surfboards", "露营道具", 22,
       sm.extract_outline_enclosed((762, 550, 838, 698), bg_seeds=[(766, 554), (834, 554), (766, 694), (834, 694)]))
sm.add("umbrella_picnic", "海风遮阳伞野餐", "Beach Umbrella Picnic", "营地设施", 22,
       sm.extract_diff((74, 564, 336, 836), thresh=28, keep_largest=False, min_area=180,
                       exclude_polys=[[(198, 564), (336, 564), (336, 664), (198, 664)],
                                      [(254, 662), (336, 662), (336, 756), (254, 756)],
                                      [(268, 798), (336, 798), (336, 836), (268, 836)]]))
sm.add("cooler_back", "迷你冷饮箱", "Mini Cooler", "露营道具", 24,
       sm.extract_diff((198, 598, 266, 664), thresh=28, keep_largest=True,
                       exclude_polys=[[(198, 598), (214, 598), (214, 622), (198, 622)]]))
sm.add("cooler_front", "沙滩冰桶", "Beach Cooler", "露营道具", 38,
       sm.extract_outline_enclosed((268, 798, 344, 878), bg_seeds=[(272, 802), (340, 802), (272, 874), (340, 874)]))
sm.add("animal_shiba", "围巾柴犬", "Shiba Inu", "可爱动物", 28,
       sm.extract_outline_enclosed((254, 662, 338, 758), bg_seeds=[(258, 666), (334, 666), (258, 754), (334, 754)]))
sm.add("char_yukata_girl", "浴衣刨冰女孩", "Yukata Girl", "卡通人物", 26,
       sm.extract_outline_enclosed((408, 456, 504, 596), bg_seeds=[(412, 460), (500, 460), (412, 592), (500, 592)]))
sm.add("char_watermelon_boy", "西瓜气球男孩", "Watermelon Balloon Boy", "卡通人物", 30,
       sm.extract_diff((326, 586, 468, 732), thresh=30, keep_largest=True))
sm.add("char_goldfish_boy", "捞金鱼男孩", "Goldfish Boy", "卡通人物", 30,
       sm.extract_diff((578, 562, 658, 686), thresh=30, keep_largest=True))
sm.add("animal_cat_fish", "馋嘴猫与小鱼", "Kitty & Fish", "可爱动物", 32,
       np.maximum(
           sm.extract_outline_enclosed((638, 660, 684, 742), bg_seeds=[(642, 664), (680, 664), (642, 738), (680, 738)]),
           sm.extract_outline_enclosed((712, 696, 762, 728), bg_seeds=[(716, 700), (758, 700), (716, 724), (758, 724)])
       ))
sm.add("char_duck_kid", "鸭子泳圈萌娃", "Duck Float Kid", "卡通人物", 32,
       sm.extract_outline_enclosed((464, 646, 554, 756), bg_seeds=[(468, 650), (550, 650), (468, 752), (550, 752)]))
sm.add("char_ukulele_boy", "尤克里里男孩", "Ukulele Boy", "卡通人物", 38,
       sm.extract_outline_enclosed((368, 742, 480, 882), bg_seeds=[(372, 746), (476, 746), (372, 878), (476, 878)]))
sm.add("char_selfie_girl", "自拍果汁女孩", "Selfie Drink Girl", "卡通人物", 38,
       sm.extract_outline_enclosed((484, 746, 578, 884), bg_seeds=[(488, 750), (574, 750), (488, 880), (574, 880)]))
sm.add("beach_balls", "沙滩彩球对", "Beach Balls", "露营道具", 40,
       sm.extract_multi_rois([(344, 864, 396, 916), (576, 732, 630, 784)], thresh=28, min_area=120))
sm.add("animal_crab", "沙滩小螃蟹", "Beach Crab", "可爱动物", 40,
       sm.extract_diff((606, 788, 680, 846), thresh=30, keep_largest=True))
sm.add("animal_turtle", "迷你小海龟", "Sea Turtle", "可爱动物", 42,
       sm.extract_outline_enclosed((524, 872, 586, 918), bg_seeds=[(528, 876), (582, 876), (528, 914), (582, 914)]))
sm.add("shells_starfish", "海星贝壳组", "Starfish & Shells", "露营道具", 44,
       sm.extract_diff((392, 896, 508, 974), thresh=28, keep_largest=False, min_area=80))
sm.add("sky_moon", "仲夏弯月", "Summer Moon", "星空光效", 50,
       sm.extract_diff((470, 204, 518, 254), thresh=28, keep_largest=True))
sm.add("sky_fireworks", "海边烟花秀", "Seaside Fireworks", "星空光效", 49,
       sm.extract_diff((105, 195, 775, 455), thresh=36, soft_glow=True,
                       exclude_polys=[[(466, 200), (522, 200), (522, 258), (466, 258)],
                                      [(190, 360), (420, 360), (420, 455), (190, 455)],
                                      [(485, 352), (668, 352), (668, 455), (485, 455)],
                                      [(635, 420), (775, 420), (775, 455), (635, 455)]]))
sm.finalize()

# ==============================================================================
# LEVEL 4: WINTER CHRISTMAS CABIN (polished)
# ==============================================================================
wc = LevelExtractor(
    "winter_cabin", "冬日圣诞小屋", "Winter Christmas Cabin", "壁炉暖光与雪地派对",
    "winter_cabin_scene_1795496213198.jpg", "winter_cabin_empty_bg_1795496259560.jpg"
)
wc.add("christmas_tree", "圣诞树与礼盒", "Christmas Tree & Gifts", "营地设施", 18,
       wc.extract_diff((400, 370, 526, 594), thresh=30, keep_largest=True,
                       exclude_polys=[[(400, 462), (426, 462), (426, 594), (400, 594)],
                                      [(476, 370), (526, 370), (526, 488), (496, 488)]]))
wc.add("fireplace_decor", "花环与圣诞袜", "Wreath & Stockings", "营地设施", 20,
       wc.extract_multi_rois([(548, 398, 616, 482), (526, 488, 622, 568)], thresh=28, min_area=120))
wc.add("gingerbread_table", "姜饼屋圆桌", "Gingerbread Table", "营地设施", 24,
       wc.extract_diff((438, 566, 578, 696), thresh=28, keep_largest=True,
                       exclude_polys=[[(438, 600), (460, 600), (460, 696), (438, 696)],
                                      [(438, 566), (484, 566), (484, 596), (438, 596)]]))
wc.add("street_lamp", "雪夜暖光路灯", "Snowy Street Lamp", "露营道具", 22,
       wc.extract_outline_enclosed((108, 524, 166, 750), bg_seeds=[(112, 528), (162, 528), (112, 746), (162, 746)]))
wc.add("char_tree_boy", "挂球男孩", "Tree Decorator Boy", "卡通人物", 22,
       wc.extract_outline_enclosed((342, 460, 434, 600), bg_seeds=[(346, 464), (430, 464), (346, 596), (430, 596)]))
wc.add("char_cookie_girl", "姜饼点心女孩", "Cookie Basket Girl", "卡通人物", 26,
       wc.extract_outline_enclosed((256, 518, 364, 652), bg_seeds=[(260, 522), (360, 522), (260, 648), (360, 648)]))
wc.add("char_cocoa_girl", "壁炉可可女孩", "Fireplace Cocoa Girl", "卡通人物", 26,
       wc.extract_outline_enclosed((598, 550, 672, 676), bg_seeds=[(602, 554), (668, 554), (602, 672), (668, 672)]))
wc.add("char_gift_girl", "拆礼物女孩", "Gift Unwrap Girl", "卡通人物", 30,
       wc.extract_diff((366, 594, 506, 748), thresh=28, keep_largest=False, min_area=200,
                       exclude_polys=[[(446, 594), (506, 594), (506, 678), (446, 678)],
                                      [(366, 660), (392, 660), (392, 748), (366, 748)]]))
wc.add("snowman_duo", "堆雪人双人组", "Snowman Builders", "卡通人物", 32,
       wc.extract_diff((190, 620, 392, 782), thresh=28, keep_largest=True,
                       exclude_polys=[[(190, 690), (212, 690), (212, 782), (190, 782)],
                                      [(204, 768), (274, 768), (274, 782), (204, 782)],
                                      [(290, 620), (392, 620), (392, 654), (290, 654)]]))
wc.add("char_bell_girl", "圣诞摇铃女孩", "Jingle Bell Girl", "卡通人物", 34,
       wc.extract_outline_enclosed((520, 670, 650, 814), bg_seeds=[(524, 674), (646, 674), (524, 810), (646, 810)]))
wc.add("char_teddy_girl", "抱小熊女孩", "Teddy Bear Girl", "卡通人物", 32,
       wc.extract_outline_enclosed((658, 632, 744, 776), bg_seeds=[(662, 636), (740, 636), (662, 772), (740, 772)]))
wc.add("char_snowball_boy", "打雪仗男孩", "Snowball Boy", "卡通人物", 38,
       wc.extract_outline_enclosed((386, 738, 494, 868), bg_seeds=[(390, 742), (490, 742), (390, 864), (490, 864)]))
wc.add("char_sled_boy", "滑雪橇男孩", "Wooden Sled Boy", "卡通人物", 40,
       wc.extract_outline_enclosed((282, 772, 428, 944), bg_seeds=[(286, 776), (424, 776), (286, 940), (424, 940)]))
wc.add("animal_reindeer", "红鼻子驯鹿", "Baby Reindeer", "可爱动物", 34,
       wc.extract_outline_enclosed((128, 688, 212, 824), bg_seeds=[(132, 692), (208, 692), (132, 820), (208, 820)]))
wc.add("animal_penguin", "围巾小企鹅", "Scarf Penguin", "可爱动物", 38,
       wc.extract_outline_enclosed((200, 770, 278, 872), bg_seeds=[(204, 774), (274, 774), (204, 868), (274, 868)]))
wc.add("animal_husky", "哈士奇小狗", "Husky Puppy", "可爱动物", 44,
       wc.extract_diff((482, 838, 590, 958), thresh=28, keep_largest=True))
wc.add("animal_bunny_front", "雪地小白兔", "Snow Bunny", "可爱动物", 45,
       wc.extract_diff((418, 894, 486, 984), thresh=28, keep_largest=True))
wc.add("animal_bunny_cookie", "姜饼小白兔", "Gingerbread Bunny", "可爱动物", 42,
       wc.extract_multi_rois([(590, 814, 636, 856), (632, 784, 708, 872)], thresh=28, min_area=100))
wc.add("animal_polar_bear", "北极熊宝宝", "Polar Bear Cub", "可爱动物", 36,
       wc.extract_diff((728, 712, 812, 806), thresh=28, keep_largest=True))
wc.add("animal_robin", "树梢知更鸟", "Winter Robin", "可爱动物", 40,
       wc.extract_outline_enclosed((792, 526, 846, 584), bg_seeds=[(796, 530), (842, 530), (796, 580), (842, 580)],
                                   exclude_polys=[[(834, 526), (846, 526), (846, 584), (834, 584)]]))
wc.add("candy_canes", "圣诞拐杖糖", "Candy Canes", "露营道具", 36,
       wc.extract_multi_rois([(94, 742, 128, 796), (488, 776, 544, 854)], thresh=28, min_area=100))
wc.finalize()
