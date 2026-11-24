import json
import os
import shutil
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
    rgba = np.dstack([img[y0:y1, x0:x1], alpha_full[y0:y1, x0:x1]])
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
        self.outline_strong = get_outline_mask(self.orig, r_max=128, g_max=106, b_max=92)
        self.out_dir = os.path.join(PIPE, f"extracted_{level_id}")
        os.makedirs(self.out_dir, exist_ok=True)
        self.stickers = []
        print(f"{level_id} diff percentiles:", np.percentile(self.diff, [10, 50, 75, 90]).round(1))

    def extract_diff(self, roi, thresh=28, keep_largest=True, min_area=120,
                     exclude_polys=None, include_outline=True, close_k=3, fill_holes=True):
        x0, y0, x1, y1 = roi
        sub_mask = (self.diff[y0:y1, x0:x1] > thresh).astype(np.uint8)
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
        print(f"  [{self.level_id}] {sid:22s} -> ({x:3d}, {y:3d}, {w:3d}x{h:3d}) z={z_index}")

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
        with open(os.path.join(self.out_dir, f"level_{self.level_id}.json"), "w", encoding="utf-8") as f:
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
        cols = 4
        rows = (n + cols - 1) // cols
        cell_w, cell_h = 220, 225
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
        shutil.copy2(os.path.join(PIPE, f"sheet_{self.level_id}.png"), os.path.join(BRAIN, f"sheet_{self.level_id}.png"))
        shutil.copy2(os.path.join(PIPE, f"verify_{self.level_id}.png"), os.path.join(BRAIN, f"verify_{self.level_id}.png"))

        ios_dest = os.path.join(IOS_LEVELS, self.level_id)
        os.makedirs(ios_dest, exist_ok=True)
        for fn in os.listdir(self.out_dir):
            shutil.copy2(os.path.join(self.out_dir, fn), os.path.join(ios_dest, fn))
        print(f"Finished {self.level_id}: {len(self.stickers)} stickers!\n")


# ==============================================================================
# LEVEL 7: ARTISAN TOAST SHOP (吐司面包屋)
# ==============================================================================
ts = LevelExtractor(
    "toast_shop", "吐司面包屋", "Artisan Toast Shop", "厚切生吐司与蜂蜜黄油",
    "toast_shop_scene_1795509656484.jpg", "toast_shop_empty_bg_1795509732373.jpg"
)
ts.add("toasters_pair", "复古吐司机对", "Pop-Up Toasters", "露营道具", 18,
       ts.extract_diff((306, 470, 424, 584), thresh=28, keep_largest=False, min_area=180,
                       exclude_polys=[[(400, 505), (424, 505), (424, 584), (400, 584)],
                                      [(306, 520), (325, 520), (325, 584), (306, 584)]]))
ts.add("baker_toaster", "烤吐司店员", "Toaster Baker", "卡通人物", 20,
       ts.extract_diff((398, 460, 496, 584), thresh=28, keep_largest=True,
                       exclude_polys=[[(398, 460), (424, 460), (424, 515), (398, 515)]]))
ts.add("clerk_packing", "打包吐司店员", "Packing Clerk", "卡通人物", 22,
       ts.extract_diff((532, 530, 640, 692), thresh=28, keep_largest=True,
                       exclude_polys=[[(568, 636), (640, 636), (640, 692), (568, 692)]]))
ts.add("chef_slicing", "切吐司主厨", "Shokupan Chef", "卡通人物", 24,
       ts.extract_diff((236, 518, 424, 706), thresh=26, keep_largest=True,
                       exclude_polys=[[(326, 518), (378, 518), (378, 578), (326, 578)],
                                      [(270, 682), (424, 682), (424, 706), (270, 706)]]))
ts.add("toast_treats", "橱窗厚吐司组", "Thick-Cut Toasts", "露营道具", 26,
       ts.extract_diff((272, 572, 538, 792), thresh=26, keep_largest=False, min_area=80, fill_holes=False,
                       exclude_polys=[[(272, 572), (422, 572), (422, 686), (272, 686)],
                                      [(450, 720), (538, 720), (538, 792), (450, 792)]]))
ts.add("toast_paper_bag", "吐司手提袋", "Toast Paper Bag", "露营道具", 24,
       ts.extract_diff((654, 602, 718, 684), thresh=28, keep_largest=True))
ts.add("animal_shiba", "吐司色柴犬", "Toast Shiba Inu", "可爱动物", 30,
       ts.extract_diff((116, 696, 224, 804), thresh=28, keep_largest=True))
ts.add("bread_basket", "吐司面包篮", "Bread Basket", "露营道具", 32,
       ts.extract_diff((212, 756, 294, 844), thresh=28, keep_largest=True,
                       exclude_polys=[[(274, 814), (294, 814), (294, 844), (274, 844)]]))
ts.add("animal_chick", "面包小黄鸡", "Baby Chick", "可爱动物", 34,
       ts.extract_outline_enclosed((272, 812, 314, 858), bg_seeds=[(276, 816), (310, 816), (276, 854), (310, 854)]))
ts.add("animal_bear_honey", "蜂蜜罐小熊", "Honey Jar Bear", "可爱动物", 36,
       ts.extract_diff((334, 806, 414, 912), thresh=28, keep_largest=True))
ts.add("animal_cat_cushion", "吐司垫白猫", "Toast Cushion Cat", "可爱动物", 40,
       ts.extract_diff((402, 882, 510, 968), thresh=28, keep_largest=True,
                       exclude_polys=[[(402, 882), (424, 882), (424, 910), (402, 910)]]))
ts.add("animal_squirrel", "啃面包小松鼠", "Crumb Squirrel", "可爱动物", 38,
       ts.extract_diff((506, 854, 568, 916), thresh=28, keep_largest=True,
                       exclude_polys=[[(506, 854), (522, 854), (522, 872), (506, 872)]]))
ts.add("girl_photo", "拍吐司女孩", "Photo Toast Girl", "卡通人物", 34,
       ts.extract_diff((446, 714, 534, 868), thresh=28, keep_largest=True))
ts.add("cafe_table_guests", "奶油吐司咖啡桌", "Toast Cafe Table", "卡通人物", 35,
       ts.extract_diff((564, 622, 790, 868), thresh=26, keep_largest=True,
                       exclude_polys=[[(656, 622), (714, 622), (714, 668), (656, 668)]]))
ts.finalize()

# ==============================================================================
# LEVEL 8: COZY SEWING STUDIO (治愈缝纫小店)
# ==============================================================================
sw = LevelExtractor(
    "sewing_shop", "治愈缝纫小店", "Cozy Sewing Studio", "复古缝纫机与布艺手作",
    "sewing_shop_scene_1795509658503.jpg", "sewing_shop_empty_bg_1795509733318.jpg"
)
sw.add("animal_bluebird", "窗帘小蓝鸟", "Curtain Bluebird", "可爱动物", 16,
       sw.extract_diff((688, 310, 738, 356), thresh=28, keep_largest=True))
sw.add("shelf_bear_ribbons", "墙架熊与线轴", "Shelf Bear & Thread", "露营道具", 18,
       sw.extract_diff((736, 404, 840, 492), thresh=28, keep_largest=False, min_area=120))
sw.add("sewing_desk_mint", "薄荷缝纫机台", "Mint Sewing Desk", "营地设施", 20,
       sw.extract_diff((206, 468, 378, 696), thresh=26, keep_largest=True,
                       exclude_polys=[[(206, 560), (268, 560), (268, 696), (206, 696)]]))
sw.add("sewing_desk_left", "粉红缝纫机女孩", "Pink Sewing Machine Girl", "卡通人物", 24,
       sw.extract_diff((44, 540, 276, 804), thresh=26, keep_largest=True,
                       exclude_polys=[[(44, 575), (112, 575), (112, 675), (44, 675)],
                                      [(162, 740), (235, 740), (235, 804), (162, 804)]]))
sw.add("mannequin_boy", "量衣男孩与人台", "Tailor Boy & Mannequin", "卡通人物", 22,
       sw.extract_diff((384, 426, 548, 622), thresh=26, keep_largest=True,
                       exclude_polys=[[(506, 548), (548, 548), (548, 622), (506, 622)]]))
sw.add("ironing_girl", "熨烫丝带女孩", "Ironing Board Girl", "卡通人物", 22,
       sw.extract_diff((534, 468, 732, 700), thresh=26, keep_largest=True,
                       exclude_polys=[[(534, 548), (586, 548), (586, 670), (534, 670)]]))
sw.add("cutting_table_girl", "裁剪台与手作女孩", "Cutting Table & Tailor", "营地设施", 28,
       sw.extract_diff((342, 548, 616, 844), thresh=24, keep_largest=True,
                       exclude_polys=[[(416, 792), (492, 792), (492, 844), (416, 844)],
                                      [(534, 768), (616, 768), (616, 844), (534, 844)]]))
sw.add("armchair_knitting", "沙发织毛衣与毛线熊", "Knitting Armchair & Bear", "卡通人物", 30,
       sw.extract_diff((682, 584, 850, 824), thresh=26, keep_largest=False, min_area=180,
                       exclude_polys=[[(682, 710), (722, 710), (722, 824), (682, 824)]]))
sw.add("boy_fabric_bolts", "抱布卷男孩", "Fabric Bolts Boy", "卡通人物", 36,
       sw.extract_outline_enclosed((600, 694, 720, 882), bg_seeds=[(604, 698), (716, 698), (604, 878), (716, 878)]))
sw.add("designer_desk", "画图小设计师", "Sketching Designer", "卡通人物", 40,
       sw.extract_diff((470, 768, 618, 974), thresh=26, keep_largest=True,
                       exclude_polys=[[(470, 768), (528, 768), (528, 820), (470, 820)]]))
sw.add("mannequin_mint", "薄荷蝴蝶结人台", "Mint Dress Mannequin", "营地设施", 42,
       sw.extract_outline_enclosed((344, 768, 428, 968), bg_seeds=[(348, 772), (424, 772), (348, 964), (424, 964)]))
sw.add("yarn_basket_bear", "毛线团小熊竹篮", "Yarn & Bear Basket", "露营道具", 46,
       sw.extract_diff((400, 904, 508, 1016), thresh=26, keep_largest=True))
sw.add("animal_hedgehog", "针插小刺猬", "Pincushion Hedgehog", "可爱动物", 32,
       sw.extract_outline_enclosed((290, 714, 364, 784), bg_seeds=[(294, 718), (360, 718), (294, 780), (360, 780)]))
sw.add("animal_bunny", "蝴蝶结小白兔", "Bow Ribbon Bunny", "可爱动物", 38,
       sw.extract_outline_enclosed((412, 790, 494, 874), bg_seeds=[(416, 794), (490, 794), (416, 870), (490, 870)]))
sw.add("animal_calico_ribbon", "玩丝带花猫与小老鼠", "Ribbon Cat & Mouse", "可爱动物", 36,
       sw.extract_diff((160, 738, 322, 872), thresh=26, keep_largest=False, min_area=100,
                       exclude_polys=[[(284, 738), (322, 738), (322, 788), (284, 788)]]))
sw.add("animal_grey_cat", "睡觉小灰猫", "Sleepy Grey Cat", "可爱动物", 40,
       sw.extract_diff((264, 862, 354, 938), thresh=26, keep_largest=False, min_area=80,
                       exclude_polys=[[(338, 862), (354, 862), (354, 902), (338, 902)]]))
sw.finalize()
