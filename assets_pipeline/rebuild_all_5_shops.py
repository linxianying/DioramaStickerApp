import json
import os
import shutil
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image

BRAIN = "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p_devbox/ff1be343-f26a-4dda-8b06-9a95d3c4f1c1"
PIPE = "/home/antigravity/DioramaStickerApp/assets_pipeline"
IOS_LEVELS = "/home/antigravity/DioramaStickerApp/iOS/DioramaSticker/DioramaSticker/Resources/Levels"
DOCS_DIR = "/home/antigravity/DioramaStickerApp/docs"


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


class ShopRebuilder:
    def __init__(self, level_id, title_zh, title_en, subtitle_zh, orig_file, bg_img):
        self.level_id = level_id
        self.title_zh = title_zh
        self.title_en = title_en
        self.subtitle_zh = subtitle_zh
        self.orig = cv2.imread(os.path.join(BRAIN, orig_file))
        self.bg = bg_img.copy()
        if self.bg.shape != self.orig.shape:
            self.bg = cv2.resize(self.bg, (self.orig.shape[1], self.orig.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        self.H, self.W = self.orig.shape[:2]
        self.diff = np.linalg.norm(self.orig.astype(np.float32) - self.bg.astype(np.float32), axis=2)
        self.outline = get_outline_mask(self.orig)
        self.outline_strong = get_outline_mask(self.orig, r_max=128, g_max=106, b_max=92)
        self.out_dir = os.path.join(PIPE, f"rebuilt_{level_id}")
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self.stickers = []
        print(f"=== {level_id} diff percentiles: {np.percentile(self.diff, [10, 50, 75, 90]).round(1)} ===")

    def extract_diff(self, roi, thresh=26, keep_largest=True, min_area=100,
                     exclude_polys=None, include_outline=True, close_k=3, fill_holes=True, custom_diff=None):
        x0, y0, x1, y1 = roi
        dmap = self.diff if custom_diff is None else custom_diff
        sub_mask = (dmap[y0:y1, x0:x1] > thresh).astype(np.uint8)
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

    def extract_outline_enclosed(self, roi, bg_seeds=None, exclude_polys=None, keep_largest=True, min_area=120, close_outline=3, dilate_out=2):
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
            if keep_largest:
                best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                cleaned[labels == best] = 1
            else:
                for i in range(1, num_l):
                    if stats[i, cv2.CC_STAT_AREA] >= min_area:
                        cleaned[labels == i] = 1
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

    def extract_poly_diff(self, poly, thresh=22, min_area=60, fill_holes=False, include_outline=True,
                          custom_diff=None, custom_outline=None, exclude_polys=None, close_k=3):
        """Extract all changed pixels strictly inside a polygon (great for isometric slanted shelves!)."""
        pts = np.array(poly, dtype=np.int32)
        xs, ys = pts[:, 0], pts[:, 1]
        x0, y0 = max(0, int(xs.min()) - 2), max(0, int(ys.min()) - 2)
        x1, y1 = min(self.W, int(xs.max()) + 3), min(self.H, int(ys.max()) + 3)
        poly_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [pts - np.array([x0, y0])], 1)
        if exclude_polys:
            for ep in exclude_polys:
                epts = np.array([[px - x0, py - y0] for px, py in ep], dtype=np.int32)
                cv2.fillPoly(poly_mask, [epts], 0)
        dmap = self.diff if custom_diff is None else custom_diff
        sub_mask = ((dmap[y0:y1, x0:x1] > thresh) & (poly_mask > 0)).astype(np.uint8)
        sub_mask = cv2.morphologyEx(sub_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        if close_k > 1:
            sub_mask = cv2.morphologyEx(sub_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
        num_l, labels, stats, _ = cv2.connectedComponentsWithStats(sub_mask)
        cleaned = np.zeros_like(sub_mask)
        for i in range(1, num_l):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 1
        if fill_holes:
            cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
        if include_outline:
            omap = self.outline if custom_outline is None else custom_outline
            sub_out = omap[y0:y1, x0:x1]
            dil = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            cleaned = np.where((cleaned > 0) | ((dil > 0) & (sub_out > 0) & (poly_mask > 0)), 1, 0).astype(np.uint8)
            if fill_holes:
                cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
        alpha_full = np.zeros((self.H, self.W), dtype=np.uint8)
        alpha_full[y0:y1, x0:x1] = smooth_alpha_from_binary(cleaned, blur_sigma=0.55)
        return alpha_full

    def add(self, sid, name_zh, name_en, category, z_index, alpha_full, source_img=None):
        out_file = f"{sid}.png"
        out_path = os.path.join(self.out_dir, out_file)
        img = self.orig if source_img is None else source_img
        x, y, w, h = save_rgba_from_mask(img, alpha_full, out_path, pad=2)
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

        # Contact sheet
        n = len(self.stickers)
        cols = 5
        rows = (n + cols - 1) // cols
        cell_w, cell_h = 200, 210
        sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        for r in range(0, rows * cell_h, 16):
            for c in range(0, cols * cell_w, 16):
                sheet[r:r+16, c:c+16] = (68, 56, 50) if ((r // 16 + c // 16) % 2 == 0) else (84, 70, 62)
        for idx, s in enumerate(self.stickers):
            r, c = divmod(idx, cols)
            rgba = cv2.imread(os.path.join(self.out_dir, s["file"]), cv2.IMREAD_UNCHANGED)
            sh, sw = rgba.shape[:2]
            max_w, max_h = cell_w - 20, cell_h - 40
            scale = min(max_w / sw, max_h / sh, 1.35)
            nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
            res = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)
            cx = c * cell_w + (cell_w - nw) // 2
            cy = r * cell_h + 8 + (max_h - nh) // 2
            alpha = res[:, :, 3:4].astype(np.float32) / 255.0
            roi = sheet[cy:cy+nh, cx:cx+nw].astype(np.float32)
            sheet[cy:cy+nh, cx:cx+nw] = (res[:, :, :3].astype(np.float32) * alpha + roi * (1.0 - alpha)).astype(np.uint8)
            cv2.rectangle(sheet, (c*cell_w, r*cell_h), ((c+1)*cell_w-1, (r+1)*cell_h-1), (110, 95, 85), 1)
            cv2.putText(sheet, f"{idx+1}. {s['id']}", (c*cell_w + 6, (r+1)*cell_h - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 240, 255), 1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(PIPE, f"sheet_{self.level_id}.png"), sheet)
        side = np.hstack([
            cv2.resize(self.bg, (448, 600)),
            cv2.resize(comp_u8, (448, 600)),
            cv2.resize(self.orig, (448, 600)),
        ])
        cv2.imwrite(os.path.join(PIPE, f"verify_{self.level_id}.png"), side)
        shutil.copy2(os.path.join(PIPE, f"sheet_{self.level_id}.png"), os.path.join(BRAIN, f"sheet_{self.level_id}.png"))
        shutil.copy2(os.path.join(PIPE, f"verify_{self.level_id}.png"), os.path.join(BRAIN, f"verify_{self.level_id}.png"))

        # Replace iOS folder completely so no stale files remain
        ios_dest = os.path.join(IOS_LEVELS, self.level_id)
        if os.path.exists(ios_dest):
            shutil.rmtree(ios_dest)
        shutil.copytree(self.out_dir, ios_dest)
        print(f"  => Rebuilt {self.level_id}: {len(self.stickers)} stickers!")


# ==============================================================================
# 1. SEWING STUDIO (治愈缝纫小店) — BARE SHELVES + RUG + ALL WALL & SHELF GOODS!
# ==============================================================================
sw_bg_v2 = cv2.imread(os.path.join(BRAIN, "sewing_shop_empty_bg_v2_1795512437821.jpg"))
sw_bg_v1 = cv2.imread(os.path.join(BRAIN, "sewing_shop_empty_bg_1795509733318.jpg"))
sw = ShopRebuilder("sewing_shop", "治愈缝纫小店", "Cozy Sewing Studio", "复古缝纫机与布艺手作",
                   "sewing_shop_scene_1795509658503.jpg", sw_bg_v2)

# Unoccluded rug, wall decor & ALL 4 fabric shelves extracted cleanly from v1 - v2!!
diff_sw_v1v2 = np.linalg.norm(sw_bg_v1.astype(np.float32) - sw_bg_v2.astype(np.float32), axis=2)
out_sw_v1 = get_outline_mask(sw_bg_v1)

sw.add("patchwork_rug", "紫色拼布地毯", "Patchwork Rug", "营地设施", 10,
       sw.extract_diff((220, 600, 725, 920), thresh=26, keep_largest=True, custom_diff=diff_sw_v1v2), source_img=sw_bg_v1)
sw.add("thread_spool_rack", "彩虹线轴壁挂架", "Rainbow Thread Rack", "露营道具", 14,
       sw.extract_diff((60, 278, 270, 566), thresh=24, keep_largest=True, custom_diff=diff_sw_v1v2), source_img=sw_bg_v1)
sw.add("button_jars", "彩色纽扣玻璃罐", "Button Glass Jars", "露营道具", 15,
       sw.extract_poly_diff([(282, 246), (464, 158), (484, 244), (308, 332)], thresh=24, min_area=120, fill_holes=True,
                            custom_diff=diff_sw_v1v2, custom_outline=out_sw_v1), source_img=sw_bg_v1)
sw.add("fabric_bolts_top", "上层碎花布卷", "Floral Fabric Bolts", "露营道具", 15,
       sw.extract_poly_diff([(324, 336), (488, 254), (492, 344), (324, 424)], thresh=24, min_area=80, fill_holes=True,
                            custom_diff=diff_sw_v1v2, custom_outline=out_sw_v1), source_img=sw_bg_v1)
sw.add("fabric_bolts_mid", "中层波点布卷", "Polka Dot Fabric Bolts", "露营道具", 15,
       sw.extract_poly_diff([(324, 426), (492, 344), (492, 434), (324, 514)], thresh=24, min_area=80, fill_holes=True,
                            custom_diff=diff_sw_v1v2, custom_outline=out_sw_v1), source_img=sw_bg_v1)
sw.add("fabric_bolts_low", "下层格子布卷", "Gingham Fabric Bolts", "露营道具", 15,
       sw.extract_poly_diff([(324, 514), (492, 434), (492, 540), (324, 622)], thresh=24, min_area=80, fill_holes=True,
                            custom_diff=diff_sw_v1v2, custom_outline=out_sw_v1), source_img=sw_bg_v1)
sw.add("wall_scissors_tapes", "墙面剪刀与软尺", "Wall Scissors & Tapes", "露营道具", 15,
       sw.extract_diff((504, 248, 592, 456), thresh=24, keep_largest=False, min_area=80, custom_diff=diff_sw_v1v2), source_img=sw_bg_v1)
sw.add("wall_notes", "墙面服装便签", "Wall Pattern Notes", "露营道具", 16,
       sw.extract_poly_diff([(96, 544), (264, 466), (266, 560), (104, 610)], thresh=22, min_area=80, fill_holes=True,
                            custom_diff=diff_sw_v1v2, custom_outline=out_sw_v1), source_img=sw_bg_v1)
sw.add("shelf_bear_ribbons", "墙架熊与线轴", "Shelf Bear & Thread", "露营道具", 18,
       sw.extract_diff((736, 404, 840, 478), thresh=26, keep_largest=False, min_area=100))
sw.add("wall_sketch_board", "挂墙服装手稿板", "Dress Sketch Board", "露营道具", 16,
       sw.extract_diff((748, 480, 834, 576), thresh=24, keep_largest=True, custom_diff=diff_sw_v1v2), source_img=sw_bg_v1)
sw.add("animal_bluebird", "窗帘小蓝鸟", "Curtain Bluebird", "可爱动物", 18,
       sw.extract_diff((688, 310, 738, 356), thresh=26, keep_largest=True))

# Floor furniture, characters & animals (clean diff over v1 for rug-sitting items!)
diff_sw_floor = np.linalg.norm(sw.orig.astype(np.float32) - sw_bg_v1.astype(np.float32), axis=2)
sw.add("sewing_desk_mint", "薄荷缝纫机台", "Mint Sewing Desk", "营地设施", 20,
       sw.extract_diff((236, 468, 378, 696), thresh=26, keep_largest=True, custom_diff=diff_sw_floor,
                       exclude_polys=[[(236, 560), (268, 560), (268, 696), (236, 696)]]))
# Left sewing desk WITH its vintage desk lamp seamlessly attached!
a_sw_desk_left = np.maximum(
    sw.extract_diff((44, 540, 276, 804), thresh=26, keep_largest=True, custom_diff=diff_sw_floor,
                    exclude_polys=[[(162, 740), (235, 740), (235, 804), (162, 804)]]),
    sw.extract_diff((52, 574, 114, 678), thresh=24, keep_largest=True)
)
sw.add("sewing_desk_left", "粉红缝纫机女孩", "Pink Sewing Machine Girl", "卡通人物", 24, a_sw_desk_left)
sw.add("mannequin_boy", "量衣男孩与人台", "Tailor Boy & Mannequin", "卡通人物", 22,
       sw.extract_diff((384, 426, 548, 622), thresh=26, keep_largest=True, custom_diff=diff_sw_floor,
                       exclude_polys=[[(506, 548), (548, 548), (548, 622), (506, 622)]]))
sw.add("ironing_girl", "熨烫丝带女孩", "Ironing Board Girl", "卡通人物", 22,
       sw.extract_diff((534, 468, 732, 700), thresh=26, keep_largest=True, custom_diff=diff_sw_floor,
                       exclude_polys=[[(534, 548), (586, 548), (586, 670), (534, 670)]]))
sw.add("cutting_table_girl", "裁剪台与手作女孩", "Cutting Table & Tailor", "营地设施", 28,
       sw.extract_diff((342, 548, 616, 844), thresh=24, keep_largest=True, custom_diff=diff_sw_floor,
                       exclude_polys=[[(416, 792), (492, 792), (492, 844), (416, 844)],
                                      [(534, 768), (616, 768), (616, 844), (534, 844)]]))
sw.add("armchair_knitting", "沙发织毛衣与毛线熊", "Knitting Armchair & Bear", "卡通人物", 30,
       sw.extract_diff((682, 584, 850, 824), thresh=26, keep_largest=False, min_area=180, custom_diff=diff_sw_floor,
                       exclude_polys=[[(682, 710), (722, 710), (722, 824), (682, 824)]]))
sw.add("boy_fabric_bolts", "抱布卷男孩", "Fabric Bolts Boy", "卡通人物", 36,
       sw.extract_outline_enclosed((600, 694, 720, 882), bg_seeds=[(604, 698), (716, 698), (604, 878), (716, 878)]))
sw.add("designer_desk", "画图小设计师", "Sketching Designer", "卡通人物", 40,
       sw.extract_diff((470, 768, 618, 974), thresh=26, keep_largest=True, custom_diff=diff_sw_floor,
                       exclude_polys=[[(470, 768), (528, 768), (528, 820), (470, 820)]]))
sw.add("mannequin_mint", "薄荷蝴蝶结人台", "Mint Dress Mannequin", "营地设施", 42,
       sw.extract_outline_enclosed((344, 768, 428, 968), bg_seeds=[(348, 772), (424, 772), (348, 964), (424, 964)]))
sw.add("yarn_basket_bear", "毛线团小熊竹篮", "Yarn & Bear Basket", "露营道具", 46,
       sw.extract_diff((400, 904, 508, 1016), thresh=26, keep_largest=True, custom_diff=diff_sw_floor))
sw.add("animal_hedgehog", "针插小刺猬", "Pincushion Hedgehog", "可爱动物", 32,
       sw.extract_outline_enclosed((290, 714, 364, 784), bg_seeds=[(294, 718), (360, 718), (294, 780), (360, 780)]))
sw.add("animal_bunny", "蝴蝶结小白兔", "Bow Ribbon Bunny", "可爱动物", 38,
       sw.extract_outline_enclosed((412, 790, 494, 874), bg_seeds=[(416, 794), (490, 794), (416, 870), (490, 870)]))
sw.add("animal_calico_ribbon", "玩丝带花猫与小老鼠", "Ribbon Cat & Mouse", "可爱动物", 36,
       sw.extract_diff((160, 738, 322, 872), thresh=26, keep_largest=False, min_area=100, custom_diff=diff_sw_floor,
                       exclude_polys=[[(284, 738), (322, 738), (322, 788), (284, 788)]]))
sw.add("animal_grey_cat", "睡觉小灰猫", "Sleepy Grey Cat", "可爱动物", 40,
       sw.extract_diff((264, 862, 354, 938), thresh=26, keep_largest=False, min_area=80, custom_diff=diff_sw_floor,
                       exclude_polys=[[(338, 862), (354, 862), (354, 902), (338, 902)]]))
sw.finalize()

# ==============================================================================
# 2. TOAST SHOP (吐司面包屋) — BARE SHELVES + UNOCCLUDED LOAVES, JAMS & FRAMES!
# ==============================================================================
ts_bg_v2 = cv2.imread(os.path.join(BRAIN, "toast_shop_empty_bg_v2_1795512436870.jpg"))
ts_bg_v1 = cv2.imread(os.path.join(BRAIN, "toast_shop_empty_bg_1795509732373.jpg"))
ts_bg_clean = ts_bg_v2.copy()
ts_bg_clean[468:586, 304:434] = ts_bg_v1[468:586, 304:434]

ts = ShopRebuilder("toast_shop", "吐司面包屋", "Artisan Toast Shop", "厚切生吐司与蜂蜜黄油",
                   "toast_shop_scene_1795509656484.jpg", ts_bg_clean)

# Unoccluded shelf & wall stocking stickers extracted cleanly from v1 - v2!!
diff_ts_v1v2 = np.linalg.norm(ts_bg_v1.astype(np.float32) - ts_bg_clean.astype(np.float32), axis=2)
out_ts_v1 = get_outline_mask(ts_bg_v1)

ts.add("jam_jars_rack", "货架果酱罐组", "Jam Jars Collection", "露营道具", 14,
       ts.extract_diff((110, 418, 224, 594), thresh=26, keep_largest=False, min_area=100, custom_diff=diff_ts_v1v2), source_img=ts_bg_v1)
ts.add("counter_loaves", "烤架双吐司", "Cooling Rack Loaves", "露营道具", 15,
       ts.extract_diff((114, 588, 236, 680), thresh=26, keep_largest=True, custom_diff=diff_ts_v1v2), source_img=ts_bg_v1)
ts.add("toast_clock", "吐司造型挂钟", "Toast Wall Clock", "露营道具", 14,
       ts.extract_diff((378, 272, 444, 356), thresh=24, keep_largest=True, custom_diff=diff_ts_v1v2), source_img=ts_bg_v1)
ts.add("loaves_shelf_top", "上层黄金吐司排", "Top Shelf Loaves", "露营道具", 15,
       ts.extract_poly_diff([(442, 306), (598, 362), (590, 434), (442, 372)], thresh=20, min_area=80, fill_holes=True,
                            custom_diff=diff_ts_v1v2, custom_outline=out_ts_v1), source_img=ts_bg_v1)
ts.add("loaves_shelf_mid", "中层圆顶吐司排", "Mid Shelf Loaves", "露营道具", 15,
       ts.extract_poly_diff([(442, 370), (582, 424), (576, 474), (442, 418)], thresh=16, min_area=60, fill_holes=True,
                            custom_diff=diff_ts_v1v2, custom_outline=out_ts_v1, close_k=5), source_img=ts_bg_v1)
ts.add("loaves_shelf_low", "下层方吐司排", "Bottom Shelf Loaves", "露营道具", 15,
       ts.extract_poly_diff([(428, 424), (574, 480), (566, 560), (428, 494)], thresh=20, min_area=80, fill_holes=True,
                            custom_diff=diff_ts_v1v2, custom_outline=out_ts_v1), source_img=ts_bg_v1)
ts.add("stand_loaves", "抹茶与原味吐司架", "Matcha & Golden Loaves", "露营道具", 16,
       ts.extract_diff((572, 432, 692, 608), thresh=22, keep_largest=False, min_area=100, custom_diff=diff_ts_v1v2), source_img=ts_bg_v1)
ts.add("wall_toast_frames", "吐司挂画双框", "Toast Wall Frames", "露营道具", 14,
       ts.extract_diff((706, 466, 788, 634), thresh=24, keep_largest=False, min_area=120, custom_diff=diff_ts_v1v2), source_img=ts_bg_v1)

# Foreground characters & animals (clean diff over v1 so ZERO shelf bleed!)
diff_ts_fg = np.linalg.norm(ts.orig.astype(np.float32) - ts_bg_v1.astype(np.float32), axis=2)
ts.add("toasters_pair", "复古吐司机对", "Pop-Up Toasters", "露营道具", 18,
       ts.extract_diff((306, 470, 424, 584), thresh=28, keep_largest=False, min_area=180, custom_diff=diff_ts_fg,
                       exclude_polys=[[(400, 505), (424, 505), (424, 584), (400, 584)],
                                      [(306, 520), (325, 520), (325, 584), (306, 584)]]))
ts.add("baker_toaster", "烤吐司店员", "Toaster Baker", "卡通人物", 20,
       ts.extract_diff((398, 460, 496, 584), thresh=28, keep_largest=True, custom_diff=diff_ts_fg,
                       exclude_polys=[[(398, 460), (424, 460), (424, 515), (398, 515)]]))
ts.add("clerk_packing", "打包吐司店员", "Packing Clerk", "卡通人物", 22,
       ts.extract_diff((532, 530, 640, 692), thresh=28, keep_largest=True, custom_diff=diff_ts_fg,
                       exclude_polys=[[(568, 636), (640, 636), (640, 692), (568, 692)]]))
ts.add("chef_slicing", "切吐司主厨", "Shokupan Chef", "卡通人物", 24,
       ts.extract_diff((236, 518, 424, 706), thresh=26, keep_largest=True, custom_diff=diff_ts_fg,
                       exclude_polys=[[(326, 518), (378, 518), (378, 578), (326, 578)],
                                      [(270, 682), (424, 682), (424, 706), (270, 706)]]))
ts.add("toast_treats", "橱窗厚吐司组", "Thick-Cut Toasts", "露营道具", 26,
       ts.extract_diff((272, 572, 538, 792), thresh=26, keep_largest=False, min_area=80, fill_holes=False, custom_diff=diff_ts_fg,
                       exclude_polys=[[(272, 572), (422, 572), (422, 686), (272, 686)],
                                      [(450, 720), (538, 720), (538, 792), (450, 792)]]))
ts.add("toast_paper_bag", "吐司手提袋", "Toast Paper Bag", "露营道具", 24,
       ts.extract_diff((654, 602, 718, 684), thresh=28, keep_largest=True, custom_diff=diff_ts_fg))
ts.add("animal_shiba", "吐司色柴犬", "Toast Shiba Inu", "可爱动物", 30,
       ts.extract_diff((116, 696, 224, 804), thresh=28, keep_largest=True, custom_diff=diff_ts_fg))
ts.add("bread_basket", "吐司面包篮", "Bread Basket", "露营道具", 32,
       ts.extract_diff((212, 756, 294, 844), thresh=28, keep_largest=True, custom_diff=diff_ts_fg,
                       exclude_polys=[[(274, 814), (294, 814), (294, 844), (274, 844)]]))
ts.add("animal_chick", "面包小黄鸡", "Baby Chick", "可爱动物", 34,
       ts.extract_outline_enclosed((272, 812, 314, 858), bg_seeds=[(276, 816), (310, 816), (276, 854), (310, 854)]))
ts.add("animal_bear_honey", "蜂蜜罐小熊", "Honey Jar Bear", "可爱动物", 36,
       ts.extract_diff((334, 806, 414, 912), thresh=28, keep_largest=True, custom_diff=diff_ts_fg))
ts.add("animal_cat_cushion", "吐司垫白猫", "Toast Cushion Cat", "可爱动物", 40,
       ts.extract_diff((402, 882, 510, 968), thresh=28, keep_largest=True, custom_diff=diff_ts_fg,
                       exclude_polys=[[(402, 882), (424, 882), (424, 910), (402, 910)]]))
ts.add("animal_squirrel", "啃面包小松鼠", "Crumb Squirrel", "可爱动物", 38,
       ts.extract_diff((506, 854, 568, 916), thresh=28, keep_largest=True, custom_diff=diff_ts_fg,
                       exclude_polys=[[(506, 854), (522, 854), (522, 872), (506, 872)]]))
ts.add("girl_photo", "拍吐司女孩", "Photo Toast Girl", "卡通人物", 34,
       ts.extract_diff((446, 714, 534, 868), thresh=28, keep_largest=True, custom_diff=diff_ts_fg))
ts.add("cafe_table_guests", "奶油吐司咖啡桌", "Toast Cafe Table", "卡通人物", 35,
       ts.extract_diff((564, 622, 790, 868), thresh=26, keep_largest=True, custom_diff=diff_ts_fg,
                       exclude_polys=[[(656, 622), (714, 622), (714, 668), (656, 668)]]))
ts.finalize()

# ==============================================================================
# 3. BAGEL SHOP (贝果铺货小店) — REAL CLEAN EMPTY BASE + FULL SHELF STOCKING!
# ==============================================================================
bg_empty = cv2.imread(os.path.join(BRAIN, "bagel_shop_empty_bg_1795512515792.jpg"))
bg = ShopRebuilder("bagel_shop", "贝果铺货小店", "Artisan Bagel Shop", "新鲜贝果补货与奶酪香",
                   "bagel_shop_scene_1795504890032.jpg", bg_empty)

# Shelf & Wall Stocking Stickers (TRUE 贝果铺货!!)
bg.add("bagel_signboard", "贝果招牌木牌", "Bagel Shop Sign", "营地设施", 14,
       bg.extract_diff((472, 248, 582, 444), thresh=24, keep_largest=True))
bg.add("wall_menu_board", "菜单小黑板", "Menu Chalkboard", "露营道具", 14,
       bg.extract_diff((584, 352, 672, 506), thresh=24, keep_largest=True,
                       exclude_polys=[[(584, 466), (610, 466), (610, 506), (584, 506)]]))
bg.add("wall_tote_bag", "挂墙帆布袋", "Hanging Tote Bag", "露营道具", 14,
       bg.extract_diff((76, 442, 126, 588), thresh=24, keep_largest=True))
bg.add("hanging_plant", "吊挂小绿植", "Hanging Plant", "露营道具", 15,
       bg.extract_diff((410, 260, 476, 344), thresh=24, keep_largest=True))
bg.add("bagels_shelf_top", "蓝莓抹茶贝果排", "Top Bagels Row", "露营道具", 16,
       bg.extract_poly_diff([(208, 390), (392, 294), (424, 346), (242, 448)], thresh=24, min_area=80, fill_holes=True))
bg.add("bagels_shelf_mid", "草莓开心果贝果排", "Mid Bagels Row", "露营道具", 16,
       bg.extract_poly_diff([(142, 506), (394, 372), (426, 424), (176, 566)], thresh=24, min_area=80, fill_holes=True))
bg.add("bagels_shelf_low", "芝麻谷物贝果排", "Lower Bagels Row", "露营道具", 16,
       bg.extract_poly_diff([(158, 594), (402, 446), (434, 492), (190, 646)], thresh=24, min_area=80, fill_holes=True,
                            exclude_polys=[[(230, 524), (316, 524), (316, 610), (230, 610)]]))
bg.add("counter_baskets_jams", "柜台贝果篮与奶酪", "Counter Bagels & Cream Cheese", "营地设施", 24,
       bg.extract_diff((290, 568, 494, 746), thresh=24, keep_largest=False, min_area=80,
                       exclude_polys=[[(348, 568), (446, 568), (446, 602), (348, 602)],
                                      [(290, 568), (334, 568), (334, 628), (290, 628)],
                                      [(466, 610), (494, 610), (494, 746), (466, 746)]]))
bg.add("animal_hamster", "啃贝果仓鼠", "Bagel Hamster", "可爱动物", 20,
       bg.extract_diff((142, 400, 208, 478), thresh=26, keep_largest=True))
bg.add("animal_bear_shelf", "货架小熊", "Shelf Bear", "可爱动物", 20,
       bg.extract_diff((424, 394, 486, 474), thresh=26, keep_largest=True))
bg.add("toaster", "复古烤面包机", "Bagel Toaster", "露营道具", 20,
       bg.extract_diff((474, 434, 550, 522), thresh=26, keep_largest=True))
bg.add("coffee_maker", "手冲咖啡壶", "Coffee Brewer", "露营道具", 20,
       bg.extract_diff((540, 450, 608, 560), thresh=26, keep_largest=True))
bg.add("stool_and_mug", "高脚凳与咖啡杯", "Stool & Coffee Cup", "营地设施", 19,
       bg.extract_diff((462, 520, 566, 608), thresh=26, keep_largest=False, min_area=120))
bg.add("potted_plant", "落地大绿植", "Potted Plant", "露营道具", 24,
       bg.extract_diff((80, 598, 174, 780), thresh=26, keep_largest=True,
                       exclude_polys=[[(136, 718), (174, 718), (174, 780), (136, 780)]]))
bg.add("animal_cat", "货箱小橘猫", "Crate Tabby Cat", "可爱动物", 28,
       bg.extract_diff((122, 714, 212, 806), thresh=26, keep_largest=True))
bg.add("bagel_crates_bear", "贝果货箱与小熊", "Bagel Crates & Bear", "营地设施", 30,
       bg.extract_diff((180, 712, 370, 882), thresh=26, keep_largest=True,
                       exclude_polys=[[(180, 712), (218, 712), (218, 774), (180, 774)]]))
bg.add("baker_stocking", "补货贝果师", "Stocking Baker", "卡通人物", 26,
       bg.extract_diff((192, 526, 344, 732), thresh=26, keep_largest=True,
                       exclude_polys=[[(290, 628), (344, 628), (344, 732), (290, 732)]]))
bg.add("clerk_spread", "抹奶酪店员", "Cream Cheese Clerk", "卡通人物", 22,
       bg.extract_diff((348, 488, 456, 610), thresh=26, keep_largest=True))
bg.add("clerk_tongs", "夹贝果店员", "Bagel Tongs Clerk", "卡通人物", 32,
       bg.extract_diff((462, 586, 584, 774), thresh=26, keep_largest=True,
                       exclude_polys=[[(536, 730), (584, 730), (584, 774), (536, 774)]]))
bg.add("delivery_boy", "送货快递小哥", "Delivery Helper", "卡通人物", 28,
       bg.extract_diff((580, 542, 718, 716), thresh=26, keep_largest=True))
bg.add("girl_coffee", "外带咖啡女孩", "Iced Coffee Girl", "卡通人物", 36,
       bg.extract_diff((600, 676, 716, 864), thresh=26, keep_largest=True))
bg.add("animal_puppy", "门口卷毛小狗", "Fluffy Puppy", "可爱动物", 34,
       bg.extract_diff((710, 684, 816, 790), thresh=26, keep_largest=True))
bg.add("table_diners", "堂食贝果双人桌", "Bagel Dining Table", "卡通人物", 38,
       bg.extract_diff((366, 728, 606, 954), thresh=26, keep_largest=True,
                       exclude_polys=[[(424, 896), (488, 896), (492, 954), (416, 954)]]))
bg.add("animal_bunny_basket", "篮中小白兔", "Basket Bunny", "可爱动物", 42,
       bg.extract_diff((414, 892, 496, 984), thresh=26, keep_largest=True))
bg.finalize()

# ==============================================================================
# 4. FRENCH BAKERY (法式面包烘焙店) — REAL CLEAN EMPTY BASE + FULL BREAD RACKS!
# ==============================================================================
bk_empty = cv2.imread(os.path.join(BRAIN, "bakery_shop_empty_bg_1795512517816.jpg"))
bk = ShopRebuilder("bakery_shop", "法式面包烘焙店", "French Boulangerie", "窑炉欧包与酥脆可颂",
                   "bakery_shop_scene_1795504894078.jpg", bk_empty)

# Awning Signboard + Shelf, Oven & Counter Stocking Stickers (NEW!!)
_sb_sub = (bk.diff[290:478, 126:350] > 32).astype(np.uint8)
_sb_sub = cv2.morphologyEx(_sb_sub, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
_sb_sub = cv2.morphologyEx(_sb_sub, cv2.MORPH_CLOSE, np.ones((19, 19), np.uint8))
_sb_n, _sb_l, _sb_s, _ = cv2.connectedComponentsWithStats(_sb_sub)
_sb_best = 1 + int(np.argmax(_sb_s[1:, cv2.CC_STAT_AREA]))
_sb_cnts, _ = cv2.findContours((_sb_l == _sb_best).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
_sb_hull = cv2.convexHull(_sb_cnts[0])
_sb_bin = np.zeros_like(_sb_sub)
cv2.fillPoly(_sb_bin, [_sb_hull], 1)
# Snap outward by 2px into dark outline of signboard frame
_sb_dil = cv2.dilate(_sb_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
_sb_bin = np.where((_sb_bin > 0) | ((_sb_dil > 0) & (bk.outline[290:478, 126:350] > 0)), 1, 0).astype(np.uint8)
_sb_alpha = np.zeros((bk.H, bk.W), dtype=np.uint8)
_sb_alpha[290:478, 126:350] = smooth_alpha_from_binary(_sb_bin, blur_sigma=0.55)
bk.add("awning_signboard", "法式烘焙招牌", "Boulangerie Signboard", "营地设施", 14, _sb_alpha)
bk.add("wall_menu", "法式菜单黑板", "Bakery Menu Board", "露营道具", 14,
       bk.extract_diff((76, 470, 146, 602), thresh=24, keep_largest=True))
bk.add("left_shelf_breads", "左侧吐司篮货架", "Left Bread Baskets", "露营道具", 16,
       bk.extract_diff((76, 580, 184, 704), thresh=24, keep_largest=False, min_area=100))
bk.add("window_bread_baskets", "窗边法棍篮组", "Window Baguette Baskets", "露营道具", 18,
       bk.extract_diff((178, 512, 344, 650), thresh=24, keep_largest=False, min_area=120,
                       exclude_polys=[[(178, 594), (254, 594), (254, 650), (178, 650)],
                                      [(306, 578), (344, 578), (344, 650), (306, 650)],
                                      [(162, 510), (210, 510), (210, 566), (162, 566)],
                                      [(254, 468), (304, 468), (304, 520), (254, 520)]]))
bk.add("oven_loaves", "砖窑热烤面包", "Oven Baking Loaves", "露营道具", 16,
       bk.extract_diff((448, 434, 516, 492), thresh=26, keep_largest=True))
bk.add("right_rack_baguettes", "右架法棍篮组", "Rack Baguettes", "露营道具", 17,
       bk.extract_poly_diff([(566, 366), (714, 412), (702, 502), (566, 456)], thresh=24, min_area=80, fill_holes=True))
bk.add("right_rack_pretzels", "右架碱水结面包", "Rack Pretzels", "露营道具", 17,
       bk.extract_poly_diff([(566, 464), (714, 512), (710, 580), (566, 530)], thresh=24, min_area=80, fill_holes=True))
bk.add("right_rack_boules", "右架乡村圆欧包", "Rack Sourdough Boules", "露营道具", 17,
       bk.extract_poly_diff([(566, 532), (708, 578), (702, 636), (566, 588)], thresh=24, min_area=80, fill_holes=True))
bk.add("cash_register_basket", "收银机与柜台篮", "Cash Register & Basket", "露营道具", 26,
       bk.extract_diff((190, 646, 302, 748), thresh=24, keep_largest=False, min_area=100,
                       exclude_polys=[[(274, 646), (302, 646), (302, 748), (274, 748)],
                                      [(190, 646), (250, 646), (250, 686), (190, 686)]]))

# Characters, Furniture & Animals (clean diff over real empty bakery!)
a_sparrows = np.maximum(
    bk.extract_diff((162, 514, 210, 566), thresh=26, keep_largest=True),
    bk.extract_diff((254, 468, 304, 518), thresh=26, keep_largest=True),
)
bk.add("animal_sparrows", "窗台小麻雀对", "Window Sparrows", "可爱动物", 18, a_sparrows)
bk.add("master_baker", "砖窑老面包师", "Master Oven Baker", "卡通人物", 22,
       bk.extract_diff((336, 406, 502, 614), thresh=26, keep_largest=True,
                       exclude_polys=[[(456, 434), (502, 434), (502, 480), (456, 480)]]))
bk.add("animal_bunny", "抱面包小白兔", "Bread Bunny", "可爱动物", 24,
       bk.extract_diff((302, 576, 364, 654), thresh=26, keep_largest=True))
bk.add("baker_dough_table", "揉面小厨与面台", "Dough Table & Chef Mouse", "营地设施", 26,
       bk.extract_diff((372, 530, 578, 804), thresh=26, keep_largest=True,
                       exclude_polys=[[(372, 704), (452, 704), (452, 804), (372, 804)]]))
bk.add("animal_cat", "暖炉三花猫", "Oven Calico Cat", "可爱动物", 24,
       bk.extract_diff((556, 610, 640, 682), thresh=26, keep_largest=True))
bk.add("animal_dachshund", "门口腊肠小狗", "Dachshund Puppy", "可爱动物", 34,
       bk.extract_diff((712, 688, 806, 762), thresh=26, keep_largest=True))
bk.add("clerk_cashier", "收银女店员", "Bakery Cashier", "卡通人物", 24,
       bk.extract_diff((174, 596, 268, 704), thresh=26, keep_largest=True,
                       exclude_polys=[[(190, 686), (268, 686), (268, 704), (190, 704)]]))
bk.add("girl_tray", "端羊角包女孩", "Pastry Tray Girl", "卡通人物", 30,
       bk.extract_diff((274, 644, 384, 798), thresh=26, keep_largest=True))
bk.add("bread_basket_floor", "落地法棍篮", "Baguette Basket", "露营道具", 32,
       bk.extract_diff((234, 744, 310, 848), thresh=26, keep_largest=True))
bk.add("boy_baguettes", "抱法棍男孩", "Baguette Boy", "卡通人物", 34,
       bk.extract_diff((366, 700, 460, 862), thresh=26, keep_largest=True))
bk.add("pastry_cart", "可颂烘焙推车", "Croissant Cart", "营地设施", 38,
       bk.extract_diff((310, 784, 426, 920), thresh=26, keep_largest=True))
bk.add("apprentice_flour", "搬面粉小徒弟", "Flour Apprentice", "卡通人物", 42,
       bk.extract_diff((412, 804, 498, 958), thresh=26, keep_largest=True))
bk.add("girl_croissant", "抱可颂女孩", "Croissant Girl", "卡通人物", 40,
       bk.extract_diff((480, 790, 564, 912), thresh=26, keep_largest=True))
bk.add("grandma_croissant", "眼镜可颂奶奶", "Croissant Grandma", "卡通人物", 38,
       bk.extract_diff((538, 722, 640, 880), thresh=26, keep_largest=True,
                       exclude_polys=[[(612, 722), (640, 722), (640, 770), (612, 770)]]))
bk.add("grandma_loaf", "抱欧包老奶奶", "Sourdough Grandma", "卡通人物", 36,
       bk.extract_diff((608, 672, 698, 832), thresh=26, keep_largest=True))
bk.finalize()

# ==============================================================================
# 5. CAT CAFE (猫咖甜品店) — BARE WALL SHELVES + ESPRESSO MACHINE + MENU!
# ==============================================================================
cc_bg_v2 = cv2.imread(os.path.join(BRAIN, "cat_cafe_empty_bg_v2_1795512593528.jpg"))
cc_bg_v1 = cv2.imread(os.path.join(BRAIN, "cat_cafe_empty_bg_1795496305551.jpg"))
cc = ShopRebuilder("cat_cafe", "猫咖甜品店", "Cat Cafe Bakery", "草莓蛋糕与软萌猫咪",
                   "cat_cafe_scene_v2_1795496255386.jpg", cc_bg_v2)

# New shelf/wall/counter stocking stickers extracted cleanly from v1 - v2 (zero character occlusion!)
diff_cc_v1v2 = np.linalg.norm(cc_bg_v1.astype(np.float32) - cc_bg_v2.astype(np.float32), axis=2)
cc.add("wall_cat_mugs", "货架猫咪马克杯", "Cat Mug Shelves", "露营道具", 14,
       cc.extract_diff((330, 286, 448, 434), thresh=24, keep_largest=False, min_area=80, custom_diff=diff_cc_v1v2), source_img=cc_bg_v1)
cc.add("espresso_machine", "薄荷意式咖啡机", "Mint Espresso Machine", "营地设施", 16,
       cc.extract_diff((342, 432, 444, 560), thresh=24, keep_largest=True, custom_diff=diff_cc_v1v2), source_img=cc_bg_v1)
cc.add("menu_chalkboard", "猫咖菜单黑板", "Cat Cafe Menu Board", "露营道具", 14,
       cc.extract_diff((470, 286, 642, 512), thresh=24, keep_largest=True, custom_diff=diff_cc_v1v2), source_img=cc_bg_v1)
cc.add("hanging_vine", "吊挂小绿藤", "Hanging Vine Plant", "露营道具", 15,
       cc.extract_diff((638, 438, 686, 540), thresh=24, keep_largest=True, custom_diff=diff_cc_v1v2), source_img=cc_bg_v1)

# Existing 16 super-clean cat cafe stickers (diff against v1 so zero overlap bleed!)
diff_cc_fg = np.linalg.norm(cc.orig.astype(np.float32) - cc_bg_v1.astype(np.float32), axis=2)
cc.add("cat_tree", "猫爬架与猫咪", "Cat Tree Tower", "营地设施", 18,
       cc.extract_diff((68, 462, 214, 792), thresh=26, keep_largest=True, custom_diff=diff_cc_fg,
                       exclude_polys=[[(196, 500), (214, 500), (214, 685), (196, 685)],
                                      [(198, 685), (214, 685), (214, 735), (198, 735)]]))
cc.add("cat_siamese", "窗台暹罗猫", "Siamese Cat", "可爱动物", 16,
       cc.extract_outline_enclosed((196, 498, 268, 572), bg_seeds=[(200, 502), (264, 502), (200, 568), (264, 568)]))
cc.add("cat_cream_milk", "窗台白猫奶罐", "Window Kitty & Milk", "可爱动物", 16,
       np.maximum(
           cc.extract_diff((686, 492, 768, 586), thresh=28, keep_largest=True, custom_diff=diff_cc_fg),
           cc.extract_diff((760, 560, 798, 608), thresh=28, keep_largest=True, custom_diff=diff_cc_fg)
       ))
cc.add("baker_girl", "烘焙蛋糕师", "Strawberry Baker", "卡通人物", 24,
       cc.extract_diff((258, 494, 372, 684), thresh=28, keep_largest=True, custom_diff=diff_cc_fg,
                       exclude_polys=[[(258, 608), (270, 608), (270, 684), (258, 684)],
                                      [(318, 666), (372, 666), (372, 684), (318, 684)]]))
cc.add("barista_boy", "拉花咖啡师", "Latte Barista", "卡通人物", 20,
       cc.extract_diff((474, 424, 574, 578), thresh=28, keep_largest=True, custom_diff=diff_cc_fg))
cc.add("pastries_display", "甜品柜点心", "Cakes & Macarons", "露营道具", 22,
       cc.extract_diff((446, 548, 580, 678), thresh=28, keep_largest=False, min_area=60, fill_holes=False, custom_diff=diff_cc_fg,
                       exclude_polys=[[(520, 548), (580, 548), (580, 576), (520, 576)]]))
cc.add("browse_girl", "挑甜品女孩", "Pastry Browser", "卡通人物", 28,
       cc.extract_diff((370, 540, 458, 704), thresh=28, keep_largest=True, custom_diff=diff_cc_fg,
                       exclude_polys=[[(445, 540), (458, 540), (458, 600), (445, 600)],
                                      [(370, 662), (398, 662), (398, 704), (370, 704)]]))
cc.add("potted_plant_tall", "落地大绿植", "Potted Plant", "露营道具", 22,
       cc.extract_diff((590, 530, 668, 670), thresh=28, keep_largest=True, custom_diff=diff_cc_fg,
                       exclude_polys=[[(648, 646), (668, 646), (668, 670), (648, 670)]]))
cc.add("potted_succulent", "迷你多肉", "Mini Succulent", "露营道具", 26,
       cc.extract_diff((562, 664, 612, 728), thresh=24, keep_largest=True, custom_diff=diff_cc_fg))
cc.add("cat_trashbin", "猫耳收纳桶", "Cat Ear Bin", "露营道具", 22,
       cc.extract_diff((200, 608, 268, 682), thresh=28, keep_largest=True, custom_diff=diff_cc_fg))
cc.add("cat_basket", "猫窝奶牛猫", "Basket Tuxedo Cat", "可爱动物", 26,
       cc.extract_diff((620, 632, 718, 715), thresh=26, keep_largest=True, custom_diff=diff_cc_fg,
                       exclude_polys=[[(620, 632), (640, 632), (640, 650), (620, 650)],
                                      [(698, 694), (718, 694), (718, 715), (698, 715)]]))
cc.add("cat_white_cushion", "粉垫小白猫", "Fluffy White Cat", "可爱动物", 28,
       cc.extract_diff((412, 694, 522, 774), thresh=28, keep_largest=True, custom_diff=diff_cc_fg,
                       exclude_polys=[[(412, 694), (452, 694), (452, 712), (412, 712)]]))
cc.add("table_left_duo", "奶茶撸猫桌", "Boba & Kitty Table", "卡通人物", 34,
       cc.extract_diff((198, 660, 404, 870), thresh=34, keep_largest=True, custom_diff=diff_cc_fg,
                       exclude_polys=[[(344, 795), (404, 795), (404, 870), (344, 870)],
                                      [(198, 745), (214, 745), (214, 790), (198, 790)]]))
cc.add("table_right_reader", "阅读撸猫桌", "Reading Cat Table", "卡通人物", 34,
       cc.extract_diff((676, 628, 808, 806), thresh=32, keep_largest=True, custom_diff=diff_cc_fg,
                       exclude_polys=[[(676, 628), (716, 628), (716, 702), (676, 702)]]))
cc.add("boy_sleeping_cat", "睡猫与男孩", "Boy & Sleepy Tabby", "卡通人物", 35,
       cc.extract_diff((562, 716, 712, 868), thresh=28, keep_largest=True, custom_diff=diff_cc_fg,
                       exclude_polys=[[(562, 716), (610, 716), (610, 730), (562, 730)]]))
cc.add("rug_cat_play", "逗猫拍照地毯", "Cat Play Rug", "卡通人物", 38,
       cc.extract_diff((340, 774, 572, 968), thresh=26, keep_largest=True, custom_diff=diff_cc_fg,
                       exclude_polys=[[(340, 774), (404, 774), (404, 802), (340, 802)]]))
cc.finalize()

# ==============================================================================
# SYNC ALL 8 LEVELS TO docs/ (GitHub Pages WebP) + GENERATE NEW 2x4 AUDIT GRID!
# ==============================================================================
with open(os.path.join(IOS_LEVELS, "levels_manifest.json"), "r", encoding="utf-8") as f:
    manifest = json.load(f)

docs_levels = os.path.join(DOCS_DIR, "levels")
web_manifest = {"levels": []}
total_stickers = 0
audit_tiles = []

for idx, item in enumerate(manifest["levels"]):
    src_folder = os.path.join(IOS_LEVELS, item["folder"])
    dst_folder = os.path.join(docs_levels, item["id"])
    with open(os.path.join(src_folder, item["jsonFile"] + ".json"), "r", encoding="utf-8") as f:
        lvl = json.load(f)

    # Rebuild WebP cleanly for all 5 upgraded shops
    if item["id"] in ("sewing_shop", "toast_shop", "bagel_shop", "bakery_shop", "cat_cafe"):
        if os.path.exists(dst_folder):
            shutil.rmtree(dst_folder)
        os.makedirs(dst_folder, exist_ok=True)
        Image.open(os.path.join(src_folder, "background.png")).save(os.path.join(dst_folder, "background.webp"), "WEBP", quality=82, method=4)
        th = Image.open(os.path.join(src_folder, "reference_complete.png"))
        th.thumbnail((440, 580), Image.LANCZOS)
        th.save(os.path.join(dst_folder, "thumb.webp"), "WEBP", quality=82, method=4)
        web_stickers = []
        for s in lvl["stickers"]:
            webp_name = os.path.splitext(s["file"])[0] + ".webp"
            Image.open(os.path.join(src_folder, s["file"])).save(os.path.join(dst_folder, webp_name), "WEBP", quality=86, method=4)
            s_copy = dict(s)
            s_copy["file"] = webp_name
            web_stickers.append(s_copy)
        lvl_web = dict(lvl)
        lvl_web["backgroundFile"] = "background.webp"
        lvl_web["stickers"] = web_stickers
        with open(os.path.join(dst_folder, "level.json"), "w", encoding="utf-8") as f:
            json.dump(lvl_web, f, ensure_ascii=False, separators=(",", ":"))

    total_stickers += len(lvl["stickers"])
    web_manifest["levels"].append({
        "id": item["id"],
        "emoji": item["emoji"],
        "title_zh": item["title_zh"],
        "subtitle_zh": item["subtitle_zh"],
        "accentHex": item["accentHex"],
        "stickerCount": len(lvl["stickers"]),
        "thumb": f"levels/{item['id']}/thumb.webp",
        "dataUrl": f"levels/{item['id']}/level.json"
    })

    # Audit grid tile
    bimg = cv2.resize(cv2.imread(os.path.join(src_folder, "background.png")), (336, 450))
    cv2.putText(bimg, f"{idx+1}. {item['id']} ({len(lvl['stickers'])})", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    audit_tiles.append(bimg)

with open(os.path.join(DOCS_DIR, "levels_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(web_manifest, f, ensure_ascii=False, indent=2)

audit_grid = np.vstack([np.hstack(audit_tiles[:4]), np.hstack(audit_tiles[4:8])])
cv2.imwrite(os.path.join(PIPE, "audit_all_8_bgs_v2.png"), audit_grid)
shutil.copy2(os.path.join(PIPE, "audit_all_8_bgs_v2.png"), os.path.join(BRAIN, "audit_all_8_bgs_v2.png"))

print(f"\nALL 8 LEVELS REBUILT & SYNCED! GRAND TOTAL: {total_stickers} STICKERS!")
for x in web_manifest["levels"]:
    print(f"  - {x['emoji']} {x['title_zh']} ({x['id']}): {x['stickerCount']} stickers")
