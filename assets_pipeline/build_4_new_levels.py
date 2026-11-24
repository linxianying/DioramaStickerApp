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
)

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


class LevelBuilder:
    def __init__(self, level_id, title_zh, title_en, subtitle_zh, orig_path, bg_path):
        self.level_id = level_id
        self.title_zh = title_zh
        self.title_en = title_en
        self.subtitle_zh = subtitle_zh
        self.orig = cv2.imread(orig_path)
        self.bg = cv2.imread(bg_path)
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

    def extract_poly_mask(self, poly):
        """Exact clean polygon mask (ideal for straight isometric wooden signboards!)."""
        pts = np.array(poly, dtype=np.int32)
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        return smooth_alpha_from_binary(mask, blur_sigma=0.5)

    def extract_poly_diff(self, poly, thresh=22, min_area=80, fill_holes=True, include_outline=True, exclude_polys=None, close_k=3):
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
        sub_mask = ((self.diff[y0:y1, x0:x1] > thresh) & (poly_mask > 0)).astype(np.uint8)
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
            sub_out = self.outline[y0:y1, x0:x1]
            dil = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            cleaned = np.where((cleaned > 0) | ((dil > 0) & (sub_out > 0) & (poly_mask > 0)), 1, 0).astype(np.uint8)
            if fill_holes:
                cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
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
            "stickers": ordered,
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

        # Sync to iOS bundle folder
        ios_dest = os.path.join(IOS_LEVELS, self.level_id)
        if os.path.exists(ios_dest):
            shutil.rmtree(ios_dest)
        shutil.copytree(self.out_dir, ios_dest)
        print(f"  => Built {self.level_id}: {len(self.stickers)} stickers!")
        return len(self.stickers)


def main():
    # Regenerate freshly polished 3 empty backgrounds first
    build_hotpot_empty_bg()
    build_desert_empty_bg()
    build_taiwan_empty_bg()

    # ==========================================================================
    # LEVEL 9: 冬季圣诞森林 (christmas_forest) — REAL AI BARE SNOWY CLEARING!
    # ==========================================================================
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
    cf.add("elves_wrapping", "包装礼物双精灵", "Gift Wrapping Elves", "卡通人物", 22,
           cf.extract_diff((64, 490, 224, 678), thresh=26, keep_largest=False, min_area=120,
                           exclude_polys=[[(204, 490), (224, 490), (224, 575), (204, 575)],
                                          [(150, 586), (224, 586), (224, 678), (150, 678)]]))
    cf.add("robin_left", "知更小鸟", "Little Robin", "可爱动物", 22,
           cf.extract_diff((330, 566, 382, 622), thresh=26, keep_largest=True))
    cf.add("robin_right", "雪地知更鸟", "Snow Robin", "可爱动物", 22,
           cf.extract_diff((578, 578, 632, 632), thresh=26, keep_largest=True))
    cf.add("hot_cocoa_cart", "热可可木推车", "Hot Chocolate Cart", "营地设施", 24,
           cf.extract_diff((618, 480, 836, 732), thresh=24, keep_largest=False, min_area=120,
                           exclude_polys=[[(750, 700), (836, 700), (836, 732), (750, 732)]]))
    cf.add("christmas_tree", "星光圣诞树与礼物", "Glowing Christmas Tree", "营地设施", 26,
           cf.extract_diff((358, 462, 540, 764), thresh=24, keep_largest=False, min_area=120,
                           exclude_polys=[[(494, 598), (540, 598), (540, 750), (494, 750)],
                                          [(358, 462), (404, 462), (404, 536), (358, 536)],
                                          [(358, 566), (382, 566), (382, 622), (358, 622)]]))
    cf.add("girl_decorating", "挂星星女孩", "Star Decorating Girl", "卡通人物", 28,
           cf.extract_diff((490, 596, 588, 752), thresh=26, keep_largest=True))
    cf.add("snowman_kids", "堆雪人双人组", "Snowman & Kids", "卡通人物", 28,
           cf.extract_diff((152, 586, 356, 760), thresh=24, keep_largest=True))
    cf.add("mini_tree_right", "小圣诞树", "Mini Pine Tree", "露营道具", 28,
           cf.extract_diff((578, 684, 638, 774), thresh=26, keep_largest=True))
    cf.add("animal_fox", "森林小红狐", "Red Forest Fox", "可爱动物", 30,
           cf.extract_diff((68, 676, 160, 780), thresh=26, keep_largest=True))
    cf.add("animal_squirrel", "抱松果小松鼠", "Pinecone Squirrel", "可爱动物", 32,
           cf.extract_diff((748, 704, 836, 792), thresh=26, keep_largest=True))
    cf.add("reindeer_sleigh", "驯鹿与礼物雪橇", "Reindeer Gift Sleigh", "营地设施", 34,
           cf.extract_diff((494, 726, 748, 932), thresh=24, keep_largest=True))
    cf.add("mini_tree_center", "雪地迷你树", "Tiny Snow Tree", "露营道具", 32,
           cf.extract_diff((438, 772, 488, 852), thresh=26, keep_largest=True))
    cf.add("animal_bunny", "雪地小白兔", "Snow Bunny", "可爱动物", 34,
           cf.extract_diff((150, 746, 218, 834), thresh=24, keep_largest=True))
    cf.add("animal_fawn", "森林小梅花鹿", "Baby Forest Fawn", "可爱动物", 36,
           cf.extract_diff((218, 770, 294, 878), thresh=26, keep_largest=True))
    cf.add("carol_singers", "唱颂歌小朋友", "Christmas Carolers", "卡通人物", 36,
           cf.extract_diff((276, 740, 434, 912), thresh=24, keep_largest=False, min_area=40,
                           exclude_polys=[[(276, 770), (296, 770), (296, 878), (276, 878)]]))
    cf.add("owl_stump", "树桩雪鸮", "Snowy Owl on Stump", "可爱动物", 40,
           cf.extract_diff((392, 868, 522, 1004), thresh=24, keep_largest=True))
    cf.add("robin_bottom", "树桩小知更鸟", "Stump Robin", "可爱动物", 42,
           cf.extract_diff((364, 918, 412, 964), thresh=26, keep_largest=True))
    cf.finalize()

    # ==========================================================================
    # LEVEL 10: 沙漠帐篷探险 (desert_camp) — BARE GOLDEN SAND + OASIS!
    # ==========================================================================
    dc = LevelBuilder(
        "desert_camp", "沙漠帐篷探险", "Desert Oasis Expedition", "绿洲星夜与波西米亚帐篷",
        os.path.join(BRAIN, "desert_camp_scene_1795514833166.jpg"),
        os.path.join(PIPE, "empty_bg_desert_camp.png"),
    )
    dc.add("boho_bell_tent", "波西米亚大帐篷", "Boho Bell Tent & Rug", "营地设施", 16,
           dc.extract_diff((146, 412, 404, 666), thresh=26, keep_largest=True,
                           exclude_polys=[[(146, 584), (214, 584), (214, 666), (146, 666)],
                                          [(146, 496), (192, 496), (192, 586), (146, 586)],
                                          [(374, 540), (404, 540), (404, 600), (374, 600)]]))
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
    dc.add("archaeologist_girl", "考古石碑女孩", "Relic Archaeologist", "卡通人物", 26,
           dc.extract_diff((130, 584, 246, 712), thresh=26, keep_largest=False, min_area=80))
    dc.add("campfire_kettle", "沙漠篝火与茶壶", "Desert Campfire & Kettle", "营地设施", 24,
           dc.extract_diff((372, 590, 494, 682), thresh=26, keep_largest=False, min_area=120,
                           exclude_polys=[[(372, 652), (430, 652), (430, 682), (372, 682)]]))
    dc.add("guitar_explorer", "弹吉他探险家", "Campfire Guitarist", "卡通人物", 26,
           dc.extract_diff((562, 600, 654, 716), thresh=26, keep_largest=True))
    dc.add("cactus_right", "帐篷仙人掌盆", "Tent Potted Cactus", "露营道具", 28,
           dc.extract_diff((694, 620, 746, 716), thresh=26, keep_largest=True))
    dc.add("map_explorer", "藏宝图木箱探险家", "Treasure Map Explorer", "卡通人物", 30,
           dc.extract_diff((248, 652, 354, 810), thresh=26, keep_largest=True))
    dc.add("animal_meerkat_left", "站哨小狐獴", "Lookout Meerkat", "可爱动物", 32,
           dc.extract_diff((200, 716, 250, 778), thresh=26, keep_largest=True))
    dc.add("marshmallow_campers", "烤棉花糖双人组", "Marshmallow Roast Duo", "卡通人物", 30,
           dc.extract_diff((358, 652, 558, 782), thresh=26, keep_largest=False, min_area=120))
    dc.add("animal_meerkats_right", "狐獴兄弟", "Meerkat Pair", "可爱动物", 30,
           dc.extract_diff((642, 696, 704, 754), thresh=26, keep_largest=False, min_area=60))
    dc.add("animal_gecko", "沙漠小壁虎", "Desert Gecko", "可爱动物", 32,
           dc.extract_diff((610, 734, 668, 780), thresh=26, keep_largest=True))
    dc.add("animal_fennec_low", "坐姿耳廓狐", "Sitting Fennec Fox", "可爱动物", 34,
           dc.extract_diff((558, 760, 634, 836), thresh=26, keep_largest=True))
    dc.add("tea_explorer", "薄荷茶探险家", "Mint Tea Camper", "卡通人物", 36,
           dc.extract_diff((356, 790, 456, 912), thresh=26, keep_largest=False, min_area=80,
                           exclude_polys=[[(424, 790), (456, 790), (456, 858), (424, 858)]]))
    dc.add("animal_camel_low", "暖阳小骆驼", "Friendly Camel", "可爱动物", 36,
           dc.extract_diff((426, 780, 560, 914), thresh=26, keep_largest=True,
                           exclude_polys=[[(426, 840), (456, 840), (456, 914), (426, 914)]]))
    dc.finalize()

    # ==========================================================================
    # LEVEL 11: 火锅食材盛宴 (hotpot_feast) — 90° CEILING-DOWN OVERHEAD VIEW!
    # ==========================================================================
    hp = LevelBuilder(
        "hotpot_feast", "火锅食材盛宴", "Overhead Hotpot Feast", "天花板俯视角鸳鸯火锅",
        os.path.join(BRAIN, "hotpot_feast_scene_1795514917338.jpg"),
        os.path.join(PIPE, "empty_bg_hotpot_feast.png"),
    )
    # Using thresh=34 eliminates 100% of subtle wood plank grain lines while keeping every dish & white halo razor-sharp!
    hp.add("yin_yang_hotpot", "鸳鸯麻辣骨汤锅", "Yin-Yang Hotpot", "营地设施", 20,
           hp.extract_diff((244, 392, 652, 770), thresh=34, keep_largest=False, min_area=180))
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
    hp.add("ladles_pair", "不锈钢汤勺漏勺", "Soup Ladle & Skimmer", "露营道具", 22,
           hp.extract_diff((56, 480, 250, 664), thresh=34, keep_largest=False, min_area=200))
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

    # ==========================================================================
    # LEVEL 12: 台湾夜市街 (taiwan_night_market) — BARE STALLS + ALL SIGNS & FOODS!
    # ==========================================================================
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

    # 4 Stall vendors & stocked counter goods!
    tw.add("stall_boba_goods", "奶茶老板娘与奶茶杯", "Boba Vendor & Cups", "卡通人物", 18,
           tw.extract_poly_diff([(130, 560), (288, 474), (296, 616), (174, 672), (130, 646)],
                                thresh=24, min_area=80, fill_holes=True))
    tw.add("stall_chicken_goods", "盐酥鸡老板与烤串", "Fried Chicken Vendor", "卡通人物", 18,
           tw.extract_poly_diff([(304, 466), (450, 384), (456, 512), (344, 578), (304, 548)],
                                thresh=24, min_area=80, fill_holes=True))
    tw.add("stall_oyster_goods", "蚵仔煎铁板与芒果冰", "Oyster Omelet & Mango Ice", "卡通人物", 18,
           tw.extract_poly_diff([(476, 418), (620, 496), (642, 570), (582, 610), (476, 544)],
                                thresh=24, min_area=80, fill_holes=True))
    tw.add("stall_ringtoss_goods", "套圈圈彩瓶与玩偶", "Ring-Toss Bottles & Plush", "露营道具", 18,
           tw.extract_poly_diff([(632, 512), (796, 540), (796, 668), (748, 704), (632, 634)],
                                thresh=24, min_area=60, fill_holes=True))

    # Street characters, props & pets (thresh=36 eliminates subtle cobblestone grout halos!)
    tw.add("tapioca_jar", "大罐黑糖珍珠", "Giant Tapioca Jar", "露营道具", 24,
           tw.extract_diff((94, 670, 154, 760), thresh=34, keep_largest=True))
    tw.add("animal_tuxedo_cat", "夜市奶牛猫", "Tuxedo Alley Cat", "可爱动物", 28,
           tw.extract_diff((146, 714, 224, 796), thresh=36, keep_largest=True))
    tw.add("couple_boba", "买奶茶情侣", "Bubble Tea Couple", "卡通人物", 26,
           tw.extract_diff((218, 590, 344, 756), thresh=36, keep_largest=True,
                           exclude_polys=[[(326, 600), (344, 600), (344, 700), (326, 700)]]))
    tw.add("animal_shiba", "盯烤串小柴犬", "Skewer Shiba Inu", "可爱动物", 24,
           tw.extract_diff((328, 606, 398, 698), thresh=36, keep_largest=True))
    tw.add("couple_snacks", "边逛边吃双人组", "Night Market Foodies", "卡通人物", 28,
           tw.extract_diff((392, 608, 534, 752), thresh=36, keep_largest=False, min_area=180))
    tw.add("kid_ringtoss", "套圈圈小朋友", "Ring-Toss Kid", "卡通人物", 28,
           tw.extract_diff((580, 630, 696, 762), thresh=36, keep_largest=False, min_area=80,
                           exclude_polys=[[(656, 630), (696, 630), (696, 672), (656, 672)]]))
    tw.add("crate_calico_cat", "木箱睡觉三花猫", "Crate Calico Cat", "可爱动物", 26,
           tw.extract_diff((742, 690, 838, 800), thresh=32, keep_largest=True))
    tw.add("mint_scooter", "薄荷复古小绵羊", "Mint Retro Scooter", "营地设施", 34,
           tw.extract_diff((204, 748, 364, 916), thresh=36, keep_largest=True))
    tw.add("girl_skewers", "撸烤串女孩", "Skewer Girl", "卡通人物", 34,
           tw.extract_diff((342, 746, 434, 888), thresh=36, keep_largest=True,
                           exclude_polys=[[(342, 820), (366, 820), (366, 888), (342, 888)]]))
    tw.add("table_mango_ice", "芒果冰红凳夜市桌", "Mango Shaved Ice Table", "卡通人物", 38,
           tw.extract_diff((412, 756, 586, 982), thresh=36, keep_largest=True))
    tw.add("girl_camera", "夜市打卡拍照女孩", "Night Market Photographer", "卡通人物", 36,
           tw.extract_diff((588, 740, 676, 886), thresh=36, keep_largest=True))
    tw.finalize()

    # ==========================================================================
    # SYNC ALL 12 LEVELS TO iOS MANIFEST + DOCS WEBP PWA + 12-BG AUDIT GRID!
    # ==========================================================================
    all_12_meta = [
        ("campfire_party", "森林篝火晚会", "Campfire Party", "星空帐篷与围炉夜话", "🏕️", "#F59E0B", "campfire_party", "level_campfire"),
        ("cat_cafe", "猫咖甜品店", "Cat Cafe Bakery", "草莓蛋糕与软萌猫咪", "🐱", "#F472B6", "cat_cafe", "level_cat_cafe"),
        ("summer_market", "海边夏日夜市", "Summer Night Market", "海浪烟花与章鱼烧摊", "🎆", "#38BDF8", "summer_market", "level_summer_market"),
        ("winter_cabin", "冬日圣诞小屋", "Christmas Cabin", "壁炉雪松与驯鹿雪橇", "🎄", "#EF4444", "winter_cabin", "level_winter_cabin"),
        ("bagel_shop", "贝果铺货小店", "Artisan Bagel Shop", "新鲜贝果补货与奶酪香", "🥯", "#FB923C", "bagel_shop", "level_bagel_shop"),
        ("bakery_shop", "法式面包烘焙店", "French Boulangerie", "窑炉欧包与酥脆可颂", "🥐", "#D97706", "bakery_shop", "level_bakery_shop"),
        ("toast_shop", "吐司面包屋", "Artisan Toast Shop", "厚切生吐司与蜂蜜黄油", "🍞", "#EAB308", "toast_shop", "level_toast_shop"),
        ("sewing_shop", "治愈缝纫小店", "Cozy Sewing Studio", "复古缝纫机与布艺手作", "🧵", "#A855F7", "sewing_shop", "level_sewing_shop"),
        ("christmas_forest", "冬季圣诞森林", "Winter Christmas Forest", "极光森林与驯鹿雪橇", "❄️", "#60A5FA", "christmas_forest", "level_christmas_forest"),
        ("desert_camp", "沙漠帐篷探险", "Desert Oasis Expedition", "绿洲星夜与波西米亚帐篷", "🐪", "#F97316", "desert_camp", "level_desert_camp"),
        ("hotpot_feast", "火锅食材盛宴", "Overhead Hotpot Feast", "天花板俯视角鸳鸯火锅", "🍲", "#DC2626", "hotpot_feast", "level_hotpot_feast"),
        ("taiwan_night_market", "台湾夜市街", "Taiwan Night Market", "珍珠奶茶、盐酥鸡与套圈圈", "🏮", "#E11D48", "taiwan_night_market", "level_taiwan_night_market"),
    ]

    ios_manifest_levels = []
    docs_manifest_levels = []
    new_ids = {"christmas_forest", "desert_camp", "hotpot_feast", "taiwan_night_market"}

    for lid, tzh, ten, sub, emoji, accent, folder, jstem in all_12_meta:
        src_dir = os.path.join(IOS_LEVELS, folder)
        with open(os.path.join(src_dir, f"{jstem}.json"), "r", encoding="utf-8") as f:
            jdata = json.load(f)
        cnt = len(jdata["stickers"])

        ios_manifest_levels.append({
            "id": lid,
            "folder": folder,
            "jsonFile": jstem,
            "title_zh": tzh,
            "title_en": ten,
            "subtitle_zh": sub,
            "emoji": emoji,
            "accentHex": accent,
        })

        # Convert the 4 new levels to optimized WebP in docs/levels/<lid>/
        if lid in new_ids:
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

        docs_manifest_levels.append({
            "id": lid,
            "title_zh": tzh,
            "title_en": ten,
            "subtitle_zh": sub,
            "emoji": emoji,
            "accentHex": accent,
            "stickerCount": cnt,
            "thumb": f"levels/{lid}/thumb.webp",
            "dataUrl": f"levels/{lid}/level.json",
        })

    with open(os.path.join(IOS_LEVELS, "levels_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"levels": ios_manifest_levels}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DOCS_DIR, "levels_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"levels": docs_manifest_levels}, f, ensure_ascii=False, indent=2)

    # Generate 4x3 Audit Grid of ALL 12 Empty Backgrounds!
    tw, th = 360, 482
    grid = np.full((th * 3, tw * 4, 3), (58, 46, 38), dtype=np.uint8)
    for idx, item in enumerate(docs_manifest_levels):
        lid = item["id"]
        bg_p = os.path.join(IOS_LEVELS, lid, "background.png")
        img = cv2.resize(cv2.imread(bg_p), (tw, th))
        r, c = divmod(idx, 4)
        grid[r*th:(r+1)*th, c*tw:(c+1)*tw] = img
        cv2.putText(grid, f"{idx+1}. {lid} ({item['stickerCount']})", (c*tw + 10, r*th + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (90, 240, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(PIPE, "audit_all_12_bgs.png"), grid)
    shutil.copy2(os.path.join(PIPE, "audit_all_12_bgs.png"), os.path.join(BRAIN, "audit_all_12_bgs.png"))

    grand_total = sum(x["stickerCount"] for x in docs_manifest_levels)
    print(f"\nALL 12 LEVELS BUILT & SYNCED! GRAND TOTAL: {grand_total} STICKERS!")
    for x in docs_manifest_levels:
        print(f"  - {x['emoji']} {x['title_zh']} ({x['id']}): {x['stickerCount']} stickers")


if __name__ == "__main__":
    main()
