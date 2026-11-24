import json
import os
import cv2
import numpy as np
from scipy import ndimage

BRAIN = "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p_devbox/ff1be343-f26a-4dda-8b06-9a95d3c4f1c1"
PIPE = "/home/antigravity/DioramaStickerApp/assets_pipeline"
IOS_LEVELS = "/home/antigravity/DioramaStickerApp/iOS/DioramaSticker/DioramaSticker/Resources/Levels"


def get_outline_mask(img, r_max=118, g_max=96, b_max=84):
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


class InpaintLevelExtractor:
    """Extracts crisp outline-enclosed cartoon stickers directly from a single illustration
    and synthesizes a clean inpainted empty background beneath all removed stickers."""

    def __init__(self, level_id, title_zh, title_en, subtitle_zh, orig_file):
        self.level_id = level_id
        self.title_zh = title_zh
        self.title_en = title_en
        self.subtitle_zh = subtitle_zh
        self.orig = cv2.imread(os.path.join(BRAIN, orig_file))
        self.H, self.W = self.orig.shape[:2]
        self.outline = get_outline_mask(self.orig, r_max=124, g_max=102, b_max=88)
        self.out_dir = os.path.join(PIPE, f"extracted_{level_id}")
        os.makedirs(self.out_dir, exist_ok=True)
        self.stickers = []
        self.union_mask = np.zeros((self.H, self.W), dtype=np.uint8)

    def extract_outline_enclosed(self, roi, bg_seeds=None, exclude_polys=None, keep_largest=True,
                                 min_area=180, close_outline=3, dilate_out=2):
        x0, y0, x1, y1 = roi
        sub_out = self.outline[y0:y1, x0:x1].copy()
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

    def extract_grabcut(self, roi, fg_rects=None, fg_pts=None, bg_pts=None, bg_polys=None, exclude_polys=None, iters=7):
        x0, y0, x1, y1 = roi
        sub = self.orig[y0:y1, x0:x1]
        sh, sw = sub.shape[:2]
        gc_mask = np.full((sh, sw), cv2.GC_PR_BGD, dtype=np.uint8)
        gc_mask[0:3, :] = cv2.GC_BGD
        gc_mask[-3:, :] = cv2.GC_BGD
        gc_mask[:, 0:3] = cv2.GC_BGD
        gc_mask[:, -3:] = cv2.GC_BGD

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

        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(sub, gc_mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
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
        sub_out = self.outline[y0:y1, x0:x1]
        dil = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
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
        self.union_mask = np.maximum(self.union_mask, (alpha_full > 30).astype(np.uint8) * 255)
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
        # Synthesize a super crisp, clean sticker-book base:
        # Fill removed sticker silhouettes with a clean warm matte cream-paper slot + delicate outline
        # (Zero blurry Telea smudging, and 100% pixel-exact match when stickers snap into place!)
        bg = self.orig.copy()
        # Sample warm paper/floor tone from bright warm pixels in the room interior
        hsv = cv2.cvtColor(self.orig, cv2.COLOR_BGR2HSV)
        warm_floor = (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 28) & (hsv[:, :, 2] > 180) & (hsv[:, :, 1] < 110)
        if np.any(warm_floor):
            base_bgr = np.median(self.orig[warm_floor], axis=0).astype(np.float32)
        else:
            base_bgr = np.array([196.0, 222.0, 242.0], dtype=np.float32)

        # Slightly desaturate & brighten for an inviting recessed paper sticker-slot look
        slot_color = np.clip(base_bgr * 0.92 + np.array([210.0, 228.0, 244.0]) * 0.08, 0, 255).astype(np.uint8)
        border_color = np.clip(base_bgr * 0.68, 0, 255).astype(np.uint8)

        # Clean mask of each sticker slot
        slot_mask = (self.union_mask > 20).astype(np.uint8)
        slot_mask = cv2.morphologyEx(slot_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        # Subtle inner shadow/rim along slot contour
        eroded = cv2.erode(slot_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        rim = (slot_mask > 0) & (eroded == 0)

        bg[slot_mask > 0] = slot_color
        # Soft blend border along slot rim for neat paper-craft feel
        bg[rim] = (slot_color.astype(np.float32) * 0.72 + border_color.astype(np.float32) * 0.28).astype(np.uint8)
        cv2.imwrite(os.path.join(self.out_dir, "background.png"), bg)

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

        # Composite verification
        comp = bg.copy().astype(np.float32)
        for s in ordered:
            rgba = cv2.imread(os.path.join(self.out_dir, s["file"]), cv2.IMREAD_UNCHANGED)
            x, y, w, h = s["x"], s["y"], s["width"], s["height"]
            alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
            comp[y:y+h, x:x+w] = rgba[:, :, :3].astype(np.float32) * alpha + comp[y:y+h, x:x+w] * (1.0 - alpha)
        comp_u8 = np.clip(comp, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(self.out_dir, "reference_complete.png"), comp_u8)

        # Contact sheet
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
            cv2.resize(bg, (448, 600)),
            cv2.resize(comp_u8, (448, 600)),
            cv2.resize(self.orig, (448, 600)),
        ])
        cv2.imwrite(os.path.join(PIPE, f"verify_{self.level_id}.png"), side)

        ios_dest = os.path.join(IOS_LEVELS, self.level_id)
        os.makedirs(ios_dest, exist_ok=True)
        for fn in os.listdir(self.out_dir):
            with open(os.path.join(self.out_dir, fn), "rb") as rf, open(os.path.join(ios_dest, fn), "wb") as wf:
                wf.write(rf.read())
        print(f"Polished {self.level_id}: {len(self.stickers)} stickers!\n")


# ==============================================================================
# LEVEL 5: BAGEL SHOP STOCKING (贝果铺货小店)
# ==============================================================================
bg = InpaintLevelExtractor(
    "bagel_shop", "贝果铺货小店", "Artisan Bagel Shop", "新鲜贝果补货与奶酪香气",
    "bagel_shop_scene_1795504890032.jpg"
)
bg.add("animal_hamster", "啃贝果仓鼠", "Bagel Hamster", "可爱动物", 20,
       bg.extract_outline_enclosed((142, 400, 208, 478), bg_seeds=[(146, 404), (204, 404), (146, 474), (204, 474)]))
bg.add("animal_bear_shelf", "货架小熊", "Shelf Bear", "可爱动物", 20,
       bg.extract_outline_enclosed((424, 396, 486, 474), bg_seeds=[(428, 400), (482, 400), (428, 470), (482, 470)]))
bg.add("hanging_plant", "吊挂小绿植", "Hanging Plant", "露营道具", 18,
       bg.extract_outline_enclosed((410, 266, 476, 344), bg_seeds=[(414, 270), (472, 270), (414, 340), (472, 340)]))
bg.add("toaster", "复古烤面包机", "Bagel Toaster", "露营道具", 20,
       bg.extract_outline_enclosed((476, 434, 548, 520), bg_seeds=[(480, 438), (544, 438), (480, 516), (544, 516)]))
bg.add("coffee_maker", "手冲咖啡壶", "Coffee Brewer", "露营道具", 20,
       bg.extract_outline_enclosed((542, 450, 608, 558), bg_seeds=[(546, 454), (604, 454), (546, 554), (604, 554)]))
bg.add("potted_plant", "落地大绿植", "Potted Plant", "露营道具", 24,
       bg.extract_grabcut((80, 598, 174, 780),
                          fg_rects=[(84, 604, 168, 772)],
                          fg_pts=[(128, 646, 22), (124, 738, 18)],
                          bg_pts=[(84, 604, 6), (168, 604, 6), (84, 772, 8), (168, 768, 8)],
                          exclude_polys=[[(146, 720), (174, 720), (174, 780), (146, 780)]]))
bg.add("animal_cat", "货箱小橘猫", "Crate Tabby Cat", "可爱动物", 28,
       bg.extract_outline_enclosed((122, 714, 212, 806), bg_seeds=[(126, 718), (208, 718), (126, 802), (208, 802)]))
bg.add("bagel_crates_bear", "贝果货箱与小熊", "Bagel Crates & Bear", "营地设施", 30,
       bg.extract_grabcut((182, 712, 370, 882),
                          fg_rects=[(188, 720, 364, 876)],
                          fg_pts=[(275, 760, 24), (275, 835, 24), (208, 792, 14), (340, 825, 16)],
                          bg_pts=[(188, 720, 6), (362, 720, 8), (355, 755, 8), (188, 872, 10), (362, 872, 8)]))
bg.add("baker_stocking", "补货贝果师", "Stocking Baker", "卡通人物", 26,
       bg.extract_grabcut((192, 526, 344, 732),
                          fg_rects=[(202, 534, 336, 722)],
                          fg_pts=[(288, 560, 18), (288, 602, 16), (242, 622, 18), (278, 675, 16), (264, 712, 10)],
                          bg_pts=[(198, 532, 8), (336, 532, 8), (198, 720, 10), (336, 720, 10), (202, 585, 8)]))
bg.add("clerk_spread", "抹奶酪店员", "Cream Cheese Clerk", "卡通人物", 22,
       bg.extract_grabcut((348, 488, 456, 610),
                          fg_rects=[(356, 496, 446, 602)],
                          fg_pts=[(394, 536, 18), (398, 584, 14), (434, 588, 8)],
                          bg_pts=[(354, 494, 6), (448, 494, 6), (354, 602, 6), (448, 550, 6)]))
bg.add("clerk_tongs", "夹贝果店员", "Bagel Tongs Clerk", "卡通人物", 32,
       bg.extract_grabcut((462, 586, 584, 774),
                          fg_rects=[(470, 594, 574, 764)],
                          fg_pts=[(534, 632, 20), (528, 686, 18), (522, 738, 12), (484, 674, 10)],
                          bg_pts=[(468, 594, 8), (576, 594, 8), (468, 764, 10), (574, 762, 12), (558, 742, 10), (474, 635, 8)],
                          exclude_polys=[[(536, 730), (584, 730), (584, 774), (536, 774)]]))
bg.add("delivery_boy", "送货快递小哥", "Delivery Helper", "卡通人物", 28,
       bg.extract_grabcut((580, 542, 718, 716),
                          fg_rects=[(586, 550, 710, 708)],
                          fg_pts=[(666, 582, 18), (674, 640, 16), (620, 646, 20), (694, 690, 10)],
                          bg_pts=[(586, 550, 8), (710, 550, 8), (586, 706, 8), (710, 640, 6), (602, 588, 10)]))
bg.add("girl_coffee", "外带咖啡女孩", "Iced Coffee Girl", "卡通人物", 36,
       bg.extract_outline_enclosed((600, 676, 716, 864), bg_seeds=[(604, 680), (712, 680), (604, 860), (712, 860)]))
bg.add("animal_puppy", "门口卷毛小狗", "Fluffy Puppy", "可爱动物", 34,
       bg.extract_outline_enclosed((710, 684, 816, 790), bg_seeds=[(714, 688), (812, 688), (714, 786), (812, 786)]))
bg.add("table_diners", "堂食贝果双人桌", "Bagel Dining Table", "卡通人物", 38,
       bg.extract_grabcut((366, 728, 606, 954),
                          fg_rects=[(374, 736, 598, 946)],
                          fg_pts=[(412, 828, 22), (414, 902, 16), (558, 782, 22), (556, 858, 18), (502, 868, 22), (502, 922, 12)],
                          bg_pts=[(372, 736, 10), (495, 745, 14), (598, 736, 8), (372, 944, 10), (598, 944, 12), (465, 938, 12)],
                          exclude_polys=[[(428, 900), (484, 900), (490, 954), (420, 954)]]))
bg.add("animal_bunny_basket", "篮中小白兔", "Basket Bunny", "可爱动物", 42,
       bg.extract_outline_enclosed((414, 892, 496, 984), bg_seeds=[(418, 896), (492, 896), (418, 980), (492, 980)]))
bg.finalize()

# ==============================================================================
# LEVEL 6: FRENCH BAKERY BOULANGERIE (法式面包烘焙店)
# ==============================================================================
bk = InpaintLevelExtractor(
    "bakery_shop", "法式面包烘焙店", "French Boulangerie", "窑炉欧包与酥脆可颂",
    "bakery_shop_scene_1795504894078.jpg"
)
bk.add("wheat_bundles", "挂墙麦穗束", "Wheat Bundles", "露营道具", 16,
       np.maximum(
           bk.extract_grabcut((332, 304, 384, 404), fg_rects=[(338, 310, 378, 396)], fg_pts=[(358, 355, 12)], bg_pts=[(336, 310, 5), (378, 310, 5), (336, 396, 5), (378, 396, 5)]),
           bk.extract_grabcut((376, 282, 424, 382), fg_rects=[(382, 288, 418, 374)], fg_pts=[(400, 330, 12)], bg_pts=[(380, 288, 5), (418, 288, 5), (380, 374, 5), (418, 374, 5)])
       ))
bk.add("animal_sparrows", "窗台小麻雀对", "Window Sparrows", "可爱动物", 18,
       np.maximum(
           bk.extract_outline_enclosed((164, 526, 206, 570), bg_seeds=[(168, 530), (202, 530), (168, 566), (202, 566)]),
           bk.extract_outline_enclosed((258, 468, 304, 514), bg_seeds=[(262, 472), (300, 472), (262, 510), (300, 510)])
       ))
bk.add("master_baker", "砖窑老面包师", "Master Oven Baker", "卡通人物", 22,
       bk.extract_grabcut((336, 406, 502, 614),
                          fg_rects=[(344, 414, 494, 606)],
                          fg_pts=[(380, 444, 18), (382, 494, 16), (382, 548, 16), (382, 594, 12), (464, 502, 20)],
                          bg_pts=[(342, 412, 8), (494, 412, 10), (455, 440, 12), (342, 604, 10), (494, 604, 14), (452, 575, 14)]))
bk.add("animal_bunny", "抱面包小白兔", "Bread Bunny", "可爱动物", 24,
       bk.extract_outline_enclosed((304, 578, 364, 654), bg_seeds=[(308, 582), (360, 582), (308, 650), (360, 650)]))
bk.add("baker_dough_table", "揉面小厨与面台", "Dough Table & Chef Mouse", "营地设施", 26,
       bk.extract_grabcut((372, 530, 578, 804),
                          fg_rects=[(380, 538, 570, 796)],
                          fg_pts=[(512, 588, 22), (512, 646, 16), (480, 692, 34), (416, 634, 12), (508, 756, 22), (458, 738, 14)],
                          bg_pts=[(378, 538, 12), (448, 548, 16), (570, 538, 10), (378, 794, 12), (570, 794, 12), (412, 742, 14)],
                          exclude_polys=[[(372, 706), (452, 706), (452, 804), (372, 804)]]))
bk.add("animal_cat", "暖炉三花猫", "Oven Calico Cat", "可爱动物", 24,
       bk.extract_outline_enclosed((556, 610, 640, 682), bg_seeds=[(560, 614), (636, 614), (560, 678), (636, 678)]))
bk.add("animal_dachshund", "门口腊肠小狗", "Dachshund Puppy", "可爱动物", 34,
       bk.extract_outline_enclosed((712, 688, 806, 762), bg_seeds=[(716, 692), (802, 692), (716, 758), (802, 758)]))
bk.add("clerk_cashier", "收银女店员", "Bakery Cashier", "卡通人物", 24,
       bk.extract_grabcut((174, 596, 268, 704),
                          fg_rects=[(182, 604, 260, 696)],
                          fg_pts=[(214, 638, 18), (218, 680, 14)],
                          bg_pts=[(180, 602, 6), (260, 602, 8), (180, 696, 6), (260, 696, 6)]))
bk.add("girl_tray", "端羊角包女孩", "Pastry Tray Girl", "卡通人物", 30,
       bk.extract_grabcut((274, 644, 384, 798),
                          fg_rects=[(282, 652, 376, 790)],
                          fg_pts=[(332, 686, 18), (336, 738, 16), (336, 778, 12), (298, 728, 12)],
                          bg_pts=[(280, 650, 8), (376, 650, 8), (280, 788, 10), (376, 788, 10)]))
bk.add("bread_basket_floor", "落地法棍篮", "Baguette Basket", "露营道具", 32,
       bk.extract_outline_enclosed((234, 744, 310, 848), bg_seeds=[(238, 748), (306, 748), (238, 844), (306, 844)]))
bk.add("boy_baguettes", "抱法棍男孩", "Baguette Boy", "卡通人物", 34,
       bk.extract_grabcut((366, 700, 460, 862),
                          fg_rects=[(374, 708, 452, 854)],
                          fg_pts=[(422, 738, 18), (416, 790, 18), (422, 838, 12), (390, 774, 12)],
                          bg_pts=[(372, 706, 8), (452, 706, 8), (372, 852, 8), (452, 852, 8)]))
bk.add("pastry_cart", "可颂烘焙推车", "Croissant Cart", "营地设施", 38,
       bk.extract_outline_enclosed((310, 784, 426, 920), bg_seeds=[(314, 788), (422, 788), (314, 916), (422, 916)]))
bk.add("apprentice_flour", "搬面粉小徒弟", "Flour Apprentice", "卡通人物", 42,
       bk.extract_outline_enclosed((412, 804, 498, 958), bg_seeds=[(416, 808), (494, 808), (416, 954), (494, 954)]))
bk.add("girl_croissant", "抱可颂女孩", "Croissant Girl", "卡通人物", 40,
       bk.extract_outline_enclosed((480, 790, 564, 912), bg_seeds=[(484, 794), (560, 794), (484, 908), (560, 908)]))
bk.add("grandma_croissant", "眼镜可颂奶奶", "Croissant Grandma", "卡通人物", 38,
       bk.extract_grabcut((538, 722, 640, 880),
                          fg_rects=[(546, 730, 632, 872)],
                          fg_pts=[(576, 762, 18), (580, 822, 18), (612, 794, 14), (582, 864, 10)],
                          bg_pts=[(544, 728, 8), (632, 728, 8), (544, 872, 8), (632, 872, 10)]))
bk.add("grandma_loaf", "抱欧包老奶奶", "Sourdough Grandma", "卡通人物", 36,
       bk.extract_outline_enclosed((608, 672, 698, 832), bg_seeds=[(612, 676), (694, 676), (612, 828), (694, 828)]))
bk.finalize()
