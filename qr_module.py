# -*- coding: utf-8 -*-
"""qr_module.py.

車検証PDF画像用 QR 高精度読み取りモジュール
(WeChatQRCode / zxingcpp / pyzbar マルチエンジン対応 ＋ 画像品質自動判別 ＋ 軽自動車対応[6個] ＋ パターン別検出 ＋ 早期判定スキップ ＋ 最終フォールバック ＋ GCメモリ解放 ＋ 詳細コンソールログ機能付き)
"""

import gc
import os
import re
import sys
import cv2
import numpy as np
from PIL import ImageDraw

# ============================================================
# QR検出ライブラリの読み込み
# ============================================================

# 1. WeChatQRCode (OpenCV dnn)
HAS_WECHAT = False
wechat_detector = None
try:
    model_dir = os.path.dirname(os.path.abspath(__file__))
    detect_proto = os.path.join(model_dir, "detect.prototxt")
    detect_caffe = os.path.join(model_dir, "detect.caffemodel")
    sr_proto = os.path.join(model_dir, "sr.prototxt")
    sr_caffe = os.path.join(model_dir, "sr.caffemodel")

    if all(os.path.exists(f) for f in [detect_proto, detect_caffe, sr_proto, sr_caffe]):
        wechat_detector = cv2.wechat_qrcode_WeChatQRCode(
            detect_proto, detect_caffe, sr_proto, sr_caffe
        )
        HAS_WECHAT = True
except Exception:
    HAS_WECHAT = False

# 2. zxingcpp
try:
    import zxingcpp

    HAS_ZXING = True
except ImportError:
    HAS_ZXING = False

# 3. pyzbar
try:
    from pyzbar.pyzbar import ZBarSymbol, decode as zbar_decode

    HAS_PYZBAR = True
except Exception:
    HAS_PYZBAR = False


# ============================================================
# 基本設定
# ============================================================

# 軽自動車の最大QR数に合わせて6に設定（登録車は5個、軽自動車は6個）
MAX_QR = 6
IGNORE_URL = True
DEBUG = True

PADDING_PX = 80

# 横型車検証QRの標準スロット位置（6分割用にも対応できるよう幅を持たせて設定）
QR_SLOT_CENTERS_X = (0.500, 0.570, 0.640, 0.710, 0.780, 0.850)
QR_SLOT_CENTER_Y = 0.895

QR_SLOT_HALF_W = 0.075
QR_SLOT_HALF_H = 0.085

TRACK_DX = (-0.060, -0.040, -0.020, 0.0, 0.020, 0.040, 0.060)
TRACK_DY = (-0.030, -0.015, 0.0, 0.015, 0.030)

SLOT_SCALES = (2.0, 3.0, 4.0)

# 誤検出防止用: 車検証QRが存在するY方向の下限閾値（全体検索時のみ使用）
MIN_QR_Y_RATIO = 0.60

_qr_cache = {}


def clear_qr_cache():
    _qr_cache.clear()
    gc.collect()


class SuppressStderr:

    def __enter__(self):
        try:
            self.err_fd = sys.stderr.fileno()
            self.saved_err_fd = os.dup(self.err_fd)
            self.devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self.devnull, self.err_fd)
        except Exception:
            self.saved_err_fd = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.saved_err_fd is not None:
            os.dup2(self.saved_err_fd, self.err_fd)
            os.close(self.saved_err_fd)
            os.close(self.devnull)


def _safe_decode_bytes(text_data):
    if text_data is None:
        return ""
    if isinstance(text_data, bytes):
        for encoding in ["shift_jis", "cp932", "utf-8", "euc-jp"]:
            try:
                return text_data.decode(encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue
        return text_data.decode("utf-8", errors="replace")
    return str(text_data)


def _normalize_qr_text(text):
    decoded = _safe_decode_bytes(text)
    return decoded.replace("\x00", "").strip()


def _is_usable_qr_text(text):
    text = _normalize_qr_text(text)
    if not text:
        return False
    if IGNORE_URL and (
        text.startswith("http://") or text.startswith("https://")
    ):
        return False
    return True


def is_kei_vehicle_qr(qr_list):
    """検出されたQRコード群から軽自動車書式かどうかを判定する."""
    if not qr_list:
        return False

    for qr in qr_list:
        raw_val = getattr(qr, "text", qr)
        text = _normalize_qr_text(raw_val)
        # 軽自動車特有の識別子チェック（/K/ や "6" 分割の記号など）
        if "/K/" in text or text.startswith("K") or "軽自動車" in text:
            return True
        parts = text.split("/")
        # 分割フォーマットが N/6（例: 1/6, 2/6...）形式の場合
        if len(parts) >= 2 and re.match(r"^\d+/6$", parts[0]):
            return True

    return False


def is_detection_sufficient(found):
    """現在の検出数で必要十分かを判定（軽自動車: 6個 / 登録車: 5個）.

    軽自動車と判定された場合は6個、それ以外（登録車）は5個で判定完了とする。
    """
    if is_kei_vehicle_qr(found):
        return len(found) >= 6

    # 登録車の場合（または未確定時点の規定値）
    return len(found) >= 5


def _point(x, y):
    return type("Point", (), {"x": float(x), "y": float(y)})()


class CustomPosition:

    def __init__(self, tl, tr, bl, br):
        self.top_left = tl
        self.top_right = tr
        self.bottom_left = bl
        self.bottom_right = br


class CustomQR:

    def __init__(self, text="", position=None, pattern_name="UNKNOWN"):
        self.text = text
        self.position = position
        self.pattern_name = pattern_name


def scale_qr_position(qr, factor):
    if factor == 1.0 or not hasattr(qr, "position") or not qr.position:
        return qr
    p = qr.position
    pos = CustomPosition(
        _point(p.top_left.x * factor, p.top_left.y * factor),
        _point(p.top_right.x * factor, p.top_right.y * factor),
        _point(p.bottom_left.x * factor, p.bottom_left.y * factor),
        _point(p.bottom_right.x * factor, p.bottom_right.y * factor),
    )
    return CustomQR(
        getattr(qr, "text", ""), pos, getattr(qr, "pattern_name", "UNKNOWN")
    )


def _qr_center(qr):
    if not hasattr(qr, "position") or not qr.position:
        return None
    p = qr.position
    xs = [p.top_left.x, p.top_right.x, p.bottom_left.x, p.bottom_right.x]
    ys = [p.top_left.y, p.top_right.y, p.bottom_left.y, p.bottom_right.y]
    return (sum(xs) / 4.0, sum(ys) / 4.0)


def _shift_qr_position(qr, offset_x, offset_y):
    if not hasattr(qr, "position") or not qr.position:
        return qr
    p = qr.position
    pos = CustomPosition(
        _point(p.top_left.x + offset_x, p.top_left.y + offset_y),
        _point(p.top_right.x + offset_x, p.top_right.y + offset_y),
        _point(p.bottom_left.x + offset_x, p.bottom_left.y + offset_y),
        _point(p.bottom_right.x + offset_x, p.bottom_right.y + offset_y),
    )
    return CustomQR(
        getattr(qr, "text", ""), pos, getattr(qr, "pattern_name", "UNKNOWN")
    )


def _convert_crop_qr_to_page(qr, origin, scale):
    if not hasattr(qr, "position") or not qr.position:
        return qr

    if scale <= 0:
        scale = 1.0

    ox, oy = origin
    p = qr.position

    pos = CustomPosition(
        _point(p.top_left.x / scale + ox, p.top_left.y / scale + oy),
        _point(p.top_right.x / scale + ox, p.top_right.y / scale + oy),
        _point(p.bottom_left.x / scale + ox, p.bottom_left.y / scale + oy),
        _point(p.bottom_right.x / scale + ox, p.bottom_right.y / scale + oy),
    )
    return CustomQR(
        getattr(qr, "text", ""), pos, getattr(qr, "pattern_name", "UNKNOWN")
    )


def _is_valid_qr_area(qr, img_height, pad=0):
    center = _qr_center(qr)
    if not center or img_height <= 0:
        return True

    actual_y = center[1] - pad
    actual_h = img_height - (pad * 2)
    if actual_h <= 0:
        return True

    y_ratio = actual_y / float(actual_h)
    return y_ratio >= MIN_QR_Y_RATIO


def analyze_image_quality(gray_img):
    if gray_img is None or gray_img.size == 0:
        return {
            "is_blurry": False,
            "is_low_contrast": False,
            "blur_score": 0.0,
            "contrast_score": 0.0,
        }

    blur_score = float(cv2.Laplacian(gray_img, cv2.CV_64F).var())
    contrast_score = float(gray_img.std())

    is_blurry = blur_score < 120.0
    is_low_contrast = contrast_score < 45.0

    return {
        "is_blurry": is_blurry,
        "is_low_contrast": is_low_contrast,
        "blur_score": round(blur_score, 1),
        "contrast_score": round(contrast_score, 1),
    }


def sort_qr_by_x(results):
    if not results:
        return []
    return sorted(
        results, key=lambda qr: _qr_center(qr)[0] if _qr_center(qr) else 0
    )


def yy_mm_to_japanese(yy_mm: str) -> str:
    if not yy_mm or len(yy_mm) != 4:
        return ""
    try:
        yy = int(yy_mm[:2])
        mm = int(yy_mm[2:])
        if not (1 <= mm <= 12):
            return ""
    except Exception:
        return ""

    year_full = (1900 + yy) if yy >= 89 else (2000 + yy)
    if year_full < 2019 or (year_full == 2019 and mm <= 4):
        return f"平成{year_full - 1988}年{mm}月"
    return f"令和{year_full - 2018}年{mm}月"


def extract_first_registration(qr1_text: str) -> str:
    if not qr1_text:
        return ""
    raw_str = (
        getattr(qr1_text, "text", str(qr1_text))
        if not isinstance(qr1_text, str)
        else qr1_text
    )
    norm_text = _normalize_qr_text(raw_str)
    parts = str(norm_text).split("/")
    if len(parts) >= 5 and re.fullmatch(r"\d{4}", parts[4]):
        return yy_mm_to_japanese(parts[4])
    match = re.search(r"\d{4}", str(norm_text))
    return yy_mm_to_japanese(match.group(0)) if match else ""


def split_vehicle_number(QR4_text, QR5_text="", QR6_text=""):
    """QRコード情報から自動車登録番号・車両番号要素（管轄・分類・かな・指定番号）を抽出."""
    t4 = getattr(QR4_text, "text", str(QR4_text)) if QR4_text else ""
    t5 = getattr(QR5_text, "text", str(QR5_text)) if QR5_text else ""
    t6 = getattr(QR6_text, "text", str(QR6_text)) if QR6_text else ""

    t4 = _normalize_qr_text(t4)
    t5 = _normalize_qr_text(t5)
    t6 = _normalize_qr_text(t6)

    if not t4 and not t5 and not t6:
        return "", "", "", ""

    # 解析用文字列抽出 (スラッシュ区切りの場合に対応)
    qr4_main = t4.split("/", 1)[1] if "/" in t4 else t4
    qr5_first = t5.split("/")[0] if "/" in t5 else ""

    # 1. 地名(管轄): 最初の漢字・ひらがな等文字列
    match_pref = re.match(r"^([^\d]+)", qr4_main)
    F = match_pref.group(1).strip() if match_pref else ""

    rest = qr4_main[len(match_pref.group(1)):] if match_pref else qr4_main

    # 2. 分類番号: 地名の直後の数字（2桁または3桁）
    match_num = re.match(r"^(\d{2,3})", rest)
    if not match_num:
        # 万が一1桁やその他の数字パターンだった場合のフォールバック
        match_num = re.match(r"^(\d+)", rest)

    G = match_num.group(1) if match_num else ""
    rest2 = rest[len(G):] if G else rest

    # 3. かな文字: 数字直後の非数字（ひらがな・アルファベット等）
    # ※ 2桁分類番号などでスペースが入っている場合や、かなが無い場合にも対応
    match_hira = re.search(r"([ぁ-んあ-んA-Za-zあ-んあ-んあ-んわ-んあ-んあ-ん])", rest2)
    
    # 日本語の一般的なナンバー用かな（あ〜ん、わ、れ、ろ、よ、等）または英字を探索
    match_kana_char = re.match(r"^\s*([^\d\s/])", rest2)
    if match_kana_char:
        H = match_kana_char.group(1).strip()
        rest3 = rest2[rest2.find(H) + len(H):]
    else:
        H = ""
        rest3 = rest2

    # 4. 一連指定番号: 残りの数字 ＋ QR5の先頭数字
    match_digit = re.search(r"(\d+)", rest3)
    I_part1 = match_digit.group(1) if match_digit else ""
    
    # もしrest3から番号が取れず、かな領域などに番号が紛れ込んでいた場合のフォロー
    if not I_part1:
        match_digit_fallback = re.search(r"(\d+)", rest2)
        if match_digit_fallback:
            I_part1 = match_digit_fallback.group(1)

    I = I_part1 + qr5_first

    return F, G, H, I

def draw_qr_boxes(pil_img, results_sorted, scale=1.0, font=None):
    draw = ImageDraw.Draw(pil_img)

    for idx, qr in enumerate(results_sorted, start=1):
        if not hasattr(qr, "position") or not qr.position:
            continue

        p = qr.position
        xs = [p.top_left.x, p.top_right.x, p.bottom_left.x, p.bottom_right.x]
        ys = [p.top_left.y, p.top_right.y, p.bottom_left.y, p.bottom_right.y]

        x0 = int(min(xs) * scale)
        y0 = int(min(ys) * scale)
        x1 = int(max(xs) * scale)
        y1 = int(max(max(ys) * scale, y0 + 1))

        draw.rectangle([x0, y0, x1, y1], outline="#00ffcc", width=3)

        label = f"QR{idx}"
        if font:
            bbox = font.getbbox(label)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            ly = max(0, y0 - th - 6)
            draw.rectangle([x0, ly, x0 + tw + 8, y0], fill="#00ffcc")
            draw.text((x0 + 4, ly + 2), label, fill="#000000", font=font)
        else:
            ly = max(0, y0 - 20)
            draw.rectangle([x0, ly, x0 + 40, y0], fill="#00ffcc")
            draw.text((x0 + 4, ly + 2), label, fill="#000000")

    return pil_img


# ============================================================
# バーコード読み取り・処理ロジック (マルチエンジン構成)
# ============================================================


def _read_barcodes(image, pattern_name):
    if image is None or getattr(image, "size", 0) == 0:
        return []

    # 画像の最小サイズチェック（C++ライブラリ内のクラッシュ防止）
    h, w = image.shape[:2] if isinstance(image, np.ndarray) else (image.height, image.width)
    if w < 20 or h < 20:
        return []

    results = []
    seen_texts = set()

    # 1. WeChatQRCode (最優先)
    if HAS_WECHAT and wechat_detector is not None:
        try:
            res, points = wechat_detector.detectAndDecode(image)
            if res and points is not None and len(points) > 0:
                for txt_raw, pts in zip(res, points):
                    txt = _normalize_qr_text(txt_raw)
                    if txt and txt not in seen_texts:
                        seen_texts.add(txt)
                        pos = CustomPosition(
                            _point(pts[0][0], pts[0][1]),
                            _point(pts[1][0], pts[1][1]),
                            _point(pts[3][0], pts[3][1]),
                            _point(pts[2][0], pts[2][1]),
                        )
                        results.append(CustomQR(text=txt, position=pos, pattern_name=pattern_name))
        except Exception:
            pass

    # 2. zxingcpp (フォールバック)
    if HAS_ZXING and len(results) < MAX_QR:
        try:
            barcodes = zxingcpp.read_barcodes(image)
            for b in barcodes:
                txt = _normalize_qr_text(getattr(b, "text", ""))
                if txt and txt not in seen_texts:
                    seen_texts.add(txt)
                    p = b.position
                    pos = CustomPosition(
                        _point(p.top_left.x, p.top_left.y),
                        _point(p.top_right.x, p.top_right.y),
                        _point(p.bottom_left.x, p.bottom_left.y),
                        _point(p.bottom_right.x, p.bottom_right.y),
                    )
                    results.append(CustomQR(text=txt, position=pos, pattern_name=pattern_name))
        except Exception:
            pass

    # 3. pyzbar (極小・ノイズ画像によるCライブラリアサーションクラッシュ防止ガルド付き)
    #if HAS_PYZBAR and len(results) < MAX_QR:
    #    try:
    #        with SuppressStderr():
    #            # 画像の縦横が50px以上ある場合のみpyzbarを安全に実行
    #            if w >= 50 and h >= 50:
    #                zbar_objs = zbar_decode(image, symbols=[ZBarSymbol.QRCODE])
    #                for obj in zbar_objs:
    #                    txt = _normalize_qr_text(obj.data)
    #                    if txt and txt not in seen_texts:
    #                        seen_texts.add(txt)
    #                        r = obj.rect
    #                        pos = CustomPosition(
    #                            _point(r.left, r.top),
    #                            _point(r.left + r.width, r.top),
    #                            _point(r.left, r.top + r.height),
    #                            _point(r.left + r.width, r.top + r.height),
    #                        )
    #                        results.append(CustomQR(text=txt, position=pos, pattern_name=pattern_name))
    #    except Exception:
    #        pass

    return results

def _add_qrs(
    found,
    seen_texts,
    qrs,
    scale_inv=1.0,
    img_height=0,
    pad=0,
    check_area=False,
):
    if not qrs:
        return 0

    added = 0
    for qr in qrs:
        text = _normalize_qr_text(getattr(qr, "text", ""))
        if not _is_usable_qr_text(text) or text in seen_texts:
            continue

        if scale_inv != 1.0:
            qr = scale_qr_position(qr, scale_inv)

        if (
            check_area
            and img_height > 0
            and not _is_valid_qr_area(qr, img_height, pad)
        ):
            continue

        seen_texts.add(text)
        found.append(qr)
        added += 1
        if len(found) >= MAX_QR:
            break

    return added


def _clip_box(cx, cy, half_w, half_h, w, h):
    x0 = max(0, int((cx - half_w) * w))
    y0 = max(0, int((cy - half_h) * h))
    x1 = min(w, int((cx + half_w) * w))
    y1 = min(h, int((cy + half_h) * h))
    return x0, y0, x1, y1


def _slot_crop(img, slot_no, dx=0.0, dy=0.0, half_w=None, half_h=None):
    h, w = img.shape[:2]
    if slot_no < 0 or slot_no >= MAX_QR:
        return None, None

    cx = QR_SLOT_CENTERS_X[slot_no] + dx
    cy = QR_SLOT_CENTER_Y + dy
    hw = QR_SLOT_HALF_W if half_w is None else half_w
    hh = QR_SLOT_HALF_H if half_h is None else half_h

    x0, y0, x1, y1 = _clip_box(cx, cy, hw, hh, w, h)
    if x1 <= x0 or y1 <= y0:
        return None, None
    return img[y0:y1, x0:x1], (x0, y0)


def _crop_qr_variants(crop, extended=False):
    if crop is None or crop.size == 0:
        return []

    gray = (
        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if len(crop.shape) == 3
        else crop.copy()
    )
    variants = []

    variants.append(gray)

    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    cl_img = clahe.apply(gray)
    variants.append(cl_img)

    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    unsharp = cv2.addWeighted(gray, 2.0, gaussian, -1.0, 0)
    variants.append(unsharp)

    _, otsu = cv2.threshold(cl_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    if extended:
        adaptive_thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            5,
        )
        variants.append(adaptive_thresh)

        kernel = np.ones((2, 2), np.uint8)
        morph_close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        variants.append(morph_close)

    return variants


def _decode_slot_once(
    img, slot_no, pattern_prefix, dx=0.0, dy=0.0, extended_variants=False, seen_texts=None
):
    crop, origin = _slot_crop(img, slot_no, dx, dy)
    if crop is None:
        return None

    for scale in SLOT_SCALES:
        scaled = cv2.resize(
            crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

        variants = _crop_qr_variants(scaled, extended=extended_variants)
        for v_idx, variant in enumerate(variants):
            p_name = f"{pattern_prefix}_s{scale}x_v{v_idx}"
            qrs = _read_barcodes(variant, p_name)
            for qr in qrs:
                txt = _normalize_qr_text(getattr(qr, "text", ""))
                if seen_texts is not None and txt in seen_texts:
                    continue
                if _is_usable_qr_text(txt):
                    return _convert_crop_qr_to_page(qr, origin, scale)

    return None


def _calculate_slot_offsets(found, img_shape):
    h, w = img_shape[:2]
    dx_list = []
    dy_list = []

    for qr in found:
        center = _qr_center(qr)
        if not center:
            continue
        cx_rel = center[0] / w
        cy_rel = center[1] / h

        best_slot = min(
            range(MAX_QR),
            key=lambda i: abs(QR_SLOT_CENTERS_X[i] - cx_rel),
        )
        dx_list.append(cx_rel - QR_SLOT_CENTERS_X[best_slot])
        dy_list.append(cy_rel - QR_SLOT_CENTER_Y)

    if not dx_list:
        return 0.0, 0.0

    return float(np.mean(dx_list)), float(np.mean(dy_list))


# --- パターン定義ブロック ---


def _pattern_p4_record_bottom_individual(img, found, seen):
    """【パターン4】自動車検査証記録事項（A4縦型）専用：下部全域の一括検出 ＋ 個別クロップ."""
    if img is None or is_detection_sufficient(found):
        return

    h, w = img.shape[:2]

    # 1. 下部全域（横幅10%〜95%、縦幅60%〜98%）を一括クロップしてWeChatQRCode等で全検出を試みる
    y1, y2 = int(h * 0.60), int(h * 0.98)
    x1, x2 = int(w * 0.05), int(w * 0.95)

    if y2 > y1 and x2 > x1:
        crop_all = img[y1:y2, x1:x2]
        if crop_all.size > 0:
            for scale in (2.0, 3.0):
                scaled = cv2.resize(
                    crop_all, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )
                # WeChat / ZXing / pyzbar で一括読み取り
                qrs = _read_barcodes(scaled, f"P4_BOTTOM_FULL_s{scale}x")
                for qr in qrs:
                    qr_page = _convert_crop_qr_to_page(qr, (x1, y1), scale)
                    text = _normalize_qr_text(getattr(qr_page, "text", ""))
                    if _is_usable_qr_text(text) and text not in seen:
                        _add_qrs(found, seen, [qr_page], check_area=False)
                
                # 必要な数（5個または6個）揃ったら終了
                if is_detection_sufficient(found):
                    return
    # 2. 6個の各スロット領域をピンポイント探索
    if not is_detection_sufficient(found):
        x_ratios = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]

        for x_r in x_ratios:
            if is_detection_sufficient(found):
                break

            cx = int(w * x_r)
            cy = int(h * 0.88)
            hw = int(w * 0.08)
            hh = int(h * 0.08)

            sub_x0, sub_x1 = max(0, cx - hw), min(w, cx + hw)
            sub_y0, sub_y1 = max(0, cy - hh), min(h, cy + hh)

            sub_crop = img[sub_y0:sub_y1, sub_x0:sub_x1]
            if sub_crop.size == 0:
                continue

            for scale in (3.0, 4.0, 5.0):
                scaled = cv2.resize(
                    sub_crop,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
                for v_idx, variant in enumerate(
                    _crop_qr_variants(scaled, extended=True)
                ):
                    p_name = f"P4_SINGLE_s{scale}x_v{v_idx}"
                    qrs = _read_barcodes(variant, p_name)
                    if qrs:
                        for qr in qrs:
                            qr_page = _convert_crop_qr_to_page(
                                qr, (sub_x0, sub_y0), scale
                            )
                            text = _normalize_qr_text(
                                getattr(qr_page, "text", "")
                            )
                            if _is_usable_qr_text(text) and text not in seen:
                                _add_qrs(
                                    found, seen, [qr_page], check_area=False
                                )
                                break


def _pattern_p1_slot_standard(
    img, found, seen, slot_status, extended_variants=False
):
    if img is None:
        return
    prefix = "P1_SLOT_EXT" if extended_variants else "P1_SLOT_STD"

    for slot_no in range(MAX_QR):
        if is_detection_sufficient(found):
            break
        if slot_status[slot_no]:
            continue

        shift_x, shift_y = _calculate_slot_offsets(found, img.shape)
        qr = _decode_slot_once(
            img,
            slot_no,
            pattern_prefix=prefix,
            dx=shift_x,
            dy=shift_y,
            extended_variants=extended_variants,
            seen_texts=seen,
        )
        if qr is not None:
            text = _normalize_qr_text(getattr(qr, "text", ""))
            if _is_usable_qr_text(text) and text not in seen:
                if _add_qrs(found, seen, [qr], check_area=False) > 0:
                    slot_status[slot_no] = True


def _pattern_p2_slot_tracking(img, found, seen, slot_status):
    if is_detection_sufficient(found):
        return

    base_dx, base_dy = _calculate_slot_offsets(found, img.shape)

    for slot_no in range(MAX_QR):
        if is_detection_sufficient(found):
            return
        if slot_status[slot_no]:
            continue

        success = False
        for dy in TRACK_DY:
            for dx in TRACK_DX:
                qr = _decode_slot_once(
                    img,
                    slot_no,
                    pattern_prefix="P2_SLOT_TRACK",
                    dx=base_dx + dx,
                    dy=base_dy + dy,
                    seen_texts=seen,
                )
                if qr is None:
                    continue

                text = _normalize_qr_text(getattr(qr, "text", ""))
                if not _is_usable_qr_text(text) or text in seen:
                    continue

                if _add_qrs(found, seen, [qr], check_area=False) > 0:
                    slot_status[slot_no] = True
                    success = True
                    break
            if success:
                break


def _pattern_p3_full_image(gray, color, found, seen, pad):
    h, w = gray.shape[:2]

    qrs_gray = _read_barcodes(gray, pattern_name="P3_FULL_GRAY")
    _add_qrs(
        found,
        seen,
        qrs_gray,
        1.0,
        img_height=h,
        pad=pad,
        check_area=True,
    )

    if not is_detection_sufficient(found) and color is not None:
        qrs_color = _read_barcodes(color, pattern_name="P3_FULL_COLOR")
        _add_qrs(
            found,
            seen,
            qrs_color,
            1.0,
            img_height=h,
            pad=pad,
            check_area=True,
        )


def _pattern_p5_grid_fallback(gray_img, found, seen, pad):
    if gray_img is None or is_detection_sufficient(found):
        return

    h, w = gray_img.shape[:2]
    y_min = int(h * MIN_QR_Y_RATIO)
    crop_bottom = gray_img[y_min:h, 0:w]
    cb_h, cb_w = crop_bottom.shape[:2]

    if cb_h <= 0 or cb_w <= 0:
        return

    cols, rows = 12, 2
    tile_w = cb_w // cols
    tile_h = cb_h // rows

    for r in range(rows):
        for c in range(cols):
            if is_detection_sufficient(found):
                return

            x0 = max(0, c * tile_w - int(tile_w * 0.2))
            y0 = max(0, r * tile_h - int(tile_h * 0.2))
            x1 = min(cb_w, (c + 1) * tile_w + int(tile_w * 0.2))
            y1 = min(cb_h, (r + 1) * tile_h + int(tile_h * 0.2))

            tile = crop_bottom[y0:y1, x0:x1]
            if tile.size == 0:
                continue

            for scale in (2.0, 3.0, 4.0):
                if is_detection_sufficient(found):
                    return

                scaled = cv2.resize(
                    tile, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )
                variants = _crop_qr_variants(scaled, extended=True)

                for v_idx, variant in enumerate(variants):
                    p_name = f"P5_GRID_FALLBACK_r{r}_c{c}_s{scale}x_v{v_idx}"
                    qrs = _read_barcodes(variant, p_name)

                    for qr in qrs:
                        crop_origin = (x0, y0 + y_min)
                        qr_page = _convert_crop_qr_to_page(qr, crop_origin, scale)
                        text = _normalize_qr_text(getattr(qr_page, "text", ""))

                        if _is_usable_qr_text(text) and text not in seen:
                            _add_qrs(
                                found,
                                seen,
                                [qr_page],
                                check_area=True,
                                img_height=h,
                                pad=pad,
                            )
                            if is_detection_sufficient(found):
                                return


# ============================================================
# エントリポイント (自動判別 ＋ パイプライン制御)
# ============================================================


def detect_best_qr(img, page_key=None):
    if img is None:
        return []

    page_label = f" [Page: {page_key}]" if page_key is not None else ""

    if page_key is not None and page_key in _qr_cache:
        return _qr_cache[page_key]

    pad = PADDING_PX
    img_padded = cv2.copyMakeBorder(
        img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    if len(img_padded.shape) == 3:
        color = img_padded
        gray = cv2.cvtColor(img_padded, cv2.COLOR_BGR2GRAY)
    else:
        color = None
        gray = img_padded

    quality = analyze_image_quality(gray)
    is_hard_image = quality["is_blurry"] or quality["is_low_contrast"]

    h_raw, w_raw = img.shape[:2]
    is_portrait = h_raw > w_raw  # A4縦型（自動車検査証記録事項）

    if DEBUG:
        status_msg = (
            "LOW_CONTRAST"
            if quality["is_low_contrast"]
            else ("BLURRY" if quality["is_blurry"] else "CLEAR")
        )
        shape_msg = "PORTRAIT" if is_portrait else "LANDSCAPE"
        print(
            f"Quality Check{page_label} -> Score (Blur: {quality['blur_score']}, Contrast: {quality['contrast_score']}) | Status: [{status_msg}] | Layout: [{shape_msg}]"
        )

    found = []
    seen = set()
    slot_status = [False] * MAX_QR

    try:
        # ① 【縦型車検証（記録事項）】最下部P4パターンを優先判定
        if is_portrait:
            _pattern_p4_record_bottom_individual(img_padded, found, seen)
            if is_detection_sufficient(found):
                if DEBUG:
                    print(
                        f"  ⚡ Early return triggered: P4 matched ({len(found)} QR detected, Kei-Check: {is_kei_vehicle_qr(found)})"
                    )
                return _finish_and_cache(found, pad, page_key, page_label)

        # ② 画質低下画像向けの拡張P1パターン
        if not is_detection_sufficient(found) and is_hard_image:
            _pattern_p1_slot_standard(
                img_padded,
                found,
                seen,
                slot_status,
                extended_variants=True,
            )
            if is_detection_sufficient(found):
                if DEBUG:
                    print(
                        f"  ⚡ Early return triggered: P1_EXT matched ({len(found)} QR detected, Kei-Check: {is_kei_vehicle_qr(found)})"
                    )
                return _finish_and_cache(found, pad, page_key, page_label)

        # ③ 標準スロット P1パターン
        if not is_detection_sufficient(found):
            _pattern_p1_slot_standard(
                img_padded,
                found,
                seen,
                slot_status,
                extended_variants=False,
            )
            if is_detection_sufficient(found):
                if DEBUG:
                    print(
                        f"  ⚡ Early return triggered: P1_STD matched ({len(found)} QR detected, Kei-Check: {is_kei_vehicle_qr(found)})"
                    )
                return _finish_and_cache(found, pad, page_key, page_label)

        # ④ ズレ補正 P2パターン
        if not is_detection_sufficient(found):
            _pattern_p2_slot_tracking(img_padded, found, seen, slot_status)
            if is_detection_sufficient(found):
                if DEBUG:
                    print(
                        f"  ⚡ Early return triggered: P2 matched ({len(found)} QR detected, Kei-Check: {is_kei_vehicle_qr(found)})"
                    )
                return _finish_and_cache(found, pad, page_key, page_label)

        # ⑤ 全体スキャン P3パターン
        if not is_detection_sufficient(found):
            _pattern_p3_full_image(gray, color, found, seen, pad)
            if is_detection_sufficient(found):
                if DEBUG:
                    print(
                        f"  ⚡ Early return triggered: P3 matched ({len(found)} QR detected, Kei-Check: {is_kei_vehicle_qr(found)})"
                    )
                return _finish_and_cache(found, pad, page_key, page_label)

        # ⑥ 最終フォールバック（グリッド超解像検出）
        if not is_detection_sufficient(found):
            if DEBUG:
                print(
                    f"  🔍 Fallback pipeline ⑥ triggered ({len(found)} QR detected so far)"
                )
            _pattern_p5_grid_fallback(gray, found, seen, pad)
            if is_detection_sufficient(found):
                if DEBUG:
                    print(
                        f"  ⚡ Early return triggered: Pipeline ⑥ matched ({len(found)} QR detected, Kei-Check: {is_kei_vehicle_qr(found)})"
                    )
                return _finish_and_cache(found, pad, page_key, page_label)

        # ⑦ A4縦型専用 縮尺・歪み補正スライドウィンドウ超解像レスキュー
        if not is_detection_sufficient(found) and is_portrait:
            if DEBUG:
                print(
                    f"  🔍 Fallback pipeline ⑦ (Portrait Dynamic Window Rescue) triggered ({len(found)} QR detected so far)"
                )
            h, w = img_padded.shape[:2]

            y_min = int(h * 0.65)
            y_max = int(h * 0.98)

            rows, cols = 2, 10
            win_w = int(w * 0.18)
            win_h = int((y_max - y_min) * 0.65)

            step_x = max(1, (w - win_w) // (cols - 1)) if cols > 1 else win_w
            step_y = (
                max(1, ((y_max - y_min) - win_h) // (rows - 1))
                if rows > 1
                else win_h
            )

            for r in range(rows):
                if is_detection_sufficient(found):
                    break
                for c in range(cols):
                    if is_detection_sufficient(found):
                        break

                    x0 = max(0, c * step_x)
                    y0 = max(y_min, y_min + r * step_y)
                    x1 = min(w, x0 + win_w)
                    y1 = min(y_max, y0 + win_h)

                    crop_target = img_padded[y0:y1, x0:x1]
                    if crop_target.size == 0:
                        continue

                    for scale in (2.5, 4.0):
                        if is_detection_sufficient(found):
                            break
                        scaled = cv2.resize(
                            crop_target,
                            None,
                            fx=scale,
                            fy=scale,
                            interpolation=cv2.INTER_CUBIC,
                        )

                        for v_idx, variant in enumerate(
                            _crop_qr_variants(scaled, extended=True)
                        ):
                            p_name = f"P7_SLIDE_r{r}_c{c}_s{scale}x_v{v_idx}"
                            qrs = _read_barcodes(variant, p_name)
                            for qr in qrs:
                                qr_page = _convert_crop_qr_to_page(
                                    qr, (x0, y0), scale
                                )
                                text = _normalize_qr_text(
                                    getattr(qr_page, "text", "")
                                )
                                if _is_usable_qr_text(text) and text not in seen:
                                    _add_qrs(
                                        found,
                                        seen,
                                        [qr_page],
                                        check_area=False,
                                    )
                                    if is_detection_sufficient(found):
                                        break

        return _finish_and_cache(found, pad, page_key, page_label)

    finally:
        del img_padded
        if color is not None:
            del color
        del gray
        gc.collect()


def _finish_and_cache(found, pad, page_key, page_label):
    found = sort_qr_by_x(found)
    final_qrs = found[:MAX_QR]

    restored_qrs = [_shift_qr_position(qr, -pad, -pad) for qr in final_qrs]

    if page_key is not None:
        _qr_cache[page_key] = restored_qrs

    if DEBUG:
        is_kei = is_kei_vehicle_qr(restored_qrs)
        kei_label = " [軽自動車]" if is_kei else " [登録車/標準]"
        print(
            f"Result Check{page_label} -> Total QR Found: {len(restored_qrs)}{kei_label}"
        )

    return restored_qrs