# -*- coding: utf-8 -*-
import difflib
import re
import unicodedata

# ============================================================
# 【最優先】車検証上の実際の表記が「三菱」になる型式リスト
# ============================================================
FORCE_MITSU_MODELS = [
    "FS74", "FP74", "FU74", "FY74",
    "FS64", "FP64", "FU64", "FY64",
    "FS54", "FP54", "FU54", "FY54",
    "BE63", "BE64", "BE",
    "FK72", "FK71", "FK"
]

# 三菱の強制判定から除外する他社特有の原動機型式
EXCLUDE_MITSU_ENGINE_PATTERNS = [
    "J05", "J07", "J08", "H06", "H07", "P09", "E13", "N04", "W04", "W06", "A05", "A09", "E13C",
    "FE6", "FD6", "PF6", "PD6", "MD92", "GE13", "4HL1", "4HK1", "4P10", "HR16"
]

# ============================================================
# 【最重要】型式プレフィックス・コードに基づく「完全一致マスター辞書」
# ============================================================
EXACT_MODEL_MAP = {
    # --- 日野 ---
    "GK8": "日野", "FR1": "日野", "EXY": "日野",
    "RX4": "日野", "HX6": "日野", "FE1": "日野", "GD1": "日野",
    "FW1": "日野", "FW": "日野", "FS1": "日野", "SH1": "日野", "FJ7": "日野", "FJ": "日野",
    "FD2": "日野", "FD": "日野",
    
    # --- いすゞ ---
    "NPR": "いすゞ", "NKR": "いすゞ", "NMR": "いすゞ", "NHR": "いすゞ",
    "FRR": "いすゞ", "FSR": "いすゞ", "CYG": "いすゞ", "CXZ": "いすゞ",
    "EXD": "いすゞ", "CYH": "いすゞ", "EXR": "いすゞ", "CXG": "いすゞ",
    
    # --- 三菱（車検証上の表記が「三菱」となる特有型式） ---
    "FS64": "三菱", "FP64": "三菱", "FU64": "三菱", "FY64": "三菱",
    "FS54": "三菱", "FP54": "三菱", "FU54": "三菱", "FY54": "三菱",
    "BE63": "三菱", "BE64": "三菱", "BE": "三菱",
    
    # --- ニッサン（単体） ---
    "F24": "ニッサン", "E26": "ニッサン", "E25": "ニッサン", "VM20": "ニッサン",
}

# 対象メーカーの一覧
TRUCK_AND_NISSAN_MAKERS = [
    "三菱",
    "三菱ふそう",
    "ニッサンディーゼル",
    "いすゞ", 
    "日野", 
    "UDトラックス",
    "UD",
    "ニッサン",
    "スカニア",
    "トヨタ"
]

# ============================================================
# 【補助】初度登録年（文字列）を西暦の整数に変換する関数
# ============================================================
def parse_year_to_int(year_str):
    if not year_str:
        return None
    s = unicodedata.normalize('NFKC', str(year_str)).strip()
    
    match_ad = re.search(r'(19\d{2}|20\d{2})', s)
    if match_ad:
        return int(match_ad.group(1))

    if '令和' in s or s.startswith('R'):
        m = re.search(r'(?:令和|R)(\d+)', s)
        if m:
            return 2018 + int(m.group(1))
    elif '平成' in s or s.startswith('H'):
        m = re.search(r'(?:平成|H)(\d+)', s)
        if m:
            return 1988 + int(m.group(1))
    elif '昭和' in s or s.startswith('S'):
        m = re.search(r'(?:昭和|S)(\d+)', s)
        if m:
            return 1925 + int(m.group(1))

    return None


# ============================================================
# 【共通】XZU / XZC 型式のOEM（トヨタ ⇔ 日野）を判定する補助関数
# ============================================================
def resolve_xzu_maker(text_clean, year_int):
    if "XZU" in text_clean or "XZC" in text_clean:
        older_prefixes = ["BDG-", "PDG-", "ADG-", "PB-", "KK-", "KL-", "KC-"]
        if any(p in text_clean for p in older_prefixes):
            return "トヨタ"
        if year_int and year_int <= 2012:
            return "トヨタ"
        return "日野"
    return None


# ============================================================
# 【最優先チェック】強制的に「三菱」にすべき型式か判定する関数
# ============================================================
def check_force_mitsu(model_text, engine_text=""):
    if not model_text:
        return False
    
    combined = f"{str(model_text)} {str(engine_text)}"
    combined_clean = unicodedata.normalize('NFKC', combined).replace(" ", "").upper()

    for excl_eng in EXCLUDE_MITSU_ENGINE_PATTERNS:
        if excl_eng in combined_clean:
            return False

    clean_model = unicodedata.normalize('NFKC', str(model_text)).strip().upper()
    for prefix in FORCE_MITSU_MODELS:
        if prefix in clean_model:
            return True
    return False


# ============================================================
# 型式コードから正確なメーカーをマスター辞書で引き当てる関数
# ============================================================
def match_maker_by_exact_model(model_text):
    if not model_text:
        return None
    clean_model = unicodedata.normalize('NFKC', str(model_text)).strip().upper()
    
    if "BPR" in clean_model or "BKR" in clean_model:
        return None

    for key, maker in EXACT_MODEL_MAP.items():
        if key in clean_model:
            return maker
    return None


# ============================================================
# QRコードのテキスト群からメーカーを判定する関数（年式考慮）
# ============================================================
def match_maker_from_qr_texts(qr_texts, initial_year=None):
    if not qr_texts:
        return None
      
    combined_text = " ".join([str(q.text if hasattr(q, 'text') else q) for q in qr_texts])
    combined_clean = unicodedata.normalize('NFKC', combined_text).replace(" ", "").upper()
    year_int = parse_year_to_int(initial_year)

    if check_force_mitsu(combined_text):
        return "三菱"
    
    if "トヨタ" in combined_text or "TOYOTA" in combined_text.upper():
        return "トヨタ"
    if "日野" in combined_text or "HINO" in combined_text.upper():
        return "日野"

    xzu_maker = resolve_xzu_maker(combined_clean, year_int)
    if xzu_maker:
        return xzu_maker

    exact = match_maker_by_exact_model(combined_text)
    if exact:
        return exact

    # ニッサンの判定（VM20等を優先）
    nissan_models = ["アトラス", "キャラバン", "バネット", "NV350", "SQ1F24", "SQ2F24", "TZF24", "VW6E26", "CSGE25", "VM20", "HR16"]
    for m in nissan_models:
        if m in combined_clean:
            return "ニッサン"

    # スカニアの判定
    if "DC13" in combined_clean or "DC16" in combined_clean or combined_clean.startswith("YS2") or "SCANIA" in combined_clean:
        return "スカニア"

    # UDトラックス / ニッサンディーゼル判定
    ud_models = ["クオン", "コンドル", "カゼット", "ビッグサム", "CG4ZA", "CG5FA", "GH5", "GH7", "GH11", "GH13", "GE13", "MK38", "MK", "GK5", "GK5X", "CW", "CD", "CG", "CV", "CK", "BVZ", "BTR", "BSR", "BRR", "BVK", "BPR", "BKR", "FEB", "FEC", "FED", "FGA", "FGB", "UD", "FE6", "FD6"]
    for m in ud_models:
        if m in combined_clean:
            old_prefixes = ["PKG-", "PDG-", "ADG-", "BDG-", "PB-", "KL-", "KC-", "U-", "P-", "K-"]
            if (year_int and year_int <= 2010) or any(p in combined_clean for p in old_prefixes):
                return "ニッサンディーゼル"
            return "UDトラックス"

    # 日野の判定
    hino_models = [
        "GK8J", "GK8", "プロフィア", "レンジャー", "デュトロ", "セレガ", 
        "RX4J", "HX6J", "HU", "HT", "HR", "KV", "FR1", "EXY", 
        "J05", "J07", "J08", "H06", "H07", "P09", "E13", "N04", "W04", "W06",
        "FW", "FS1", "SH1", "FJ", "FD", "A05C", "A09C", "E13C", "A09", "FE1"
    ]
    for m in hino_models:
        if m in combined_clean:
            return "日野"

    # いすゞの判定
    isuzu_models = ["ギガ", "フォワード", "エルフ", "CYJ", "CYG", "CXZ", "EXD", "CYH", "EXR", "CXG", "BPR", "NPR", "NKR", "NHR", "-LR", "-LV", "NMR"]
    for m in isuzu_models:
        if m in combined_clean:
            return "いすゞ"

    # 三菱ふそう系
    mitsu_keywords = ["MITSUBISHI", "FUSO", "ふそう", "ローザ", "スーパーグレート", "ファイター", "キャンター", "BE63", "BE64", "4P10", "4M50", "4M40", "6R10", "6R20", "6S10", "6M60", "FK"]
    for mk in mitsu_keywords:
        if mk in combined_clean:
            if year_int and year_int < 2003:
                return "三菱"
            return "三菱ふそう"

    return None


# ============================================================
# 【最重要】型式・原動機型式・年式を組み合わせて判定するメインロジック
# ============================================================
def match_maker_with_model_and_engine(model_text, engine_text, initial_year=None, vin_text=None):
    if check_force_mitsu(model_text, engine_text):
        return "三菱"

    combined = f"{str(model_text)} {str(engine_text)} {str(vin_text)}"
    combined_clean = unicodedata.normalize('NFKC', combined).replace(" ", "").upper()
    year_int = parse_year_to_int(initial_year)

    xzu_maker = resolve_xzu_maker(combined_clean, year_int)
    if xzu_maker:
        return xzu_maker

    # 1. ニッサン単体の判定（VM20, HR16等を最優先チェック）
    nissan_patterns = ["F24", "E26", "E25", "ZD30", "YD25", "VM20", "HR16"]
    if any(k in combined_clean for k in nissan_patterns):
        return "ニッサン"

    exact = match_maker_by_exact_model(model_text)
    if exact:
        return exact

    # スカニア特有の判定
    if any(k in combined_clean for k in ["DC13", "DC16", "SCANIA"]) or combined_clean.startswith("YS2"):
        return "スカニア"

    # UD / ニッサンディーゼル判定
    ud_patterns = ["GH5", "GH7", "GH11", "GH13", "GE13", "MD92", "MK38", "MK", "GK5", "GK5X", "CD", "CW", "CV", "CG", "CK", "BVZ", "BTR", "BSR", "BRR", "BVK", "BPR", "BKR", "FEB", "FEC", "FED", "FGA", "FGB", "FE6", "FD6"]
    if any(k in combined_clean for k in ud_patterns):
        old_prefixes = ["PKG-", "PDG-", "ADG-", "BDG-", "PB-", "KL-", "KC-", "U-", "P-", "K-"]
        if (year_int and year_int <= 2010) or any(p in combined_clean for p in old_prefixes):
            return "ニッサンディーゼル"
        return "UDトラックス"  

    # 2. 日野特有
    hino_patterns = [
        "GK8J", "GK8", "RX4J", "HX6J", "HX", "J05", "J07", "J08", "H06", "H07", "P09", "E13", 
        "N04", "W04", "W06", "FR1", "EXY", "SH", "FH", "FP", "FN", 
        "FE1J", "GD1J", "FW", "FS1", "SH1", "FJ", "FD", "A05C", "A09C", "E13C", "A09", "FE1"
    ]
    if any(k in combined_clean for k in hino_patterns):
        return "日野"
   
    # 3. いすゞ特有
    if any(k in combined_clean for k in ["NPR", "NKR", "NMR", "NHR", "FRR", "FSR", "CYG", "CXZ", "EXD", "CYH", "4HK1", "6HK1", "6UZ1"]):
        return "いすゞ"

    # 4. その他 三菱ふそう系トラック・バス
    mitsu_patterns = ["BE", "FS", "FP", "FU", "FK", "FE", "4P10", "4M50", "4M40", "6R10", "6R20", "6S10", "6M60", "MITSUBISHI", "FUSO"]
    if any(k in combined_clean for k in mitsu_patterns):
        if year_int and year_int < 2003:
            return "三菱"
        return "三菱ふそう"

    return None


# ============================================================
# メインエントリーポイント（OCR不使用・スペック＆QR判定専用）
# ============================================================
def extract_car_name_ocr(img_cv=None, page_num=None, pdf_path=None, qr_texts=None, model_text=None, engine_text=None, initial_year=None, vin_text=None):
    """
    OCRスキャンを行わず、QRテキスト、型式、原動機型式、初度登録年（年式）、車台番号を
    ベースにした純ロジックによりメーカー名を確定して返します。
    """
    try:
        if model_text and check_force_mitsu(model_text, engine_text):
            return "三菱"

        if model_text:
            exact_maker = match_maker_by_exact_model(model_text)
            if exact_maker:
                return exact_maker

        if model_text or engine_text or vin_text:
            maker_from_spec = match_maker_with_model_and_engine(model_text, engine_text, initial_year, vin_text)
            if maker_from_spec and maker_from_spec != "（取得できず）":
                return maker_from_spec

        if qr_texts:
            maker_from_qr = match_maker_from_qr_texts(qr_texts, initial_year)
            if maker_from_qr:
                return maker_from_qr
            
    except Exception:
        pass

    return "（取得できず）"