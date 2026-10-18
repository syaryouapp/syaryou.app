# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")  # すべての警告メッセージを抑制

import os
import sys
import io
import gc
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import re
import pymupdf  # PyMuPDF推奨インポート形式
from PIL import Image, ImageTk, ImageFont
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import zxingcpp

################-----------------------------------------------
# ログオーバーフロー・バッファ詰まり防止対策
################-----------------------------------------------
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)
sys.stderr = io.TextIOWrapper(
    sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
)

# 外部モジュール（同一ディレクトリに配置してください）
import maker_parser
import qr_module
import excel_module  # Excel出力を担当するモジュール

################-----------------------------------------------
# 高速化された傾き補正処理（縮小画像で角度判定）
################-----------------------------------------------
def deskew_image_fast(img_cv):
    try:
        h, w = img_cv.shape[:2]
        # 角度計算用に1/2に縮小して計算負荷を削減
        small = cv2.resize(img_cv, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 10:
            return img_cv
            
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        # 傾きが軽微ならスキップ
        if abs(angle) < 0.2 or abs(angle) > 15.0:
            return img_cv
            
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_cv, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return rotated
    except Exception as e:
        print(f"傾き補正例外: {e}", flush=True)
        return img_cv

ver = 'V09L03E24_qr_strict_fix'

BASE_EXCEL = os.path.join(os.getcwd(), "車両情報調査表(+++ ++++++様）20250827-01.xlsx")
RESULT_FOLDER = os.path.join(os.getcwd(), "結果")
os.makedirs(RESULT_FOLDER, exist_ok=True)

DISPLAY_WIDTH = 950
DISPLAY_HEIGHT = 850
CANVAS_BG = "#121212"

class PDFViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"車両情報調査表作成アプリ {ver}")
        self.root.state("zoomed")
        self.root.configure(bg="#1e1e1e")

        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)  # スレッドプール（並列実行用）

        try:
            self.font = ImageFont.truetype("meiryo.ttc", 14)
        except:
            try:
                self.font = ImageFont.load_default()
            except:
                self.font = None

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#252526", foreground="#ffffff", fieldbackground="#252526", bordercolor="#333333", rowheight=24)
        style.configure("Treeview.Heading", background="#333333", foreground="#ffffff", font=("Meiryo", 9, "bold"))
        style.map("Treeview", background=[("selected", "#007acc")])

        self.pdf_doc = None
        self.total_pages = 0
        self.pdf_path = None

        self.pil_pages = []
        self.qr_results_pages = []
        self.cv_images = []
        self.display_images = []
        self.page_scales = []
        self.page_offsets = []
        self.ocr_cache = {}
        
        self.company_name_common = ""
        self.office_name_common = ""
        self.office_list = []

        self.selected_model_var = tk.StringVar(value="__NONE__")
        self.model_list = []

        self.current_page = 0
        self.is_editing = False
        self.is_processing = False
        self._pdf_opening = False
        self._is_updating_panel = False
        self.current_processing_idx = None 

        toolbar = tk.Frame(self.root, bg="#252526", height=65)
        toolbar.pack(side="top", fill="x")
        toolbar.pack_propagate(False)

        btn_config = {"font": ("Meiryo", 10, "bold"), "bg": "#007acc", "fg": "white", "activebackground": "#005999", "activeforeground": "white", "bd": 0, "padx": 15, "pady": 6}
        
        tk.Button(toolbar, text="📂 PDFを開く", font=("Meiryo",12), bg="#3c3c3c", fg="white", command=self.open_pdf).pack(side="left", padx=10, pady=8)
        tk.Button(toolbar, text="⏪ 前ページ", command=self.prev_page, **{**btn_config, "font": ("Meiryo", 12, "bold"), "bg": "#3c3c3c"}).pack(side="left", padx=4, pady=13)
        tk.Button(toolbar, text="⏩ 次ページ", command=self.next_page, **{**btn_config, "font": ("Meiryo", 12, "bold"), "bg": "#3c3c3c"}).pack(side="left", padx=4, pady=13)
        tk.Button(toolbar, text="💾 Excelに出力", font=("Meiryo",12), bg="#3c3c3c", fg="white", command=self._export_all_pages_to_excel).pack(side="left", padx=10, pady=8)        
        tk.Button(toolbar, text="❌ 終了", command=root.quit, **{**btn_config, "font": ("Meiryo", 12, "bold"), "bg": "#d9534f"}).pack(side="right", padx=12, pady=12)
        
        self.page_label = tk.Label(toolbar, text="ページ: 0 / 0", font=("Meiryo", 11, "bold"), bg="#252526", fg="#00ffcc")
        self.page_label.pack(side="right", padx=20)

        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        canvas_container = tk.Frame(main_frame, bg="#252526", bd=1, relief="solid")
        canvas_container.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.canvas = tk.Canvas(canvas_container, bg=CANVAS_BG, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=5, pady=(5, 0))

        self.status_label = tk.Label(canvas_container, text="準備完了", font=("Meiryo", 10, "bold"), bg="#252526", fg="#f39c12", anchor="w")
        self.status_label.pack(fill="x", padx=10, pady=6)

        self.info_frame = tk.Frame(main_frame, bg="#252526", width=460, bd=1, relief="solid")
        self.info_frame.pack(side="right", fill="y")
        self.info_frame.pack_propagate(False)

        basic_frame = tk.LabelFrame(self.info_frame, text=" 基本情報 ", font=("Meiryo", 12, "bold"), bg="#252526", fg="#ffffff", bd=1, relief="solid")
        basic_frame.pack(fill="x", padx=12, pady=10)

        tk.Label(basic_frame, text="会社名（全ページ共通）", font=("Meiryo", 11), bg="#252526", fg="#cccccc").pack(anchor="w", padx=10, pady=(4, 1))
        self.company_entry = tk.Entry(basic_frame, font=("Meiryo", 10), bg="#3c3c3c", fg="white", insertbackground="white", bd=1, relief="solid")
        self.company_entry.pack(fill="x", padx=10, pady=(0, 4), ipady=2)

        tk.Label(basic_frame, text="事業所名（全ページ共通）", font=("Meiryo", 11), bg="#252526", fg="#cccccc").pack(anchor="w", padx=10, pady=(4, 1))
        self.office_entry = tk.Entry(basic_frame, font=("Meiryo", 10), bg="#3c3c3c", fg="white", insertbackground="white", bd=1, relief="solid")
        self.office_entry.pack(fill="x", padx=10, pady=(0, 8), ipady=2)

        model_frame_outer = tk.LabelFrame(self.info_frame, text=" 機種選択 ", font=("Meiryo", 12, "bold"), bg="#252526", fg="#ffffff", bd=1, relief="solid")
        model_frame_outer.pack(fill="x", padx=12, pady=5)

        models_left = {"DTS-G1O": "FV710G1DO", "DTS-G1D": "FV710G1D", "DTS-G1D3": "FV710G1D3"}
        models_right = {"DTS-D2D": "FV710D2D", "DTS-D2A": "FV710D2A", "DTS-D1D": "FV710D1D", "DTS-D1A": "FV710D1A"}

        m_inner = tk.Frame(model_frame_outer, bg="#252526")
        m_inner.pack(fill="x", padx=10, pady=6)
        left_f = tk.Frame(m_inner, bg="#252526")
        right_f = tk.Frame(m_inner, bg="#252526")
        left_f.pack(side="left", expand=True, fill="x")
        right_f.pack(side="left", expand=True, fill="x")

        for text, val in models_left.items():
            tk.Radiobutton(left_f, text=text, variable=self.selected_model_var, value=val, font=("Meiryo", 11), bg="#252526", fg="white", activebackground="#252526", activeforeground="white", selectcolor="#252526", command=self.on_model_changed).pack(anchor="w", pady=1)
        for text, val in models_right.items():
            tk.Radiobutton(right_f, text=text, variable=self.selected_model_var, value=val, font=("Meiryo", 11), bg="#252526", fg="white", activebackground="#252526", activeforeground="white", selectcolor="#252526", command=self.on_model_changed).pack(anchor="w", pady=1)

        qr_frame = tk.LabelFrame(self.info_frame, text=" 検出QRコード一覧 ", font=("Meiryo", 12, "bold"), bg="#252526", fg="#00ffcc", bd=1, relief="solid")
        qr_frame.pack(fill="x", padx=12, pady=8)

        tree_container = tk.Frame(qr_frame, bg="#252526")
        tree_container.pack(fill="x", padx=8, pady=8)

        self.tree = ttk.Treeview(tree_container, columns=("No", "Data"), show="headings", height=5, selectmode="browse")
        self.tree.heading("No", text="No")
        self.tree.heading("Data", text="QRデータ内容")
        self.tree.column("No", width=40, anchor="center", stretch=False)
        self.tree.column("Data", width=365, anchor="w")
        
        tree_scroll = ttk.Scrollbar(tree_container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

        preview_frame = tk.LabelFrame(self.info_frame, text=" 出力データプレビュー ", font=("Meiryo", 12, "bold"), bg="#252526", fg="#00ffcc", bd=1, relief="solid")
        preview_frame.pack(fill="both", expand=True, padx=12, pady=(2, 10))

        self.data_text = tk.Text(preview_frame, font=("Consolas", 9), bg="#121212", fg="#a9b7c6", insertbackground="white", bd=0, wrap="none")
        preview_scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=self.data_text.yview)
        self.data_text.configure(yscrollcommand=preview_scroll.set)
        self.data_text.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        preview_scroll.pack(side="right", fill="y", pady=4)

        self.company_entry.bind("<KeyRelease>", self.on_company_changed)
        self.office_entry.bind("<KeyRelease>", self.on_office_changed)
        self.company_entry.bind("<FocusIn>", lambda e: self._start_edit())
        self.company_entry.bind("<FocusOut>", lambda e: self._end_edit())
        self.office_entry.bind("<FocusIn>", lambda e: self._start_edit())
        self.office_entry.bind("<FocusOut>", lambda e: self._end_edit())

    def _start_edit(self):
        self.is_editing = True

    def _end_edit(self):
        self.is_editing = False
        self.update_data_panel()

    def on_company_changed(self, event=None):
        self.company_name_common = self.company_entry.get()
        self.update_data_panel()

    def on_office_changed(self, event=None):
        self.office_name_common = self.office_entry.get()
        if self.total_pages > 0:
            self.office_list = [self.office_name_common] * self.total_pages
        self.update_data_panel()

    def on_model_changed(self):
        val = self.selected_model_var.get()
        if self.total_pages > 0:
            self.model_list = [val] * self.total_pages
        self.update_data_panel()

    def open_pdf(self):
        if getattr(self, "_pdf_opening", False) or self.is_processing:
            return
        self._pdf_opening = True

        try:
            file_path = filedialog.askopenfilename(
                title="車検証PDFを選択",
                filetypes=[("PDFファイル", "*.pdf"), ("すべてのファイル", "*.*")]
            )
            if not file_path:
                self._pdf_opening = False
                return

            self.pdf_path = file_path
            
            if hasattr(qr_module, "clear_qr_cache"):
                qr_module.clear_qr_cache()

            if self.pdf_doc:
                self.pdf_doc.close()
            
            self.pdf_doc = pymupdf.open(file_path)
            self.total_pages = len(self.pdf_doc)

            self.pil_pages = [None] * self.total_pages
            self.cv_images = [None] * self.total_pages
            self.qr_results_pages = [None] * self.total_pages
            self.page_scales = [None] * self.total_pages
            self.page_offsets = [None] * self.total_pages
            self.display_images = [None] * self.total_pages
            self.ocr_cache = {}
            
            self.office_list = [self.office_name_common] * self.total_pages
            initial_model = self.selected_model_var.get()
            self.model_list = [initial_model] * self.total_pages

            if self.total_pages > 0:
                self.status_label.config(text=f"⚡ ページ 1 / {self.total_pages} を解析中...")
                threading.Thread(target=self._load_first_page_and_continue, daemon=True).start()
        except Exception as e:
            print(f"❌ PDF例外: {e}", flush=True)
            traceback.print_exc()
            messagebox.showerror("エラー", f"PDF読み込みに失敗しました:\n{e}")
            self._pdf_opening = False

    def _load_first_page_and_continue(self):
        try:
            self._process_single_page(0)
            self.root.after(0, lambda: self._show_page_index(0))
            
            if self.total_pages > 1:
                self._load_remaining_pages_async(1)
            else:
                self.root.after(0, lambda: self.status_label.config(text="✅ すべてのページの解析が完了しました。"))
        except Exception as e:
            print(f"❌ 初回ロードエラー: {e}", flush=True)
            traceback.print_exc()
        finally:
            self._pdf_opening = False

    def _process_single_page(self, idx):
        with self.lock:
            if idx < 0 or idx >= self.total_pages:
                return
            if self.cv_images[idx] is not None:
                return

            try:
                page = self.pdf_doc[idx]

                # 200 DPI でレンダリング（QR認識・OCRに十分な精度で処理負荷軽減）
                pix = page.get_pixmap(dpi=200)

                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 4:
                    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                else:
                    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # 軽量・高速な傾き補正関数を適用
                img_cv = deskew_image_fast(img_cv)

                # QRコード検出・ソート
                qrs = qr_module.detect_best_qr(img_cv, page_key=idx)
                results_sorted = qr_module.sort_qr_by_x(qrs)
                qr_texts_temp = (
                    [qr.text for qr in results_sorted] if results_sorted else []
                )

                # 表示用サイズとオフセットの設定
                h_orig, w_orig = img_cv.shape[:2]
                max_w, max_h = DISPLAY_WIDTH, DISPLAY_HEIGHT
                scale = min(max_w / w_orig, max_h / h_orig)
                disp_w, disp_h = int(w_orig * scale), int(h_orig * scale)
                offset_x = max(0, (max_w - disp_w) // 2)
                offset_y = max(0, (max_h - disp_h) // 2)

                # 高速補間 (INTER_NEAREST / INTER_LINEAR)
                img_cv_disp = cv2.resize(
                    img_cv, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR
                )
                page_disp = Image.fromarray(
                    cv2.cvtColor(img_cv_disp, cv2.COLOR_BGR2RGB)
                )
                
                if hasattr(qr_module, "draw_qr_boxes"):
                    page_disp = qr_module.draw_qr_boxes(
                        page_disp, results_sorted, scale, self.font
                    )

                # OCRによる車名抽出
                try:
                    carname = maker_parser.extract_car_name_ocr(
                        img_cv,
                        page_num=idx + 1,
                        pdf_path=self.pdf_path,
                        qr_texts=qr_texts_temp,
                    )
                    ocr_result = carname if carname else "（取得できず）"
                except Exception as e:
                    print(f"⚠️ OCRエラー (ページ {idx+1}): {e}", flush=True)
                    ocr_result = "（OCRエラー）"

                # データの格納
                self.cv_images[idx] = img_cv
                self.pil_pages[idx] = page_disp
                self.display_images[idx] = ImageTk.PhotoImage(page_disp)
                self.qr_results_pages[idx] = results_sorted
                self.page_scales[idx] = scale
                self.page_offsets[idx] = (offset_x, offset_y)
                self.ocr_cache[idx] = ocr_result

            except Exception as e:
                print(f"❌ ページ処理例外 (ページ {idx+1}): {e}", flush=True)
                traceback.print_exc()

    def _load_remaining_pages_async(self, start_idx):
        self.is_processing = True
        try:
            # 並列処理で全ページを一気に解析
            futures = []
            for idx in range(start_idx, self.total_pages):
                if self.cv_images[idx] is None:
                    futures.append((idx, self.executor.submit(self._process_single_page, idx)))

            for idx, future in futures:
                self.current_processing_idx = idx
                msg = f"⚡ ページ {idx + 1} / {self.total_pages} を解析中... (ページ移動は自由に行えます)"
                self.root.after(0, lambda m=msg: self.status_label.config(text=m))
                try:
                    future.result()  # 完了を待機
                except Exception as fe:
                    print(f"❌ 非同期解析例外 (ページ {idx+1}): {fe}", flush=True)
                    traceback.print_exc()
                
                if self.current_page == idx:
                    self.root.after(0, lambda i=idx: self._show_page_index(i))

            self.root.after(0, lambda: self.status_label.config(text=f"✅ すべてのページの解析が完了しました（全 {self.total_pages} ページ）。"))
        except Exception as e:
            print(f"❌ バックグラウンドロード例外: {e}", flush=True)
            traceback.print_exc()
        finally:
            self.current_processing_idx = None
            self.is_processing = False

    def _show_page_index(self, idx):
        if self.is_editing:
            return

        if 0 <= idx < self.total_pages:
            if self.current_processing_idx == idx and self.cv_images[idx] is None:
                self.status_label.config(text=f"⚠️ ページ {idx + 1} は現在バックグラウンドで解析中です。少々お待ちください。")
                messagebox.showinfo("お知らせ", f"ページ {idx + 1} は現在バックグラウンドで解析中です。\n解析終了後に自動で表示されます。")
                return

            self.current_page = idx
            
            if self.cv_images[idx] is None:
                self.status_label.config(text=f"⚡ ページ {idx + 1} / {self.total_pages} を優先解析中...")
                self._process_single_page(idx)
                self.status_label.config(text=f"✅ ページ {idx + 1} の解析完了")

            photo = self.display_images[idx]
            if photo is not None:
                self.canvas.delete("all")
                offset_x, offset_y = self.page_offsets[idx] if self.page_offsets[idx] else (0, 0)
                self.canvas.create_image(offset_x, offset_y, image=photo, anchor="nw")
                self.canvas.image = photo

            self.page_label.config(text=f"ページ: {idx+1} / {self.total_pages}")
            self.update_data_panel()

    def prev_page(self):
        if self.current_page > 0:
            self._show_page_index(self.current_page - 1)

    def next_page(self):
        if self.current_page < self.total_pages - 1:
            self._show_page_index(self.current_page + 1)

    def update_data_panel(self):
        if getattr(self, "_is_updating_panel", False):
            return
        self._is_updating_panel = True

        try:
            idx = self.current_page

            if not self.is_editing:
                self.company_entry.delete(0, tk.END)
                self.company_entry.insert(0, self.company_name_common)

                office_val = self.office_name_common
                self.office_entry.delete(0, tk.END)
                self.office_entry.insert(0, office_val)

                model_val = self.selected_model_var.get()
                self.selected_model_var.set(model_val)

            self.data_text.delete(1.0, tk.END)

            for item in self.tree.get_children():
                self.tree.delete(item)

            results_sorted = self.qr_results_pages[idx] if idx < len(self.qr_results_pages) and self.qr_results_pages[idx] is not None else []
            qr_texts = [qr.text for qr in results_sorted] if results_sorted else []

            for i in range(1, 6):
                qr_txt = qr_texts[i-1] if (i-1) < len(qr_texts) else ""
                self.tree.insert("", "end", values=(f"QR{i}", qr_txt))

            QR1 = qr_texts[0] if len(qr_texts) >= 1 else ""
            QR2 = qr_texts[1] if len(qr_texts) >= 2 else ""
            L = f"{QR1.split('/')[-1] if QR1 else ''}{QR2.split('/')[0] if QR2 else ''}"
            
            qr_count = len(qr_texts)
            is_keijidousha = False
            if qr_count >= 2:
                if QR1 and ("K/" in QR1 or "軽自動車" in QR1):
                    is_keijidousha = True
                else:
                    is_keijidousha = len(L) <= 6 and not any(c.isalpha() for c in L[:2])

            # 軽自動車なら6未満、普通車なら5未満、あるいはQR未検出（0）の場合はエラーとする
            is_invalid_kei_qr = is_keijidousha and (qr_count < 6)
            is_invalid_normal_qr = (not is_keijidousha) and (qr_count < 5)
            is_read_error = (qr_count < 1) or is_invalid_kei_qr or is_invalid_normal_qr

            if is_read_error:
                J = "読み取りできません"
                L = "読み取りできません"
                M_replace = "読み取りできません"
                N = "読み取りできません"
                carname = "読み取りできません"
                F_col = G_col = H_col = I_col = "読み取りできません"
            else:
                QR3 = qr_texts[2] if len(qr_texts) >= 3 else ""
                QR4 = qr_texts[3] if len(qr_texts) >= 4 else ""
                QR5 = qr_texts[4] if len(qr_texts) >= 5 else ""

                detected_num_text = ""
                for t in qr_texts:
                    if any(h in t for h in ['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ', 'さ', 'し', 'す', 'せ', 'そ', 'た', 'ち', 'つ', 'て', 'と', 'な', 'に', 'ぬ', 'ね', 'の', 'は', 'ひ', 'ふ', 'へ', 'ほ', 'ま', 'み', 'む', 'め', 'も', 'や', 'ゆ', 'よ', 'ら', 'り', 'る', 'れ', 'ろ', 'わ', 'ん']) or '三重' in t or '伊勢志摩' in t:
                        detected_num_text = t
                        break
                
                parse_target_qr4 = detected_num_text if detected_num_text else QR4

                J = qr_module.extract_first_registration(QR1) if hasattr(qr_module, "extract_first_registration") else ""
                QR5_parts = QR5.split("/") if QR5 else []
                M = QR5_parts[2] if len(QR5_parts) >= 3 else ""
                N = QR5_parts[3] if len(QR5_parts) >= 4 else ""
                M_replace = M.replace("[01]", "国[01]")
                
                res_split = qr_module.split_vehicle_number(parse_target_qr4, QR5) if hasattr(qr_module, "split_vehicle_number") else ("", "", "", "")
                if isinstance(res_split, (list, tuple)) and len(res_split) >= 4:
                    F_col, G_col, H_col, I_col = res_split[:4]
                else:
                    F_col = G_col = H_col = I_col = ""

                carname = self.ocr_cache.get(idx, "")

            current_co = self.company_name_common
            current_of = self.office_name_common
            current_mo = self.selected_model_var.get()

            has_qr = qr_count > 0

            p_text = f" 🛠 【基本情報】\n"
            if not has_qr or is_read_error:
                p_text += f" ├ 状態 : ❌ 読み取りできません\n"
            
            if has_qr and is_keijidousha and not is_read_error:
                p_text += f" ├ 種別 : 🔴 軽自動車\n"
                
            p_text += f" ├ 会社名 (共通) : {current_co}\n"
            p_text += f" └ 事業所名 (共通): {current_of}\n\n"
            
            p_text += f" 📑 【車検証・QR解析データ】\n"
            p_text += f" ├ 初度登録 : {J}\n"
            p_text += f" ├ 型式番号 : {L}\n"
            p_text += f" ├ 車台番号 : {M_replace}\n"
            p_text += f" ├ 原動機型式: {N}\n"
            p_text += f" └ 車名(OCR): {carname}\n\n"
            p_text += f" 🚘 【登録番号・ナンバー情報】\n"
            p_text += f" ├ 管轄（F） : {F_col}\n"
            p_text += f" ├ 分類（G） : {G_col}\n"
            p_text += f" ├ かな（H） : {H_col}\n"
            p_text += f" └ 番号（I） : {I_col}\n\n"
            p_text += f" ⚙ 選定機種 (共通): {current_mo}\n"

            self.data_text.insert(tk.END, p_text)
            
            self.data_text.tag_remove("red_text", "1.0", tk.END)
            
            target_words = []
            if not has_qr or is_read_error:
                target_words.append("❌ 読み取りできません")
            if has_qr and is_keijidousha and not is_read_error:
                target_words.append("🔴 軽自動車")
            
            for target_word in target_words:
                idx_start = "1.0"
                while True:
                    idx_start = self.data_text.search(target_word, idx_start, stopindex=tk.END)
                    if not idx_start:
                        break
                    idx_end = f"{idx_start}+{len(target_word)}c"
                    self.data_text.tag_add("red_text", idx_start, idx_end)
                    idx_start = idx_end

            self.data_text.tag_config("red_text", foreground="#ff4444")            
        except Exception as e:
            print(f"❌ パネル更新例外: {e}", flush=True)
            traceback.print_exc()
        finally:
            self._is_updating_panel = False

    def _export_all_pages_to_excel(self):
        if self.total_pages == 0:
            messagebox.showwarning("警告", "出力対象のPDFが読み込まれていません。")
            return

        try:
            company = self.company_name_common.strip()

            for idx in range(self.total_pages):
                if self.cv_images[idx] is None:
                    self.status_label.config(text=f"⚡ ページ {idx + 1} / {self.total_pages} を優先解析中...")
                    self._process_single_page(idx)

            self.status_label.config(text="Excelファイルを作成中...")
            all_pages_data = []

            for idx in range(self.total_pages):
                office = self.office_name_common
                model = self.selected_model_var.get()

                results_sorted = self.qr_results_pages[idx] if idx < len(self.qr_results_pages) and self.qr_results_pages[idx] is not None else []
                qr_texts = [qr.text for qr in results_sorted] if results_sorted else []

                QR1 = qr_texts[0] if len(qr_texts) >= 1 else ""
                QR2 = qr_texts[1] if len(qr_texts) >= 2 else ""
                QR3 = qr_texts[2] if len(qr_texts) >= 3 else ""
                QR4 = qr_texts[3] if len(qr_texts) >= 4 else ""
                QR5 = qr_texts[4] if len(qr_texts) >= 5 else ""

                qr_count = len(qr_texts)
                is_keijidousha = False
                if qr_count >= 2:
                    if QR1 and ("K/" in QR1 or "軽自動車" in QR1):
                        is_keijidousha = True
                    else:
                        L_temp = f"{QR1.split('/')[-1] if QR1 else ''}{QR2.split('/')[0] if QR2 else ''}"
                        is_keijidousha = len(L_temp) <= 6 and not any(c.isalpha() for c in L_temp[:2])

                is_invalid_kei_qr = is_keijidousha and (qr_count < 6)
                is_invalid_normal_qr = (not is_keijidousha) and (qr_count < 5)
                is_read_error = (qr_count < 1) or is_invalid_kei_qr or is_invalid_normal_qr

                if is_read_error:
                    J = L = M_replace = N = carname = F_col = G_col = H_col = I_col = "読み取りできません"
                else:
                    detected_num_text = ""
                    for t in qr_texts:
                        if any(h in t for h in ['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ', 'さ', 'し', 'す', 'せ', 'そ', 'た', 'ち', 'つ', 'て', 'と', 'な', 'に', 'ぬ', 'ね', 'の', 'は', 'ひ', 'ふ', 'へ', 'ほ', 'ま', 'み', 'む', 'め', 'も', 'や', 'ゆ', 'よ', 'ら', 'り', 'る', 'れ', 'ろ', 'わ', 'ん']) or '三重' in t or '伊勢志摩' in t:
                            detected_num_text = t
                            break
                    
                    parse_target_qr4 = detected_num_text if detected_num_text else QR4

                    J = qr_module.extract_first_registration(QR1) if hasattr(qr_module, "extract_first_registration") else ""
                    QR5_parts = QR5.split("/") if QR5 else []
                    M = QR5_parts[2] if len(QR5_parts) >= 3 else ""
                    N = QR5_parts[3] if len(QR5_parts) >= 4 else ""

                    L = f"{QR1.split('/')[-1] if QR1 else ''}{QR2.split('/')[0] if QR2 else ''}"
                    
                    carname = self.ocr_cache.get(idx, "")
                    M_replace = M.replace("[01]", "国[01]")
                    
                    res_split = qr_module.split_vehicle_number(parse_target_qr4, QR5) if hasattr(qr_module, "split_vehicle_number") else ("", "", "", "")
                    if isinstance(res_split, (list, tuple)) and len(res_split) >= 4:
                        F_col, G_col, H_col, I_col = res_split[:4]
                    else:
                        F_col = G_col = H_col = I_col = ""

                page_data = {
                    "company": company,
                    "office": office,
                    "car_name": carname,
                    "first_registration": J,
                    "model_number": L,
                    "chassis_number": M_replace,
                    "engine_number": N,
                    "kankan_f": F_col,
                    "bunrui_g": G_col,
                    "kana_h": H_col,
                    "number_i": I_col,
                    "model_code": model
                }
                all_pages_data.append(page_data)

            if hasattr(excel_module, "export_to_excel"):
                excel_module.export_to_excel(all_pages_data, BASE_EXCEL, RESULT_FOLDER)

            self.status_label.config(text="✅ Excel出力が完了しました。")
            messagebox.showinfo("成功", f"Excelファイルを出力しました。\n保存先: {RESULT_FOLDER}")

        except Exception as e:
            print(f"❌ Excel出力例外: {e}", flush=True)
            traceback.print_exc()
            messagebox.showerror("エラー", f"Excel出力に失敗しました:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFViewerApp(root)
    root.mainloop()