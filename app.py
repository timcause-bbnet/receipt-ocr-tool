import streamlit as st
import pytesseract
from PIL import Image, ImageGrab
import pandas as pd
import re
import os
import shutil

# ==========================================
# 🔧 跨平台 Tesseract 路徑設定
# ==========================================
if os.name == 'nt':
    # Windows 本機路徑
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    # 雲端 Linux 路徑
    tesseract_cmd = shutil.which("tesseract")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

# ==========================================
# 頁面與樣式設定
# ==========================================
st.set_page_config(page_title="全能證件辨識系統", layout="wide", page_icon="🕵️")

# ==========================================
# 核心邏輯：防呆驗證函式
# ==========================================
def validate_image_content(text, doc_type):
    """
    根據 OCR 結果判斷是否上傳了正確的證件
    回傳: (是否通過, 錯誤訊息)
    """
    # 移除雜訊方便比對
    clean_text = text.replace(" ", "").upper()
    
    # 1. 如果在【健保卡模式】
    if doc_type == "health_card":
        if "全民健康保險" in clean_text or "健保" in clean_text:
            return True, ""
        # 偵測是否誤傳為其他證件
        if "PASSPORT" in clean_text: return False, "⚠️ 錯誤：偵測到這是【護照】，請切換模式！"
        if "身分證" in clean_text: return False, "⚠️ 錯誤：偵測到這是【身分證】，請切換模式！"
        return False, "⚠️ 錯誤：無法識別為健保卡，請確認照片清晰或包含「全民健康保險」字樣。"

    # 2. 如果在【護照模式】
    elif doc_type == "passport":
        if "PASSPORT" in clean_text or "REPUBLICOFCHINA" in clean_text or "P<TWN" in clean_text:
            return True, ""
        if "全民健康保險" in clean_text: return False, "⚠️ 錯誤：偵測到這是【健保卡】，請切換模式！"
        if "身分證" in clean_text: return False, "⚠️ 錯誤：偵測到這是【身分證】，請切換模式！"
        return False, "⚠️ 錯誤：無法識別為護照，請確認照片包含「PASSPORT」字樣。"

    # 3. 如果在【身分證模式】
    elif doc_type == "id_card":
        if "身分證" in clean_text or "出生" in clean_text:
            return True, ""
        if "全民健康保險" in clean_text: return False, "⚠️ 錯誤：偵測到這是【健保卡】，請切換模式！"
        if "PASSPORT" in clean_text: return False, "⚠️ 錯誤：偵測到這是【護照】，請切換模式！"
        return False, "⚠️ 錯誤：無法識別為身分證，請確認照片清晰。"

    return True, ""

# ==========================================
# 側邊欄選單
# ==========================================
st.sidebar.title("🕵️ 全能辨識系統")
app_mode = st.sidebar.radio("請選擇辨識項目：", 
    ["💳 悠遊卡報表產生器", "🪪 身分證辨識", "🏥 健保卡辨識", "✈️ 護照辨識"]
)

# 初始化 Session
if 'ocr_df' not in st.session_state: st.session_state['ocr_df'] = None
if 'current_image' not in st.session_state: st.session_state['current_image'] = None

# ==========================================
# 模式 A: 悠遊卡報表 (維持原本功能)
# ==========================================
if app_mode == "💳 悠遊卡報表產生器":
    st.title("💳 悠遊卡報表產生器")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("📂 上傳截圖", type=['png', 'jpg', 'jpeg'])
        if uploaded_file: st.session_state['current_image'] = Image.open(uploaded_file)
    with col2:
        if st.button("📋 貼上剪貼簿 (限本機)"):
            try: st.session_state['current_image'] = ImageGrab.grabclipboard()
            except: st.error("雲端無法讀取剪貼簿")
            
    # (此處省略詳細悠遊卡解析代碼，與上一版相同，若需要請告知)
    # ... 您可以保留上一版的 parse_easycard 函式與 HTML 生成邏輯 ...
    if st.session_state['current_image']:
        st.image(st.session_state['current_image'], width=600)
        st.info("請參考上一版代碼填入悠遊卡解析邏輯，或專注於下方新功能測試。")

# ==========================================
# 模式 B, C, D: 證件辨識通用區
# ==========================================
else:
    # 根據模式設定標題與變數
    if app_mode == "🪪 身分證辨識":
        st.title("🪪 台灣身分證 OCR")
        target_type = "id_card"
    elif app_mode == "🏥 健保卡辨識":
        st.title("🏥 健保卡 OCR")
        target_type = "health_card"
    elif app_mode == "✈️ 護照辨識":
        st.title("✈️ 護照 OCR (Passport)")
        target_type = "passport"

    st.markdown("---")
    uploaded_file = st.file_uploader(f"請上傳 **{app_mode.split(' ')[1]}** 照片", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='已上傳照片', width=400)

        if st.button("🔍 開始智慧辨識"):
            with st.spinner('正在分析影像特徵...'):
                # 1. 全文 OCR
                text = pytesseract.image_to_string(image, lang='chi_tra+eng')
                
                # 2. 🛡️ 防呆驗證：檢查是否上傳錯誤
                is_valid, error_msg = validate_image_content(text, target_type)
                
                if not is_valid:
                    # ❌ 驗證失敗：顯示紅色錯誤警告
                    st.error(error_msg)
                    st.toast(error_msg, icon="❌")
                else:
                    # ✅ 驗證成功：開始解析資料
                    clean_text = text.replace(" ", "").replace("\n", "")
                    
                    # --- 🪪 身分證解析邏輯 ---
                    if target_type == "id_card":
                        name_match = re.search(r'姓名(.{2,4})', clean_text)
                        id_match = re.search(r'[A-Z][12]\d{8}', clean_text)
                        dob_match = re.search(r'民國\d{2,3}年\d{1,2}月\d{1,2}日', clean_text)
                        
                        st.success("✅ 這是有效的身分證！")
                        st.subheader("辨識結果")
                        with st.form("id_form"):
                            c1, c2 = st.columns(2)
                            c1.text_input("姓名", value=name_match.group(1) if name_match else "")
                            c2.text_input("身分證字號", value=id_match.group(0) if id_match else "")
                            st.text_input("出生年月日", value=dob_match.group(0) if dob_match else "")
                            st.form_submit_button("確認存檔")

                    # --- 🏥 健保卡解析邏輯 ---
                    elif target_type == "health_card":
                        # 健保卡號通常是 12 碼數字
                        card_num_match = re.search(r'\d{4}\d{4}\d{4}', clean_text)
                        if not card_num_match: card_num_match = re.search(r'\d{12}', clean_text)
                        
                        # 身分證字號 (健保卡上也有)
                        id_match = re.search(r'[A-Z][12]\d{8}', clean_text)
                        
                        # 姓名 (通常在 "姓名" 後面)
                        name_match = re.search(r'姓名(.{2,4})', clean_text)
                        
                        st.success("✅ 這是有效的健保卡！")
                        st.subheader("辨識結果")
                        with st.form("health_form"):
                            c1, c2 = st.columns(2)
                            c1.text_input("姓名", value=name_match.group(1) if name_match else "")
                            c2.text_input("身分證字號", value=id_match.group(0) if id_match else "")
                            st.text_input("健保卡卡號 (12碼)", value=card_num_match.group(0) if card_num_match else "")
                            st.form_submit_button("確認存檔")

                    # --- ✈️ 護照解析邏輯 ---
                    elif target_type == "passport":
                        # 護照號碼 (通常 9 碼數字)
                        passport_no_match = re.search(r'[0-9]{9}', clean_text)
                        
                        # 英文姓名 (尋找全大寫英文，且有逗號分隔 EX: WANG, XIAO-MING)
                        # 這邊用比較寬鬆的 regex
                        eng_name_match = re.search(r'[A-Z]+,[A-Z\-]+', text) # 注意：這裡用有空格的 original text 比較好抓
                        
                        # 機器讀碼區 (MRZ) 的身分證字號
                        id_in_passport = re.search(r'[A-Z][12]\d{8}', clean_text)

                        st.success("✅ 這是有效的護照！")
                        st.subheader("辨識結果")
                        with st.form("passport_form"):
                            c1, c2 = st.columns(2)
                            c1.text_input("英文姓名", value=eng_name_match.group(0) if eng_name_match else "")
                            c2.text_input("護照號碼", value=passport_no_match.group(0) if passport_no_match else "")
                            st.text_input("身分證字號 (從護照)", value=id_in_passport.group(0) if id_in_passport else "")
                            st.form_submit_button("確認存檔")