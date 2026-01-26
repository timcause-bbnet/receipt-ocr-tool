import streamlit as st
import pytesseract
from PIL import Image, ImageGrab, ImageEnhance, ImageOps
import pandas as pd
import re
import os
import shutil

# ==========================================
# 🔧 跨平台 Tesseract 路徑設定
# ==========================================
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    tesseract_cmd = shutil.which("tesseract")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

st.set_page_config(page_title="全能證件辨識系統 (修復版)", layout="wide", page_icon="🕵️")

# ==========================================
# 📷 影像預處理函式 (關鍵修復！)
# ==========================================
def preprocess_image(image):
    """
    對證件照片進行增強，提高 OCR 成功率
    特別針對身分證背面的防偽底紋進行過濾
    """
    # 1. 轉灰階
    img_gray = ImageOps.grayscale(image)
    # 2. 增加對比度 (讓文字更黑，背景更白)
    enhancer = ImageEnhance.Contrast(img_gray)
    img_contrast = enhancer.enhance(2.0) 
    # 3. 銳利化
    enhancer_sharp = ImageEnhance.Sharpness(img_contrast)
    img_final = enhancer_sharp.enhance(1.5)
    return img_final

# ==========================================
# 核心邏輯：防呆驗證
# ==========================================
def validate_image_content(text, doc_type):
    # 移除空白與標點，轉大寫
    clean_text = re.sub(r'\s+', '', text).upper()
    
    if doc_type == "health_card":
        if any(x in clean_text for x in ["全民健康保險", "健保", "IC卡"]): return True, "health_card"
        if "PASSPORT" in clean_text: return False, "⚠️ 錯誤：這是【護照】"
        if "父母" in clean_text: return False, "⚠️ 錯誤：這是【身分證背面】"
        return False, "⚠️ 讀取不到健保卡特徵"

    elif doc_type == "passport":
        if any(x in clean_text for x in ["PASSPORT", "REPUBLIC", "TWN"]): return True, "passport"
        if "健保" in clean_text: return False, "⚠️ 錯誤：這是【健保卡】"
        return False, "⚠️ 讀取不到護照特徵"

    elif doc_type == "id_card":
        # 1. 檢查正面特徵
        if any(x in clean_text for x in ["身分證", "出生", "性別", "統一編號"]):
            return True, "id_card_front"
        
        # 2. 檢查背面特徵 (針對背面讀取困難優化判定)
        if any(x in clean_text for x in ["配偶", "役別", "住址", "父母", "出生地"]):
            return True, "id_card_back"
            
        if "健保" in clean_text: return False, "⚠️ 錯誤：這是【健保卡】"
        
        # 如果都沒中，但有讀到大量文字，可能是 OCR 失敗，回傳警告
        if len(clean_text) > 10:
            return False, "⚠️ 讀取到文字但無法識別特徵，請嘗試重新拍照 (避開反光)"
        return False, "⚠️ 無法讀取任何文字，請確認照片清晰度"

    return True, doc_type

# ==========================================
# 核心邏輯：資料提取 (Regex 修正版)
# ==========================================
def extract_data(text, doc_type, specific_type=None):
    # 基礎清理
    clean_text = text.replace(" ", "").replace("\n", "")
    # 數字專用清理 (O->0, I/l->1)
    num_clean_text = clean_text.upper().replace("O", "0").replace("I", "1").replace("L", "1")
    data = {}

    # === 身分證系列 ===
    if doc_type == "id_card":
        if specific_type == "id_card_front":
            # 1. 姓名 (嘗試多種模式)
            name_match = re.search(r'姓名[:\s]*([\u4e00-\u9fa5]{2,4})', clean_text)
            if not name_match: name_match = re.search(r'([\u4e00-\u9fa5]{2,4})性別', clean_text)
            data['name'] = name_match.group(1) if name_match else ""

            # 2. 身分證字號
            id_match = re.search(r'[A-Z][12]\d{8}', num_clean_text)
            data['id_no'] = id_match.group(0) if id_match else ""

            # 3. 生日 (修正：嚴格限定在"出生"或"年月日"之後)
            # 這樣就不會抓到發證日期了
            dob_match = re.search(r'(?:出生|年月日).*?(民國\d{2,3}年\d{1,2}月\d{1,2}日)', clean_text)
            if not dob_match:
                # 備用方案：抓取第一個出現的日期 (通常生日在上面)
                dob_match = re.search(r'(民國\d{2,3}年\d{1,2}月\d{1,2}日)', clean_text)
            
            data['dob'] = dob_match.group(1) if dob_match else ""
            data['type_label'] = "身分證 (正面)"

        elif specific_type == "id_card_back":
            # 1. 配偶
            spouse_match = re.search(r'配偶([\u4e00-\u9fa5]{2,4})', clean_text)
            data['spouse'] = spouse_match.group(1) if spouse_match else ""
            
            # 2. 父母 (處理 父XXX母XXX 的連在一起狀況)
            # 邏輯：找 "父" 後面的字，直到遇到 "母"
            parents_match = re.search(r'父([\u4e00-\u9fa5]+)母', clean_text)
            if parents_match:
                data['father'] = parents_match.group(1)
                # 找 "母" 後面的字，直到遇到 "配偶" 或 "役別" 或換行
                mother_match = re.search(r'母([\u4e00-\u9fa5]+)(?:配偶|役別|$)', clean_text)
                data['mother'] = mother_match.group(1) if mother_match else ""
            else:
                # 備用：如果沒抓到連在一起的，分開抓
                f_match = re.search(r'父([\u4e00-\u9fa5]{2,4})', clean_text)
                m_match = re.search(r'母([\u4e00-\u9fa5]{2,4})', clean_text)
                data['father'] = f_match.group(1) if f_match else ""
                data['mother'] = m_match.group(1) if m_match else ""

            # 3. 住址 (抓取 住址 後面所有的中文與數字)
            addr_match = re.search(r'住址([\u4e00-\u9fa50-9\-\(\)鄰里巷弄號樓]+)', clean_text)
            data['address'] = addr_match.group(1) if addr_match else ""
            data['type_label'] = "身分證 (背面)"

    # === 健保卡 ===
    elif doc_type == "health_card":
        name_match = re.search(r'姓名[:\s]*([\u4e00-\u9fa5]{2,4})', clean_text)
        id_match = re.search(r'[A-Z][12]\d{8}', num_clean_text)
        card_match = re.search(r'\d{12}', num_clean_text)
        
        data['name'] = name_match.group(1) if name_match else ""
        data['id_no'] = id_match.group(0) if id_match else ""
        data['card_no'] = card_match.group(0) if card_match else ""
        data['type_label'] = "健保卡"

    # === 護照 ===
    elif doc_type == "passport":
        pass_match = re.search(r'[0-9]{9}', num_clean_text)
        id_match = re.search(r'[A-Z][12]\d{8}', num_clean_text)
        eng_match = re.search(r'([A-Z]+,\s?[A-Z\-]+)', text) # 用原始 text 抓英文
        
        data['eng_name'] = eng_match.group(1).replace("\n", "") if eng_match else ""
        data['passport_no'] = pass_match.group(0) if pass_match else ""
        data['id_no'] = id_match.group(0) if id_match else ""
        data['type_label'] = "護照"

    return data

# ==========================================
# 介面顯示
# ==========================================
st.sidebar.title("🧰 工具箱")
app_mode = st.sidebar.radio("請選擇功能：", 
    ["💳 悠遊卡報表產生器", "🪪 身分證辨識", "🏥 健保卡辨識", "✈️ 護照辨識"]
)

if 'current_image' not in st.session_state: st.session_state['current_image'] = None

if app_mode == "💳 悠遊卡報表產生器":
    st.title("💳 悠遊卡報表產生器")
    uploaded_file = st.file_uploader("📂 上傳截圖", type=['png', 'jpg'])
    if uploaded_file: 
        st.session_state['current_image'] = Image.open(uploaded_file)
        st.image(st.session_state['current_image'], width=600)
        st.info("⚠️ 請使用之前的完整代碼來執行悠遊卡功能。")

else:
    doc_map = {"🪪 身分證辨識": "id_card", "🏥 健保卡辨識": "health_card", "✈️ 護照辨識": "passport"}
    target_type = doc_map[app_mode]
    
    st.title(app_mode)
    uploaded_file = st.file_uploader(f"請上傳 {app_mode.split(' ')[1]} (支援正反面)", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # 顯示原始圖 vs 處理後的圖 (讓使用者知道發生了什麼)
        c1, c2 = st.columns(2)
        c1.image(image, caption='原始照片', use_container_width=True)
        
        # === 關鍵步驟：執行影像預處理 ===
        processed_image = preprocess_image(image)
        c2.image(processed_image, caption='AI 增強後 (去除背面雜訊)', use_container_width=True)

        if st.button("🔍 開始辨識"):
            with st.spinner('正在分析影像特徵...'):
                # 1. OCR (使用處理後的圖片辨識！)
                # --psm 6 假設是一個統一的文字塊，對於身分證背面這種表格形式特別有效
                raw_text = pytesseract.image_to_string(processed_image, lang='chi_tra+eng', config='--psm 6')
                
                # 2. 驗證
                is_valid, status_or_msg = validate_image_content(raw_text, target_type)
                
                if not is_valid:
                    st.error(status_or_msg)
                else:
                    specific_type = status_or_msg 
                    st.success(f"✅ 成功識別！偵測為：{specific_type}")
                    
                    # 3. 提取資料
                    data = extract_data(raw_text, target_type, specific_type)
                    
                    # 4. 結果表單
                    st.subheader(f"📝 {data.get('type_label', '結果')} (可修改)")
                    with st.form("result_form"):
                        c1, c2 = st.columns(2)
                        
                        if specific_type == "id_card_front":
                            c1.text_input("姓名", value=data.get('name', ''))
                            c2.text_input("身分證字號", value=data.get('id_no', ''))
                            st.text_input("出生年月日", value=data.get('dob', ''))

                        elif specific_type == "id_card_back":
                            st.text_input("住址", value=data.get('address', ''))
                            c1.text_input("配偶", value=data.get('spouse', ''))
                            c2.text_input("父親", value=data.get('father', ''))
                            st.text_input("母親", value=data.get('mother', ''))
                            
                        elif target_type == "health_card":
                            c1.text_input("姓名", value=data.get('name', ''))
                            c2.text_input("身分證字號", value=data.get('id_no', ''))
                            st.text_input("健保卡號 (12碼)", value=data.get('card_no', ''))
                            
                        elif target_type == "passport":
                            c1.text_input("英文姓名", value=data.get('eng_name', ''))
                            c2.text_input("護照號碼", value=data.get('passport_no', ''))
                            st.text_input("身分證字號", value=data.get('id_no', ''))

                        st.form_submit_button("💾 確認存檔")

                # === 除錯區 ===
                with st.expander("🛠️ 查看原始 OCR 文字 (除錯用)"):
                    st.text_area("OCR Raw Text", raw_text, height=150)