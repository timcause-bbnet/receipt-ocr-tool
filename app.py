import streamlit as st
import pytesseract
from PIL import Image, ImageGrab, ImageEnhance, ImageOps
import pandas as pd
import re
import os
import shutil

# ==========================================
# 🔧 Tesseract 路徑設定
# ==========================================
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    tesseract_cmd = shutil.which("tesseract")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

st.set_page_config(page_title="全能證件辨識系統 (v3.0 終極修復)", layout="wide", page_icon="🕵️")

# ==========================================
# 📷 影像預處理 (參數微調)
# ==========================================
def preprocess_image(image):
    # 1. 轉灰階
    img_gray = ImageOps.grayscale(image)
    # 2. 增加對比度 (稍微降低強度，避免斷字)
    enhancer = ImageEnhance.Contrast(img_gray)
    img_contrast = enhancer.enhance(1.8) 
    # 3. 銳利化 (稍微降低強度)
    enhancer_sharp = ImageEnhance.Sharpness(img_contrast)
    img_final = enhancer_sharp.enhance(1.2) # 從 1.5 降到 1.2
    return img_final

# ==========================================
# 核心邏輯：寬鬆版防呆驗證
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
        # 1. 檢查正面 (特徵明確)
        if re.search(r'[A-Z][12]\d{8}', clean_text):
            return True, "id_card_front"
        if any(x in clean_text for x in ["身分證", "出生", "性別", "統一編號"]):
            return True, "id_card_front"
        
        # 2. 檢查背面 (改用單字計分法，因為背面文字容易破碎)
        # 只要出現以下關鍵字中的 2 個，就認定是背面
        back_keywords = ["配偶", "役別", "住址", "父母", "出生地", "父親", "母親", "鄉", "鎮", "鄰", "里", "區"]
        hit_count = sum(1 for k in back_keywords if k in clean_text)
        
        if hit_count >= 2:
            return True, "id_card_back"
            
        if "健保" in clean_text: return False, "⚠️ 錯誤：這是【健保卡】"
        
        return False, f"⚠️ 特徵不足 (命中關鍵字: {hit_count})，請嘗試重新拍照"

    return True, doc_type

# ==========================================
# 核心邏輯：資料提取 (排除發證日期)
# ==========================================
def extract_data(text, doc_type, specific_type=None):
    clean_text = text.replace(" ", "").replace("\n", "")
    num_clean_text = clean_text.upper().replace("O", "0").replace("I", "1").replace("L", "1")
    data = {}

    if doc_type == "id_card":
        if specific_type == "id_card_front":
            # 1. 姓名
            name_match = re.search(r'姓名[:\s]*([\u4e00-\u9fa5]{2,4})', clean_text)
            if not name_match: name_match = re.search(r'([\u4e00-\u9fa5]{2,4})性別', clean_text)
            data['name'] = name_match.group(1) if name_match else ""

            # 2. 身分證字號
            id_match = re.search(r'[A-Z][12]\d{8}', num_clean_text)
            data['id_no'] = id_match.group(0) if id_match else ""

            # 3. 生日 (邏輯大修：排除法)
            # 找出文中"所有"符合日期的字串
            all_dates = re.finditer(r'民國\d{2,3}年\d{1,2}月\d{1,2}日', clean_text)
            
            potential_dob = ""
            for match in all_dates:
                date_str = match.group(0)
                start_idx = match.start()
                # 檢查這個日期前面 10 個字內，有沒有"發證"、"換發"、"補發"
                context_before = clean_text[max(0, start_idx-10):start_idx]
                
                if any(x in context_before for x in ["發證", "換發", "補發", "日期"]):
                    continue # 跳過這個日期，因為它是發證日期
                
                # 如果前面有 "出生" 或 "年月日"，那一定是它
                if any(x in context_before for x in ["出生", "年月"]):
                    potential_dob = date_str
                    break
                
                # 如果還沒決定，暫定第一個遇到的非發證日期為生日
                if not potential_dob:
                    potential_dob = date_str
            
            data['dob'] = potential_dob
            data['type_label'] = "身分證 (正面)"

        elif specific_type == "id_card_back":
            # 1. 配偶
            spouse_match = re.search(r'配偶([\u4e00-\u9fa5]{2,4})', clean_text)
            data['spouse'] = spouse_match.group(1) if spouse_match else ""
            
            # 2. 父母 (寬鬆模式)
            # 嘗試找 "父" 開頭
            father_match = re.search(r'父([\u4e00-\u9fa5]{2,4})', clean_text)
            data['father'] = father_match.group(1) if father_match else ""
            
            # 嘗試找 "母" 開頭
            mother_match = re.search(r'母([\u4e00-\u9fa5]{2,4})', clean_text)
            data['mother'] = mother_match.group(1) if mother_match else ""

            # 3. 住址 (特徵：通常含有 縣/市/區/路/街/號)
            addr_match = re.search(r'住址([\u4e00-\u9fa50-9\-\(\)鄰里巷弄號樓]+)', clean_text)
            data['address'] = addr_match.group(1) if addr_match else ""
            data['type_label'] = "身分證 (背面)"

    elif doc_type == "health_card":
        name_match = re.search(r'姓名[:\s]*([\u4e00-\u9fa5]{2,4})', clean_text)
        id_match = re.search(r'[A-Z][12]\d{8}', num_clean_text)
        card_match = re.search(r'\d{12}', num_clean_text)
        data['name'] = name_match.group(1) if name_match else ""
        data['id_no'] = id_match.group(0) if id_match else ""
        data['card_no'] = card_match.group(0) if card_match else ""
        data['type_label'] = "健保卡"

    elif doc_type == "passport":
        pass_match = re.search(r'[0-9]{9}', num_clean_text)
        id_match = re.search(r'[A-Z][12]\d{8}', num_clean_text)
        eng_match = re.search(r'([A-Z]+,\s?[A-Z\-]+)', text)
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
    st.info("⚠️ 請使用完整版代碼執行悠遊卡功能。")
else:
    doc_map = {"🪪 身分證辨識": "id_card", "🏥 健保卡辨識": "health_card", "✈️ 護照辨識": "passport"}
    target_type = doc_map[app_mode]
    
    st.title(app_mode)
    uploaded_file = st.file_uploader(f"請上傳 {app_mode.split(' ')[1]}", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        processed_image = preprocess_image(image) # 執行影像增強
        
        # 預覽區
        c1, c2 = st.columns(2)
        c1.image(image, caption='原始照片')
        c2.image(processed_image, caption='AI 增強後')

        if st.button("🔍 開始辨識"):
            with st.spinner('AI 正在讀取並過濾雜訊...'):
                # OCR 辨識
                raw_text = pytesseract.image_to_string(processed_image, lang='chi_tra+eng', config='--psm 6')
                
                # 防呆驗證
                is_valid, status_or_msg = validate_image_content(raw_text, target_type)
                
                if not is_valid:
                    st.error(status_or_msg)
                else:
                    specific_type = status_or_msg 
                    st.success(f"✅ 成功識別！({specific_type})")
                    
                    # 提取資料
                    data = extract_data(raw_text, target_type, specific_type)
                    
                    # 顯示結果
                    st.subheader(f"📝 {data.get('type_label', '結果')} (可修改)")
                    with st.form("result_form"):
                        c1, c2 = st.columns(2)
                        
                        if specific_type == "id_card_front":
                            c1.text_input("姓名", value=data.get('name', ''))
                            c2.text_input("身分證字號", value=data.get('id_no', ''))
                            st.text_input("出生年月日", value=data.get('dob', '')) # 這裡現在應該會顯示正確的生日

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

                with st.expander("🛠️ 查看原始 OCR 文字"):
                    st.text_area("OCR Raw Text", raw_text, height=150)