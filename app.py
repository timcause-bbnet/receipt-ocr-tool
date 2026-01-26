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

st.set_page_config(page_title="全能證件辨識 (V5.0 高清修復)", layout="wide", page_icon="🕵️")

# ==========================================
# 📷 影像預處理 (改為：放大 + 灰階)
# ==========================================
def preprocess_image(image):
    # 1. 轉灰階 (保留漸層，不要變成死黑死白)
    img_gray = ImageOps.grayscale(image)
    
    # 2. 【關鍵】放大圖片 2 倍 (讓文字變大，細節更清楚)
    # 使用 LANCZOS 演算法保持清晰度
    new_size = (int(img_gray.width * 2), int(img_gray.height * 2))
    img_resized = img_gray.resize(new_size, Image.Resampling.LANCZOS)
    
    # 3. 溫柔的對比度增強 (不要太強)
    enhancer = ImageEnhance.Contrast(img_resized)
    img_final = enhancer.enhance(1.5) 
    
    return img_final

# ==========================================
# 核心邏輯：防呆驗證
# ==========================================
def validate_image_content(text, doc_type):
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
        # 正面特徵
        if re.search(r'[A-Z][12]\d{8}', clean_text) or \
           any(x in clean_text for x in ["身分證", "出生", "性別", "統一編號"]):
            return True, "id_card_front"
        
        # 背面特徵 (地址字根)
        back_keywords = ["配偶", "役別", "父母", "出生地", "父親", "母親", "鄉", "鎮", "鄰", "里", "區", "路", "街", "巷", "樓"]
        hit_count = sum(1 for k in back_keywords if k in clean_text)
        
        if hit_count >= 2:
            return True, "id_card_back"
            
        if "健保" in clean_text: return False, "⚠️ 錯誤：這是【健保卡】"
        
        if len(clean_text) > 10:
             return False, f"⚠️ 特徵不足 (命中數:{hit_count})。請確保照片對焦清晰。"
        return False, "⚠️ 讀不到文字"

    return True, doc_type

# ==========================================
# 核心邏輯：資料提取
# ==========================================
def extract_data(text, doc_type, specific_type=None):
    raw_text = text
    # 用於比對身分證字號 (移除所有空格與干擾符號)
    clean_text_nospace = re.sub(r'[\s\.\-\_]+', '', text).upper().replace("O", "0").replace("I", "1").replace("L", "1")
    
    data = {}

    if doc_type == "id_card":
        if specific_type == "id_card_front":
            # 1. 姓名 (放寬搜尋條件，支援空格)
            # 抓取 "姓名" 後面的中文字
            name_match = re.search(r'姓\s*名[:\s\.]*([\u4e00-\u9fa5\s]{2,10})', raw_text)
            if name_match:
                data['name'] = name_match.group(1).replace(" ", "").replace("\n", "")
            else:
                # 備用：直接在全文前幾行找 2-4 個字的中文名
                lines = raw_text.split('\n')
                for line in lines[:6]:
                    c_line = re.sub(r'[^\u4e00-\u9fa5]', '', line) # 只留中文
                    if 2 <= len(c_line) <= 4 and "中華" not in c_line and "身分" not in c_line and "出生" not in c_line:
                        data['name'] = c_line
                        break

            # 2. 身分證字號
            id_match = re.search(r'[A-Z][12]\d{8}', clean_text_nospace)
            data['id_no'] = id_match.group(0) if id_match else ""

            # 3. 生日 (抓取所有日期並排序，取最小的)
            date_pattern = r'民\s*國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日'
            all_dates = []
            for match in re.finditer(date_pattern, raw_text):
                y, m, d = match.groups()
                all_dates.append({
                    "str": f"民國{y}年{m}月{d}日",
                    "val": int(y)*10000 + int(m)*100 + int(d)
                })
            
            if all_dates:
                all_dates.sort(key=lambda x: x['val'])
                data['dob'] = all_dates[0]['str']
            else:
                # 備用：有時候OCR讀不到"民國"，只讀到數字
                num_date = re.search(r'(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', raw_text)
                if num_date:
                    y, m, d = num_date.groups()
                    data['dob'] = f"民國{y}年{m}月{d}日"
            
            data['type_label'] = "身分證 (正面)"

        elif specific_type == "id_card_back":
            # 1. 住址
            addr_match = re.search(r'址[:\s\.]*([\u4e00-\u9fa50-9\-\(\)鄰里巷弄號樓\s]+)', raw_text)
            if addr_match:
                data['address'] = addr_match.group(1).replace(" ", "").replace("\n", "")
            else:
                # 備用：掃描地址特徵
                scan = re.search(r'[\u4e00-\u9fa5]+[縣市][\u4e00-\u9fa5]+[區鄉鎮市][\u4e00-\u9fa50-9]+', clean_text_nospace)
                data['address'] = scan.group(0) if scan else ""

            # 2. 配偶 (支援空格)
            spouse_match = re.search(r'偶[:\s\.]*([\u4e00-\u9fa5\s]{2,5})', raw_text)
            if spouse_match:
                sp = spouse_match.group(1).replace(" ", "")
                data['spouse'] = sp if "役" not in sp else ""
            
            # 3. 父母
            # 這裡用 clean_text_nospace 因為父母欄位通常很擠
            f_match = re.search(r'父([\u4e00-\u9fa5]{2,4})', clean_text_nospace)
            m_match = re.search(r'母([\u4e00-\u9fa5]{2,4})', clean_text_nospace)
            data['father'] = f_match.group(1) if f_match else ""
            data['mother'] = m_match.group(1) if m_match else ""
            
            data['type_label'] = "身分證 (背面)"

    elif doc_type == "health_card":
        name_match = re.search(r'姓\s*名[:\s]*([\u4e00-\u9fa5\s]{2,10})', raw_text)
        if name_match: data['name'] = name_match.group(1).replace(" ", "")
        
        id_match = re.search(r'[A-Z][12]\d{8}', clean_text_nospace)
        data['id_no'] = id_match.group(0) if id_match else ""
        
        card_match = re.search(r'\d{12}', clean_text_nospace)
        data['card_no'] = card_match.group(0) if card_match else ""
        data['type_label'] = "健保卡"

    elif doc_type == "passport":
        pass_match = re.search(r'[0-9]{9}', clean_text_nospace)
        id_match = re.search(r'[A-Z][12]\d{8}', clean_text_nospace)
        # 英文姓名 (修正 Regex 以抓取常見護照格式)
        eng_match = re.search(r'([A-Z]+,\s?[A-Z\-]+)', raw_text)
        
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
        
        # 顯示處理結果 (讓您看看這次是不是變清晰了)
        processed_image = preprocess_image(image)
        c1, c2 = st.columns(2)
        c1.image(image, caption='原始照片')
        c2.image(processed_image, caption='AI 放大與灰階優化 (非二值化)')

        if st.button("🔍 開始辨識"):
            with st.spinner('AI 正在讀取...'):
                # OCR 辨識
                raw_text = pytesseract.image_to_string(processed_image, lang='chi_tra+eng', config='--psm 6')
                
                # 驗證
                is_valid, status_or_msg = validate_image_content(raw_text, target_type)
                
                if not is_valid:
                    st.error(status_or_msg)
                else:
                    specific_type = status_or_msg 
                    st.success(f"✅ 成功識別！({specific_type})")
                    
                    # 提取資料
                    data = extract_data(raw_text, target_type, specific_type)
                    
                    # 顯示結果
                    st.subheader(f"📝 {data.get('type_label', '結果')}")
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

                with st.expander("🛠️ 查看原始 OCR 文字 (Debug)"):
                    st.text_area("Raw Text", raw_text, height=200)