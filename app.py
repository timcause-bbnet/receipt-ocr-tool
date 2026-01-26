import streamlit as st
import pytesseract
from PIL import Image, ImageGrab, ImageEnhance, ImageOps
import pandas as pd
import re
import os
import shutil
from datetime import datetime

# ==========================================
# 🔧 Tesseract 路徑設定
# ==========================================
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    tesseract_cmd = shutil.which("tesseract")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

st.set_page_config(page_title="全能證件辨識 (V4.0 強力修復)", layout="wide", page_icon="🕵️")

# ==========================================
# 📷 影像預處理 (針對線條干擾的最佳化)
# ==========================================
def preprocess_image(image):
    # 轉灰階
    img_gray = ImageOps.grayscale(image)
    # 提高對比度 (讓字變黑)
    enhancer = ImageEnhance.Contrast(img_gray)
    img_contrast = enhancer.enhance(2.0)
    # 稍微銳利化 (不要太強，避免斷字)
    enhancer_sharp = ImageEnhance.Sharpness(img_contrast)
    img_final = enhancer_sharp.enhance(1.1)
    # 二值化 (將灰色轉為純黑白，去除淺色底紋)
    thresh = 140
    fn = lambda x : 255 if x > thresh else 0
    img_binary = img_final.convert('L').point(fn, mode='1')
    return img_binary

# ==========================================
# 核心邏輯：寬鬆版防呆驗證
# ==========================================
def validate_image_content(text, doc_type):
    # 移除所有空白，轉大寫，方便比對
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
        # 1. 正面判定：有身分證字號 OR 有"正面"關鍵字
        if re.search(r'[A-Z][12]\d{8}', clean_text) or \
           any(x in clean_text for x in ["身分證", "出生", "性別", "統一編號"]):
            return True, "id_card_front"
        
        # 2. 背面判定 (改用地址特徵，因為地址佔最大面積)
        # 只要出現以下任意 2 個字，就認定是背面
        back_keywords = ["配偶", "役別", "父母", "出生地", "父親", "母親", "鄉", "鎮", "鄰", "里", "區", "路", "街", "巷", "樓"]
        hit_count = sum(1 for k in back_keywords if k in clean_text)
        
        if hit_count >= 2:
            return True, "id_card_back"
            
        if "健保" in clean_text: return False, "⚠️ 錯誤：這是【健保卡】"
        
        # 如果有讀到字但判定失敗，顯示提示
        if len(clean_text) > 10:
             return False, f"⚠️ 特徵不足 (命中數:{hit_count})。請嘗試：\n1. 避開反光\n2. 讓證件填滿畫面"
        return False, "⚠️ 讀不到文字，請確認照片清晰度"

    return True, doc_type

# ==========================================
# 核心邏輯：資料提取 (支援空格與排序)
# ==========================================
def extract_data(text, doc_type, specific_type=None):
    # 保留原始格式 (有空格) 用於 Regex
    raw_text = text
    # 移除空格版 (用於抓 ID)
    clean_text_nospace = re.sub(r'\s+', '', text).upper().replace("O", "0").replace("I", "1").replace("L", "1")
    
    data = {}

    if doc_type == "id_card":
        if specific_type == "id_card_front":
            # 1. 姓名 (支援空格：陳 筱 玲)
            # 邏輯：找 "姓名" 後面的 2~5 個中文字 (允許中間有空格)
            name_match = re.search(r'姓\s*名[:\s]*([\u4e00-\u9fa5\s]{2,10})', raw_text)
            if name_match:
                # 抓到後把空格去掉
                data['name'] = name_match.group(1).replace(" ", "").replace("\n", "")
            else:
                # 備用：如果沒讀到"姓名"，抓第一行看起來像名字的 (2-4個中文字)
                lines = raw_text.split('\n')
                for line in lines[:5]: # 只看前5行
                    cleaned_line = re.sub(r'\s+', '', line)
                    if 2 <= len(cleaned_line) <= 4 and re.match(r'^[\u4e00-\u9fa5]+$', cleaned_line):
                        if "中華" not in cleaned_line and "身分" not in cleaned_line:
                            data['name'] = cleaned_line
                            break

            # 2. 身分證字號
            id_match = re.search(r'[A-Z][12]\d{8}', clean_text_nospace)
            data['id_no'] = id_match.group(0) if id_match else ""

            # 3. 生日 (抓取所有日期並排序)
            # Regex 允許 "民 國" 這種空格
            date_pattern = r'民\s*國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日'
            all_dates_found = []
            
            for match in re.finditer(date_pattern, raw_text):
                y, m, d = match.groups()
                # 轉成整數方便比較
                all_dates_found.append({
                    "original": f"民國{y}年{m}月{d}日",
                    "value": int(y)*10000 + int(m)*100 + int(d)
                })
            
            if all_dates_found:
                # 排序：生日一定是最小的數字 (發證日期一定比較晚)
                all_dates_found.sort(key=lambda x: x['value'])
                data['dob'] = all_dates_found[0]['original'] # 取最小的
            else:
                data['dob'] = ""
            
            data['type_label'] = "身分證 (正面)"

        elif specific_type == "id_card_back":
            # 1. 住址 (最重要) - 抓取包含縣市路街的長字串
            addr_match = re.search(r'址[:\s]*([\u4e00-\u9fa50-9\-\(\)鄰里巷弄號樓\s]+)', raw_text)
            if addr_match:
                data['address'] = addr_match.group(1).replace(" ", "").replace("\n", "")
            else:
                # 備用：直接找很長的地址特徵
                scan_addr = re.search(r'[\u4e00-\u9fa5]+[縣市][\u4e00-\u9fa5]+[區鄉鎮市]', clean_text_nospace)
                data['address'] = scan_addr.group(0) if scan_addr else ""

            # 2. 配偶
            spouse_match = re.search(r'偶[:\s]*([\u4e00-\u9fa5\s]{2,5})', raw_text)
            if spouse_match:
                clean_spouse = spouse_match.group(1).replace(" ", "")
                data['spouse'] = clean_spouse if "役" not in clean_spouse else "" # 避免抓到役別
            
            # 3. 父母
            clean_nospace = re.sub(r'\s+', '', raw_text)
            f_match = re.search(r'父([\u4e00-\u9fa5]{2,4})', clean_nospace)
            m_match = re.search(r'母([\u4e00-\u9fa5]{2,4})', clean_nospace)
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
    st.info("⚠️ 請使用之前的完整代碼執行悠遊卡功能。")
else:
    doc_map = {"🪪 身分證辨識": "id_card", "🏥 健保卡辨識": "health_card", "✈️ 護照辨識": "passport"}
    target_type = doc_map[app_mode]
    
    st.title(app_mode)
    uploaded_file = st.file_uploader(f"請上傳 {app_mode.split(' ')[1]}", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # 顯示處理結果 (Debug 用)
        processed_image = preprocess_image(image)
        c1, c2 = st.columns(2)
        c1.image(image, caption='原始照片')
        c2.image(processed_image, caption='AI 增強 (二值化)')

        if st.button("🔍 開始辨識"):
            with st.spinner('正在進行深度 OCR 分析...'):
                # OCR 辨識 (使用二值化後的圖)
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

                with st.expander("🛠️ 查看原始 OCR 文字 (如果還是空白請看這)"):
                    st.text_area("Raw Text", raw_text, height=200)