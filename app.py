import streamlit as st
import pytesseract
from PIL import Image, ImageGrab, ImageEnhance, ImageOps
import pandas as pd
import re
import os
import shutil

# ==========================================
# 🔧 Tesseract 路徑
# ==========================================
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    tesseract_cmd = shutil.which("tesseract")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

st.set_page_config(page_title="全能證件辨識 (V6.0 濾鏡全開)", layout="wide", page_icon="🕵️")

# ==========================================
# 📷 影像預處理 (紅光濾鏡技術)
# ==========================================
def preprocess_image(image):
    # 1. 確保是 RGB 模式
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # 2. 【核心密技】分離通道，只取「紅色通道 (R)」
    # 原理：紅色的浮水印/印章在紅色通道下會變得很亮(接近白)，黑字則保持黑
    r, g, b = image.split()
    
    # 3. 放大 2 倍 (LANCZOS 高品質放大)
    new_size = (int(r.width * 2), int(r.height * 2))
    img_resized = r.resize(new_size, Image.Resampling.LANCZOS)
    
    # 4. 對比度增強 (讓黑字更深)
    enhancer = ImageEnhance.Contrast(img_resized)
    img_final = enhancer.enhance(2.0)
    
    # 5. 稍微銳利化
    enhancer_sharp = ImageEnhance.Sharpness(img_final)
    img_final = enhancer_sharp.enhance(1.2)

    return img_final

# ==========================================
# 核心邏輯：驗證
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
        if re.search(r'[A-Z][12]\d{8}', clean_text) or \
           any(x in clean_text for x in ["身分證", "出生", "性別", "統一編號"]):
            return True, "id_card_front"
        
        # 背面特徵
        back_keywords = ["配偶", "役別", "父母", "出生地", "父親", "母親", "鄉", "鎮", "鄰", "里", "區", "路", "街", "巷", "樓"]
        hit_count = sum(1 for k in back_keywords if k in clean_text)
        if hit_count >= 2: return True, "id_card_back"
            
        if "健保" in clean_text: return False, "⚠️ 錯誤：這是【健保卡】"
        
        if len(clean_text) > 10:
             return False, f"⚠️ 特徵不足 (命中數:{hit_count})。請嘗試避開浮水印或反光。"
        return False, "⚠️ 讀不到文字"

    return True, doc_type

# ==========================================
# 核心邏輯：資料提取
# ==========================================
def extract_data(text, doc_type, specific_type=None):
    raw_text = text
    clean_text_nospace = re.sub(r'[\s\.\-\_]+', '', text).upper().replace("O", "0").replace("I", "1").replace("L", "1")
    data = {}

    if doc_type == "id_card":
        if specific_type == "id_card_front":
            # 1. 姓名 (針對有"樣本"干擾的狀況優化)
            # 策略：抓取 "姓名" 後面的字，但排除 "樣本"、"樣"、"本"
            name_match = re.search(r'姓\s*名[:\s\.]*([\u4e00-\u9fa5\s]{2,10})', raw_text)
            if name_match:
                raw_name = name_match.group(1).replace(" ", "").replace("\n", "")
                # 過濾掉浮水印常見字
                filtered_name = raw_name.replace("樣本", "").replace("樣", "").replace("本", "")
                data['name'] = filtered_name
            else:
                # 備用：掃描前幾行
                lines = raw_text.split('\n')
                for line in lines[:6]:
                    c_line = re.sub(r'[^\u4e00-\u9fa5]', '', line) 
                    if 2 <= len(c_line) <= 4 and "中華" not in c_line and "身分" not in c_line:
                        data['name'] = c_line.replace("樣本", "")
                        break

            # 2. 身分證字號
            id_match = re.search(r'[A-Z][12]\d{8}', clean_text_nospace)
            data['id_no'] = id_match.group(0) if id_match else ""

            # 3. 生日 (紅色濾鏡後，這裡應該能讀到了)
            date_pattern = r'(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日'
            all_dates = []
            
            # 使用 Regex 掃描所有日期
            for match in re.finditer(date_pattern, raw_text):
                y, m, d = match.groups()
                # 簡單驗證：年份要在合理範圍 (例如 10~100) 且不能太大 (大於 100 可能是發證日)
                # 但為了保險，我們還是存下來排序
                all_dates.append({
                    "str": f"民國{y}年{m}月{d}日",
                    "val": int(y)*10000 + int(m)*100 + int(d)
                })
            
            if all_dates:
                # 生日一定是最小的那個數字 (例如 57 < 94)
                all_dates.sort(key=lambda x: x['val'])
                data['dob'] = all_dates[0]['str']
            else:
                data['dob'] = ""
            
            data['type_label'] = "身分證 (正面)"

        elif specific_type == "id_card_back":
            # 背面邏輯維持不變 (上版已修復)
            addr_match = re.search(r'址[:\s\.]*([\u4e00-\u9fa50-9\-\(\)鄰里巷弄號樓\s]+)', raw_text)
            if addr_match:
                data['address'] = addr_match.group(1).replace(" ", "").replace("\n", "")
            else:
                scan = re.search(r'[\u4e00-\u9fa5]+[縣市][\u4e00-\u9fa5]+[區鄉鎮市][\u4e00-\u9fa50-9]+', clean_text_nospace)
                data['address'] = scan.group(0) if scan else ""

            spouse_match = re.search(r'偶[:\s\.]*([\u4e00-\u9fa5\s]{2,5})', raw_text)
            if spouse_match:
                sp = spouse_match.group(1).replace(" ", "")
                data['spouse'] = sp if "役" not in sp else ""
            
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
    st.info("⚠️ 悠遊卡功能請使用先前版本。")
else:
    doc_map = {"🪪 身分證辨識": "id_card", "🏥 健保卡辨識": "health_card", "✈️ 護照辨識": "passport"}
    target_type = doc_map[app_mode]
    
    st.title(app_mode)
    uploaded_file = st.file_uploader(f"請上傳 {app_mode.split(' ')[1]}", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # 顯示處理對比圖 (讓您見證紅色濾鏡的威力)
        processed_image = preprocess_image(image)
        c1, c2 = st.columns(2)
        c1.image(image, caption='原始照片 (含紅字干擾)')
        c2.image(processed_image, caption='紅色濾鏡+放大 (紅字消失，黑字保留)')

        if st.button("🔍 開始辨識"):
            with st.spinner('紅光濾鏡分析中...'):
                # OCR
                raw_text = pytesseract.image_to_string(processed_image, lang='chi_tra+eng', config='--psm 6')
                
                is_valid, status_or_msg = validate_image_content(raw_text, target_type)
                
                if not is_valid:
                    st.error(status_or_msg)
                else:
                    specific_type = status_or_msg 
                    st.success(f"✅ 成功識別！({specific_type})")
                    
                    data = extract_data(raw_text, target_type, specific_type)
                    
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

                with st.expander("🛠️ 查看原始 OCR 文字"):
                    st.text_area("Raw Text", raw_text, height=200)