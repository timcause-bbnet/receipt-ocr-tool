import streamlit as st
import pytesseract
from PIL import Image, ImageGrab, ImageEnhance, ImageOps, ImageFilter
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

st.set_page_config(page_title="全能證件辨識 (V9.0 終極調校)", layout="wide", page_icon="🕵️")

# ==========================================
# 📷 影像預處理 (V9: 3倍放大 + 銳利化 + 高對比)
# ==========================================
def preprocess_image(image):
    # 1. 轉 RGB 並取紅色通道 (過濾印章)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    r, g, b = image.split()
    
    # 2. 自動色階 (拉開對比，讓字最黑、紙最白)
    img_contrasted = ImageOps.autocontrast(r, cutoff=2)
    
    # 3. 【關鍵】放大 3 倍 (讓文字筆畫分離，Tesseract 對小字很苦手)
    new_size = (int(r.width * 3), int(r.height * 3))
    img_resized = img_contrasted.resize(new_size, Image.Resampling.LANCZOS)
    
    # 4. 銳利化 (讓文字邊緣清晰)
    enhancer_sharp = ImageEnhance.Sharpness(img_resized)
    img_sharp = enhancer_sharp.enhance(2.0) # 強力銳利化
    
    # 5. 再次增強對比
    enhancer_contrast = ImageEnhance.Contrast(img_sharp)
    img_final = enhancer_contrast.enhance(1.5)
    
    # 保持灰階，不進行二值化，保留筆畫細節
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
        
        # 背面特徵
        back_keywords = ["配偶", "役別", "父母", "出生地", "父親", "母親", "鄉", "鎮", "鄰", "里", "區", "路", "街", "巷", "樓"]
        hit_count = sum(1 for k in back_keywords if k in clean_text)
        if hit_count >= 2: return True, "id_card_back"
            
        if "健保" in clean_text: return False, "⚠️ 錯誤：這是【健保卡】"
        
        if len(clean_text) > 5:
             return False, f"⚠️ 特徵不足 (命中數:{hit_count})。請確保照片清晰且無反光。"
        return False, "⚠️ 讀不到文字"

    return True, doc_type

# ==========================================
# 核心邏輯：資料提取 (Regex 再優化)
# ==========================================
def extract_data(text, doc_type, specific_type=None):
    raw_text = text
    clean_text_nospace = re.sub(r'[\s\.\-\_]+', '', text).upper().replace("O", "0").replace("I", "1").replace("L", "1")
    data = {}

    if doc_type == "id_card":
        if specific_type == "id_card_front":
            # 1. 姓名 (支援空格，並過濾掉"樣本")
            name_match = re.search(r'姓\s*名[:\s\.]*([\u4e00-\u9fa5\s]{2,10})', raw_text)
            if name_match:
                raw_name = name_match.group(1).replace(" ", "").replace("\n", "")
                data['name'] = raw_name.replace("樣本", "").replace("樣", "").replace("本", "")
            else:
                # 備用：掃描前幾行，找 2-4 個中文字
                lines = raw_text.split('\n')
                for line in lines[:8]: # 掃描範圍擴大到前8行
                    c_line = re.sub(r'[^\u4e00-\u9fa5]', '', line) 
                    if 2 <= len(c_line) <= 4 and "中華" not in c_line and "身分" not in c_line:
                        data['name'] = c_line.replace("樣本", "")
                        break

            # 2. 身分證字號
            id_match = re.search(r'[A-Z][12]\d{8}', clean_text_nospace)
            data['id_no'] = id_match.group(0) if id_match else ""

            # 3. 生日 (抓取所有日期並排序)
            date_pattern = r'(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日'
            all_dates = []
            for match in re.finditer(date_pattern, raw_text):
                y, m, d = match.groups()
                # 只有民國10年到100多內的才算，避免抓到奇怪的數字
                if 10 < int(y) < 150:
                    all_dates.append({
                        "str": f"民國{y}年{m}月{d}日",
                        "val": int(y)*10000 + int(m)*100 + int(d)
                    })
            
            if all_dates:
                all_dates.sort(key=lambda x: x['val'])
                data['dob'] = all_dates[0]['str']
            else:
                data['dob'] = ""
            
            data['type_label'] = "身分證 (正面)"

        elif specific_type == "id_card_back":
            # 背面：住址
            addr_match = re.search(r'址[:\s\.]*([\u4e00-\u9fa50-9\-\(\)鄰里巷弄號樓\s]+)', raw_text)
            if addr_match:
                data['address'] = addr_match.group(1).replace(" ", "").replace("\n", "")
            else:
                scan = re.search(r'[\u4e00-\u9fa5]+[縣市][\u4e00-\u9fa5]+[區鄉鎮市][\u4e00-\u9fa50-9]+', clean_text_nospace)
                data['address'] = scan.group(0) if scan else ""

            # 配偶
            spouse_match = re.search(r'偶[:\s\.]*([\u4e00-\u9fa5\s]{2,5})', raw_text)
            if spouse_match:
                sp = spouse_match.group(1).replace(" ", "")
                data['spouse'] = sp if "役" not in sp else ""
            
            # 父母
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
        
        # 【護照姓名 Regex 修正】
        # 允許逗號、句號、空格、連字號，確保 LIN, MEI-HUA 能被抓到
        eng_match = re.search(r'([A-Z]+[,\.]\s*[-A-Z\s]+)', raw_text)
        
        data['eng_name'] = eng_match.group(1).replace("\n", "").strip() if eng_match else ""
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
        
        # 預覽處理結果
        processed_image = preprocess_image(image)
        c1, c2 = st.columns(2)
        c1.image(image, caption='原始照片')
        c2.image(processed_image, caption='V9 終極處理 (3倍放大 + 銳利化)')

        if st.button("🔍 開始辨識"):
            with st.spinner('V9 引擎強力運算中...'):
                # V9: 強制使用 psm 6 (假設單一文字區塊)
                # 這對身分證这种有背景圖案的文件非常重要！
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

                with st.expander("🛠️ 查看原始 OCR 文字 (Debug)"):
                    st.text_area("Raw Text", raw_text, height=200)