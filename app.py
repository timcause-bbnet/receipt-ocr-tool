import streamlit as st
import pytesseract
from PIL import Image, ImageGrab, ImageEnhance, ImageOps, ImageFilter
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

st.set_page_config(page_title="全能證件辨識 (V8.0 柔和增強)", layout="wide", page_icon="🕵️")

# ==========================================
# 📷 影像預處理 (V8: 紅色通道 + 柔和降噪)
# ==========================================
def preprocess_image(image):
    # 1. 確保 RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # 2. 取紅色通道 (過濾紅色印章)
    r, g, b = image.split()
    
    # 3. 放大 2 倍 (LANCZOS 高品質放大)
    # 放大可以讓文字筆畫分離，避免黏在一起
    new_size = (int(r.width * 2), int(r.height * 2))
    img_resized = r.resize(new_size, Image.Resampling.LANCZOS)
    
    # 4. 柔和降噪 (關鍵步驟！)
    # 使用 MedianFilter 去除椒鹽雜訊(細小的黑點)，但保留文字邊緣
    img_blurred = img_resized.filter(ImageFilter.MedianFilter(size=3))
    
    # 5. 增強對比 (讓字變深，但不要變成死黑)
    enhancer = ImageEnhance.Contrast(img_blurred)
    img_final = enhancer.enhance(1.8)
    
    # V8修正：不執行二值化(threshold)，保持灰階，讓 OCR 自己判斷邊緣
    return img_final

# ==========================================
# 核心邏輯：防呆驗證 (積分制)
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
        # === V8 改進：積分制判定 ===
        # 正面關鍵字池
        front_keywords = ["身", "分", "證", "出", "生", "性", "別", "統", "一", "編", "號", "民", "國"]
        # 背面關鍵字池
        back_keywords = ["配", "偶", "役", "別", "父", "母", "鄉", "鎮", "鄰", "里", "區", "路", "街", "巷", "樓"]
        
        front_score = sum(1 for k in front_keywords if k in clean_text)
        back_score = sum(1 for k in back_keywords if k in clean_text)
        
        # 只要身分證字號 Regex 吻合，直接視為正面 (最強特徵)
        if re.search(r'[A-Z][12]\d{8}', clean_text):
            return True, "id_card_front"

        # 根據分數判定
        if front_score >= 2: return True, "id_card_front"
        if back_score >= 2: return True, "id_card_back"
            
        if "健保" in clean_text: return False, "⚠️ 錯誤：這是【健保卡】"
        
        if len(clean_text) > 5:
             return False, f"⚠️ 特徵不足 (正面分數:{front_score}, 背面分數:{back_score})。請避開反光。"
        return False, "⚠️ 讀不到文字，請確認照片解析度"

    return True, doc_type

# ==========================================
# 核心邏輯：資料提取
# ==========================================
def extract_data(text, doc_type, specific_type=None):
    raw_text = text
    # 清理後的文字 (無空格)
    clean_text_nospace = re.sub(r'[\s\.\-\_]+', '', text).upper().replace("O", "0").replace("I", "1").replace("L", "1")
    data = {}

    if doc_type == "id_card":
        if specific_type == "id_card_front":
            # 1. 姓名
            # 嘗試抓 "姓名" 後面的字
            name_match = re.search(r'姓\s*名[:\s\.]*([\u4e00-\u9fa5\s]{2,10})', raw_text)
            if name_match:
                raw_name = name_match.group(1).replace(" ", "").replace("\n", "")
                data['name'] = raw_name.replace("樣本", "").replace("樣", "").replace("本", "")
            else:
                # 備用：直接找第 2~6 行看起來像名字的 (排除包含"身分"或"中華"的行)
                lines = raw_text.split('\n')
                found_name = ""
                for line in lines[:6]:
                    c_line = re.sub(r'[^\u4e00-\u9fa5]', '', line) 
                    if 2 <= len(c_line) <= 4 and "中華" not in c_line and "身分" not in c_line and "出生" not in c_line:
                        found_name = c_line
                        break
                data['name'] = found_name.replace("樣本", "")

            # 2. 身分證字號
            id_match = re.search(r'[A-Z][12]\d{8}', clean_text_nospace)
            data['id_no'] = id_match.group(0) if id_match else ""

            # 3. 生日 (抓所有日期並排序)
            date_pattern = r'(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日'
            all_dates = []
            for match in re.finditer(date_pattern, raw_text):
                y, m, d = match.groups()
                # 簡單過濾：年份要在合理範圍 (例如 10~100)
                if len(y) <= 3:
                    all_dates.append({
                        "str": f"民國{y}年{m}月{d}日",
                        "val": int(y)*10000 + int(m)*100 + int(d)
                    })
            
            if all_dates:
                # 排序取最小 (生日)
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
        # 護照姓名：允許 空格 和 連字號
        eng_match = re.search(r'([A-Z]+,\s*[-A-Z\s]+)', raw_text)
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
        
        # 顯示處理後的效果
        processed_image = preprocess_image(image)
        c1, c2 = st.columns(2)
        c1.image(image, caption='原始照片')
        c2.image(processed_image, caption='V8 柔和降噪 (保留灰階層次)')

        if st.button("🔍 開始辨識"):
            with st.spinner('正在分析...'):
                # V8: 使用 psm 6 (統一區塊) 或 psm 3 (自動分割)
                # 這裡改回預設的 psm 3，因為我們沒有二值化，讓 Tesseract 自己判斷版面
                raw_text = pytesseract.image_to_string(processed_image, lang='chi_tra+eng', config='--psm 3')
                
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