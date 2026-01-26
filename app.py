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

st.set_page_config(page_title="全能證件辨識 (V10 回歸初心版)", layout="wide", page_icon="🕵️")

# ==========================================
# 📷 影像預處理 (V10: 紅色濾鏡 + Gamma加粗)
# ==========================================
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # 1. 取紅色通道 (這招對去除紅色「樣本」印章最有效)
    r, g, b = image.split()
    
    # 2. Gamma 校正 (關鍵！把變淡的灰字強制加粗)
    # 數值 < 1.0 會讓中間調變暗(字變粗)，解決V9字太細的問題
    def gamma_correction(pixel_val):
        return int(255 * (pixel_val / 255) ** 0.6)
    
    img_gamma = r.point(gamma_correction)
    
    # 3. 放大 2 倍 (適中就好，太大會失真)
    new_size = (int(r.width * 2), int(r.height * 2))
    img_resized = img_gamma.resize(new_size, Image.Resampling.LANCZOS)
    
    # 4. 對比度增強
    enhancer = ImageEnhance.Contrast(img_resized)
    img_final = enhancer.enhance(2.0)
    
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
        # 只要能抓到身分證字號，直接放行 (針對樣本圖優化)
        if re.search(r'[A-Z][12]\d{8}', clean_text):
            return True, "id_card_front"

        # 關鍵字判定
        front_keywords = ["身", "分", "證", "出", "生", "性", "別", "統", "一", "編", "號", "民", "國"]
        back_keywords = ["配", "偶", "役", "別", "父", "母", "鄉", "鎮", "鄰", "里", "區", "路", "街", "巷", "樓"]
        
        front_score = sum(1 for k in front_keywords if k in clean_text)
        back_score = sum(1 for k in back_keywords if k in clean_text)
        
        if front_score >= 2: return True, "id_card_front"
        if back_score >= 2: return True, "id_card_back"
            
        if "健保" in clean_text: return False, "⚠️ 錯誤：這是【健保卡】"
        
        # 為了讓樣本能過，放寬錯誤提示
        if len(clean_text) > 5:
             return False, f"⚠️ 特徵不足 (正面:{front_score}, 背面:{back_score})。請確認是否為樣本圖干擾。"
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
            # 1. 姓名 (針對樣本的「圈胡」問題)
            # 優先找 "姓名" 後面的字
            name_match = re.search(r'姓\s*名[:\s\.]*([\u4e00-\u9fa5\s]{2,10})', raw_text)
            if name_match:
                raw_name = name_match.group(1).replace(" ", "").replace("\n", "")
                # 把可能誤判的浮水印字過濾掉
                data['name'] = raw_name.replace("樣本", "").replace("樣", "").replace("本", "").replace("圈", "").replace("胡", "")
            else:
                lines = raw_text.split('\n')
                found_name = ""
                for line in lines[:8]:
                    c_line = re.sub(r'[^\u4e00-\u9fa5]', '', line) 
                    if 2 <= len(c_line) <= 4 and "中華" not in c_line and "身分" not in c_line:
                        found_name = c_line
                        break
                data['name'] = found_name.replace("樣本", "").replace("圈", "")

            # 2. 身分證字號
            id_match = re.search(r'[A-Z][12]\d{8}', clean_text_nospace)
            data['id_no'] = id_match.group(0) if id_match else ""

            # 3. 生日
            date_pattern = r'(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日'
            all_dates = []
            for match in re.finditer(date_pattern, raw_text):
                y, m, d = match.groups()
                if 10 < int(y) < 150: # 合理年份
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
            # 住址
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
        
        # 【護照姓名 - 貪婪修正版】
        # 邏輯：抓取 "大寫字母 + 逗號" 開頭，然後抓取後面"所有東西"，直到遇到換行
        # 這樣就算有空格、連字號、甚至雜訊，都會先抓進來
        eng_match = re.search(r'([A-Z]+,\s*.*)', raw_text)
        
        if eng_match:
            # 抓到後，過濾掉非英文字母/連字號/空格的雜訊
            raw_eng = eng_match.group(1)
            # 只保留 A-Z, 逗號, 連字號, 空格
            clean_eng = re.sub(r'[^A-Z,\-\s]', '', raw_eng).strip()
            # 再次確保是 "名字, 名字" 的格式
            if "," in clean_eng:
                 data['eng_name'] = clean_eng
            else:
                 data['eng_name'] = ""
        else:
             data['eng_name'] = ""

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
        
        # 顯示效果：紅色濾鏡 + Gamma加粗
        processed_image = preprocess_image(image)
        c1, c2 = st.columns(2)
        c1.image(image, caption='原始照片')
        c2.image(processed_image, caption='V10 紅色濾鏡+Gamma加粗 (字體回歸)')

        if st.button("🔍 開始辨識"):
            with st.spinner('V10 引擎啟動...'):
                # 這裡使用 psm 6 (假設單一區塊)，這對樣本圖最有效
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