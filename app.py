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
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    tesseract_cmd = shutil.which("tesseract")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

st.set_page_config(page_title="全能證件辨識系統", layout="wide", page_icon="🕵️")

# ==========================================
# 核心邏輯：防呆驗證 (新增背面特徵)
# ==========================================
def validate_image_content(text, doc_type):
    clean_text = text.replace(" ", "").upper()
    
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
        # 1. 檢查是否為正面
        if any(x in clean_text for x in ["身分證", "出生", "性別"]) and re.search(r'[A-Z][12]\d{8}', clean_text):
            return True, "id_card_front" # 判定為正面
        
        # 2. 檢查是否為背面 (父, 母, 配偶, 住址, 役別)
        if any(x in clean_text for x in ["配偶", "役別", "住址", "父母"]):
            return True, "id_card_back" # 判定為背面
            
        if "健保" in clean_text: return False, "⚠️ 錯誤：這是【健保卡】"
        return False, "⚠️ 讀取不到身分證特徵 (請確認是否反光)"

    return True, doc_type

# ==========================================
# 核心邏輯：資料提取 (新增背面解析)
# ==========================================
def extract_data(text, doc_type, specific_type=None):
    clean_text = text.replace(" ", "").replace("\n", "")
    num_clean_text = clean_text.upper().replace("O", "0").replace("I", "1").replace("L", "1")
    data = {}

    # === 身分證系列 ===
    if doc_type == "id_card":
        if specific_type == "id_card_front":
            # 正面解析邏輯
            name_match = re.search(r'姓名[:\s]*([\u4e00-\u9fa5]{2,4})', clean_text)
            if not name_match: name_match = re.search(r'([\u4e00-\u9fa5]{2,4})性別', clean_text)
            
            id_match = re.search(r'[A-Z][12]\d{8}', num_clean_text)
            dob_match = re.search(r'民國\d{2,3}年\d{1,2}月\d{1,2}日', clean_text)

            data['name'] = name_match.group(1) if name_match else ""
            data['id_no'] = id_match.group(0) if id_match else ""
            data['dob'] = dob_match.group(0) if dob_match else ""
            data['type_label'] = "身分證 (正面)"

        elif specific_type == "id_card_back":
            # 背面解析邏輯
            # 抓配偶
            spouse_match = re.search(r'配偶([\u4e00-\u9fa5]{2,4})', clean_text)
            data['spouse'] = spouse_match.group(1) if spouse_match else ""
            
            # 抓住址 (通常在 "住址" 到 數字串(條碼) 之間，或者到最後)
            addr_match = re.search(r'住址([\u4e00-\u9fa50-9\-\(\)鄰里巷弄號樓]+)', clean_text)
            data['address'] = addr_match.group(1) if addr_match else ""

            # 抓父母 (父...母...)
            father_match = re.search(r'父([\u4e00-\u9fa5]{2,4})', clean_text)
            mother_match = re.search(r'母([\u4e00-\u9fa5]{2,4})', clean_text)
            data['father'] = father_match.group(1) if father_match else ""
            data['mother'] = mother_match.group(1) if mother_match else ""
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
        st.image(image, caption='已上傳照片', width=400)

        if st.button("🔍 開始辨識"):
            with st.spinner('正在分析影像特徵...'):
                # 1. OCR
                raw_text = pytesseract.image_to_string(image, lang='chi_tra+eng')
                
                # 2. 驗證 (並判斷是正面還是背面)
                is_valid, status_or_msg = validate_image_content(raw_text, target_type)
                
                if not is_valid:
                    st.error(status_or_msg)
                else:
                    specific_type = status_or_msg # 這裡會拿到 "id_card_front" 或 "id_card_back"
                    st.success(f"✅ 成功識別！偵測為：{specific_type}")
                    
                    # 3. 提取資料
                    data = extract_data(raw_text, target_type, specific_type)
                    
                    # 4. 動態顯示結果表單
                    st.subheader(f"📝 {data.get('type_label', '結果')} (可修改)")
                    with st.form("result_form"):
                        c1, c2 = st.columns(2)
                        
                        # 根據正面/背面顯示不同欄位
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
                with st.expander("🛠️ 查看原始 OCR 文字"):
                    st.text_area("OCR Raw Text", raw_text, height=150)