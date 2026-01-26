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
# 核心邏輯：防呆驗證
# ==========================================
def validate_image_content(text, doc_type):
    clean_text = text.replace(" ", "").upper()
    
    if doc_type == "health_card":
        if any(x in clean_text for x in ["全民健康保險", "健保", "IC卡"]): return True, ""
        if "PASSPORT" in clean_text: return False, "⚠️ 錯誤：這是【護照】，請切換模式！"
        if "身分證" in clean_text: return False, "⚠️ 錯誤：這是【身分證】，請切換模式！"
        return False, "⚠️ 讀取不到「全民健康保險」字樣，請確認照片清晰。"

    elif doc_type == "passport":
        if any(x in clean_text for x in ["PASSPORT", "REPUBLIC", "TWN"]): return True, ""
        if "健保" in clean_text: return False, "⚠️ 錯誤：這是【健保卡】，請切換模式！"
        return False, "⚠️ 讀取不到「PASSPORT」字樣，請確認照片清晰。"

    elif doc_type == "id_card":
        if any(x in clean_text for x in ["身分證", "出生", "姓名"]): return True, ""
        if "健保" in clean_text: return False, "⚠️ 錯誤：這是【健保卡】，請切換模式！"
        return False, "⚠️ 讀取不到身分證特徵，請確認照片清晰。"

    return True, ""

# ==========================================
# 核心邏輯：強力資料提取 (Regex)
# ==========================================
def extract_data(text, doc_type):
    # 1. 基礎清理：移除空格換行，並把常見混淆字替換 (O->0)
    clean_text = text.replace(" ", "").replace("\n", "")
    # 針對數字欄位的優化清理 (把誤判的英文轉回數字)
    num_clean_text = clean_text.upper().replace("O", "0").replace("I", "1").replace("L", "1")

    data = {}

    if doc_type == "id_card":
        # 姓名：嘗試找「姓名」後面的 2-4 個字，如果找不到，就嘗試找「性別」前面的字
        name_match = re.search(r'姓名[:\s]*([\u4e00-\u9fa5]{2,4})', clean_text)
        if not name_match:
             # fallback: 找「性別」前面
             name_match = re.search(r'([\u4e00-\u9fa5]{2,4})性別', clean_text)
        data['name'] = name_match.group(1) if name_match else ""

        # 身分證字號
        id_match = re.search(r'[A-Z][12]\d{8}', num_clean_text)
        data['id_no'] = id_match.group(0) if id_match else ""

        # 生日
        dob_match = re.search(r'民國\d{2,3}年\d{1,2}月\d{1,2}日', clean_text)
        data['dob'] = dob_match.group(0) if dob_match else ""

    elif doc_type == "health_card":
        # 姓名
        name_match = re.search(r'姓名[:\s]*([\u4e00-\u9fa5]{2,4})', clean_text)
        data['name'] = name_match.group(1) if name_match else ""

        # 身分證字號
        id_match = re.search(r'[A-Z][12]\d{8}', num_clean_text)
        data['id_no'] = id_match.group(0) if id_match else ""

        # 健保卡號 (12碼)
        card_match = re.search(r'\d{12}', num_clean_text)
        data['card_no'] = card_match.group(0) if card_match else ""

    elif doc_type == "passport":
        # 護照號碼 (9碼)
        pass_match = re.search(r'[0-9]{9}', num_clean_text)
        data['passport_no'] = pass_match.group(0) if pass_match else ""

        # 身分證 (從護照內找)
        id_match = re.search(r'[A-Z][12]\d{8}', num_clean_text)
        data['id_no'] = id_match.group(0) if id_match else ""
        
        # 英文姓名 (抓取 逗號分隔的大寫英文)
        # 注意：這裡用原始 text 比較好抓，因為有空格
        eng_match = re.search(r'([A-Z]+,\s?[A-Z\-]+)', text)
        data['eng_name'] = eng_match.group(1).replace("\n", "") if eng_match else ""

    return data

# ==========================================
# 介面顯示
# ==========================================
st.sidebar.title("🧰 工具箱")
app_mode = st.sidebar.radio("請選擇功能：", 
    ["💳 悠遊卡報表產生器", "🪪 身分證辨識", "🏥 健保卡辨識", "✈️ 護照辨識"]
)

if 'current_image' not in st.session_state: st.session_state['current_image'] = None

# --- 模式 A: 悠遊卡 (保持簡化，您已有完整版代碼) ---
if app_mode == "💳 悠遊卡報表產生器":
    st.title("💳 悠遊卡報表產生器")
    uploaded_file = st.file_uploader("📂 上傳截圖", type=['png', 'jpg'])
    if uploaded_file: st.session_state['current_image'] = Image.open(uploaded_file)
    # (此處為節省篇幅省略 HTML 生成邏輯，請沿用您上一版的悠遊卡代碼)
    if st.session_state['current_image']:
        st.image(st.session_state['current_image'], width=500)
        st.info("⚠️ 請使用上一版提供的完整程式碼來執行悠遊卡功能，本頁面專注於展示修復後的證件辨識。")

# --- 模式 B/C/D: 證件辨識 ---
else:
    doc_map = {"🪪 身分證辨識": "id_card", "🏥 健保卡辨識": "health_card", "✈️ 護照辨識": "passport"}
    target_type = doc_map[app_mode]
    
    st.title(app_mode)
    uploaded_file = st.file_uploader(f"請上傳 {app_mode.split(' ')[1]}", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='已上傳照片', width=400)

        if st.button("🔍 開始辨識"):
            with st.spinner('正在分析並提取資料...'):
                # 1. OCR
                raw_text = pytesseract.image_to_string(image, lang='chi_tra+eng')
                
                # 2. 驗證
                is_valid, err_msg = validate_image_content(raw_text, target_type)
                
                if not is_valid:
                    st.error(err_msg)
                else:
                    st.success(f"✅ 成功識別為 {app_mode.split(' ')[1]}！")
                    
                    # 3. 提取資料
                    data = extract_data(raw_text, target_type)
                    
                    # 4. 顯示結果表單
                    st.subheader("📝 辨識結果 (可直接修改)")
                    with st.form("result_form"):
                        c1, c2 = st.columns(2)
                        
                        if target_type == "id_card":
                            new_name = c1.text_input("姓名", value=data.get('name', ''))
                            new_id = c2.text_input("身分證字號", value=data.get('id_no', ''))
                            new_dob = st.text_input("出生年月日", value=data.get('dob', ''))
                            
                        elif target_type == "health_card":
                            new_name = c1.text_input("姓名", value=data.get('name', ''))
                            new_id = c2.text_input("身分證字號", value=data.get('id_no', ''))
                            new_card = st.text_input("健保卡號 (12碼)", value=data.get('card_no', ''))
                            
                        elif target_type == "passport":
                            new_name = c1.text_input("英文姓名", value=data.get('eng_name', ''))
                            new_id = c2.text_input("護照號碼", value=data.get('passport_no', ''))
                            st.text_input("身分證字號 (若有)", value=data.get('id_no', ''))

                        submitted = st.form_submit_button("💾 確認存檔")
                        if submitted:
                            st.balloons()
                            st.success("資料已保存！")

                # === 除錯專區 (關鍵功能) ===
                with st.expander("🛠️ 抓不到資料？點此查看原始 OCR 文字"):
                    st.text_area("電腦讀到的內容：", raw_text, height=200)
                    st.caption("說明：若此處沒看到您的名字，代表照片可能太模糊，或字體被反光遮住了。")