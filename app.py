import streamlit as st
from rapidocr_onnxruntime import RapidOCR
from PIL import Image, ImageGrab, ImageEnhance, ImageOps
import pandas as pd
import numpy as np
import re
import cv2

st.set_page_config(page_title="全能 OCR (V13 雙重引擎版)", layout="wide", page_icon="🚀")

# ==========================================
# 🔧 初始化 RapidOCR (輕量級)
# ==========================================
@st.cache_resource
def load_engine():
    engine = RapidOCR()
    return engine

engine = load_engine()

# ==========================================
# 🛠️ 影像處理工具
# ==========================================
def preprocess_red_filter(image):
    """ 紅色濾鏡：專門用來去除紅色印章，讓黑字浮現 """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    r, g, b = image.split()
    
    # Gamma 加粗 (避免字變太淡)
    def gamma_correction(pixel_val):
        return int(255 * (pixel_val / 255) ** 0.6)
    img_gamma = r.point(gamma_correction)
    return img_gamma

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def pil_to_cv(img_pil):
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==========================================
# 核心：執行 OCR
# ==========================================
def run_ocr(image_pil):
    img_cv = pil_to_cv(image_pil)
    result, _ = engine(img_cv)
    if not result: return "", []
    
    # 組合所有文字
    all_text = "\n".join([line[1] for line in result])
    raw_lines = [line[1] for line in result]
    return all_text, raw_lines

# ==========================================
# 邏輯 1: 悠遊卡 (復活並修復)
# ==========================================
def parse_easycard(text_lines):
    data = []
    for line in text_lines:
        line = line.strip()
        # 抓取日期 (支援 2025-01-01 或 2025/01/01)
        date_match = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2})', line)
        time_match = re.search(r'(\d{2}:\d{2}:\d{2})', line)
        
        # 抓取金額 (從後面找數字)
        # RapidOCR 有時會把 "-" 讀成其他符號，這裡簡單處理
        amount_match = re.search(r'[-]?\d+', line[::-1]) 
        
        if date_match and time_match:
            full_date = date_match.group(1).replace("/", "-")
            time_part = time_match.group(1)
            
            amount = 0
            if amount_match:
                amount = amount_match.group(0)[::-1]
            
            # 清理地點雜訊
            loc_raw = line
            for useless in [full_date, full_date.replace("-", "/"), time_part, str(amount), "扣款", "交易", "連線"]:
                loc_raw = loc_raw.replace(useless, "")
            loc_raw = loc_raw.strip()

            if "加值" in loc_raw: continue
            
            transport_type = "捷運"
            if "台鐵" in loc_raw: transport_type = "台鐵"
            elif "客運" in loc_raw: transport_type = "客運"
            elif "高鐵" in loc_raw: transport_type = "高鐵"
            elif "路" in loc_raw or "車" in loc_raw: transport_type = "公車"

            data.append({
                "選取": True,
                "完整日期": f"{full_date} {time_part}",
                "短日期": full_date[5:].replace("-", "/"),
                "交通": transport_type,
                "訖點": loc_raw,
                "金額": str(amount).replace("-", "")
            })
    return data

# ==========================================
# 邏輯 2: 證件智慧解析 (雙重引擎)
# ==========================================
def extract_id_passport_dual(img_original):
    data = {}
    
    # --- 第一掃：用原圖 (讀取紅色身分證字號、護照) ---
    text_orig, lines_orig = run_ocr(img_original)
    clean_orig = re.sub(r'[\s\.\-\_]+', '', text_orig).upper().replace("O", "0").replace("I", "1").replace("L", "1")

    # --- 自動判斷類型 ---
    doc_type = "unknown"
    if "PASSPORT" in clean_orig or "REPUBLIC" in clean_orig:
        doc_type = "passport"
    elif any(x in clean_orig for x in ["身分證", "出生", "性別", "統一編號"]):
        doc_type = "id_card_front"
    elif any(x in clean_orig for x in ["配偶", "役別", "父母", "鄉鎮", "市區", "住址"]):
        doc_type = "id_card_back"
    # 如果都沒中，但有身分證格式，預設為正面
    elif re.search(r'[A-Z][12]\d{8}', clean_orig):
        doc_type = "id_card_front"

    data['type_label'] = doc_type

    # --- 第二掃：用濾鏡圖 (讀取被印章遮住的姓名) ---
    # 只有身分證正面需要這一步
    text_filter, lines_filter = "", []
    if doc_type == "id_card_front":
        img_filter = preprocess_red_filter(img_original)
        text_filter, lines_filter = run_ocr(img_filter)

    # === 開始提取資料 ===
    
    if doc_type == "passport":
        data['type_label'] = "護照"
        # 護照號碼
        pass_match = re.search(r'[0-9]{9}', clean_orig)
        data['passport_no'] = pass_match.group(0) if pass_match else ""
        
        # 英文姓名 (從原圖找)
        # 邏輯：找全大寫，且包含逗號
        for line in lines_orig:
            if "," in line and re.search(r'[A-Z]', line):
                if "MINISTRY" not in line and "REPUBLIC" not in line:
                    # 修正 OCR 常見錯誤 (例如把 I 讀成 l)
                    data['eng_name'] = line.replace("1", "I").replace("|", "I")
                    break
        
        id_match = re.search(r'[A-Z][12]\d{8}', clean_orig)
        data['id_no'] = id_match.group(0) if id_match else ""

    elif doc_type == "id_card_front":
        data['type_label'] = "身分證 (正面)"
        
        # 1. 身分證字號 (絕對要從「原圖」抓，因為它是紅色的！)
        id_match = re.search(r'[A-Z][12]\d{8}', clean_orig)
        data['id_no'] = id_match.group(0) if id_match else ""
        
        # 2. 姓名 (優先從「濾鏡圖」抓，因為可能被印章遮住)
        # 策略：找 "姓名" 關鍵字，如果濾鏡圖沒抓到，再回原圖找
        def find_name(lines):
            for i, line in enumerate(lines):
                if "姓名" in line:
                    val = line.replace("姓名", "").strip()
                    if len(val) > 1: return val
                    if i+1 < len(lines): return lines[i+1]
            return ""

        name_candidate = find_name(lines_filter) # 先試濾鏡圖
        if not name_candidate:
            name_candidate = find_name(lines_orig) # 再試原圖
            
        data['name'] = name_candidate.replace("樣本", "").replace("樣", "").replace("本", "").strip()

        # 3. 生日 (原圖通常比較準，除非被印章蓋住)
        dob_match = re.search(r'民國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', text_orig)
        if not dob_match: # 原圖沒抓到，試試濾鏡圖
             dob_match = re.search(r'民國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', text_filter)
        data['dob'] = dob_match.group(0) if dob_match else ""

    elif doc_type == "id_card_back":
        data['type_label'] = "身分證 (背面)"
        # 背面通常沒有紅字干擾，用原圖即可
        addr = ""
        for line in lines_orig:
            if any(k in line for k in ["縣", "市", "區", "路", "街", "里", "鄰"]):
                addr += line
        data['address'] = addr.replace("住址", "")
        
        # 父母/配偶
        parents_line = "".join([l for l in lines_orig if "父" in l or "母" in l])
        f_match = re.search(r'父\s*([\u4e00-\u9fa5]+)', parents_line)
        m_match = re.search(r'母\s*([\u4e00-\u9fa5]+)', parents_line)
        data['father'] = f_match.group(1) if f_match else ""
        data['mother'] = m_match.group(1) if m_match else ""
        
        spouse_line = "".join([l for l in lines_orig if "配偶" in l])
        data['spouse'] = spouse_line.replace("配偶", "")

    return data

# ==========================================
# 介面顯示
# ==========================================
st.sidebar.title("🚀 RapidOCR 工具箱")
app_mode = st.sidebar.radio("功能選單", ["💳 悠遊卡報表", "🪪 證件辨識 (自動分類)"])

if 'ocr_df' not in st.session_state: st.session_state['ocr_df'] = None

# --- 功能 1: 悠遊卡 ---
if app_mode == "💳 悠遊卡報表":
    st.title("💳 悠遊卡報表產生器")
    st.info("💡 支援截圖上傳與剪貼簿貼上 (需在本地端)。")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("📂 上傳截圖", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, width=500)
            if st.button("🚀 開始辨識"):
                text, lines = run_ocr(image)
                parsed_data = parse_easycard(lines)
                if parsed_data:
                    st.session_state['ocr_df'] = pd.DataFrame(parsed_data)
                else:
                    st.error("辨識失敗或無資料。")

    with col2:
        # 剪貼簿功能在雲端環境受限，但在本地端可用
        if st.button("📋 讀取剪貼簿"):
            try:
                img = ImageGrab.grabclipboard()
                if img:
                    st.toast("已讀取剪貼簿！")
                    text, lines = run_ocr(img)
                    parsed_data = parse_easycard(lines)
                    st.session_state['ocr_df'] = pd.DataFrame(parsed_data)
                else:
                    st.warning("剪貼簿為空。")
            except:
                st.warning("雲端版不支援直接讀取剪貼簿，請使用 Ctrl+V 上傳或存檔後上傳。")

    if st.session_state['ocr_df'] is not None:
        st.subheader("👇 編輯資料")
        edited_df = st.data_editor(
            st.session_state['ocr_df'],
            column_config={
                "選取": st.column_config.CheckboxColumn("列入", width="small"),
                "交通": st.column_config.SelectboxColumn("交通", options=["捷運", "台鐵", "高鐵", "公車"]),
            },
            hide_index=True,
            use_container_width=True
        )
        
        if st.button("產生 HTML"):
            final_data = edited_df[edited_df["選取"] == True]
            html = final_data.to_html(classes='table', index=False)
            st.download_button("下載報表", html, "report.html")

# --- 功能 2: 證件辨識 ---
else:
    st.title("🪪 智慧證件辨識 (V13)")
    st.info("💡 自動判斷身分證(正反)或護照。針對網路樣本圖進行雙重掃描優化。")
    
    uploaded_file = st.file_uploader("上傳證件", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="已上傳圖片", width=400)

        if st.button("🚀 開始辨識"):
            with st.spinner('AI 雙重引擎分析中 (原圖 + 濾鏡圖)...'):
                # 呼叫雙重引擎
                data = extract_id_passport_dual(image)
            
            # 顯示結果
            doc_label = data.get('type_label', '未知')
            if doc_label == "unknown":
                st.warning("⚠️ 無法識別證件類型，請確認照片清晰度。")
            else:
                st.success(f"✅ 成功識別：{doc_label}")
                
                with st.form("result"):
                    c1, c2 = st.columns(2)
                    if doc_label == "護照":
                        c1.text_input("英文姓名", data.get('eng_name', ''))
                        c2.text_input("護照號碼", data.get('passport_no', ''))
                        st.text_input("身分證字號", data.get('id_no', ''))
                    
                    elif doc_label == "身分證 (正面)":
                        c1.text_input("姓名", data.get('name', ''))
                        c2.text_input("身分證字號", data.get('id_no', ''))
                        st.text_input("出生年月日", data.get('dob', ''))
                        
                    elif doc_label == "身分證 (背面)":
                        st.text_input("住址", data.get('address', ''))
                        c1.text_input("父親", data.get('father', ''))
                        c2.text_input("母親", data.get('mother', ''))
                        st.text_input("配偶", data.get('spouse', ''))

                    st.form_submit_button("💾 確認存檔")