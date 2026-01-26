import streamlit as st
# 改用 RapidOCR (ONNXRuntime版)，輕量又強大
from rapidocr_onnxruntime import RapidOCR
from PIL import Image, ImageGrab, ImageEnhance, ImageOps
import pandas as pd
import numpy as np
import re
import cv2

st.set_page_config(page_title="全能 OCR (V12 RapidOCR版)", layout="wide", page_icon="🚀")

# ==========================================
# 🔧 初始化 RapidOCR
# ==========================================
@st.cache_resource
def load_engine():
    # det_use_cuda=False (雲端只有 CPU)
    # 第一次執行會自動下載輕量模型 (約 10MB)，非常快
    engine = RapidOCR()
    return engine

engine = load_engine()

# ==========================================
# 🛠️ 輔助工具：把 RapidOCR 結果轉成文字
# ==========================================
def run_ocr(image):
    # 轉換 PIL Image -> OpenCV 格式 (numpy)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_np = np.array(image)
    # RGB -> BGR (因為 OpenCV 吃 BGR)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # 執行辨識
    result, elapse = engine(img_cv)
    
    if not result:
        return "", []
        
    # result 結構: [[座標], '文字', 信心度]
    # 我們把它接成一個大字串，模擬以前 Tesseract 的輸出，方便 Regex 處理
    all_text = "\n".join([line[1] for line in result])
    raw_lines = [line[1] for line in result]
    
    return all_text, raw_lines

# ==========================================
# 📷 影像預處理 (Gamma 加粗 - 選用)
# ==========================================
def preprocess_image_gamma(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # 分離紅色通道 (過濾印章)
    r, g, b = image.split()
    # Gamma 加粗 (讓被洗淡的字變黑)
    def gamma_correction(pixel_val):
        return int(255 * (pixel_val / 255) ** 0.6)
    img_gamma = r.point(gamma_correction)
    return img_gamma

# ==========================================
# 邏輯 1: 悠遊卡解析 (復活版)
# ==========================================
def parse_easycard(text_lines):
    data = []
    # 針對每一行文字進行分析
    for line in text_lines:
        line = line.strip()
        # Regex 找日期時間 + 金額
        # RapidOCR 斷句比較準，通常一行就是一筆
        # 尋找: 2025-xx-xx 或 2025/xx/xx
        date_match = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2})', line)
        time_match = re.search(r'(\d{2}:\d{2}:\d{2})', line)
        amount_match = re.search(r'[-]?\d+', line[::-1]) # 從後面找金額
        
        if date_match and time_match:
            full_date = date_match.group(1).replace("/", "-")
            time_part = time_match.group(1)
            
            # 金額處理 (反轉回來)
            amount = 0
            if amount_match:
                amt_str = amount_match.group(0)[::-1]
                amount = amt_str
            
            # 地點處理 (移除日期、時間、金額、扣款等字眼)
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
# 邏輯 2: 證件解析 (RapidOCR版)
# ==========================================
def extract_id_passport(all_text, raw_lines, doc_type):
    # 移除空格方便找 ID
    clean_text = re.sub(r'[\s\.\-\_]+', '', all_text).upper().replace("O", "0").replace("I", "1").replace("L", "1")
    data = {}

    if doc_type == "passport":
        data['type_label'] = "護照"
        # 護照號碼
        pass_match = re.search(r'[0-9]{9}', clean_text)
        data['passport_no'] = pass_match.group(0) if pass_match else ""
        # 英文姓名 (RapidOCR 讀英文很準)
        # 找全大寫且有逗號的行
        for line in raw_lines:
            if "," in line and re.search(r'[A-Z]', line):
                if "MINISTRY" not in line and "REPUBLIC" not in line:
                     data['eng_name'] = line
                     break
        # 身分證字號
        id_match = re.search(r'[A-Z][12]\d{8}', clean_text)
        data['id_no'] = id_match.group(0) if id_match else ""

    elif doc_type == "id_card":
        # 判斷正反面
        is_back = any(x in clean_text for x in ["配偶", "役別", "父母", "鄉鎮", "市區", "住址"])
        
        if is_back:
            data['type_label'] = "身分證 (背面)"
            # 住址: 找含有縣/市/區/路的行
            addr = ""
            for line in raw_lines:
                if any(k in line for k in ["縣", "市", "區", "路", "街", "里", "鄰"]):
                    addr += line
            data['address'] = addr.replace("住址", "")
            
            # 父母/配偶: 簡單關鍵字抓取
            parents_line = "".join([l for l in raw_lines if "父" in l or "母" in l])
            f_match = re.search(r'父\s*([\u4e00-\u9fa5]+)', parents_line)
            m_match = re.search(r'母\s*([\u4e00-\u9fa5]+)', parents_line)
            data['father'] = f_match.group(1) if f_match else ""
            data['mother'] = m_match.group(1) if m_match else ""
            
            spouse_line = "".join([l for l in raw_lines if "配偶" in l])
            data['spouse'] = spouse_line.replace("配偶", "")
            
        else:
            data['type_label'] = "身分證 (正面)"
            # 姓名: 找 "姓名" 附近的字
            for i, line in enumerate(raw_lines):
                if "姓名" in line:
                    potential_name = line.replace("姓名", "").strip()
                    if len(potential_name) > 1:
                        data['name'] = potential_name
                    elif i+1 < len(raw_lines):
                        data['name'] = raw_lines[i+1]
                    break
            # 去除樣本字樣
            if 'name' in data:
                data['name'] = data['name'].replace("樣本", "").replace("樣", "").replace("本", "")

            # ID
            id_match = re.search(r'[A-Z][12]\d{8}', clean_text)
            data['id_no'] = id_match.group(0) if id_match else ""
            
            # 生日
            dob_match = re.search(r'民國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', all_text)
            data['dob'] = dob_match.group(0) if dob_match else ""

    return data

# ==========================================
# 介面顯示
# ==========================================
st.sidebar.title("🚀 RapidOCR 工具箱")
app_mode = st.sidebar.radio("功能選單", ["💳 悠遊卡報表", "🪪 證件辨識 (身分證/護照)"])

if 'ocr_df' not in st.session_state: st.session_state['ocr_df'] = None

# --- 功能 1: 悠遊卡 (完全回歸) ---
if app_mode == "💳 悠遊卡報表":
    st.title("💳 悠遊卡報表產生器 (RapidOCR版)")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("📂 上傳截圖", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, width=500)
            if st.button("🚀 開始辨識", key="btn_easy"):
                full_text, lines = run_ocr(image)
                # 解析
                parsed_data = parse_easycard(lines)
                if parsed_data:
                    st.session_state['ocr_df'] = pd.DataFrame(parsed_data)
                else:
                    st.error("無法辨識資料，請確認截圖是否清晰。")
                    st.text(full_text) # debug

    if st.session_state['ocr_df'] is not None:
        st.subheader("👇 編輯資料")
        edited_df = st.data_editor(
            st.session_state['ocr_df'],
            column_config={
                "選取": st.column_config.CheckboxColumn("列入", width="small"),
                "交通": st.column_config.SelectboxColumn("交通", options=["捷運", "台鐵", "高鐵", "公車", "計程車"]),
            },
            hide_index=True,
            use_container_width=True
        )
        
        if st.button("產生 HTML"):
            final_data = edited_df[edited_df["選取"] == True]
            # (這裡簡化 HTML 生成邏輯，您可以把之前美美的 HTML 貼回來)
            html = final_data.to_html() 
            st.download_button("下載報表", html, "report.html")

# --- 功能 2: 證件辨識 ---
else:
    st.title("🪪 證件辨識 (支援樣本圖)")
    doc_type_ui = st.selectbox("證件類型", ["身分證 (自動正反)", "護照"])
    doc_map = {"身分證 (自動正反)": "id_card", "護照": "passport"}
    
    uploaded_file = st.file_uploader("上傳證件", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # 樣本圖特殊處理開關
        use_filter = st.checkbox("開啟紅字濾鏡 (針對網路樣本)", value=True)
        
        if use_filter:
            # 針對紅色樣本，用紅光濾鏡處理後再辨識
            proc_img = preprocess_image_gamma(image)
            st.image(proc_img, caption="預處理後 (過濾紅章)", width=400)
            target_img = proc_img
        else:
            st.image(image, caption="原始圖", width=400)
            target_img = image

        if st.button("🚀 開始辨識"):
            full_text, lines = run_ocr(target_img)
            
            data = extract_id_passport(full_text, lines, doc_map[doc_type_ui])
            
            st.success(f"辨識完成！類型: {data.get('type_label', '未知')}")
            
            with st.form("result"):
                c1, c2 = st.columns(2)
                if "name" in data: c1.text_input("姓名", data['name'])
                if "eng_name" in data: c1.text_input("英文姓名", data['eng_name'])
                if "id_no" in data: c2.text_input("身分證字號", data['id_no'])
                if "dob" in data: st.text_input("生日", data['dob'])
                if "address" in data: st.text_input("住址", data['address'])
                if "passport_no" in data: c2.text_input("護照號碼", data['passport_no'])
                
                st.form_submit_button("存檔")
                
            with st.expander("查看原始文字"):
                st.text(full_text)