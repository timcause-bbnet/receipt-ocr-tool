import streamlit as st
from rapidocr_onnxruntime import RapidOCR
from PIL import Image, ImageGrab, ImageEnhance, ImageOps
import pandas as pd
import numpy as np
import re
import cv2
from opencc import OpenCC

# 簡轉繁轉換器
cc = OpenCC('s2t')

def to_traditional(text):
    """簡體轉繁體"""
    return cc.convert(text)

st.set_page_config(page_title="全能 OCR (V15 三卡完美版)", layout="wide", page_icon="🚀")

# ==========================================
# 🔧 初始化 RapidOCR
# ==========================================
@st.cache_resource
def load_engine():
    engine = RapidOCR()
    return engine

engine = load_engine()

# ==========================================
# 🛠️ 影像處理工具 (保持不變)
# ==========================================
def preprocess_red_filter(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    r, g, b = image.split()
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

def run_ocr(image_pil):
    img_cv = pil_to_cv(image_pil)
    result, _ = engine(img_cv)
    if not result: return "", []
    # 簡轉繁處理
    all_text = "\n".join([to_traditional(line[1]) for line in result])
    raw_lines = [to_traditional(line[1]) for line in result]
    return all_text, raw_lines

# ==========================================
# 悠遊卡功能 (保持不變)
# ==========================================
def parse_easycard(text_lines):
    data = []
    for line in text_lines:
        line = line.strip()
        date_match = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2})', line)
        time_match = re.search(r'(\d{2}:\d{2}:\d{2})', line)
        amount_match = re.search(r'[-]?\d+', line[::-1]) 
        
        if date_match and time_match:
            full_date = date_match.group(1).replace("/", "-")
            time_part = time_match.group(1)
            amount = 0
            if amount_match: amount = amount_match.group(0)[::-1]
            
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
# 邏輯: 證件解析 (V15 健保卡補完)
# ==========================================
def extract_id_passport_dual(img_original):
    data = {}
    
    # 第一掃：原圖
    text_orig, lines_orig = run_ocr(img_original)
    clean_orig = re.sub(r'[\s\.\-\_]+', '', text_orig).upper().replace("O", "0").replace("I", "1").replace("L", "1")

    # --- 類型判斷 (修正順序：護照 -> 健保卡 -> 身分證) ---
    doc_type = "unknown"
    
    # 1. 護照 (優先)
    if "PASSPORT" in clean_orig or "REPUBLIC" in clean_orig or "TWN" in clean_orig or "MINISTRY" in clean_orig:
        doc_type = "passport"
    # 2. 【新增】健保卡 (必須在身分證之前判斷)
    elif "全民健康保險" in clean_orig or "健保" in clean_orig:
        doc_type = "health_card"
    # 3. 身分證
    elif any(x in clean_orig for x in ["身分證", "出生", "性別", "統一編號"]):
        doc_type = "id_card_front"
    elif any(x in clean_orig for x in ["配偶", "役別", "父母", "鄉鎮", "市區", "住址"]):
        doc_type = "id_card_back"
    # 4. 依照特徵猜測 (Fallback)
    elif re.search(r'[A-Z][12]\d{8}', clean_orig):
        # 有身分證字號，但沒看到中文標題，這時候要小心
        # 如果有 "NHI" 或卡號格式，也可能是健保卡
        if re.search(r'\d{12}', clean_orig) or re.search(r'\d{4}\s\d{4}\s\d{4}', text_orig):
             doc_type = "health_card"
        elif not re.search(r'[\u4e00-\u9fa5]', clean_orig):
             doc_type = "passport"
        else:
             doc_type = "id_card_front"

    data['type_label'] = doc_type

    # --- 身分證濾鏡準備 (保持不動) ---
    text_filter, lines_filter = "", []
    if doc_type == "id_card_front":
        img_filter = preprocess_red_filter(img_original)
        text_filter, lines_filter = run_ocr(img_filter)

    # === 資料提取 ===
    
    # --- 1. 健保卡邏輯 (新增) ---
    if doc_type == "health_card":
        data['type_label'] = "健保卡"
        
        # 姓名: 健保卡通常名字很大，且沒有"姓名"標籤
        # 策略: 排除掉標題字，找 2-4 個字的中文
        for line in lines_orig:
            clean_line = re.sub(r'[^\u4e00-\u9fa5]', '', line) # 只留中文
            if "全民" in clean_line or "保險" in clean_line or "樣本" in clean_line:
                continue
            if 2 <= len(clean_line) <= 4:
                data['name'] = clean_line
                break
        
        # 身分證字號
        id_match = re.search(r'[A-Z][12]\d{8}', clean_orig)
        data['id_no'] = id_match.group(0) if id_match else ""
        
        # 健保卡卡號 (12碼)
        # 格式可能是 0000 0000 0000 或 000000000000
        card_match = re.search(r'\d{4}\s*\d{4}\s*\d{4}', text_orig)
        if not card_match:
             card_match = re.search(r'\d{12}', clean_orig)
        data['card_no'] = card_match.group(0) if card_match else ""

    # --- 2. 護照邏輯 (V14 保持不動) ---
    elif doc_type == "passport":
        data['type_label'] = "護照"
        pass_match = re.search(r'[0-9]{9}', clean_orig)
        data['passport_no'] = pass_match.group(0) if pass_match else ""
        
        found_name = ""
        for line in lines_orig:
            if "," in line and re.search(r'[A-Z]', line):
                line_upper = line.upper()
                if any(bad_word in line_upper for bad_word in ["NAME", "SURNAME", "GIVEN", "MINISTRY", "REPUBLIC", "BIRTH"]):
                    continue
                clean_line = line.replace("1", "I").replace("|", "I").strip()
                found_name = clean_line
                break
        data['eng_name'] = found_name
        id_match = re.search(r'[A-Z][12]\d{8}', clean_orig)
        data['id_no'] = id_match.group(0) if id_match else ""

    # --- 3. 身分證邏輯 (V13/V14 保持不動) ---
    elif doc_type == "id_card_front":
        data['type_label'] = "身分證 (正面)"
        id_match = re.search(r'[A-Z][12]\d{8}', clean_orig)
        data['id_no'] = id_match.group(0) if id_match else ""
        
        def find_name(lines):
            for i, line in enumerate(lines):
                if "姓名" in line:
                    val = line.replace("姓名", "").strip()
                    if len(val) > 1: return val
                    if i+1 < len(lines): return lines[i+1]
            return ""

        name_candidate = find_name(lines_filter) 
        if not name_candidate: name_candidate = find_name(lines_orig) 
        data['name'] = name_candidate.replace("樣本", "").replace("樣", "").replace("本", "").strip()

        dob_match = re.search(r'民國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', text_orig)
        if not dob_match: dob_match = re.search(r'民國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', text_filter)
        data['dob'] = dob_match.group(0) if dob_match else ""

    elif doc_type == "id_card_back":
        data['type_label'] = "身分證 (背面)"
        addr = ""
        for line in lines_orig:
            if any(k in line for k in ["縣", "市", "區", "路", "街", "里", "鄰"]):
                addr += line
        data['address'] = addr.replace("住址", "")
        
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

if app_mode == "💳 悠遊卡報表":
    st.title("💳 悠遊卡報表產生器")
    
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
        if st.button("📋 讀取剪貼簿 (限本地)"):
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
                st.warning("雲端版無法直接存取剪貼簿，請使用上傳功能。")

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

else:
    st.title("🪪 智慧證件辨識 (V15 三卡合一)")
    st.info("💡 完美支援：身分證 (正/反)、護照、健保卡 (自動判斷)。")
    
    uploaded_file = st.file_uploader("上傳證件", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="已上傳圖片", width=400)

        if st.button("🚀 開始辨識"):
            with st.spinner('AI 分析中...'):
                data = extract_id_passport_dual(image)
            
            doc_label = data.get('type_label', '未知')
            
            if doc_label == "unknown":
                st.warning("⚠️ 無法識別證件，請確認照片清晰。")
            else:
                st.success(f"✅ 成功識別：{doc_label}")
                
                with st.form("result"):
                    c1, c2 = st.columns(2)
                    
                    if doc_label == "護照":
                        c1.text_input("英文姓名", data.get('eng_name', ''))
                        c2.text_input("護照號碼", data.get('passport_no', ''))
                        st.text_input("身分證字號", data.get('id_no', ''))
                    
                    elif doc_label == "健保卡":
                        c1.text_input("姓名", data.get('name', ''))
                        c2.text_input("身分證字號", data.get('id_no', ''))
                        st.text_input("健保卡號 (12碼)", data.get('card_no', ''))
                    
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