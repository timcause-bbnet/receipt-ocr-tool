import streamlit as st
from rapidocr_onnxruntime import RapidOCR
from PIL import Image, ImageGrab, ImageEnhance, ImageOps
import pandas as pd
import numpy as np
import re
import cv2
from opencc import OpenCC

cc = OpenCC('s2t')
def to_traditional(text):
    return cc.convert(text)

st.set_page_config(page_title="全能 OCR (V23 視覺暴力版)", layout="wide", page_icon="🌍")

# ==========================================
# 🌍 證件設定 (邏輯優化：護照優先權最高)
# ==========================================
DOCUMENT_CONFIG = [
    # 1. 健保卡
    {
        "id": "twn_health",
        "label": "🇹🇼 台灣健保卡",
        "keywords": ["全民健康保險", "健保"],
        "parser": "twn_health"
    },
    # 2. 國際護照 (權重最高，包含所有國家)
    {
        "id": "passport_universal",
        "label": "🌍 國際護照",
        "keywords": ["PASSPORT", "P<", "I<", "REPUBLIC", "JAPAN", "USA", "DEUTSCHLAND", "CHINA"], 
        "parser": "universal_passport"
    },
    # 3. 台灣身分證 (排除外國關鍵字，防止誤判)
    {
        "id": "twn_id_front",
        "label": "🇹🇼 台灣身分證 (正面)",
        "keywords": ["中華民國", "國民身分證", "統一編號"],
        "exclude": ["PASSPORT", "USA", "JAPAN", "GERMANY", "DEUTSCHLAND", "共和國", "CHINA", "PEOPLE"], 
        "parser": "twn_id"
    },
    {
        "id": "twn_id_back",
        "label": "🇹🇼 台灣身分證 (背面)",
        "keywords": ["配偶", "役別", "父母", "出生地", "住址"],
        "exclude": ["PASSPORT", "REPUBLIC", "CHINA", "MINISTRY"], 
        "parser": "twn_id_back"
    }
]

# ==========================================
# 🔧 核心引擎
# ==========================================
@st.cache_resource
def load_engine():
    return RapidOCR()

engine = load_engine()

def preprocess_red_filter(image):
    if image.mode != 'RGB': image = image.convert('RGB')
    r, g, b = image.split()
    return r.point(lambda p: int(255 * (p / 255) ** 0.6))

def run_ocr(image_pil):
    img_np = np.array(image_pil.convert('RGB'))
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    result, _ = engine(img_cv)
    if not result: return "", []
    
    all_text = "\n".join([to_traditional(line[1]) for line in result])
    raw_lines = [to_traditional(line[1]) for line in result]
    return all_text, raw_lines

# ==========================================
# 🧠 智慧分類核心
# ==========================================
def detect_document_type(clean_text):
    # 1. 絕對優先：只要有 PASSPORT 或 共和國(針對中國護照)，直接鎖定護照
    if "PASSPORT" in clean_text or "REPUBLIC" in clean_text or "P<" in clean_text:
        # 再次確認不是台灣身分證 (台灣身分證雖然有 REPUBLIC OF CHINA，但通常比較小)
        # 如果同時有 "身分證" 字樣，才轉回去，否則預設護照
        if "身分證" not in clean_text:
            return next((d for d in DOCUMENT_CONFIG if d["id"] == "passport_universal"), None)

    best_match = None
    max_score = 0
    
    for doc in DOCUMENT_CONFIG:
        if "exclude" in doc:
            if any(ex in clean_text for ex in doc["exclude"]):
                continue
        
        score = 0
        for kw in doc["keywords"]:
            if kw in clean_text:
                score += 1
        
        if score > max_score:
            max_score = score
            best_match = doc
            
    # Fallback: 只有完全沒特徵時，才允許猜台灣 ID
    if not best_match and re.search(r'[A-Z][12]\d{8}', clean_text):
        if not any(k in clean_text for k in ["USA", "CHINA", "JAPAN", "GERMANY"]):
            return next((d for d in DOCUMENT_CONFIG if d["id"] == "twn_id_front"), None)
    
    return best_match

# ==========================================
# 📝 解析器：視覺暴力抓取 (Visual Extraction)
# ==========================================

def parse_universal_passport(clean_text, raw_lines):
    data = {}
    
    # === 1. 護照號碼 (針對右上角紅框處暴力抓取) ===
    # 我們不依賴 "Passport No" 標籤，直接找符合格式的字串
    
    # 候選清單：抓出所有可能的號碼
    candidates = []
    
    # 格式 A: 美國卡/德國 (C開頭 + 8-9碼)
    candidates += re.findall(r'[C][0-9A-Z]{8,9}', clean_text)
    # 格式 B: 日本 (雙字母 + 7碼數字)
    candidates += re.findall(r'[A-Z]{2}\d{7}', clean_text)
    # 格式 C: 中國外交 (DE + 數字) 或 一般 (E/G + 數字)
    candidates += re.findall(r'D[E]\d{7}', clean_text)
    candidates += re.findall(r'[EG]\d{8}', clean_text)
    # 格式 D: 台灣/其他 (9碼數字)
    candidates += re.findall(r'\d{9}', clean_text)
    # 格式 E: 德國/通用 (9碼英數混合)
    candidates += re.findall(r'[C-Z0-9]{9}', clean_text)

    # 過濾候選名單
    for cand in candidates:
        # 排除關鍵字
        if any(x in cand for x in ["PASS", "PORT", "REP", "CHN", "USA", "TWN", "JPN", "CODE", "TYPE"]):
            continue
        # 德國護照號碼特徵 (不會有母音，避免組成單字)
        # 這裡簡單判斷：如果長度是 9 且包含數字，優先度高
        data['passport_no'] = cand
        break # 抓到第一個符合的就停 (通常右上角會先被 OCR 讀到)

    # === 2. 英文姓名 (回歸逗號邏輯) ===
    # 這是您覺得最準的邏輯
    for line in raw_lines:
        # 條件：全大寫 + 逗號 (LIN, MEI-HUA)
        if re.search(r'[A-Z]', line) and "," in line:
            # 排除黑名單 (標題)
            line_upper = line.upper()
            blacklist = ["NAME", "SURNAME", "GIVEN", "MINISTRY", "REPUBLIC", "BIRTH", "PASSPORT", "SEX", "AUTHORITY", "DATE", "NATIONALITY"]
            if any(bad in line_upper for bad in blacklist): continue
            if re.search(r'\d', line): continue 
            
            # 找到名字
            data['eng_name'] = line
            break
    
    # 如果沒逗號 (像德國護照可能是分兩行)，嘗試找 Name 下面的字
    if "eng_name" not in data:
        for i, line in enumerate(raw_lines):
            if "NAME" in line.upper() or "SURNAME" in line.upper():
                if i + 1 < len(raw_lines):
                    potential_name = raw_lines[i+1]
                    # 簡單驗證：全大寫且無數字
                    if re.match(r'^[A-Z\s]+$', potential_name) and len(potential_name) > 3:
                        data['eng_name'] = potential_name
                        break

    # === 3. 台灣身分證字號 (特例) ===
    if "TAIWAN" in clean_text:
        id_match = re.search(r'[A-Z][12]\d{8}', clean_text)
        data['id_no'] = id_match.group(0) if id_match else ""
        
    return data

# === 以下台灣證件邏輯保持不動 ===

def parse_twn_id(clean_text, raw_lines, img_orig):
    data = {}
    id_match = re.search(r'[A-Z][12]\d{8}', clean_text)
    data['id_no'] = id_match.group(0) if id_match else ""
    
    img_filter = preprocess_red_filter(img_orig)
    _, lines_filter = run_ocr(img_filter)
    
    def find_name(lines):
        for i, line in enumerate(lines):
            if "姓名" in line:
                val = line.replace("姓名", "").strip()
                if len(val) > 1: return val
                if i+1 < len(lines): return lines[i+1]
        return ""
    
    name = find_name(lines_filter)
    if not name: name = find_name(raw_lines)
    data['name'] = name.replace("樣本", "").replace("樣", "").replace("本", "").strip()
    
    dob_match = re.search(r'民國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', clean_text)
    data['dob'] = dob_match.group(0) if dob_match else ""
    return data

def parse_twn_id_back(clean_text, raw_lines):
    data = {}
    addr = "".join([l for l in raw_lines if any(k in l for k in ["縣", "市", "區", "路", "街"])])
    data['address'] = addr.replace("住址", "")
    return data

def parse_twn_health(clean_text, raw_lines):
    data = {}
    for line in raw_lines:
        c_line = re.sub(r'[^\u4e00-\u9fa5]', '', line)
        if "全民" in c_line or "保險" in c_line: continue
        if 2 <= len(c_line) <= 4:
            data['name'] = c_line
            break
    id_match = re.search(r'[A-Z][12]\d{8}', clean_text)
    data['id_no'] = id_match.group(0) if id_match else ""
    card_match = re.search(r'\d{12}', clean_text)
    data['card_no'] = card_match.group(0) if card_match else ""
    return data

PARSERS = {
    "twn_id": parse_twn_id,
    "twn_id_back": parse_twn_id_back,
    "twn_health": parse_twn_health,
    "universal_passport": parse_universal_passport
}

# ==========================================
# 悠遊卡 (保持不變)
# ==========================================
def parse_easycard_func(text_lines):
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
# 介面
# ==========================================
st.sidebar.title("🌍 萬國 OCR")
app_mode = st.sidebar.radio("功能選單", ["💳 悠遊卡報表", "🪪 證件辨識 (自動多國)"])

if app_mode == "💳 悠遊卡報表":
    st.title("💳 悠遊卡報表")
    uploaded_file = st.file_uploader("上傳截圖", type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, width=500)
        if st.button("🚀 辨識"):
            _, lines = run_ocr(image)
            df = pd.DataFrame(parse_easycard_func(lines))
            if not df.empty: st.data_editor(df, use_container_width=True)
            else: st.error("無資料")

else:
    st.title("🪪 智慧證件辨識 (V23 視覺暴力版)")
    supported = ", ".join([d['label'] for d in DOCUMENT_CONFIG])
    st.caption(f"目前支援：{supported}")
    
    uploaded_file = st.file_uploader("上傳證件", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="已上傳", width=400)
        
        if st.button("🚀 開始辨識"):
            with st.spinner('AI 正在分析特徵...'):
                full_text, lines = run_ocr(image)
                clean_text = re.sub(r'[\s\.\-\_]+', '', full_text).upper().replace("O", "0").replace("I", "1").replace("L", "1")
                
                doc_conf = detect_document_type(clean_text)
                
                if not doc_conf:
                    st.error("⚠️ 無法識別證件類型，請確認照片清晰度。")
                    with st.expander("除錯"): st.text(full_text)
                else:
                    st.success(f"✅ 識別成功：{doc_conf['label']}")
                    parser_name = doc_conf['parser']
                    parser_func = PARSERS[parser_name]
                    
                    if parser_name == "twn_id":
                        data = parser_func(clean_text, lines, image)
                    else:
                        data = parser_func(clean_text, lines)
                    
                    st.subheader("📝 辨識結果")
                    with st.form("res"):
                        c1, c2 = st.columns(2)
                        
                        if "name" in data: c1.text_input("姓名 (中文)", data['name'])
                        if "eng_name" in data: c1.text_input("姓名 (英文)", data['eng_name'])
                        
                        if "id_no" in data: c2.text_input("身分證/公民號", data['id_no'])
                        if "passport_no" in data: c2.text_input("護照號碼", data['passport_no'])
                        if "card_no" in data: c2.text_input("健保卡號", data['card_no'])
                        
                        if "dob" in data: st.text_input("出生日期", data['dob'])
                        if "address" in data: st.text_input("住址", data['address'])
                        
                        if "father" in data: 
                            c1.text_input("父親", data['father'])
                            c2.text_input("母親", data['mother'])
                        if "spouse" in data: st.text_input("配偶", data['spouse'])
                        
                        st.form_submit_button("💾 存檔")