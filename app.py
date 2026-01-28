import streamlit as st
from rapidocr_onnxruntime import RapidOCR
from PIL import Image, ImageGrab, ImageEnhance, ImageOps
import pandas as pd
import numpy as np
import re
import cv2
from opencc import OpenCC

# 初始化簡轉繁
cc = OpenCC('s2t')
def to_traditional(text):
    return cc.convert(text)

st.set_page_config(page_title="全能 OCR (V24 特徵鎖定版)", layout="wide", page_icon="🌍")

# ==========================================
# 🌍 證件設定
# ==========================================
DOCUMENT_CONFIG = [
    # 1. 健保卡
    {
        "id": "twn_health",
        "label": "🇹🇼 台灣健保卡",
        "keywords": ["全民健康保險", "健保"],
        "parser": "twn_health"
    },
    # 2. 國際護照 (優先權最高)
    {
        "id": "passport_universal",
        "label": "🌍 國際護照",
        "keywords": ["PASSPORT", "P<", "I<", "REPUBLIC", "JAPAN", "USA", "DEUTSCHLAND", "CHINA", "PEOPLE"], 
        "parser": "universal_passport"
    },
    # 3. 台灣身分證 (嚴格限制)
    {
        "id": "twn_id_front",
        "label": "🇹🇼 台灣身分證 (正面)",
        "keywords": ["中華民國", "國民身分證", "統一編號"],
        # 排除所有可能出現在護照上的國家關鍵字
        "exclude": ["PASSPORT", "USA", "JAPAN", "GERMANY", "DEUTSCHLAND", "PEOPLE", "MINISTRY", "DIPLOMATIC"], 
        "parser": "twn_id"
    },
    {
        "id": "twn_id_back",
        "label": "🇹🇼 台灣身分證 (背面)",
        "keywords": ["配偶", "役別", "父母", "出生地", "住址"],
        "exclude": ["PASSPORT", "REPUBLIC", "CHINA", "MINISTRY", "AUTHORITY"], 
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
# 🧠 智慧分類核心 (修正誤判)
# ==========================================
def detect_document_type(clean_text):
    # 1. 絕對優先：護照特徵
    # 只要出現 PASSPORT, P<, I<, 或特定的國家英文名，直接鎖定護照
    passport_triggers = ["PASSPORT", "P<", "I<", "DEUTSCHLAND", "JAPAN", "USA", "PEOPLE'SREPUBLIC", "DIPLOMATIC"]
    if any(t in clean_text for t in passport_triggers):
        # 除非有非常明確的「國民身分證」字樣，否則都是護照
        if "國民身分證" not in clean_text:
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
            
    # Fallback
    if not best_match and re.search(r'[A-Z][12]\d{8}', clean_text):
        if not any(k in clean_text for k in ["USA", "CHINA", "JAPAN", "GERMANY"]):
            return next((d for d in DOCUMENT_CONFIG if d["id"] == "twn_id_front"), None)
    
    return best_match

# ==========================================
# 📝 解析器：特徵鎖定 (Feature Locking)
# ==========================================

def parse_universal_passport(clean_text, raw_lines):
    data = {}
    
    # === 1. 護照號碼 (分國籍精準打擊) ===
    # 我們不再用通用的 regex，而是針對每種可能，一個一個試
    
    # [A] 台灣 (9碼純數字) - 最嚴格，不准有英文
    # 排除掉日期格式 (如 20 Feb 2000)
    twn_candidates = re.findall(r'(?<!\d)\d{9}(?!\d)', clean_text)
    for c in twn_candidates:
        # 台灣護照號碼通常以 1, 2, 3, 8, 9 開頭
        if c[0] in ['1', '2', '3', '8', '9']: 
            data['passport_no'] = c
            break
            
    # [B] 日本 (2英文 + 7數字)
    if "passport_no" not in data:
        jpn_match = re.search(r'[A-Z]{2}\d{7}', clean_text)
        if jpn_match: data['passport_no'] = jpn_match.group(0)
        
    # [C] 美國卡 (C + 8數字)
    if "passport_no" not in data:
        usa_match = re.search(r'C\d{8}', clean_text)
        if usa_match: data['passport_no'] = usa_match.group(0)
        
    # [D] 中國 (E/G/D + 8數字)
    if "passport_no" not in data:
        chn_match = re.search(r'[EGD]\d{8}', clean_text) # 包含 DE + 7數字
        if not chn_match: chn_match = re.search(r'DE\d{7}', clean_text)
        if chn_match: data['passport_no'] = chn_match.group(0)
        
    # [E] 德國/通用 (9碼英數，但排除容易混淆的字)
    if "passport_no" not in data:
        # 德國護照號碼只有: C,F,G,H,J,K,L,M,N,P,R,T,V,W,X,Y,Z 和 0-9
        deu_candidates = re.findall(r'[CFGHJKLMNPRTVWXYZ0-9]{9}', clean_text)
        for c in deu_candidates:
            # 排除關鍵字干擾 (如 PASSPORT, AUTHORITY)
            if not any(bad in c for bad in ["PASS", "AUTH", "TYPE", "CODE"]):
                data['passport_no'] = c
                break

    # === 2. 英文姓名 (逗號優先) ===
    # 您的要求：之前的版本抓得很好 -> 回歸該邏輯
    for line in raw_lines:
        # 條件：包含逗號 + 大寫英文
        if "," in line and re.search(r'[A-Z]', line):
            # 排除黑名單
            line_upper = line.upper()
            blacklist = ["NAME", "SURNAME", "GIVEN", "MINISTRY", "REPUBLIC", "BIRTH", "PASSPORT", "SEX", "AUTHORITY", "DATE", "NATIONALITY", "CHINESE", "AMERICA"]
            
            # 如果這一行包含黑名單字眼，就跳過
            if any(bad in line_upper for bad in blacklist): continue
            # 如果這一行有數字 (例如地址或日期)，就跳過
            if re.search(r'\d', line): continue 
            
            data['eng_name'] = line
            break
            
    # 如果沒逗號 (例如德國護照分兩行)，才用備用邏輯
    if "eng_name" not in data:
        for i, line in enumerate(raw_lines):
            # 找 "Name" 或 "Surname" 下面那行
            if "SURNAME" in line.upper() or "NAME" in line.upper():
                if i + 1 < len(raw_lines):
                    potential = raw_lines[i+1]
                    # 檢查：全大寫、無數字、長度 > 2
                    if re.match(r'^[A-Z\s\-]+$', potential) and len(potential) > 2:
                        # 再次確認不是標題
                        if not any(k in potential.upper() for k in ["GIVEN", "GEB", "BIRTH"]):
                            data['eng_name'] = potential
                            break

    # 3. 台灣身分證字號 (只有台灣護照才抓)
    if "TAIWAN" in clean_text:
        id_match = re.search(r'[A-Z][12]\d{8}', clean_text)
        data['id_no'] = id_match.group(0) if id_match else ""
        
    return data

# === 以下台灣證件邏輯完全不動 (保留您滿意的設定) ===

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
    parents = "".join([l for l in raw_lines if "父" in l or "母" in l])
    f = re.search(r'父\s*([\u4e00-\u9fa5]+)', parents)
    m = re.search(r'母\s*([\u4e00-\u9fa5]+)', parents)
    data['father'] = f.group(1) if f else ""
    data['mother'] = m.group(1) if m else ""
    spouse = "".join([l for l in raw_lines if "配偶" in l])
    data['spouse'] = spouse.replace("配偶", "")
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
    st.title("🪪 智慧證件辨識 (V24 特徵鎖定)")
    
    uploaded_file = st.file_uploader("上傳證件", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="已上傳", width=400)
        
        if st.button("🚀 開始辨識"):
            with st.spinner('AI 正在鎖定特徵...'):
                full_text, lines = run_ocr(image)
                clean_text = re.sub(r'[\s\.\-\_]+', '', full_text).upper().replace("O", "0").replace("I", "1").replace("L", "1")
                
                doc_conf = detect_document_type(clean_text)
                
                if not doc_conf:
                    st.error("⚠️ 無法識別類型，請確認圖片清晰。")
                    with st.expander("OCR 文字內容"): st.text(full_text)
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