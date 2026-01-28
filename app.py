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

st.set_page_config(page_title="全能 OCR (V22 完美護照版)", layout="wide", page_icon="🌍")

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
    # 2. 國際護照 (包含中國、美國、日本、德國)
    {
        "id": "passport_universal",
        "label": "🌍 國際護照",
        "keywords": ["PASSPORT", "P<", "I<", "P[A-Z]<", "REPUBLIC"], 
        "parser": "universal_passport"
    },
    # 3. 台灣身分證 (嚴格限制)
    {
        "id": "twn_id_front",
        "label": "🇹🇼 台灣身分證 (正面)",
        "keywords": ["中華民國", "國民身分證", "統一編號"],
        # 排除任何外國關鍵字
        "exclude": ["PASSPORT", "USA", "JAPAN", "GERMANY", "CHINA", "PEOPLE", "REPUBLIC", "DIPLOMATIC"], 
        "parser": "twn_id"
    },
    {
        "id": "twn_id_back",
        "label": "🇹🇼 台灣身分證 (背面)",
        "keywords": ["配偶", "役別", "父母", "出生地", "住址"],
        # 排除護照常見字，避免中國護照(有出生地/住址)被誤判
        "exclude": ["PASSPORT", "REPUBLIC", "CHINA", "MINISTRY", "AUTHORITY", "有效期"], 
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
    # 1. 絕對優先：MRZ 特徵
    # P<... 或 I<... 或 PDCHN... (中國外交護照)
    if re.search(r'[PI]<[A-Z]{3}', clean_text) or "PDCHN" in clean_text or "PASSPORT" in clean_text:
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
            
    # Fallback: 只有在完全沒有外國關鍵字時，才允許猜台灣 ID
    if not best_match and re.search(r'[A-Z][12]\d{8}', clean_text):
        if not any(k in clean_text for k in ["USA", "CHINA", "JAPAN", "REPUBLIC", "PEOPLE"]):
            return next((d for d in DOCUMENT_CONFIG if d["id"] == "twn_id_front"), None)
    
    return best_match

# ==========================================
# 📝 解析器：MRZ 優先 + 逗號回歸
# ==========================================

def parse_mrz(clean_text):
    """
    強大的 MRZ 解析器 (支援 TD3/TD1/外交護照)
    """
    mrz_data = {}
    lines = clean_text.split('\n')
    clean_lines = [l.replace(" ", "").upper() for l in lines]
    
    for i, l in enumerate(clean_lines):
        # 模式 1: 標準護照 (P<TWN, P<JPN, P<D, PDCHN)
        if len(l) > 30 and (l.startswith("P") or l.startswith("V") or l.startswith("I")):
            # 抓名字: 位於第一個國碼之後，直到下一個數字或行尾
            # 格式通常是: P<CCCSURNAME<<GIVEN<NAME<<<<
            if "<<" in l:
                try:
                    # 分割出各個區塊
                    parts = l.split("<")
                    # 過濾掉國碼 (前3-5碼通常是國碼) 和 P
                    # 簡單策略：取長度大於 1 且不是純數字的區塊
                    valid_parts = []
                    for p in parts:
                        if len(p) >= 2 and not any(c.isdigit() for c in p):
                            # 排除常見國碼
                            if p not in ["TWN", "CHN", "JPN", "USA", "D", "DEU", "FRA"]:
                                valid_parts.append(p)
                    
                    if valid_parts:
                        # 這是 Surname, Given Name
                        mrz_data['eng_name'] = ", ".join(valid_parts)
                except: pass

            # 抓號碼: 通常在下一行
            if i+1 < len(clean_lines):
                l2 = clean_lines[i+1]
                # 號碼特徵: 前9碼是英數混合 (中國外交護照是 DE...)
                pass_no = re.search(r'[A-Z0-9]{7,9}', l2)
                if pass_no: 
                    mrz_data['passport_no'] = pass_no.group(0)

        # 模式 2: 美國卡 (I<USA)
        if l.startswith("I<") or l.startswith("C<"):
            # Line 1: 號碼在國碼後
            pass_no = re.search(r'(?<=<)[A-Z0-9]{9}', l)
            if not pass_no: pass_no = re.search(r'[A-Z0-9]{9}', l[2:])
            if pass_no: mrz_data['passport_no'] = pass_no.group(0)
            
            # Line 3: 名字 (往下找)
            for j in range(i+1, min(i+4, len(clean_lines))):
                if "<<" in clean_lines[j]:
                    parts = clean_lines[j].split("<<")
                    names = [p.replace("<", " ") for p in parts if p]
                    mrz_data['eng_name'] = ", ".join(names)
                    break

    return mrz_data

def parse_universal_passport(clean_text, raw_lines):
    data = {}
    
    # 1. MRZ 解析 (最準確，優先使用)
    data.update(parse_mrz(clean_text))
    
    # 2. 視覺補強 (如果 MRZ 沒抓到)
    
    # [護照號碼]
    if "passport_no" not in data:
        # 排除標題 (Passport No)
        cands = re.findall(r'[A-Z0-9]{7,9}', clean_text)
        for c in cands:
            # 必須包含數字 (避免抓到單純英文單字) 且不是關鍵字
            if any(char.isdigit() for char in c) and "PASS" not in c and "CODE" not in c:
                data['passport_no'] = c
                break

    # [英文姓名] - 回歸逗號邏輯 (最穩)
    if "eng_name" not in data:
        for line in raw_lines:
            # 條件：全大寫 + 包含逗號 + 長度夠
            if re.search(r'[A-Z]', line) and "," in line and len(line) > 5:
                # 排除黑名單
                line_upper = line.upper()
                blacklist = ["NAME", "SURNAME", "GIVEN", "MINISTRY", "REPUBLIC", "BIRTH", "PASSPORT", "SEX", "AUTHORITY", "DATE", "NATIONALITY"]
                if any(bad in line_upper for bad in blacklist): continue
                if re.search(r'\d', line): continue # 不能有數字
                
                # 找到 LIN, MEI-HUA
                data['eng_name'] = line
                break
                
        # 備用：如果沒逗號 (像德國護照有時分兩行)
        if "eng_name" not in data:
             # 找全大寫行，排除標題
             for line in raw_lines:
                 if re.match(r'^[A-Z\s\-]+$', line) and len(line) > 4:
                     if not any(k in line for k in ["NAME", "REP", "MIN", "PASS", "TYPE"]):
                         data['eng_name'] = line
                         break

    # 3. 台灣身分證字號 (特例)
    if "TAIWAN" in clean_text:
        id_match = re.search(r'[A-Z][12]\d{8}', clean_text)
        data['id_no'] = id_match.group(0) if id_match else ""
        
    return data

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
    st.title("🪪 智慧證件辨識 (V22 完美版)")
    supported = ", ".join([d['label'] for d in DOCUMENT_CONFIG])
    st.caption(f"目前支援：{supported}")
    
    uploaded_file = st.file_uploader("上傳證件", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="已上傳", width=400)
        
        if st.button("🚀 開始辨識"):
            with st.spinner('AI 正在分析...'):
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