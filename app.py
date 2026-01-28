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

st.set_page_config(page_title="全能 OCR (V18 國際標準版)", layout="wide", page_icon="🌍")

# ==========================================
# 🌍 萬國證件設定檔 (優化關鍵字與權重)
# ==========================================
DOCUMENT_CONFIG = [
    # --- 台灣專區 ---
    {
        "id": "twn_id_front",
        "label": "🇹🇼 台灣身分證 (正面)",
        "keywords": ["中華民國", "國民身分證", "統一編號"], # 移除太通用的"出生"
        "exclude": ["配偶", "役別", "USA", "UNITEDSTATES", "JAPAN", "CHINA"], # 排除外國關鍵字
        "parser": "twn_id"
    },
    {
        "id": "twn_id_back",
        "label": "🇹🇼 台灣身分證 (背面)",
        "keywords": ["配偶", "役別", "父母", "出生地", "住址"],
        "parser": "twn_id_back"
    },
    {
        "id": "twn_health",
        "label": "🇹🇼 台灣健保卡",
        "keywords": ["全民健康保險", "健保", "IC卡"],
        "parser": "twn_health"
    },
    # --- 護照專區 (利用 MRZ 邏輯) ---
    {
        "id": "passport_twn",
        "label": "🇹🇼 台灣護照",
        "keywords": ["TAIWAN", "REPUBLICOFCHINA", "TWN"],
        "parser": "universal_passport"
    },
    {
        "id": "passport_chn",
        "label": "🇨🇳 中國護照",
        "keywords": ["CHINA", "CHN", "PEOPLE", "REPUBLIC"], # 拆成單字
        "parser": "universal_passport"
    },
    {
        "id": "passport_usa",
        "label": "🇺🇸 美國護照/卡",
        "keywords": ["USA", "UNITEDSTATES", "AMERICA"],
        "parser": "universal_passport"
    },
    {
        "id": "passport_jpn",
        "label": "🇯🇵 日本護照",
        "keywords": ["JAPAN", "JPN", "GAIMU"],
        "parser": "universal_passport"
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
    """紅光濾鏡"""
    if image.mode != 'RGB': image = image.convert('RGB')
    r, g, b = image.split()
    return r.point(lambda p: int(255 * (p / 255) ** 0.6))

def run_ocr(image_pil):
    img_np = np.array(image_pil.convert('RGB'))
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    result, _ = engine(img_cv)
    if not result: return "", []
    
    # 轉繁體並保留原始行結構
    all_text = "\n".join([to_traditional(line[1]) for line in result])
    raw_lines = [to_traditional(line[1]) for line in result]
    return all_text, raw_lines

# ==========================================
# 🧠 智慧分類核心
# ==========================================
def detect_document_type(clean_text):
    best_match = None
    max_score = 0
    
    # 1. 優先檢查是否為護照 (PASSPORT 權重最高)
    is_passport = "PASSPORT" in clean_text
    
    for doc in DOCUMENT_CONFIG:
        score = 0
        # 排除機制
        if "exclude" in doc:
            if any(ex in clean_text for ex in doc["exclude"]):
                continue
        
        for kw in doc["keywords"]:
            if kw in clean_text:
                score += 1
        
        # 如果有 PASSPORT 字樣，護照類別加分
        if is_passport and "passport" in doc["id"]:
            score += 2
            
        if score > max_score:
            max_score = score
            best_match = doc
            
    # Fallback 機制 (修正：避免美國卡 V 開頭號碼誤判為台灣 ID)
    if not best_match:
        # 只有在完全沒有 USA/CHINA/JAPAN 等外國關鍵字時，才允許用 Regex 猜台灣 ID
        foreign_keys = ["USA", "UNITED", "JAPAN", "CHINA", "PEOPLE"]
        if not any(k in clean_text for k in foreign_keys):
            if re.search(r'[A-Z][12]\d{8}', clean_text):
                return next((d for d in DOCUMENT_CONFIG if d["id"] == "twn_id_front"), None)
    
    return best_match

# ==========================================
# 📝 解析器 (Parser)
# ==========================================

# 1. MRZ 解析器 (護照神器) def parse_mrz(clean_text):
    """
    嘗試解析護照下方的 P<TWN... 或 P<JPN... 兩行代碼
    這是最準確的資料來源
    """
    lines = clean_text.split('\n')
    mrz_data = {}
    
    for line in lines:
        # 尋找 P<XXX 開頭的行
        if line.startswith("P<") or line.startswith("PZM") or (len(line) > 30 and "<" in line):
            # 這是 MRZ 第一行: P<TWNLIN<<MEI<HUA<<<<<<<<
            # 提取名字: 去掉 P<XXX, 把 << 變成逗號
            try:
                # 簡單提取邏輯：抓取兩個 < 之間或之後的文字
                name_part = line[5:].replace("<", " ").strip()
                mrz_data['eng_name'] = name_part
            except:
                pass
                
        # 尋找含有大量數字和 < 的行 (第二行)
        # 例如: 1234567897TWN880101...
        if re.search(r'\d{7,}.*<', line):
            # 提取護照號碼 (通常在前9碼)
            pass_no = re.search(r'[A-Z0-9]{7,9}', line)
            if pass_no:
                mrz_data['passport_no'] = pass_no.group(0)

    return mrz_data

def parse_universal_passport(clean_text, raw_lines):
    """通用護照解析 (MRZ 優先 + 視覺備援)"""
    data = {}
    
    # 策略 A: 先試著解 MRZ (最準)
    mrz_data = parse_mrz(clean_text)
    data.update(mrz_data)
    
    # 策略 B: 視覺補充 (如果 MRZ 沒抓到)
    
    # 1. 護照號碼
    if "passport_no" not in data:
        # 排除 "PASSPORT" 這個字被當成號碼
        candidates = re.findall(r'[A-Z0-9]{7,9}', clean_text)
        for cand in candidates:
            if "PASSPORT" not in cand and "REPUBLIC" not in cand:
                data['passport_no'] = cand
                break
    
    # 2. 英文姓名
    if "eng_name" not in data:
        for line in raw_lines:
            # 規則：全大寫、長度夠、不是黑名單
            if re.search(r'[A-Z]', line) and len(line) > 3:
                line_upper = line.upper()
                # 黑名單擴充
                blacklist = ["NAME", "SURNAME", "GIVEN", "MINISTRY", "REPUBLIC", "BIRTH", "PASSPORT", "JAPAN", "SEX", "TYPE", "CODE", "ISSUING", "AUTHORITY", "DATE", "NATIONALITY"]
                if any(bad in line_upper for bad in blacklist): continue
                if re.search(r'\d', line): continue 
                
                # 日本護照特徵：Surname 和 Given name 可能分行
                # 如果有逗號最穩
                if "," in line:
                    data['eng_name'] = line
                    break
                
                # 暫存可能是名字的行
                if "eng_name" not in data: data['eng_name'] = line

    # 3. 台灣身分證 (只有台灣護照有)
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

def parse_chn_id(clean_text, raw_lines):
    data = {}
    id_match = re.search(r'\d{17}[\dXx]', clean_text)
    data['id_no'] = id_match.group(0) if id_match else ""
    for line in raw_lines:
        if "姓名" in line:
            data['name'] = line.replace("姓名", "").strip()
            break
    addr = ""
    start = False
    for line in raw_lines:
        if "住址" in line:
            start = True
            addr += line.replace("住址", "")
        elif "公民" in line: start = False
        elif start: addr += line
    data['address'] = addr
    return data

PARSERS = {
    "twn_id": parse_twn_id,
    "twn_id_back": parse_twn_id_back,
    "twn_health": parse_twn_health,
    "universal_passport": parse_universal_passport,
    "chn_id": parse_chn_id
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
# 🖥️ 介面
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
    st.title("🪪 智慧證件辨識 (國際強化版)")
    supported = ", ".join([d['label'] for d in DOCUMENT_CONFIG])
    st.caption(f"目前支援：{supported}")
    
    uploaded_file = st.file_uploader("上傳證件", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="已上傳", width=400)
        
        if st.button("🚀 開始辨識"):
            with st.spinner('AI 正在分析 MRZ 與 特徵...'):
                full_text, lines = run_ocr(image)
                clean_text = re.sub(r'[\s\.\-\_]+', '', full_text).upper().replace("O", "0").replace("I", "1").replace("L", "1")
                
                doc_conf = detect_document_type(clean_text)
                
                if not doc_conf:
                    st.error("⚠️ 無法識別證件類型，請確認照片清晰度。")
                    with st.expander("除錯資訊"): st.text(full_text)
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
                        if "name" in data: c1.text_input("姓名", data['name'])
                        if "eng_name" in data: c1.text_input("英文姓名", data['eng_name'])
                        
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