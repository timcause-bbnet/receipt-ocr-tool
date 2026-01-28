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

st.set_page_config(page_title="全能 OCR (V19 修復版)", layout="wide", page_icon="🌍")

# ==========================================
# 🌍 萬國證件設定檔 (修正關鍵字邏輯)
# ==========================================
DOCUMENT_CONFIG = [
    # --- 優先檢查護照 (權重高) ---
    {
        "id": "passport_twn",
        "label": "🇹🇼 台灣護照",
        "keywords": ["TAIWAN", "REPUBLICOFCHINA", "TWN"],
        "parser": "universal_passport"
    },
    {
        "id": "passport_chn",
        "label": "🇨🇳 中國護照",
        "#": "必須包含中文全名，防止誤判",
        "keywords": ["中華人民共和國", "POEPLESREPUBLIC", "CHINA", "CHN"], 
        "parser": "universal_passport"
    },
    {
        "id": "passport_usa",
        "label": "🇺🇸 美國護照/卡",
        "keywords": ["UNITEDSTATES", "AMERICA", "USA"],
        "parser": "universal_passport"
    },
    {
        "id": "passport_jpn",
        "label": "🇯🇵 日本護照",
        "keywords": ["JAPAN", "JPN", "GAIMU"],
        "parser": "universal_passport"
    },
    # --- 台灣證件 ---
    {
        "id": "twn_health",
        "label": "🇹🇼 台灣健保卡",
        "keywords": ["全民健康保險", "健保", "IC卡"],
        "parser": "twn_health"
    },
    {
        "id": "twn_id_front",
        "label": "🇹🇼 台灣身分證 (正面)",
        "keywords": ["中華民國", "國民身分證", "統一編號"],
        # 強力排除：只要出現這些國家字眼，絕對不是台灣身分證
        "exclude": ["配偶", "役別", "USA", "UNITED", "JAPAN", "CHINA", "共和國", "PASSPORT"], 
        "parser": "twn_id"
    },
    {
        "id": "twn_id_back",
        "label": "🇹🇼 台灣身分證 (背面)",
        "keywords": ["配偶", "役別", "父母", "出生地", "住址"],
        "exclude": ["PASSPORT", "共和國", "CHINA", "USA"], # 避免誤判中國護照
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
    
    # 轉繁體並去除空格，方便比對
    all_text = "\n".join([to_traditional(line[1]) for line in result])
    raw_lines = [to_traditional(line[1]) for line in result]
    return all_text, raw_lines

# ==========================================
# 🧠 智慧分類核心
# ==========================================
def detect_document_type(clean_text):
    best_match = None
    max_score = 0
    
    for doc in DOCUMENT_CONFIG:
        score = 0
        
        # 1. 排除機制 (一票否決)
        if "exclude" in doc:
            if any(ex in clean_text for ex in doc["exclude"]):
                continue
        
        # 2. 計算關鍵字
        for kw in doc["keywords"]:
            if kw in clean_text:
                score += 1
        
        # 3. 護照 MRZ 特徵加分 (P<TWN, P<JPN)
        if "passport" in doc["id"] and ("P<" in clean_text or "PKA" in clean_text):
             score += 1

        if score > max_score:
            max_score = score
            best_match = doc
            
    # Fallback: 如果特徵不足，但有 ID 格式
    if not best_match:
        # 如果有 "共和國" 或 "CHINA"，強制轉中國護照 (修復誤判)
        if "共和國" in clean_text or "CHINA" in clean_text:
             return next((d for d in DOCUMENT_CONFIG if d["id"] == "passport_chn"), None)
             
        if re.search(r'[A-Z][12]\d{8}', clean_text):
            return next((d for d in DOCUMENT_CONFIG if d["id"] == "twn_id_front"), None)
    
    return best_match

# ==========================================
# 📝 解析器 (補回 parse_mrz)
# ==========================================

def parse_mrz(clean_text):
    """
    解析護照下方的機器讀碼區 (MRZ)
    例如: P<JPNSAKURA<<GAIMU<<<<<<<<
    """
    mrz_data = {}
    lines = clean_text.split('\n')
    
    for line in lines:
        # 移除空格以便分析
        l = line.replace(" ", "")
        
        # 抓名字 (第一行通常以 P, V, I 開頭，含有 <<)
        if (l.startswith("P") or l.startswith("V") or l.startswith("I")) and "<<" in l:
            # 找到國碼後的區塊 (P<TWN 或 P<JPN)
            # 簡單暴力的解法：找第一個 < 之後的字
            try:
                parts = l.split("<")
                # 組合名字：通常是 Surname<<Given<Name
                # 過濾掉空字串和國碼
                valid_parts = [p for p in parts if len(p) > 2 and not any(c.isdigit() for c in p)]
                if valid_parts:
                    # 這是很粗略的抓法，但對大部分護照有效
                    full_str = " ".join(valid_parts).replace("PASSPORT", "").strip()
                    if len(full_str) > 3:
                        mrz_data['eng_name'] = full_str
            except:
                pass
                
        # 抓護照號碼 (通常在第二行，包含數字)
        # 格式: 號碼 + 國碼 + 生日
        # 尋找連續的數字與字母組合
        pass_match = re.search(r'[A-Z0-9]{8,9}', l)
        if pass_match and not "PASSPORT" in l:
             # 簡單驗證：護照號碼通常會有數字
             if any(c.isdigit() for c in pass_match.group(0)):
                 mrz_data['passport_no'] = pass_match.group(0)

    return mrz_data

def parse_universal_passport(clean_text, raw_lines):
    """通用護照解析"""
    data = {}
    
    # 1. 先用 MRZ 嘗試解析 (補回此功能)
    mrz_data = parse_mrz(clean_text)
    data.update(mrz_data)
    
    # 2. 視覺解析 (備援)
    if "passport_no" not in data:
        # 排除標題字
        cands = re.findall(r'[A-Z0-9]{7,9}', clean_text)
        for c in cands:
            if "PASSPORT" not in c and "REPUBLIC" not in c and "CHINA" not in c:
                data['passport_no'] = c
                break
                
    if "eng_name" not in data:
        for line in raw_lines:
            # 找全大寫英文
            if re.search(r'[A-Z]', line) and len(line) > 3:
                line_upper = line.upper()
                blacklist = ["NAME", "SURNAME", "GIVEN", "MINISTRY", "REPUBLIC", "BIRTH", "PASSPORT", "JAPAN", "SEX", "TYPE", "CODE", "ISSUING", "AUTHORITY", "DATE", "NATIONALITY", "CHINESE", "AMERICA"]
                if any(bad in line_upper for bad in blacklist): continue
                if re.search(r'\d', line): continue 
                
                if "," in line:
                    data['eng_name'] = line
                    break
                if not data.get('eng_name'): 
                    data['eng_name'] = line

    # 3. 台灣護照才有身分證
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

# Parser 路由
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
    st.title("🪪 智慧證件辨識 (V19 萬國修復版)")
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
                    st.error("⚠️ 無法識別類型")
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