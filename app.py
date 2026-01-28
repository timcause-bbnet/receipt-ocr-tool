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

st.set_page_config(page_title="全能 OCR (V20 萬國通用版)", layout="wide", page_icon="🌍")

# ==========================================
# 🌍 證件設定 (簡化為三大類：身分證、健保卡、通用護照)
# ==========================================
DOCUMENT_CONFIG = [
    # 1. 台灣健保卡 (特徵明顯，優先判斷)
    {
        "id": "twn_health",
        "label": "🇹🇼 台灣健保卡",
        "keywords": ["全民健康保險", "健保", "IC卡"],
        "parser": "twn_health"
    },
    # 2. 通用護照 (只要是護照，不分國籍，全部走這裡)
    {
        "id": "passport_universal",
        "label": "🌍 國際護照 (自動偵測國籍)",
        "keywords": ["PASSPORT", "P<", "REPUBLIC", "TYPE/CODE"], 
        "parser": "universal_passport"
    },
    # 3. 台灣身分證 (嚴格限制：必須有中文標題)
    {
        "id": "twn_id_front",
        "label": "🇹🇼 台灣身分證 (正面)",
        "keywords": ["中華民國", "國民身分證", "統一編號"],
        "exclude": ["PASSPORT", "USA", "JAPAN", "GERMANY", "DEUTSCHLAND"], 
        "parser": "twn_id"
    },
    {
        "id": "twn_id_back",
        "label": "🇹🇼 台灣身分證 (背面)",
        "keywords": ["配偶", "役別", "父母", "出生地", "住址"],
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
# 🧠 智慧分類 (邏輯優化)
# ==========================================
def detect_document_type(clean_text):
    best_match = None
    max_score = 0
    
    # 預處理：將 MRZ 特徵 (P<) 的權重拉到最高
    # 只要看到 P< 開頭的字串，99% 是護照
    if "P<" in clean_text or re.search(r'P[A-Z]<', clean_text):
        return next((d for d in DOCUMENT_CONFIG if d["id"] == "passport_universal"), None)

    for doc in DOCUMENT_CONFIG:
        score = 0
        if "exclude" in doc:
            if any(ex in clean_text for ex in doc["exclude"]):
                continue
        
        for kw in doc["keywords"]:
            if kw in clean_text:
                score += 1
        
        if score > max_score:
            max_score = score
            best_match = doc
            
    # Fallback (嚴格版)：只有在確定看到中文字時，才允許猜台灣 ID
    if not best_match:
        if re.search(r'[A-Z][12]\d{8}', clean_text):
            # 必須包含至少一個台灣特有關鍵字才能放行
            if any(k in clean_text for k in ["民國", "年", "月", "日", "發證", "換發", "補發"]):
                return next((d for d in DOCUMENT_CONFIG if d["id"] == "twn_id_front"), None)
            else:
                # 否則假設是外國證件/護照 (避免德國護照誤判)
                return next((d for d in DOCUMENT_CONFIG if d["id"] == "passport_universal"), None)
    
    return best_match

# ==========================================
# 📝 解析器
# ==========================================

def parse_mrz(clean_text):
    """通用 MRZ 解析 (支援所有符合 ICAO 標準的護照)"""
    mrz_data = {}
    lines = clean_text.split('\n')
    
    for line in lines:
        l = line.replace(" ", "").upper()
        
        # 識別 MRZ 第一行 (P<...)
        # 格式: P < 國碼 (3碼) < 姓 << 名
        if len(l) > 30 and (l.startswith("P<") or l.startswith("P") and "<" in l):
            try:
                # 抓國碼 (通常在 index 2~5)
                # 例如 P<D<<... (德國是 D), P<TWN... (台灣), P<USA...
                # 這裡做簡單提取
                parts = l.split('<')
                raw_parts = [p for p in parts if p] # 去除空字串
                
                # 國碼通常是第一個 < 之後的字，或者 P 之後的字
                # 簡單判斷：如果有多個部分，第二部分通常是名字
                
                # 嘗試提取名字 (Surname + Given names)
                name_parts = []
                for p in parts:
                    if len(p) > 1 and not any(c.isdigit() for c in p) and p != "P":
                        name_parts.append(p)
                
                if len(name_parts) >= 1:
                    # 排除掉國碼 (通常是 3 碼以下，如 D, TWN, USA)
                    # 但名字也可能很短，所以這裡主要靠排除法
                    real_names = [n for n in name_parts if len(n) > 3 or n not in ["TWN", "USA", "CHN", "JPN", "DEU", "FRA", "GBR"]]
                    if not real_names and name_parts: real_names = name_parts # 如果都刪光了，就全加回來
                    
                    mrz_data['eng_name'] = " ".join(real_names)
                    
            except:
                pass

        # 識別 MRZ 第二行 (護照號碼 + 生日 + 效期)
        # 特徵：包含大量數字
        if len(l) > 30 and re.search(r'\d', l) and "<" in l:
            # 護照號碼通常在前 9 碼
            pass_no_match = re.search(r'[A-Z0-9]{7,9}', l)
            if pass_no_match:
                # 簡單驗證：不要抓到 PASSPORT 字樣
                if "PASSPORT" not in pass_no_match.group(0):
                    mrz_data['passport_no'] = pass_no_match.group(0)

    return mrz_data

def parse_universal_passport(clean_text, raw_lines):
    """萬國護照通用解析"""
    data = {}
    
    # 1. MRZ 解析 (最優先)
    data.update(parse_mrz(clean_text))
    
    # 2. 視覺補充解析
    if "passport_no" not in data:
        # 德國護照號碼特徵 (可能包含 C, F, G, H, J, K, L, M, N, P, R, T, V, W, X, Y, Z 和 0-9)
        # 排除掉 "PASSPORT", "REPUBLIC" 等字
        cands = re.findall(r'[A-Z0-9]{9}', clean_text)
        for c in cands:
            if not any(x in c for x in ["PASS", "PUBL", "NAME", "TYPE"]):
                data['passport_no'] = c
                break
    
    if "eng_name" not in data:
        for line in raw_lines:
            # 尋找全大寫英文名 (避開標題)
            if re.search(r'[A-Z]', line) and len(line) > 3:
                line_upper = line.upper()
                blacklist = ["NAME", "SURNAME", "GIVEN", "PASSPORT", "REPUBLIC", "DEUTSCHLAND", "GERMANY", "TYPE", "CODE", "NATIONALITY", "BIRTH", "DATE"]
                if any(bad in line_upper for bad in blacklist): continue
                if re.search(r'\d', line): continue
                
                # 德國/歐洲護照通常名字在 "Name / Surname" 下方
                if "," in line: # 如果有逗號 (LIN, MEI)
                    data['eng_name'] = line
                    break
                # 如果沒有逗號，可能是單行名字 (ERIKA MUSTERMANN)
                if not data.get('eng_name'):
                    data['eng_name'] = line

    # 3. 台灣身分證字號特例
    if "TAIWAN" in clean_text or "TWN" in clean_text:
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
    st.title("🪪 智慧證件辨識 (V20 萬國版)")
    st.info("💡 支援：台灣身分證/健保卡 + 全世界護照 (德/美/日/中...)。")
    
    uploaded_file = st.file_uploader("上傳證件", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="已上傳", width=400)
        
        if st.button("🚀 開始辨識"):
            with st.spinner('AI 正在分析國籍...'):
                full_text, lines = run_ocr(image)
                clean_text = re.sub(r'[\s\.\-\_]+', '', full_text).upper().replace("O", "0").replace("I", "1").replace("L", "1")
                
                doc_conf = detect_document_type(clean_text)
                
                if not doc_conf:
                    st.error("⚠️ 無法識別證件，請確認清晰度。")
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
                        
                        # 姓名
                        if "name" in data: c1.text_input("姓名 (中文)", data['name'])
                        if "eng_name" in data: c1.text_input("姓名 (英文)", data['eng_name'])
                        
                        # 號碼
                        if "id_no" in data: c2.text_input("身分證字號", data['id_no'])
                        if "passport_no" in data: c2.text_input("護照號碼", data['passport_no'])
                        if "card_no" in data: c2.text_input("健保卡號", data['card_no'])
                        
                        # 其他
                        if "dob" in data: st.text_input("出生日期", data['dob'])
                        if "address" in data: st.text_input("住址", data['address'])
                        
                        st.form_submit_button("💾 存檔")