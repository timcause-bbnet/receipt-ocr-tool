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

st.set_page_config(page_title="全能 OCR (V21 錨點定位版)", layout="wide", page_icon="🌍")

# ==========================================
# 🌍 證件設定 (邏輯優化)
# ==========================================
DOCUMENT_CONFIG = [
    # 1. 健保卡 (特徵最明顯，優先)
    {
        "id": "twn_health",
        "label": "🇹🇼 台灣健保卡",
        "keywords": ["全民健康保險", "健保"],
        "parser": "twn_health"
    },
    # 2. 國際護照 (權重最高，只要有 PASSPORT 就是它)
    {
        "id": "passport_universal",
        "label": "🌍 國際護照",
        "keywords": ["PASSPORT", "P<", "I<", "C<", "REPUBLIC", "JAPAN", "USA"], 
        "parser": "universal_passport"
    },
    # 3. 台灣身分證 (最後判定，且排除護照關鍵字)
    {
        "id": "twn_id_front",
        "label": "🇹🇼 台灣身分證 (正面)",
        "keywords": ["中華民國", "國民身分證", "統一編號"],
        "exclude": ["PASSPORT", "USA", "JAPAN", "GERMANY", "DEUTSCHLAND", "共和國", "CHINA"], 
        "parser": "twn_id"
    },
    {
        "id": "twn_id_back",
        "label": "🇹🇼 台灣身分證 (背面)",
        "keywords": ["配偶", "役別", "父母", "出生地", "住址"],
        "exclude": ["PASSPORT", "REPUBLIC", "CHINA", "MINISTRY"], # 關鍵：排除中國護照
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
    
    # 轉繁體
    all_text = "\n".join([to_traditional(line[1]) for line in result])
    raw_lines = [to_traditional(line[1]) for line in result]
    return all_text, raw_lines

# ==========================================
# 🧠 智慧分類核心
# ==========================================
def detect_document_type(clean_text):
    # 1. 絕對優先：MRZ 特徵 (P<, I<, C<) 或 PASSPORT 關鍵字
    if re.search(r'[PIC]<[A-Z]{3}', clean_text) or "PASSPORT" in clean_text:
        return next((d for d in DOCUMENT_CONFIG if d["id"] == "passport_universal"), None)

    best_match = None
    max_score = 0
    
    for doc in DOCUMENT_CONFIG:
        # 排除邏輯 (一票否決)
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
        # 再次確認沒有護照特徵
        if "REPUBLIC" not in clean_text and "CHINA" not in clean_text:
            return next((d for d in DOCUMENT_CONFIG if d["id"] == "twn_id_front"), None)
    
    return best_match

# ==========================================
# 📝 解析器：錨點定位法 + 多格式 MRZ
# ==========================================

def get_value_by_anchor(lines, anchors):
    """
    錨點定位法：找到標籤 (如 Surname)，回傳它「下面」或「旁邊」的字
    """
    for i, line in enumerate(lines):
        # 如果這一行包含錨點 (例如 "Surname")
        if any(anchor in line.upper() for anchor in anchors):
            # 情況 A: 值在同一行 (Surname: LIN)
            # 移除錨點字眼
            clean = line
            for a in anchors: clean = re.sub(a, "", clean, flags=re.IGNORECASE)
            clean = clean.replace(":", "").strip()
            # 如果剩下的字夠長，那就是答案
            if len(clean) > 1: return clean
            
            # 情況 B: 值在下一行 (常見於護照)
            if i + 1 < len(lines):
                val = lines[i+1].strip()
                # 過濾掉可能是其他標籤的字
                if len(val) > 1 and not any(k in val.upper() for k in ["GIVEN", "SEX", "DATE", "NO"]):
                    return val
    return ""

def parse_mrz_advanced(clean_text):
    """
    進階 MRZ 解析：支援 TD3 (2行) 與 TD1 (3行-卡式)
    """
    mrz_data = {}
    lines = clean_text.split('\n')
    clean_lines = [l.replace(" ", "").upper() for l in lines]
    
    for i, l in enumerate(clean_lines):
        # === 格式 TD3 (標準護照: 2行, 44字) ===
        # P<TWNLIN<<MEI<HUA<<<<<<<<<<
        if l.startswith("P<") and len(l) > 30:
            try:
                parts = l.split("<")
                names = [p for p in parts if len(p) > 1 and not any(c.isdigit() for c in p)]
                # 排除 P 和 國碼
                if len(names) > 1:
                    real_names = [n for n in names[1:]] # 跳過 P
                    # 再次過濾國碼 (CHN, TWN, JPN, USA, D)
                    real_names = [n for n in real_names if n not in ["CHN", "TWN", "JPN", "USA", "D", "DEU"]]
                    mrz_data['eng_name'] = ", ".join(real_names).replace(" ,", ",")
            except: pass
            
            # 找下一行 (號碼)
            if i+1 < len(clean_lines):
                l2 = clean_lines[i+1]
                pass_no = re.search(r'[A-Z0-9]{7,9}', l2)
                if pass_no: mrz_data['passport_no'] = pass_no.group(0)

        # === 格式 TD1 (卡式/美國卡: 3行, 30字) ===
        # I<USA000000000<<<<<<<<<<<<<<< (Line 1)
        # ... (Line 2)
        # HAPPY<<TRAVELER<<<<<<<<<<<<<< (Line 3: 名字在這裡)
        if (l.startswith("I<") or l.startswith("C<") or l.startswith("A<")) and len(l) > 15:
            # Line 1 包含號碼 (通常在國碼後)
            # I<USA C03005988 5
            pass_no = re.search(r'(?<=<)[A-Z0-9]{9}', l) # 找 < 後面的9碼
            if not pass_no: pass_no = re.search(r'[A-Z0-9]{9}', l[2:]) # 備用
            if pass_no: mrz_data['passport_no'] = pass_no.group(0)
            
            # 往下找名字 (通常在第3行，或者含有 << 的行)
            for j in range(i+1, min(i+4, len(clean_lines))):
                next_l = clean_lines[j]
                if "<<" in next_l and not any(c.isdigit() for c in next_l):
                    parts = next_l.split("<<")
                    valid_names = [p.replace("<", " ") for p in parts if p]
                    mrz_data['eng_name'] = ", ".join(valid_names)
                    break

    return mrz_data

def parse_universal_passport(clean_text, raw_lines):
    data = {}
    
    # 1. 絕對優先：MRZ 解析
    data.update(parse_mrz_advanced(clean_text))
    
    # 2. 視覺解析 (Anchor Method) - 補充 MRZ 沒抓到的
    if "passport_no" not in data:
        # 找 "Passport No." 下面或旁邊的字
        anchors = ["PASSPORTNO", "PASSPORTNUMBER", "PASSNR", "DOCNO"]
        val = get_value_by_anchor(raw_lines, anchors)
        if val: 
            # 清理 (只留英數)
            val = re.sub(r'[^A-Z0-9]', '', val)
            data['passport_no'] = val
        else:
            # 德國護照特例 (右上角 C01X...)
            matches = re.findall(r'[CFGHJKLMNPRTVWXYZ0-9]{9}', clean_text)
            for m in matches:
                if not any(x in m for x in ["PASS", "TYPE", "CODE"]):
                    data['passport_no'] = m
                    break

    if "eng_name" not in data:
        # 找 "Surname" (姓) 和 "Given names" (名)
        surname = get_value_by_anchor(raw_lines, ["SURNAME", "NAME", "NOM"])
        given = get_value_by_anchor(raw_lines, ["GIVEN", "VORNAME", "PRENOMS"])
        
        if surname:
            if given:
                data['eng_name'] = f"{surname}, {given}"
            else:
                data['eng_name'] = surname
        else:
            # 備用：抓大寫英文行
            for line in raw_lines:
                if re.search(r'[A-Z]', line) and len(line) > 3:
                    if "," in line and not any(k in line.upper() for k in ["MINISTRY", "REPUBLIC"]):
                        data['eng_name'] = line
                        break

    # 3. 台灣身分證字號
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
    st.title("🪪 智慧證件辨識 (V21 錨點定位)")
    supported = ", ".join([d['label'] for d in DOCUMENT_CONFIG])
    st.caption(f"目前支援：{supported}")
    
    uploaded_file = st.file_uploader("上傳證件", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="已上傳", width=400)
        
        if st.button("🚀 開始辨識"):
            with st.spinner('AI 正在分析特徵 (MRZ優先)...'):
                full_text, lines = run_ocr(image)
                # 預處理文字，方便分類
                clean_text = re.sub(r'[\s\.\-\_]+', '', full_text).upper().replace("O", "0").replace("I", "1").replace("L", "1")
                
                doc_conf = detect_document_type(clean_text)
                
                if not doc_conf:
                    st.error("⚠️ 無法識別證件類型，請確認照片清晰。")
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