import streamlit as st
from rapidocr_onnxruntime import RapidOCR
from PIL import Image, ImageGrab, ImageEnhance, ImageOps
import pandas as pd
import numpy as np
import re
import cv2
from opencc import OpenCC

# 初始化
cc = OpenCC('s2t')
def to_traditional(text):
    return cc.convert(text)

st.set_page_config(page_title="全能 OCR (V17 萬國通用版)", layout="wide", page_icon="🌍")

# ==========================================
# 🌍 萬國證件設定檔 (以後改這裡就好！)
# ==========================================
# 邏輯：程式會由上往下檢查，只要命中 "keywords" 裡的 2 個關鍵字，就認定是該證件
DOCUMENT_CONFIG = [
    # --- 台灣專區 ---
    {
        "id": "twn_id_front",
        "label": "🇹🇼 台灣身分證 (正面)",
        "keywords": ["中華民國", "國民身分證", "統一編號", "出生年月日"],
        "exclude": ["配偶", "役別"], # 如果出現這些字，就絕對不是這個
        "parser": "twn_id" # 指定使用哪種解析邏輯
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
    {
        "id": "twn_passport",
        "label": "🇹🇼 台灣護照",
        "keywords": ["REPUBLIC OF CHINA", "TAIWAN", "PASSPORT", "MINISTRY"],
        "parser": "universal_passport"
    },

    # --- 中國專區 (新增) ---
    {
        "id": "chn_id",
        "label": "🇨🇳 中國居民身分證",
        "keywords": ["居民身份证", "公民身份", "汉族", "出生"],
        "parser": "chn_id"
    },
    {
        "id": "chn_passport",
        "label": "🇨🇳 中國護照",
        "keywords": ["PEOPLE'S REPUBLIC OF CHINA", "PASSPORT", "CHN"],
        "parser": "universal_passport"
    },

    # --- 美國專區 (新增) ---
    {
        "id": "usa_passport",
        "label": "🇺🇸 美國護照",
        "keywords": ["UNITED STATES OF AMERICA", "USA", "PASSPORT"],
        "parser": "universal_passport"
    },
    # 這裡可以繼續加駕照等...

    # --- 日本專區 ---
    {
        "id": "jpn_passport",
        "label": "🇯🇵 日本護照",
        "keywords": ["JAPAN", "JPN", "PASSPORT"],
        "parser": "universal_passport"
    },
]

# ==========================================
# 🔧 核心引擎與工具
# ==========================================
@st.cache_resource
def load_engine():
    return RapidOCR()

engine = load_engine()

def preprocess_red_filter(image):
    """紅光濾鏡 (去印章)"""
    if image.mode != 'RGB': image = image.convert('RGB')
    r, g, b = image.split()
    return r.point(lambda p: int(255 * (p / 255) ** 0.6))

def run_ocr(image_pil):
    """執行 OCR 並回傳繁體中文結果"""
    img_np = np.array(image_pil.convert('RGB'))
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    result, _ = engine(img_cv)
    if not result: return "", []
    
    # 全部轉繁體方便比對
    all_text = "\n".join([to_traditional(line[1]) for line in result])
    raw_lines = [to_traditional(line[1]) for line in result]
    return all_text, raw_lines

# ==========================================
# 🧠 智慧分類核心 (不用再寫 if else 了)
# ==========================================
def detect_document_type(clean_text):
    """根據設定檔自動判斷證件類型"""
    best_match = None
    max_score = 0
    
    for doc in DOCUMENT_CONFIG:
        score = 0
        # 檢查排除關鍵字 (一票否決)
        if "exclude" in doc:
            if any(ex in clean_text for ex in doc["exclude"]):
                continue
        
        # 計算關鍵字命中數
        for kw in doc["keywords"]:
            if kw in clean_text:
                score += 1
        
        # 護照權重加成 (避免被身分證搶走)
        if "PASSPORT" in clean_text and "keywords" in doc and "PASSPORT" in doc["keywords"]:
            score += 2
            
        if score >= 2 and score > max_score:
            max_score = score
            best_match = doc
            
    # Fallback: 如果都沒中，但有身分證字號，猜是台灣身分證
    if not best_match and re.search(r'[A-Z][12]\d{8}', clean_text):
        if "PASSPORT" in clean_text:
            return next((d for d in DOCUMENT_CONFIG if d["id"] == "twn_passport"), None)
        return next((d for d in DOCUMENT_CONFIG if d["id"] == "twn_id_front"), None)
        
    return best_match

# ==========================================
# 📝 各國證件解析器 (Parser)
# ==========================================

def parse_twn_id(clean_text, raw_lines, img_orig):
    """台灣身分證解析"""
    data = {}
    # 1. 字號
    id_match = re.search(r'[A-Z][12]\d{8}', clean_text)
    data['id_no'] = id_match.group(0) if id_match else ""
    
    # 2. 姓名 (嘗試用濾鏡)
    # 只有這裡需要跑第二次 OCR (紅光濾鏡)
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
    
    # 3. 生日
    dob_match = re.search(r'民國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', clean_text)
    data['dob'] = dob_match.group(0) if dob_match else ""
    
    return data

def parse_twn_id_back(clean_text, raw_lines):
    """台灣身分證背面"""
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

def parse_chn_id(clean_text, raw_lines):
    """🇨🇳 中國身分證解析"""
    data = {}
    # 1. 公民身分號碼 (18碼數字, 最後一位可能是X)
    id_match = re.search(r'\d{17}[\dXx]', clean_text)
    data['id_no'] = id_match.group(0) if id_match else ""
    
    # 2. 姓名
    for line in raw_lines:
        if "姓名" in line:
            data['name'] = line.replace("姓名", "").strip()
            break
            
    # 3. 住址
    addr = ""
    start_addr = False
    for line in raw_lines:
        if "住址" in line:
            start_addr = True
            addr += line.replace("住址", "")
        elif "公民" in line:
            start_addr = False
        elif start_addr:
            addr += line
    data['address'] = addr
    
    return data

def parse_universal_passport(clean_text, raw_lines):
    """🌍 通用護照解析 (美/中/台/日皆可用)"""
    data = {}
    # 1. 護照號碼 (通常是 7-9 位英數)
    # 台灣/美國/中國: 9碼數字 或 字母+數字
    pass_match = re.search(r'[A-Z0-9]{7,9}', clean_text)
    data['passport_no'] = pass_match.group(0) if pass_match else ""
    
    # 2. 英文姓名 (排除標題字)
    found_name = ""
    for line in raw_lines:
        if re.search(r'[A-Z]', line) and len(line) > 3:
            line_upper = line.upper()
            if any(bad in line_upper for bad in ["NAME", "SURNAME", "GIVEN", "MINISTRY", "REPUBLIC", "BIRTH", "PASSPORT", "JAPAN", "SEX", "UNITED", "STATES"]):
                continue
            if re.search(r'\d', line): continue # 排除有數字的行
            
            # 有逗號最優先 (LIN, MEI-HUA)
            if "," in line:
                found_name = line
                break
            # 沒逗號但看起來像名字 (MAY LIN)
            if not found_name:
                found_name = line
    data['eng_name'] = found_name
    
    # 3. 身分證字號 (只有台灣護照才有，其他國家留白)
    if "TAIWAN" in clean_text or "REPUBLICOFCHINA" in clean_text:
        id_match = re.search(r'[A-Z][12]\d{8}', clean_text)
        data['id_no'] = id_match.group(0) if id_match else ""
        
    return data

def parse_twn_health(clean_text, raw_lines):
    """台灣健保卡"""
    data = {}
    # 姓名 (排除標題)
    for line in raw_lines:
        c_line = re.sub(r'[^\u4e00-\u9fa5]', '', line)
        if "全民" in c_line or "保險" in c_line: continue
        if 2 <= len(c_line) <= 4:
            data['name'] = c_line
            break
            
    # 字號
    id_match = re.search(r'[A-Z][12]\d{8}', clean_text)
    data['id_no'] = id_match.group(0) if id_match else ""
    
    # 卡號
    card_match = re.search(r'\d{12}', clean_text)
    data['card_no'] = card_match.group(0) if card_match else ""
    return data

# 解析器路由
PARSERS = {
    "twn_id": parse_twn_id,
    "twn_id_back": parse_twn_id_back,
    "twn_health": parse_twn_health,
    "chn_id": parse_chn_id,
    "universal_passport": parse_universal_passport
}

# ==========================================
# 悠遊卡功能 (保持不變)
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
# 🖥️ 介面顯示
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
            if not df.empty:
                st.data_editor(df, use_container_width=True)
            else:
                st.error("無資料")

else:
    st.title("🪪 智慧證件辨識 (支援 中/台/美/日)")
    
    # 顯示目前支援的國家 (動態讀取設定檔)
    supported = ", ".join([d['label'] for d in DOCUMENT_CONFIG])
    st.caption(f"目前支援：{supported}")
    
    uploaded_file = st.file_uploader("上傳證件", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="已上傳", width=400)
        
        if st.button("🚀 開始辨識"):
            with st.spinner('AI 正在分析特徵與國別...'):
                # 1. 執行 OCR
                full_text, lines = run_ocr(image)
                clean_text = re.sub(r'[\s\.\-\_]+', '', full_text).upper().replace("O", "0").replace("I", "1").replace("L", "1")
                
                # 2. 自動判斷類型
                doc_conf = detect_document_type(clean_text)
                
                if not doc_conf:
                    st.error("⚠️ 無法識別證件類型，請確認照片是否清晰。")
                    with st.expander("查看原始文字"):
                        st.text(full_text)
                else:
                    st.success(f"✅ 識別成功：{doc_conf['label']}")
                    
                    # 3. 呼叫對應的解析器
                    parser_name = doc_conf['parser']
                    parser_func = PARSERS[parser_name]
                    
                    # 針對台灣身分證正面，需要傳入原圖做濾鏡
                    if parser_name == "twn_id":
                        data = parser_func(clean_text, lines, image)
                    else:
                        data = parser_func(clean_text, lines)
                    
                    # 4. 顯示結果表單
                    st.subheader("📝 辨識結果")
                    with st.form("res"):
                        c1, c2 = st.columns(2)
                        
                        # 通用欄位
                        if "name" in data: c1.text_input("姓名", data['name'])
                        if "eng_name" in data: c1.text_input("英文姓名", data['eng_name'])
                        
                        # 證件號碼
                        if "id_no" in data: c2.text_input("身分證/公民號", data['id_no'])
                        if "passport_no" in data: c2.text_input("護照號碼", data['passport_no'])
                        if "card_no" in data: c2.text_input("健保卡號", data['card_no'])
                        
                        # 其他資料
                        if "dob" in data: st.text_input("出生日期", data['dob'])
                        if "address" in data: st.text_input("住址", data['address'])
                        
                        if "father" in data: 
                            c1.text_input("父親", data['father'])
                            c2.text_input("母親", data['mother'])
                        if "spouse" in data: st.text_input("配偶", data['spouse'])
                        
                        st.form_submit_button("💾 存檔")