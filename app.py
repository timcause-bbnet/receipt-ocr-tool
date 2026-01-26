import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image
import pandas as pd
import numpy as np
import re
import os
import cv2

st.set_page_config(page_title="PaddleOCR 強力辨識版", layout="wide", page_icon="🚀")

# ==========================================
# 🔧 初始化 PaddleOCR (加上快取，避免重複載入)
# ==========================================
@st.cache_resource
def load_ocr_model():
    # lang='ch' 代表支援中英文混合
    # use_angle_cls=True 會自動轉正圖片
    st.toast("正在載入深度學習模型，第一次啟動需約 1-2 分鐘...", icon="⏳")
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
    return ocr

# ==========================================
# 核心邏輯：PaddleOCR 資料提取
# ==========================================
def extract_data_paddle(ocr_result, doc_type):
    # Paddle 的結果格式是巢狀 list，我們先把它攤平成單純的文字串
    # 格式: [[[[x,y],..], ("文字", 信心度)], ...]
    all_text = ""
    lines = []
    
    # 信心度閥值 (過濾掉太不像字的雜訊)
    CONFIDENCE_THRESHOLD = 0.7 

    if ocr_result and ocr_result[0]:
        for line in ocr_result[0]:
            text = line[1][0]
            score = line[1][1]
            if score > CONFIDENCE_THRESHOLD:
                all_text += text + "\n"
                lines.append(text)
    
    # 移除空白與符號方便 Regex
    clean_text_nospace = re.sub(r'[\s\.\-\_]+', '', all_text).upper().replace("O", "0").replace("I", "1").replace("L", "1")
    raw_text_oneline = all_text.replace("\n", " ")
    
    data = {}

    # === 身分證 ===
    if doc_type == "id_card":
        # 1. 判斷正反面 (Paddle 讀這類關鍵字非常準)
        if any(x in clean_text_nospace for x in ["身分證", "出生", "性別", "統一編號"]):
            specific_type = "id_card_front"
        elif any(x in clean_text_nospace for x in ["配偶", "役別", "父母", "鄉鎮市區", "住址"]):
            specific_type = "id_card_back"
        elif re.search(r'[A-Z][12]\d{8}', clean_text_nospace):
             specific_type = "id_card_front"
        else:
             specific_type = "unknown"

        if specific_type == "id_card_front":
            # 姓名：Paddle 通常會把 "姓名" 和 "陳筱玲" 分成不同行，或者在同一行
            # 我們直接找 "姓名" 關鍵字附近的字
            data['name'] = ""
            for i, line in enumerate(lines):
                if "姓名" in line:
                    # 如果同一行有字 (ex: 姓名陳筱玲)
                    clean_line = line.replace("姓名", "").replace(" ", "")
                    if len(clean_line) > 1:
                        data['name'] = clean_line
                    # 如果在下一行 (ex: 姓名 \n 陳筱玲)
                    elif i + 1 < len(lines):
                        data['name'] = lines[i+1].replace(" ", "")
                    break
            # 樣本過濾
            if data.get('name'):
                data['name'] = data['name'].replace("樣本", "").replace("樣", "").replace("本", "")

            # 身分證字號
            id_match = re.search(r'[A-Z][12]\d{8}', clean_text_nospace)
            data['id_no'] = id_match.group(0) if id_match else ""

            # 生日
            # Paddle 讀數字很準，直接抓民國xx年
            dob_match = re.search(r'民國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', raw_text_oneline)
            if dob_match:
                data['dob'] = dob_match.group(0)
            else:
                data['dob'] = ""
            
            data['type_label'] = "身分證 (正面)"

        elif specific_type == "id_card_back":
            # 住址
            # Paddle 常常把地址拆成兩行，我們試著合併包含"市/縣/區/路"的行
            addr_str = ""
            for line in lines:
                if any(k in line for k in ["縣", "市", "區", "路", "街", "里", "鄰"]):
                    if "住址" not in line and "出生地" not in line:
                        addr_str += line
                elif "住址" in line:
                     addr_str += line.replace("住址", "")
            data['address'] = addr_str

            # 配偶
            for line in lines:
                if "配偶" in line:
                    data['spouse'] = line.replace("配偶", "").strip()
            
            # 父母
            # 父母通常在同一行或分兩行
            parents_line = ""
            for line in lines:
                if "父" in line or "母" in line:
                    parents_line += line
            
            f_match = re.search(r'父\s*([\u4e00-\u9fa5]+)', parents_line)
            m_match = re.search(r'母\s*([\u4e00-\u9fa5]+)', parents_line)
            data['father'] = f_match.group(1) if f_match else ""
            data['mother'] = m_match.group(1) if m_match else ""
            
            data['type_label'] = "身分證 (背面)"
        else:
             return {}, "unknown"

    # === 護照 ===
    elif doc_type == "passport":
        data['type_label'] = "護照"
        
        # 英文姓名 (LIN, MEI-HUA)
        # PaddleOCR 對英文大寫辨識能力很強
        for line in lines:
            # 尋找全大寫且有逗號的行
            if "," in line and re.search(r'[A-Z]', line):
                # 排除雜訊
                if "MINISTRY" not in line and "REPUBLIC" not in line:
                    data['eng_name'] = line
                    break
        
        # 護照號碼
        pass_match = re.search(r'[0-9]{9}', clean_text_nospace)
        data['passport_no'] = pass_match.group(0) if pass_match else ""
        
        # 身分證字號
        id_match = re.search(r'[A-Z][12]\d{8}', clean_text_nospace)
        data['id_no'] = id_match.group(0) if id_match else ""

    # === 健保卡 ===
    elif doc_type == "health_card":
        data['type_label'] = "健保卡"
        
        # 姓名
        for line in lines:
            if "姓名" in line:
                data['name'] = line.replace("姓名", "").strip()
            elif len(line) <= 4 and re.match(r'^[\u4e00-\u9fa5]+$', line):
                 # 可能是名字單獨一行
                 if "全民" not in line and "保險" not in line:
                     if 'name' not in data: data['name'] = line

        # 健保卡號
        card_match = re.search(r'\d{4}\s*\d{4}\s*\d{4}', raw_text_oneline)
        data['card_no'] = card_match.group(0) if card_match else ""
        
        # 身分證
        id_match = re.search(r'[A-Z][12]\d{8}', clean_text_nospace)
        data['id_no'] = id_match.group(0) if id_match else ""

    return data, "success"

# ==========================================
# 介面顯示
# ==========================================
st.sidebar.title("🚀 PaddleOCR (深度學習版)")
app_mode = st.sidebar.radio("請選擇功能：", ["🪪 身分證辨識", "🏥 健保卡辨識", "✈️ 護照辨識"])

doc_map = {"🪪 身分證辨識": "id_card", "🏥 健保卡辨識": "health_card", "✈️ 護照辨識": "passport"}
target_type = doc_map[app_mode]

st.title(app_mode + " (AI 強力版)")
st.info("💡 使用 PaddleOCR 引擎。無需任何影像處理，直接上傳原圖即可。樣本圖、浮水印、斜拍皆可辨識。")

uploaded_file = st.file_uploader("請上傳照片", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    # 轉換圖片格式 (Streamlit Upload -> OpenCV 格式)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_cv = cv2.imdecode(file_bytes, 1) # BGR format
    
    # 顯示圖片 (轉回 RGB 給 st.image 顯示)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption='原始照片 (無需濾鏡)', width=500)

    if st.button("🚀 啟動 AI 辨識"):
        try:
            with st.spinner('正在呼叫 PaddleOCR 深度學習模型 (若為第一次啟動請稍候 1-2 分鐘)...'):
                # 載入模型
                ocr = load_ocr_model()
                
                # 執行辨識
                result = ocr.ocr(image_cv, cls=True)
                
                # 提取資料
                data, status = extract_data_paddle(result, target_type)
                
                if status == "unknown":
                    st.warning("⚠️ 能夠讀取文字，但無法判斷是正面還是背面，或特徵不足。")
                    # 顯示原始文字供除錯
                    st.write("讀到的所有文字：", [line[1][0] for line in result[0]])
                else:
                    st.success(f"✅ 成功識別！ ({data.get('type_label')})")
                    
                    st.subheader("📝 辨識結果")
                    with st.form("result_form"):
                        c1, c2 = st.columns(2)
                        
                        if data.get('type_label') == "身分證 (正面)":
                            c1.text_input("姓名", value=data.get('name', ''))
                            c2.text_input("身分證字號", value=data.get('id_no', ''))
                            st.text_input("出生年月日", value=data.get('dob', ''))

                        elif data.get('type_label') == "身分證 (背面)":
                            st.text_input("住址", value=data.get('address', ''))
                            c1.text_input("父親", value=data.get('father', ''))
                            c2.text_input("母親", value=data.get('mother', ''))
                            st.text_input("配偶", value=data.get('spouse', ''))
                            
                        elif data.get('type_label') == "健保卡":
                            c1.text_input("姓名", value=data.get('name', ''))
                            c2.text_input("身分證字號", value=data.get('id_no', ''))
                            st.text_input("健保卡號", value=data.get('card_no', ''))
                            
                        elif data.get('type_label') == "護照":
                            c1.text_input("英文姓名", value=data.get('eng_name', ''))
                            c2.text_input("護照號碼", value=data.get('passport_no', ''))
                            st.text_input("身分證字號", value=data.get('id_no', ''))

                        st.form_submit_button("💾 確認存檔")
                        
                    # 除錯用：顯示 AI 看到的文字位置
                    with st.expander("👁️ AI 看到的文字與信心度"):
                        for line in result[0]:
                            st.write(f"文字: **{line[1][0]}** (信心度: {line[1][1]:.2f})")
                            
        except Exception as e:
            st.error(f"發生錯誤：{e}")
            st.info("若出現 Memory 錯誤，代表 Streamlit 免費版記憶體不足，請重新整理頁面再試一次。")