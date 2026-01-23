import streamlit as st
import pytesseract
from PIL import Image, ImageGrab
import pandas as pd
import re
import os
import shutil

# ==========================================
# 🔧 關鍵修正：跨平台 Tesseract 路徑設定
# ==========================================
if os.name == 'nt':
    # 情況 1：在您的 Windows 電腦上執行
    # 請確保路徑正確指向您的安裝位置
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    # 情況 2：在 Streamlit Cloud (Linux) 上執行
    # 使用 shutil.which 自動尋找系統安裝的 tesseract 指令位置
    tesseract_cmd = shutil.which("tesseract")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    else:
        # 如果找不到，通常是因為 packages.txt 沒設定好
        st.error("⚠️ 錯誤：在系統中找不到 Tesseract。若您正在雲端部署，請確認 `packages.txt` 已包含 `tesseract-ocr`。")

# ==========================================
# 頁面設定
# ==========================================
st.set_page_config(page_title="悠遊卡報表 (雲端通用版)", layout="wide", page_icon="☁️")

st.title("☁️ 悠遊卡報表產生器 (雲端通用版)")
st.markdown("""
本工具支援 **Windows 本機** 與 **Streamlit Cloud 雲端** 執行。
- **雲端模式**：請使用「上傳圖片」功能。
- **本機模式**：可使用「貼上剪貼簿」功能。
""")

# 初始化 Session
if 'ocr_df' not in st.session_state:
    st.session_state['ocr_df'] = None
if 'current_image' not in st.session_state:
    st.session_state['current_image'] = None

# =======================
# 1. 圖片來源區 (新增上傳功能以支援雲端)
# =======================
col1, col2, col3 = st.columns([2, 1, 3])

with col1:
    # 雲端版最穩定的輸入方式
    uploaded_file = st.file_uploader("📂 上傳截圖檔案 (雲端推薦)", type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        st.session_state['current_image'] = Image.open(uploaded_file)
        # 重置資料以觸發重新辨識
        if st.session_state.get('last_uploaded') != uploaded_file.name:
            st.session_state['ocr_df'] = None
            st.session_state['last_uploaded'] = uploaded_file.name

with col2:
    # 本機版方便的功能 (雲端可能因瀏覽器權限失效)
    if st.button("📋 貼上剪貼簿 (限本機)", type="secondary"):
        try:
            image = ImageGrab.grabclipboard()
            if isinstance(image, Image.Image):
                st.session_state['current_image'] = image
                st.session_state['ocr_df'] = None 
                st.toast("圖片已從剪貼簿載入！")
            else:
                st.warning("剪貼簿為空或非圖片格式。")
        except Exception as e:
            st.error(f"讀取剪貼簿失敗 (雲端環境請改用上傳)：{e}")

with col3:
    if st.session_state['current_image'] and st.button("🗑️ 清除重來"):
        st.session_state['current_image'] = None
        st.session_state['ocr_df'] = None
        st.rerun()

# =======================
# 2. 辨識邏輯
# =======================
def parse_ocr_text(text):
    data = []
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Regex: 精準抓取 完整日期時間 / 地點 / 金額
        match = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2})\s+(\d{2}:\d{2}:\d{2}).*?([\u4e00-\u9fa5].*?)(?=\s+\d+)\s+(\d+)', line)
        
        if match:
            full_date = match.group(1).replace("/", "-") 
            time_part = match.group(2)
            loc_raw = match.group(3).replace("扣款", "").replace("交易", "").strip() 
            amount = match.group(4)
            
            if "加值" in loc_raw: continue

            # 資料拆解
            short_date = full_date[5:].replace("-", "/") 
            transport_type = "捷運"
            simple_loc = loc_raw 
            
            if "台鐵" in loc_raw: 
                transport_type = "台鐵"
                simple_loc = loc_raw.replace("台鐵", "").replace("車站", "")
            elif "捷運" in loc_raw:
                transport_type = "捷運"
                simple_loc = loc_raw.replace("台北捷運", "").replace("高雄捷運", "")
            elif "客運" in loc_raw:
                transport_type = "客運"
            elif "高鐵" in loc_raw:
                transport_type = "高鐵"
            
            data.append({
                "選取": True,
                "完整日期": f"{full_date} {time_part}",
                "短日期": short_date,
                "交通": transport_type,
                "訖點": simple_loc,
                "金額": amount,
                "地點原始": loc_raw 
            })
    return data

# =======================
# 3. 執行辨識與介面
# =======================
if st.session_state['current_image']:
    st.image(st.session_state['current_image'], caption='預覽截圖', width=600)

    if st.session_state['ocr_df'] is None:
        with st.spinner('正在雲端進行 OCR 辨識...'):
            try:
                # 執行 OCR
                text = pytesseract.image_to_string(st.session_state['current_image'], lang='chi_tra+eng', config='--psm 4')
                parsed_data = parse_ocr_text(text)
                
                if parsed_data:
                    st.session_state['ocr_df'] = pd.DataFrame(parsed_data)
                else:
                    st.error("無法辨識有效資料。請確認：\n1. 圖片清晰度\n2. 若在雲端，packages.txt 是否已安裝中文包 (tesseract-ocr-chi-tra)")
            except Exception as e:
                st.error(f"OCR 執行錯誤：{e}")

    if st.session_state['ocr_df'] is not None:
        st.info("👇 預覽辨識結果 (修改請在產生後的 HTML 報表中進行)：")
        
        edited_df = st.data_editor(
            st.session_state['ocr_df'],
            column_config={
                "選取": st.column_config.CheckboxColumn("列入", width="small"),
                "完整日期": st.column_config.TextColumn(width="medium", disabled=True),
                "交通": st.column_config.SelectboxColumn("交通預判", options=["捷運", "台鐵", "高鐵", "公車"]),
                "訖點": st.column_config.TextColumn("訖點"),
                "金額": st.column_config.TextColumn(width="small"),
            },
            hide_index=True,
            use_container_width=True
        )

        if st.button("🚀 產生 HTML 報表", type="primary"):
            final_data = edited_df[edited_df["選取"] == True]
            
            if final_data.empty:
                st.warning("請至少勾選一筆！")
            else:
                # 準備下拉選單資料
                all_locs = set(final_data["訖點"].tolist())
                all_locs.update(["台北車站", "板橋", "南港", "桃園機場", "公司", "住家", "左營", "森福德"])
                datalist_options = "".join([f'<option value="{loc}"></option>' for loc in all_locs])

                rows_html = ""
                for index, row in final_data.iterrows():
                    rows_html += f"""
                    <tr>
                        <td style="width: 180px;">{row['完整日期']}</td>
                        <td style="width: 60px;">扣款</td>
                        <td style="text-align: left; padding-left: 15px;">{row['地點原始']}</td>
                        <td style="width: 60px;">{row['金額']}</td>
                        
                        <td class="black-cell">
                            <div class="black-container">
                                <input type="text" class="blk-input short-date" value="{row['短日期']}">
                                <select class="blk-select">
                                    <option value="捷運" {'selected' if row['交通']=='捷運' else ''}>捷運</option>
                                    <option value="台鐵" {'selected' if row['交通']=='台鐵' else ''}>台鐵</option>
                                    <option value="高鐵" {'selected' if row['交通']=='高鐵' else ''}>高鐵</option>
                                    <option value="公車" {'selected' if row['交通']=='客運' else ''}>公車</option>
                                    <option value="計程車" {'selected' if row['交通']=='計程車' else ''}>計程車</option>
                                </select>
                                <input type="text" list="locList" class="blk-input loc-input" placeholder="[起點]">
                                <span style="margin: 0 2px;">到</span>
                                <input type="text" list="locList" class="blk-input loc-input" value="{row['訖點']}">
                            </div>
                        </td>
                    </tr>
                    """

                full_html = f"""
                <!DOCTYPE html>
                <html lang="zh-TW">
                <head>
                    <meta charset="UTF-8">
                    <title>差旅報表</title>
                    <style>
                        body {{ font-family: "Microsoft JhengHei", Arial, sans-serif; margin: 20px; -webkit-print-color-adjust: exact; print-color-adjust: exact; background-color: #f4f4f4; }}
                        table {{ width: 100%; max-width: 1000px; border-collapse: collapse; background-color: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                        td {{ border: 1px solid #e0e0e0; padding: 8px; text-align: center; vertical-align: middle; color: #333; font-size: 15px; }}
                        tr:nth-child(even) td:not(.black-cell) {{ background-color: #fcfcfc; }}
                        .black-cell {{ padding: 0 !important; border: none !important; width: 420px; background-color: black !important; }}
                        .black-container {{ display: flex; align-items: center; justify-content: flex-start; padding: 12px 10px; background-color: black; color: white; height: 100%; font-weight: bold; font-size: 18px; }}
                        .blk-input, .blk-select {{ background-color: black; color: white; border: none; outline: none; font-family: "Microsoft JhengHei", sans-serif; font-size: 18px; font-weight: bold; text-align: center; }}
                        .short-date {{ width: 60px; }}
                        .blk-select {{ width: 70px; cursor: pointer; }} 
                        .loc-input {{ width: 110px; text-align: left; border-bottom: 1px dashed #555; }}
                        .loc-input:focus {{ border-bottom: 1px solid white; }}
                        @media print {{
                            .no-print {{ display: none !important; }}
                            body {{ margin: 0; background-color: #fff; }}
                            table {{ box-shadow: none; max-width: none; }}
                            .blk-select {{ appearance: none; -webkit-appearance: none; padding-right: 0; }}
                            .loc-input {{ border-bottom: none; }}
                        }}
                    </style>
                </head>
                <body>
                    <div class="no-print" style="background:#e9ecef; padding:15px; margin-bottom:20px; border-radius:5px; max-width:1000px;">
                        <h3 style="margin-top:0;">報表預覽</h3>
                        <p>請點擊黑色區塊進行編輯，完成後點擊按鈕列印。</p>
                        <button onclick="window.print()" style="background:#0056b3; color:white; border:none; padding:10px 20px; cursor:pointer; font-size:16px; border-radius:4px;">🖨️ 列印報表</button>
                    </div>
                    <table><tbody>{rows_html}</tbody></table>
                    <datalist id="locList">{datalist_options}</datalist>
                </body>
                </html>
                """
                
                st.components.v1.html(full_html, height=600, scrolling=True)
                st.download_button("📥 下載 HTML", full_html, "report.html")

else:
    st.info("👆 請先上傳圖片或貼上剪貼簿")