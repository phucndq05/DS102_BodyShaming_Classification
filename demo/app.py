import streamlit as st
import joblib
import os
import sys
import numpy as np
import torch
import re
import emoji
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Body Shaming Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS TÙY CHỈNH (XANH LÁ & CHUYÊN NGHIỆP) ---
st.markdown("""
<style>
    /* Tiêu đề chính */
    h1 {
        color: #2E7D32 !important;
        text-align: center;
        font-family: 'Helvetica', sans-serif;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 0px;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    /* Ô nhập liệu */
    .stTextArea textarea {
        font-size: 16px !important;
        border: 2px solid #A5D6A7 !important;
        border-radius: 10px !important;
        background-color: #FAFAFA;
    }
    .stTextArea textarea:focus {
        border-color: #2E7D32 !important;
        box-shadow: 0 0 5px rgba(46, 125, 50, 0.2);
    }
    
    /* Nút bấm */
    .stButton button {
        width: 100%;
        border-radius: 25px !important;
        height: 50px;
        font-weight: bold;
        font-size: 16px !important;
        text-transform: uppercase;
        border: none;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F1F8E9;
    }
    
    /* Label Input */
    .stTextArea label {
        color: #2E7D32 !important;
        font-weight: bold;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. HỆ THỐNG & MODEL ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def local_clean_text(text, mode='statistical'):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = emoji.demojize(text, delimiters=(' ', ' '))
    text = text.replace(':', '').replace('_', ' ')
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@[a-zA-Z0-9_.]+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r'(\d+)\s*kg\b', r'\1 kilogram', text)
    text = re.sub(r'\bkg\b', 'không', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\.{3,}', ' ... ', text)
    text = re.sub(r'[,\-*~()"]', ' ', text)
    text = re.sub(r'(?<!\.)\.(?!\.)', ' ', text)
    text = re.sub(r'([!?]+)', r' \1 ', text)
    text = ViTokenizer.tokenize(text.strip())
    return text

ARTIFACTS_DIR = os.path.join(current_dir, "artifacts")
MODEL_CONFIG = {
    "SVM": "svm.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Logistic Regression": "logreg.pkl",
    "PhoBERT": "phobert_final"
}

@st.cache_resource
def load_model(model_name):
    model, tokenizer = None, None
    path = os.path.join(ARTIFACTS_DIR, MODEL_CONFIG[model_name])

    if model_name in ["SVM", "Naive Bayes", "Logistic Regression"]:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
            except Exception as e:
                st.error(f"Lỗi file: {e}")
        else:
            st.error(f"Thiếu file: {path}")
    elif model_name == "PhoBERT":
        if os.path.exists(path):
            try:
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=3)
                model.to("cpu")
                model.eval()
            except Exception as e:
                st.error(f"Lỗi PhoBERT: {e}")
        else:
            st.error(f"Thiếu folder: {path}")
    return model, tokenizer

def predict(model_obj, tokenizer_obj, text, model_name):
    mode = 'deep_learning' if model_name == 'PhoBERT' else 'statistical'
    clean_txt = local_clean_text(text, mode=mode)
    label, confidence = 0, 0.0
    
    if model_name == "PhoBERT":
        if model_obj is None: return 0, 0.0
        inputs = tokenizer_obj(clean_txt, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model_obj(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs_np = probs.numpy()[0]
        label = np.argmax(probs_np)
        confidence = probs_np[label]
    else:
        if model_obj is None: return 0, 0.0
        try:
            proba = model_obj.predict_proba([clean_txt])[0]
            label = np.argmax(proba)
            confidence = proba[label]
        except:
            label = model_obj.predict([clean_txt])[0]
            confidence = 1.0
    return label, confidence

# --- 4. GIAO DIỆN CHÍNH ---
def main():
    # Sidebar
    st.sidebar.markdown("### ⚙️ Cấu hình Model")
    model_option = st.sidebar.selectbox("Chọn Thuật toán:", list(MODEL_CONFIG.keys()))
    
    with st.spinner("Loading..."):
        model, tokenizer = load_model(model_option)
    
    info_texts = {
        "SVM": "SVM (Support Vector Machine): Tìm siêu phẳng tối ưu. Ổn định với dataset nhỏ.",
        "Naive Bayes": "Naive Bayes: Tốc độ xử lý cực nhanh, baseline hiệu quả.",
        "Logistic Regression": "Logistic Regression: Mô hình tuyến tính đơn giản, dễ giải thích.",
        "PhoBERT": "PhoBERT: Transformer SOTA cho tiếng Việt, hiểu ngữ cảnh sâu sắc."
    }
    
    if model:
        st.sidebar.success(f"📌 **{model_option}**: {info_texts.get(model_option, '')}", icon=None)

    # Main Content
    st.markdown("<h1>Body Shaming Detector</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Hệ thống lọc bình luận văn minh - Vì một môi trường mạng tích cực</div>", unsafe_allow_html=True)
    
    st.write("")

    # placeholder ví dụ 
    text_input = st.text_area(
        "🌱 Nhập bình luận cần kiểm tra:", 
        height=100,
        placeholder="Ví dụ: Bạn này nhìn xinh quá" 
    )

    col_btn, col_empty = st.columns([1, 3])
    with col_btn:
        analyze_btn = st.button("KIỂM TRA NGAY", type="primary")

    if analyze_btn:
        if not text_input.strip():
            st.warning("Vui lòng nhập nội dung!")
        else:
            with st.spinner("Đang phân tích..."):
                pred_label, conf = predict(model, tokenizer, text_input, model_option)
                
                result_map = {
                    0: ("KHÔNG XÚC PHẠM", "success", "✅"),
                    1: ("MỈA MAI / ẨN Ý", "warning", "⚠️"),
                    2: ("XÚC PHẠM", "error", "🚫")
                }
                
                res_text, res_type, res_icon = result_map.get(pred_label)
                
                st.markdown("### 📊 Kết quả phân tích:")
                msg = f"{res_icon} {res_text}"
                
                if res_type == "success":
                    st.success(msg)
                elif res_type == "warning":
                    st.warning(msg)
                else:
                    st.error(msg)
                
                st.caption(f"Độ tin cậy: {conf*100:.2f}%")
                st.progress(float(conf))

if __name__ == "__main__":
    main()