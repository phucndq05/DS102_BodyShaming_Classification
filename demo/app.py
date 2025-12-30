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

# --- 1. CẤU HÌNH HỆ THỐNG ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

st.set_page_config(page_title="Body Shaming Detection", page_icon="🛡️", layout="centered")

# --- 2. HÀM XỬ LÝ TEXT (Giữ nguyên logic chuẩn) ---
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
    
    # Tách từ
    text = ViTokenizer.tokenize(text.strip())
    return text

# --- 3. LOAD MODEL ---
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
                st.error(f"❌ Lỗi file {model_name}: {e}")
        else:
            st.error(f"❌ Thiếu file: {path}")

    elif model_name == "PhoBERT":
        if os.path.exists(path):
            try:
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=3)
                model.to("cpu")
                model.eval()
            except Exception as e:
                st.error(f"❌ Lỗi load PhoBERT: {e}")
        else:
            st.error(f"❌ Không tìm thấy thư mục: {path}")
            
    return model, tokenizer

# --- 4. HÀM DỰ ĐOÁN ---
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

# --- 5. GIAO DIỆN CHÍNH (Clean Version) ---
def main():
    # Sidebar
    st.sidebar.header("⚙️ Cấu hình Model")
    model_option = st.sidebar.selectbox("Chọn Thuật toán:", list(MODEL_CONFIG.keys()))
    
    with st.spinner(f"Đang khởi động {model_option}..."):
        model, tokenizer = load_model(model_option)
    if model: st.sidebar.success(f"✅ Đã load {model_option}")

    # Main UI
    st.title("🛡️ Demo Body Shaming Detection")
    st.markdown("---")

    # Chỉ còn ô nhập liệu đơn giản
    text_input = st.text_area(
        "📝 Nhập bình luận cần kiểm tra:", 
        height=100, 
        placeholder="Ví dụ: Bạn này nhìn đẹp quá"
    )

    # Button Phân tích
    if st.button("🔍 Phân tích ngay", type="primary"):
        if not text_input.strip():
            st.warning("Vui lòng nhập nội dung!")
        else:
            with st.spinner("AI đang phân tích..."):
                pred_label, conf = predict(model, tokenizer, text_input, model_option)
                
                result_map = {
                    0: ("KHÔNG XÚC PHẠM", "success", "✅"),
                    1: ("MỈA MAI", "warning", "⚠️"),
                    2: ("XÚC PHẠM", "error", "🚫")
                }
                txt, color, icon = result_map.get(pred_label)
                
                st.markdown(f"### Kết quả:")
                if color == "success": st.success(f"{icon} {txt}")
                elif color == "warning": st.warning(f"{icon} {txt}")
                else: st.error(f"{icon} {txt}")
                
                st.progress(float(conf), text=f"Độ tin cậy: {conf*100:.2f}%")

if __name__ == "__main__":
    main()