
import streamlit as st
# import joblib
# from src.preprocessing import DataPreprocessor

def load_model(model_path):
    """
    Hàm load model từ file artifact.
    """
    pass

def predict(text):
    """
    Hàm dự đoán nhãn cho 1 câu text.
    """
    pass

def main():
    st.title("🛡️ Demo Body Shaming Detection")
    st.write("Hệ thống phân loại bình luận tiếng Việt.")
    
    # --- Sidebar: Model Selection ---
    # option = st.sidebar.selectbox("Chọn mô hình:", ["SVM", "Naive Bayes"])
    
    # --- Main Interface ---
    # text_input = st.text_area("Nhập bình luận:")
    
    # if st.button("Kiểm tra"):
    #     result = predict(text_input)
    #     st.write(f"Kết quả: {result}")

if __name__ == "__main__":
    main()
