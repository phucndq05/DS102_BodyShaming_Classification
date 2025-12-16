# [GHI CHÚ DÀNH CHO NHÓM PHÁT TRIỂN]
# -----------------------------------
# Mã nguồn ứng dụng Demo sử dụng thư viện Streamlit.
# Giao diện và chức năng có thể được mở rộng tùy theo yêu cầu báo cáo.

import streamlit as st

def main():
    st.set_page_config(page_title="DS102 Demo", page_icon="🛡️")
    
    st.title("🛡️ Demo Phân loại Body Shaming")
    st.markdown("Vui lòng nhập nội dung bình luận cần kiểm tra vào ô bên dưới:")
    
    user_input = st.text_area("Nội dung bình luận:", height=100)
    
    if st.button("Phân tích", type="primary"):
        if not user_input:
            st.warning("Vui lòng nhập nội dung trước khi kiểm tra.")
        else:
            st.info("Đang xử lý dữ liệu... ([TODO]: Kết nối Mô hình)")
            # [TODO]: Gọi hàm dự đoán từ mô hình đã huấn luyện
            # result = model.predict(user_input)
            
            # Hiển thị kết quả giả lập
            st.success("Kết quả dự đoán: ...")

if __name__ == "__main__":
    main()