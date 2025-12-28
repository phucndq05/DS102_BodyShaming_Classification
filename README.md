# DS102 - PHÂN LOẠI BÌNH LUẬN BODY SHAMING (BODY SHAMING DETECTION)

## 1. Tổng quan Dự án
Đây là đồ án môn học DS102 - Học máy Thống kê.
Mục tiêu nghiên cứu: Xây dựng hệ thống tự động phân loại bình luận trên mạng xã hội tiếng Việt thành 3 nhãn:
- Nhãn 0: Bình thường (Normal)
- Nhãn 1: Mỉa mai / Ẩn ý (Sarcasm)
- Nhãn 2: Xúc phạm rõ rệt (Direct Harassment)

## 2. Cấu trúc Tổ chức Thư mục
Dự án được tổ chức theo tiêu chuẩn Khoa học Dữ liệu (Data Science) nhằm đảm bảo tính tái lập:

| Thư mục | Mô tả chức năng |
| :--- | :--- |
| **data/** | Kho lưu trữ dữ liệu. <br> - `raw`: Dữ liệu thô gốc. <br> - `processed`: Dữ liệu đã làm sạch. <br> - `dictionaries`: Từ điển Teencode/Stopwords. |
| **src/** | Mã nguồn chính (Source Code). Chứa các lớp xử lý dữ liệu và huấn luyện mô hình. |
| **notebooks/** | Các tệp Jupyter Notebook dùng cho phân tích khám phá (EDA) và thử nghiệm. |
| **demo/** | Mã nguồn ứng dụng Web Demo (sử dụng thư viện Streamlit). |
| **docs/** | Tài liệu báo cáo đồ án và các tài liệu tham khảo liên quan. |

## 3. Hướng dẫn Cài đặt và Triển khai
1. Sao chép mã nguồn về máy (Clone repository).
2. Cài đặt các thư viện phụ thuộc:
   \`pip install -r requirements.txt\`
3. Khởi chạy ứng dụng Demo:
   \`streamlit run demo/app.py\`