# DS102 - Phân loại bình luận body shaming (Body Shaming Detection)

## 1. Tổng quan dự án
Đây là đồ án môn học DS102 - Học máy Thống kê.
Mục tiêu nghiên cứu: Xây dựng hệ thống tự động phân loại bình luận trên mạng xã hội tiếng Việt thành 3 nhãn:
- **Nhãn 0**: Không xúc phạm (Bình thường/An toàn)
- **Nhãn 1**: Mỉa mai / Ẩn ý (Body Shaming gián tiếp)
- **Nhãn 2**: Xúc phạm (Body Shaming trực diện)

Hệ thống sử dụng và so sánh hiệu quả của **4 thuật toán** huấn luyện:
1. Support Vector Machine (SVM)
2. Naive Bayes
3. Logistic Regression
4. PhoBERT (Fine-tuned Transformer)

## 2. Cấu trúc tổ chức thư mục
Dự án được tổ chức theo tiêu chuẩn Khoa học Dữ liệu (Data Science) nhằm đảm bảo tính tái lập:

- **data/**: Kho lưu trữ dữ liệu.
  - `raw`: Dữ liệu thô gốc.
  - `processed`: Dữ liệu đã làm sạch.
  - `dictionaries`: Từ điển Teencode/Stopwords.
- **src/**: Mã nguồn chính (Source Code). Chứa các lớp xử lý dữ liệu và huấn luyện mô hình.
- **notebooks/**: Các tệp Jupyter Notebook dùng cho phân tích khám phá (EDA) và thử nghiệm.
- **demo/**: Mã nguồn ứng dụng Web Demo (sử dụng thư viện Streamlit).
- **docs/**: Tài liệu báo cáo đồ án và các tài liệu tham khảo liên quan.

## 3. Hướng dẫn cài đặt và triển khai (chi tiết)

### Yêu cầu hệ thống
- Python 3.8 trở lên.
- Đã cài đặt `pip`.

### Bước 1: Thiết lập môi trường
Khuyến khích sử dụng môi trường ảo (Virtual Environment) để tránh xung đột thư viện.

**Trên macOS / Linux:**
```bash
# 1. Tạo môi trường ảo (tên là venv)
python3 -m venv venv

# 2. Kích hoạt môi trường (BẮT BUỘC mỗi lần chạy lại)
source venv/bin/activate
```

**Trên Windows:**
```bash
# 1. Tạo môi trường ảo
python -m venv venv

# 2. Kích hoạt môi trường
.\venv\Scripts\activate
```

### Bước 2: Cài đặt thư viện
Sau khi kích hoạt `venv` (thấy chữ `(venv)` ở đầu dòng lệnh), chạy lệnh sau:
```bash
pip install -r requirements.txt
```

### Bước 3: Cấu hình Model PhoBERT (Quan trọng)
Do giới hạn dung lượng GitHub, file model PhoBERT (nặng >500MB) không được lưu trong repo này.
- **Để chạy tính năng PhoBERT:** Bạn cần tải file `model.safetensors` từ [Link Google Drive của nhóm] và đặt vào thư mục:
  `demo/artifacts/phobert_final/model.safetensors`
- **Nếu không có file này:** App vẫn chạy bình thường với các model SVM, Naive Bayes, nhưng tính năng PhoBERT sẽ chạy ở chế độ giả lập (Demo UI).

### Bước 4: Khởi chạy ứng dụng
```bash
streamlit run demo/app.py
```
Truy cập đường dẫn hiển thị trên terminal (thường là `http://localhost:8501`) để trải nghiệm.