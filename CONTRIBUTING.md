# QUY ĐỊNH ĐÓNG GÓP MÃ NGUỒN (CONTRIBUTION GUIDELINES)

Tài liệu này quy định các nguyên tắc làm việc chung cho nhóm nghiên cứu nhằm đảm bảo tính đồng bộ của dự án.

## 1. Quy định về Cấu trúc Lưu trữ
Các thành viên vui lòng tuân thủ vị trí lưu trữ tệp tin như sau:

* **data/raw/**: Chỉ chứa file CSV gốc tải về. [LƯU Ý]: Không chỉnh sửa trực tiếp file tại đây.
* **data/processed/**: Nơi lưu file dữ liệu sạch (train.csv, test.csv) đầu ra từ quá trình xử lý.
* **src/preprocessing.py**: Chứa lớp \`DataPreprocessor\`. Mã nguồn làm sạch và tách từ đặt tại đây.
* **src/train.py**: Mã nguồn thực thi quy trình huấn luyện mô hình.
* **demo/app.py**: Mã nguồn giao diện Web.
* **notebooks/**: Nơi chạy thử nghiệm. Đặt tên file theo định dạng số: \`1.0_eda.ipynb\`, \`2.0_test_model.ipynb\`.

## 2. Quy ước Đặt tên (Naming Conventions)
Thống nhất sử dụng tiếng Anh cho toàn bộ định danh trong mã nguồn:

* **Biến (Variables):** Sử dụng \`snake_case\` (chữ thường, gạch dưới).
    * [ĐÚNG]: \`user_input\`, \`clean_data\`
    * [SAI]: \`userInput\`, \`CleanData\`
* **Hàm (Functions):** Sử dụng \`snake_case\`. Tên hàm phải bắt đầu bằng động từ.
    * [ĐÚNG]: \`def clean_text(text):\`, \`def load_model():\`
    * [SAI]: \`def CleanText(text):\`, \`def text_cleaning(text):\`
* **Lớp (Classes):** Sử dụng \`PascalCase\` (Viết hoa chữ cái đầu mỗi từ).
    * [ĐÚNG]: \`class DataPreprocessor:\`
* **Hằng số (Constants):** Sử dụng \`UPPER_CASE\`.
    * [ĐÚNG]: \`MAX_LENGTH = 256\`

## 3. Quy chuẩn Tài liệu hóa (Documentation)
Mỗi hàm chức năng quan trọng bắt buộc phải có Docstring mô tả 3 nội dung: Nhiệm vụ, Đầu vào (Input), Đầu ra (Output).

\`\`\`python
def clean_text(text):
    """
    Nhiệm vụ: Loại bỏ các thẻ HTML và đường dẫn URL trong văn bản.
    Input: str (văn bản thô)
    Output: str (văn bản đã làm sạch)
    """
    # Mã xử lý...
\`\`\`

## 4. Quy trình Kiểm soát Phiên bản (Git Workflow)
1. Không cam kết (commit) trực tiếp lên nhánh \`main\`.
2. Khởi tạo nhánh tính năng riêng: \`git checkout -b feature/ten-tinh-nang\`.
3. Ghi chú commit rõ ràng, nghiêm túc.