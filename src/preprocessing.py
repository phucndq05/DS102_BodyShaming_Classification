# [GHI CHÚ DÀNH CHO NHÓM PHÁT TRIỂN]
# -----------------------------------
# Tệp tin này cung cấp khung sườn (template) cơ bản cho lớp xử lý dữ liệu.
# Các thành viên có quyền chỉnh sửa, tối ưu hóa logic bên trong các hàm
# để phù hợp với yêu cầu thực tế của dự án.
# Khuyến nghị giữ nguyên tên Lớp và các phương thức chính (process, clean_text)
# để đảm bảo tính tương thích khi tích hợp hệ thống.

import pandas as pd
import re

class DataPreprocessor:
    def __init__(self, mode='baseline'):
        """
        Khởi tạo bộ xử lý dữ liệu.
        Tham số:
            mode (str): Chế độ xử lý. 
                        - 'baseline': Sử dụng thư viện PyVi (cho mô hình thống kê).
                        - 'deep_learning': Sử dụng PhoBERT Tokenizer (cho mô hình học sâu).
        """
        self.mode = mode
        # [TODO]: Tải các tài nguyên cần thiết (từ điển teencode, danh sách stopwords) tại đây
        pass

    def clean_text(self, text):
        """
        Nhiệm vụ: Làm sạch nhiễu kỹ thuật (HTML, URL, @User).
        Input: str
        Output: str
        """
        if not isinstance(text, str):
            return ""
        # [TODO]: Cài đặt logic làm sạch dữ liệu (Regex)
        return text

    def normalize(self, text):
        """
        Nhiệm vụ: Chuẩn hóa văn bản (Unicode, chữ thường, chuyển đổi Teencode).
        Input: str
        Output: str
        """
        # [TODO]: Cài đặt logic chuẩn hóa
        return text.lower()

    def process(self, text):
        """
        Phương thức xử lý chính (Main Pipeline).
        """
        text = self.clean_text(text)
        text = self.normalize(text)
        # [TODO]: Cài đặt logic tách từ (Tokenization) tùy thuộc vào self.mode
        return text