# [GHI CHÚ DÀNH CHO NHÓM PHÁT TRIỂN]
# -----------------------------------
# Tệp tin này cung cấp khung sườn (template) cơ bản cho lớp xử lý dữ liệu.
# Các thành viên có quyền chỉnh sửa, tối ưu hóa logic bên trong các hàm
# để phù hợp với yêu cầu thực tế của dự án.
# Khuyến nghị giữ nguyên tên Lớp và các phương thức chính (process, clean_text)
# để đảm bảo tính tương thích khi tích hợp hệ thống.

import pandas as pd
import re
import unicodedata  # [FIX]: Thêm thư viện này để xử lý Unicode NFC
from pyvi import ViTokenizer # [FIX]: Thêm thư viện này để tách từ tiếng Việt

# Optional: PhoBERT
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

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
        # 1. Từ điển Teencode (Cập nhật theo gợi ý của thành viên nhóm)
        self.teencode_dict = {
            # Buồn, khóc
            "hjx": "hic hic", "hjxhjx": "hic hic", "hix": "hic hic", "hu hu": "hic hic",
            "huhu": "hic hic", "khoc": "khóc", "khóc": "khóc", "buồn": "buồn",
            
            # Cười, vui
            "kkk": "cười", "haha": "cười", "hihi": "cười", "hehe": "cười",
            "lol": "cười", "xỉu": "cười ngất", "xjxu": "cười ngất", "dead": "cười chết",
            
            # Không, được, không được
            "ko": "không", "k": "không", "kh": "không", "khum": "không", "hong": "không",
            "dc": "được", "đc": "được", "kdc": "không được", "hok": "không",
            
            # Mình, bạn, người
            "mik": "mình", "mjk": "mình", "t": "tôi", "tui": "tôi", "tao": "tôi",
            "mày": "bạn", "may": "bạn", "cậu": "bạn",
            
            # Các từ phổ biến khác
            "ntn": "như thế nào", "ntnao": "như thế nào", "ns": "nói sao", "j": "gì",
            "z": "vậy", "zay": "vậy", "vs": "với", "ak": "à", "ik": "đi", "bjz": "bây giờ",
            "bh": "bây giờ", "r": "rồi", "okela": "ok", "sp": "sản phẩm", "shop": "cửa hàng",
            "mn": "mọi người", "ae": "anh em", "ngta": "người ta", "wa": "qua",
            "h": "giờ", "ms": "mới", "luôn": "luôn", "đmx": "đẹp lắm", "xink": "xinh",
            "cute": "dễ thương", "hen": "hẹn", "njz": "nhỉ", "tr": "trời", "troi": "trời",
            
            # Bổ sung từ Guideline v1.2 để bắt nhãn mỉa mai
            "dị": "vậy", "í": "ý", "ha": "hả", "bả": "bà ấy"
        }

        # 2. Danh sách stopwords (Để trống hoặc load file tùy sau này)
        self.stopwords = set() # Chưa EDA → để trống, sau thêm
    
        # 3. Load PhoBERT nếu dùng mode deep_learning
        if self.mode == 'deep_learning':
            try:
                from transformers import AutoTokenizer
                self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            except ImportError:
                print("Warning: Cần pip install transformers để dùng mode deep_learning")

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
        # 1. Chuyển đổi Icon mỉa mai thành token (Theo Guideline v1.2)
        # Giữ lại tính năng quan trọng cho nhãn (1) Mỉa mai
        text = text.replace(":)))", " icon_cuoi_deu ")
        text = text.replace(":))", " icon_cuoi_deu ")
        text = text.replace("=))", " icon_cuoi_lan_lon ")
        text = text.replace("^^", " icon_hi_hi ") 
        
        # 2. Xóa HTML tags
        text = re.sub(r'<[^>]*>', ' ', text)
        
        # 3. Xóa URL và Email
        text = re.sub(r'http[s]?://\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # 4. Xóa Mentions (@user)
        text = re.sub(r'@\w+', ' ', text)
        
        # 5. Xóa ký tự đặc biệt NHƯNG giữ lại dấu câu cơ bản và số (cho trường hợp "2m", "3p")
        text = re.sub(r'[^\w\s.,?!]', ' ', text)
        
        # 6. Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def normalize(self, text):
        """
        Nhiệm vụ: Chuẩn hóa văn bản (Unicode, chữ thường, chuyển đổi Teencode).
        Input: str
        Output: str
        """
        # [TODO]: Cài đặt logic chuẩn hóa
        # [QUAN TRỌNG 1]: Phải chuẩn hóa Unicode trước
        text = unicodedata.normalize('NFC', text)
        
        # [QUAN TRỌNG 2]: Phải chuyển về chữ thường thì mới khớp được với từ điển teencode
        text = text.lower()
        
        # [QUAN TRỌNG 3]: Rút gọn ký tự lặp (beoooo -> beoo)
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        
        if self.teencode_dict:
            # Sắp xếp key dài trước để tránh lỗi replace nhầm
            sorted_keys = sorted(self.teencode_dict.keys(), key=len, reverse=True)
            
            # Tạo pattern Regex
            pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted_keys) + r')\b')
            
            # Thực hiện thay thế
            text = pattern.sub(lambda x: self.teencode_dict[x.group()], text)
       
        # Clean space lần cuối
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text  # Không cần .lower() lần nữa

    def process(self, text):
        """
        Phương thức xử lý chính (Main Pipeline).
        """
        text = self.clean_text(text)
        text = self.normalize(text)
        # [TODO]: Cài đặt logic tách từ (Tokenization) tùy thuộc vào self.mode
        if self.mode == 'baseline':
            # Sử dụng PyVi cho mô hình thống kê
            text = ViTokenizer.tokenize(text)
            
        elif self.mode == 'deep_learning':
            # Quan trọng: phải word-segment trước khi PhoBERT tokenize
            segmented = ViTokenizer.tokenize(text)
            tokens = self.phobert_tokenizer.tokenize(segmented)
            text = ' '.join(tokens)
        return text