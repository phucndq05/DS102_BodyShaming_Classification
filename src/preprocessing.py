# [GHI CHÚ DÀNH CHO NHÓM PHÁT TRIỂN]
# -----------------------------------
# Tệp tin này cung cấp khung sườn (template) cơ bản cho lớp xử lý dữ liệu.
# Các thành viên có quyền chỉnh sửa, tối ưu hóa logic bên trong các hàm
# để phù hợp với yêu cầu thực tế của dự án.
# Khuyến nghị giữ nguyên tên Lớp và các phương thức chính (process, clean_text)
# để đảm bảo tính tương thích khi tích hợp hệ thống.
# Chức năng: Class xử lý dữ liệu chuẩn (Stopwords .txt + Teencode .csv)

import pandas as pd
import re
import unicodedata
import emoji  # Cần pip install emoji
from pyvi import ViTokenizer 
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

class DataPreprocessor:
    def __init__(self, mode='baseline', stopwords_path=None, teencode_path=None):
        """
        Khởi tạo bộ xử lý dữ liệu.
        Tham số:
            stopwords_path: Nếu None -> Chế độ Deep Learning (Giữ stopwords).
                        Nếu có path -> Chế độ Statistical (Xóa stopwords).
        """
        self.stopwords = set()
        self.teencode_dict = {} # Khởi tạo rỗng, sẽ load từ CSV

        # Load stopwords từ file .txt
        if stopwords_path and os.path.exists(stopwords_path):
            try:
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    # splitlines() tự động cắt dòng, strip() để xóa khoảng trắng thừa đầu đuôi
                    self.stopwords = set(line.strip() for line in f if line.strip())
                print(f"[Statistical Mode] Đã load {len(self.stopwords)} stopwords.")
            except Exception as e:
                print(f" Lỗi load stopwords: {e}")
        else:
            print(f"[Deep Learning Mode] Không dùng Stopwords (Giữ nguyên văn bản).")

        # Load teencode từ file .csv
        if teencode_path and os.path.exists(teencode_path):
            try:
                df = pd.read_csv(teencode_path)
                # Kiểm tra xem file có đúng 2 cột cần thiết không
                if 'Word' in df.columns and 'Meaning' in df.columns:
                    # Chuyển thành Dictionary {Word: Meaning}
                    # ép kiểu str để tránh lỗi nếu file csv có số
                    self.teencode_dict = dict(zip(df['Word'].astype(str), df['Meaning'].astype(str)))
                    print(f"Đã load {len(self.teencode_dict)} teencode từ file .csv")
                else:
                    print("File CSV thiếu cột 'Word' hoặc 'Meaning'")
            except Exception as e:
                print(f"Lỗi load teencode CSV: {e}")
        else:
            print(f"Không tìm thấy file teencode tại: {teencode_path}")

        # Compile Regex
        if self.teencode_dict:
            # Sắp xếp từ dài trước ngắn sau để replace đúng (vd: 'ko' trước 'k')
            sorted_keys = sorted(self.teencode_dict.keys(), key=len, reverse=True)
            self.teencode_pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted_keys) + r')\b')
        else:
            self.teencode_pattern = None

    def clean_text(self, text):
        """
        Bước 1: Basic Cleaning & Formatting
        Mục tiêu: Làm sạch nhiễu bề mặt trước khi chuẩn hóa sâu.
        Bao gồm:
        - Chuyển toàn bộ văn bản về chữ thường (lowercase)
        - Loại bỏ HTML tags
        - Loại bỏ URL / Link
        - Loại bỏ Mentions (@user) theo regex ASCII (tránh ảnh hưởng tiếng Việt)
        - Loại bỏ Hashtag (#topic)
        - Chuẩn hóa ký tự xuống dòng, tab về khoảng trắng
        - Chuẩn hóa khoảng trắng dư thừa
        """
        
        if not isinstance(text, str): 
            return ""
        
        # 1. Chuyển về chữ thường
        text = text.lower()
        
        # 2. Xóa HTML tags
        text = re.sub(r'<[^>]*>', ' ', text)
        
        # 3. Xóa URL/Link
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # 4. Xóa Mentions (@user) – dùng regex ASCII để không dính chữ Việt
        text = re.sub(r'@[a-zA-Z0-9_.]+', '', text)
        
        # 5. Xóa Hashtag (#trend)
        text = re.sub(r'#\S+', '', text)
        
        # 6. Chuẩn hóa newline, tab về khoảng trắng
        text = re.sub(r'[\n\t]', ' ', text)
        
        # 7. Xóa khoảng trắng dư thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def handle_emoji(self, text):
        """
        BƯỚC 4: EMOJI HANDLING
        Mục tiêu: Chuyển emoji sang dạng text để mô hình học được cảm xúc.
        Cách làm:
        - Sử dụng thư viện emoji.demojize
        - Chuyển 😭 → loudly crying face
        - Thay dấu : và _ thành khoảng trắng để tách từ"""
        
        # demojize: chuyển 😭 -> :loudly_crying_face:
        text = emoji.demojize(text, delimiters=(' ', ' '))
        
        # loudly_crying_face: -> loudly crying face
        text = text.replace(':', '').replace('_', ' ')
        
        return text

    def normalize(self, text):
        """
        BƯỚC 2 → 6 : NORMALIZATION PIPELINE
        Bao gồm:
        - Bước 2: Chuẩn hóa Unicode (NFC)
        - Bước 3: Chuẩn hóa ký tự lặp (spam characters)
        - Bước 4: Xử lý Emoji
        - Bước 5: Chuẩn hóa Teencode / Slang
        - Bước 6: Lọc & chuẩn hóa dấu câu
        """
        # Bước 2: Chuẩn hóa Unicode (NFC)
        text = unicodedata.normalize('NFC', text)
        
        # Bước 4: Xử lý Emoji 
        text = self.handle_emoji(text)
        
        # Xử lí riêng cho từ "kg": 'kilogram' hoặc là 'không'
        # Case A: Đơn vị đo (5kg → 5 kilogram)
        text = re.sub(r'(\d+)\s*kg\b', r'\1 kilogram', text)
        # Case B: Nghĩa phủ định (kg → không)
        text = re.sub(r'\bkg\b', 'không', text)

        # Bước 5: Map Teencode (Từ file CSV đã load)
        if self.teencode_pattern:
            text = self.teencode_pattern.sub(lambda x: self.teencode_dict[x.group()], text)
        
        # Bước 3: Spam Character Handling
        # Rút gọn ký tự lặp > 2 lần về 1 ký tự gốc (VD: đẹpppp -> đẹp)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # Bước 6: Punctuation Filtering
        # 1. Chuẩn hóa dấu ba chấm: ... hoặc .... -> về chuẩn '...'
        text = re.sub(r'\.{3,}', ' ... ', text)
        
        # Xóa dấu câu nhiễu:  , - * ~ ( ) 
        # Giữ lại: ! ? ...
        text = re.sub(r'[,\-*~()"]', ' ', text)

        # Xóa dấu chấm đơn (.) nhưng không ảnh hưởng dấu ba chấm (...)
        text = re.sub(r'(?<!\.)\.(?!\.)', ' ', text)

        # 2. Tách dấu câu (Giữ lại ! ? để model học cảm xúc)
        # VD: "quá!" -> "quá !"
        text = re.sub(r'([!?]+)', r' \1 ', text)
        
        # Xóa khoảng trắng thừa sinh ra
        return re.sub(r'\s+', ' ', text).strip()

    def remove_stopwords(self, text):
        """
        BƯỚC 8 – STOPWORDS REMOVAL
        Áp dụng cho mô hình Statistical (TF-IDF, ML truyền thống).
        Không áp dụng cho Deep Learning để giữ ngữ cảnh.
        """
        if not self.stopwords:
            return text
        
        words = text.split()
        # Giữ lại từ không nằm trong stopwords
        words = [w for w in words if w not in self.stopwords]
        return ' '.join(words)


    def process(self, text, target_model='statistical'):
        """
        MAIN PREPROCESSING PIPELINE (NHIỆM VỤ 1)
        Luồng xử lý chuẩn:
        - Bước 1: Basic Cleaning
        - Bước 2–6: Normalization
        - Bước 7: Word Segmentation (ViTokenizer)
        - Bước 8: Stopwords Removal (chỉ cho Statistical Model)
        """
        # Bước 1
        text = self.clean_text(text) 
        # Bước 2-6 
        text = self.normalize(text)   
        
        # Bước 7: Tách từ (bắt buộc cho cả 2 mode)
        text = ViTokenizer.tokenize(text)
        
        # Bước 8: Phân nhánh xử lý Stopwords
        if target_model == 'statistical':
            # Mode Statistical: Chạy Full 11 bước (Xóa Stopwords)
            text = self.remove_stopwords(text)

        # Mode Deep Learning: Không làm gì thêm (Giữ nguyên text đã tách từ)
        # Vì PhoBERT cần ngữ cảnh đầy đủ của câu.

        return text

   
if __name__ == "__main__":
    # 1. Khởi tạo
    preprocessor = DataPreprocessor(
        stopwords_path=os.path.join(DATA_DIR, 'dictionaries', 'vietnamese_stopwords.txt'),
        teencode_path=os.path.join(DATA_DIR, 'dictionaries', 'teencode.csv')
    )
    
    # 2. Đọc dữ liệu thô
    input_file = os.path.join(DATA_DIR, 'raw', 'dataset_raw.csv') 
    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
        df.rename(columns={'comment_text': 'text', 'comment_id': 'id'}, inplace=True)
        
        # 3. Chạy 2 lần Pipeline cho 2 Mode
        modes = ['statistical', 'deep_learning']
        tqdm.pandas(desc="Tiền xử lý dữ liệu")
        for mode in modes:
            print(f"\n Đang xử lý cho chế độ: {mode}")
            temp_df = df.copy()
            
            # Tiền xử lý text theo mode
            temp_df['text'] = temp_df['text'].progress_apply(lambda x: preprocessor.process(x, target_model=mode))
            
            # Map nhãn từ chữ sang số
            label_map = {'Không xúc phạm': 0, 'Mỉa mai': 1, 'Xúc phạm': 2}
            temp_df['label'] = temp_df['label'].map(label_map)
        
            # Xóa các dòng có nhãn (label) bị trống
            temp_df = temp_df.dropna(subset=['label'])
        
            # Xóa các dòng có văn bản bị trống sau khi xử lý (ví dụ: comment chỉ có emoji bị xóa hết)
            temp_df = temp_df[temp_df['text'].str.strip() != '']
        
            # Đảm bảo label là kiểu số nguyên (Integer)
            temp_df['label'] = temp_df['label'].astype(int)
        
            # Bước 9: Deduplication (Lọc trùng)
            temp_df = temp_df.drop_duplicates(subset=['text'], keep='first')
            
            # Bước 11: Data Splitting (70-15-15)
            # Split 1: Tách Test (15%)
            train_val, test = train_test_split(
                temp_df, test_size=0.15, stratify=temp_df['label'], random_state=42
            )
            # Split 2: Tách Train và Val (Tỷ lệ 0.15/0.85 approx 0.1765)
            train, val = train_test_split(
                train_val, test_size=0.1765, stratify=train_val['label'], random_state=42
            )
            
            # Xác định các cột cần giữ lại
            output_cols = ['id', 'text', 'label']

            # Xuất file (Nhiệm vụ 2 - Đủ 6 file)
            suffix = 'stat' if mode == 'statistical' else 'dl'
            processed_dir = os.path.join(DATA_DIR, 'processed')
            os.makedirs(processed_dir, exist_ok=True)
            
            train[output_cols].to_csv(f'{processed_dir}/train_{suffix}.csv', index=False)
            val[output_cols].to_csv(f'{processed_dir}/val_{suffix}.csv', index=False)
            test[output_cols].to_csv(f'{processed_dir}/test_{suffix}.csv', index=False)
            
        print("\nHoàn thành xuất 6 file output tại data/processed/!")
    else:
        print(f"Lỗi: Không tìm thấy file đầu vào tại {input_file}")