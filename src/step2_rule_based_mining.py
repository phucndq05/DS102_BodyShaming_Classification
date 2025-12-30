import pandas as pd
import re
import os
from collections import Counter
from preprocessing import DataPreprocessor

def get_potential_teencode(data_path, text_col='comment', output_path='teencode_candidates.csv', top_n=200):
    """
    Hàm thống kê và phát hiện các từ có khả năng là teencode từ dữ liệu.
    
    Args:
        data_path (str): Đường dẫn đến file dữ liệu (csv hoặc excel).
        text_col (str): Tên cột chứa văn bản.
        output_path (str): Đường dẫn lưu file kết quả.
        top_n (int): Số lượng từ xuất hiện nhiều nhất muốn xem.
    """
    print(f"-> Đang đọc dữ liệu từ: {data_path}")
    
    # 1. Load dữ liệu
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        raise ValueError("Chỉ hỗ trợ file .csv hoặc .xlsx")
    
    if text_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{text_col}' trong dữ liệu.")

    # 2. Khởi tạo Preprocessor để dùng lại logic làm sạch cơ bản
    preprocessor = DataPreprocessor(mode='baseline')
    
    # Lấy danh sách teencode đã biết để loại trừ
    known_teencode = set(preprocessor.teencode_dict.keys())
    
    all_tokens = []
    
    print("-> Đang xử lý và tách từ...")
    # 3. Duyệt qua từng dòng dữ liệu
    for text in df[text_col].dropna():
        # Fix: Tự làm sạch tại chỗ để KHÔNG bị dính Emoji

        # 1. Chuyển về chữ thường & ép kiểu string
        text = str(text).lower()

        # 2. Xóa HTML & URL (Copy logic của clean_text nhưng viết thẳng vào đây)
        text = re.sub(r'<[^>]*>', ' ', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)

        # 3. Chuẩn hóa Unicode (để máy tính đọc đúng tiếng Việt)
        import unicodedata
        clean_text = unicodedata.normalize('NFC', text)

        # LƯU Ý: Tuyệt đối KHÔNG gọi preprocessor.normalize() ở đây nữa!
        # -------------------------------------------------------
        
        # Bước C: Tách từ đơn giản bằng khoảng trắng (không dùng PyVi ở đây vì PyVi có thể gộp sai teencode)
        tokens = clean_text.split()
        
        for token in tokens:
            # Chỉ lấy các từ không phải là số thuần túy và độ dài > 1
            if not token.isdigit() and len(token) > 1:
                all_tokens.append(token)

    # 4. Thống kê tần suất
    word_counts = Counter(all_tokens)
    
    # 5. Lọc và chấm điểm "Teencode"
    candidates = []
    
    # Regex nhận diện đặc điểm teencode:
    # - Chứa j, z, w, f (ít gặp trong tiếng Việt chuẩn)
    # - Chứa số nằm giữa chữ (vd: chao2)
    # - Ký tự lặp lại quá 2 lần (vd: haizzz)
    pattern_suspicious = re.compile(r'[jzfw]|\d|(.)\1{2,}')
    
    print("-> Đang phân tích các ứng viên teencode...")
    
    for word, count in word_counts.most_common():
        # Bỏ qua nếu từ đã có trong từ điển teencode hiện tại
        if word in known_teencode:
            continue
            
        is_suspicious = False
        note = ""
        
        # Kiểm tra các dấu hiệu teencode
        if pattern_suspicious.search(word):
            is_suspicious = True
            note = "Chứa ký tự lạ/lặp/số"
        elif word.endswith('k') and word not in ['jack', 'facebook', 'tiktok', 'book']: # Heuristic đơn giản
            is_suspicious = True
            note = "Kết thúc bằng k"
            
        # Lưu lại nếu là từ đáng ngờ HOẶC tần suất xuất hiện cao (để bắt các từ viết tắt như 'nt', 'ms')
        if is_suspicious or count >= 5: 
            candidates.append({
                'word': word,
                'count': count,
                'is_suspicious': is_suspicious,
                'note': note
            })

    # 6. Tạo DataFrame kết quả
    result_df = pd.DataFrame(candidates)
    
    # Ưu tiên hiển thị: Từ đáng ngờ trước, sau đó đến tần suất cao
    if not result_df.empty:
        result_df = result_df.sort_values(by=['is_suspicious', 'count'], ascending=[False, False])
        
        # Lưu file
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        result_df.head(top_n).to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"-> Đã lưu {top_n} từ tiềm năng nhất vào file: {output_path}")
        print("-> Hãy mở file này, kiểm tra cột 'word' và thêm vào 'teencode_dict' trong preprocessing.py")
    else:
        print("-> Không tìm thấy từ lạ nào đáng kể.")

if __name__ == "__main__":
    # [CẤU HÌNH ĐƯỜNG DẪN TẠI ĐÂY]
    # Bạn hãy thay đổi đường dẫn bên dưới trỏ tới file dữ liệu thực tế của bạn
    DATA_PATH = 'data/raw/dataset_raw.csv'
    OUTPUT_PATH = 'data/dictionaries/teencode_candidates.csv'
    
    # Kiểm tra file tồn tại trước khi chạy
    if os.path.exists(DATA_PATH):
        get_potential_teencode(DATA_PATH, text_col='comment_text', output_path=OUTPUT_PATH)
    else:
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {DATA_PATH}. Vui lòng sửa đường dẫn trong code.")
