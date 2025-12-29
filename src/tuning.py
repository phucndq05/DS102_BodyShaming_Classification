"""
Script tìm kiếm tham số tối ưu (Hyperparameter Tuning) sử dụng GridSearchCV.
Hỗ trợ SVM và Logistic Regression.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Thêm đường dẫn root để đảm bảo import đúng
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Hằng số đường dẫn (Chỉ dùng tập Train để Tuning)
DATA_PATH = os.path.join('data', 'processed', 'train_stat.csv')


def load_data(file_path: str) -> pd.DataFrame:
    """Load dữ liệu huấn luyện sạch."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    # Đọc file và bỏ dòng null để tránh lỗi training
    return pd.read_csv(file_path, encoding='utf-8').dropna(subset=['text', 'label'])


def tune_svm(X, y) -> dict:
    """Chạy GridSearch cho SVM."""
    print("\n" + "="*40)
    print(">>> BẮT ĐẦU TUNING SVM (SUPPORT VECTOR MACHINE)...")
    print("="*40)

    # 1. Định nghĩa Pipeline
    # Lưu ý: Luôn dùng class_weight='balanced' vì dữ liệu lệch
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3))),
        ('clf', SVC(class_weight='balanced', random_state=42))
    ])

    # 2. Định nghĩa không gian tham số cần thử
    # C: Quản lý mức độ "phạt" lỗi. C càng lớn -> càng ít sai trên tập train (dễ overfitting)
    # kernel: Loại hàm nhân
    param_grid = {
        'clf__C': [0.1, 1, 10, 100],
        'clf__kernel': ['linear', 'rbf'] 
    }

    # 3. GridSearch
    print(">> Đang chạy 3-Fold Cross-Validation (Có thể mất vài phút)...")
    grid = GridSearchCV(
        pipeline, param_grid,
        cv=3,               # Chia 3 phần kiểm định chéo
        scoring='f1_macro', # QUAN TRỌNG: Tối ưu theo Macro F1
        verbose=1,          # Hiện tiến độ
        n_jobs=-1           # Chạy song song (dùng hết CPU)
    )

    grid.fit(X, y)

    print(f"\n[KẾT QUẢ SVM]")
    print(f"► Best Params: {grid.best_params_}")
    print(f"► Best Macro F1: {grid.best_score_:.4f}")
    
    return grid.best_params_


def tune_logreg(X, y) -> dict:
    """Chạy GridSearch cho Logistic Regression."""
    print("\n" + "="*40)
    print(">>> BẮT ĐẦU TUNING LOGISTIC REGRESSION...")
    print("="*40)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3))),
        ('clf', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
    ])

    param_grid = {
        'clf__C': [0.1, 1, 10, 100],
        'clf__solver': ['lbfgs', 'liblinear']
    }

    print(">> Đang chạy 3-Fold Cross-Validation...")
    grid = GridSearchCV(
        pipeline, param_grid,
        cv=3,
        scoring='f1_macro',
        verbose=1,
        n_jobs=-1
    )

    grid.fit(X, y)

    print(f"\n[KẾT QUẢ LOGISTIC REGRESSION]")
    print(f"► Best Params: {grid.best_params_}")
    print(f"► Best Macro F1: {grid.best_score_:.4f}")
    
    return grid.best_params_


def main() -> None:
    """Hàm chính điều phối quá trình Tuning."""
    try:
        print(">> Đang tải dữ liệu Train...")
        df = load_data(DATA_PATH)
        X = df['text']
        y = df['label']
        print(f"   Số lượng mẫu dùng để Tuning: {len(X)}")

        # Chạy Tuning lần lượt
        tune_svm(X, y)
        tune_logreg(X, y)
        
        print("\n" + "="*40)
        print("HƯỚNG DẪN CẬP NHẬT:")
        print("Hãy lấy giá trị 'Best Params' ở trên và sửa vào file src/models.py")
        print("Ví dụ: Sửa clf = SVC(C=10, kernel='rbf', ...)")
        print("="*40)

    except Exception as e:
        print(f"LỖI: {e}")


if __name__ == "__main__":
    main()