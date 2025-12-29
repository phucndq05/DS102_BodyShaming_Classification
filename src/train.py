"""
Script huấn luyện mô hình Body Shaming Detection.
Quy trình: Load Data (Clean) -> Train -> Validate -> Save.
"""

import argparse
import os
import sys
import pandas as pd

# Thêm root path để import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import BaselineModel

# Đường dẫn mặc định (Constants)
DATA_DIR = os.path.join('data', 'processed')
ARTIFACTS_DIR = os.path.join('demo', 'artifacts')
DEFAULT_TRAIN_PATH = os.path.join(DATA_DIR, 'train_stat.csv')
DEFAULT_VAL_PATH = os.path.join(DATA_DIR, 'val_stat.csv')


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads dataset directly.
    Assumes input file exists and has correct format (columns: text, label).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy file tại {file_path}")
    return pd.read_csv(file_path, encoding='utf-8')


def main() -> None:
    """Main execution function."""
    print("\n" + "="*50)
    print("--- QUÁ TRÌNH HUẤN LUYỆN (TRAINING PIPELINE) ---")
    print("="*50)

    # 1. Cấu hình tham số
    parser = argparse.ArgumentParser(description="Train and validate models.")
    
    parser.add_argument("--model_type", type=str, default="svm",
                        choices=["svm", "naive_bayes", "logreg"],
                        help="Loại thuật toán cần train")
    
    parser.add_argument("--train_path", type=str, default=DEFAULT_TRAIN_PATH,
                        help="Đường dẫn file Train")
    
    parser.add_argument("--val_path", type=str, default=DEFAULT_VAL_PATH,
                        help="Đường dẫn file Validation")
    
    parser.add_argument("--output_dir", type=str, default=ARTIFACTS_DIR,
                        help="Thư mục lưu model")

    args = parser.parse_args()

    print(f">> Model:      {args.model_type.upper()}")
    print(f">> Train Data: {args.train_path}")
    print(f">> Val Data:   {args.val_path}")

    # 2. Load Data
    try:
        print("\n>> Đang tải dữ liệu...")
        df_train = load_data(args.train_path)
        df_val = load_data(args.val_path)

        X_train = df_train['text']
        y_train = df_train['label']
        X_val = df_val['text']
        y_val = df_val['label']

        print(f"   + Train size: {len(X_train)}")
        print(f"   + Val size:   {len(X_val)}")
    except Exception as e:
        print(f"LỖI LOAD DATA: {e}")
        return

    # 3. Training
    print(f"\n>> Bắt đầu huấn luyện model {args.model_type}...")
    model = BaselineModel(model_type=args.model_type)
    model.train(X_train, y_train)

    # 4. Validation & Full Reporting
    print(f"\n>> Đánh giá trên tập Validation:")
    # Hàm evaluate() trong models.py đã in Classification Report và Confusion Matrix
    results = model.evaluate(X_val, y_val)

    # In thêm bảng tổng hợp metrics chi tiết
    if results:
        print("\n" + "-"*30)
        print(" TỔNG HỢP KẾT QUẢ (METRICS SUMMARY)")
        print("-" * 30)
        
        # 1. Accuracy
        acc = results.get('accuracy', 0)
        print(f"► Accuracy (Độ chính xác): {acc:.4f}")

        # 2. Macro F1 & Weighted F1
        if 'report' in results:
            report = results['report']
            macro_f1 = report['macro avg']['f1-score']
            weighted_f1 = report['weighted avg']['f1-score']
            
            print(f"► Macro F1-Score:        {macro_f1:.4f} (Quan trọng nhất)")
            print(f"► Weighted F1-Score:     {weighted_f1:.4f}")
            
            print("\n► Chi tiết F1-Score từng lớp:")
            # Duyệt qua các label 0, 1, 2 để in F1 riêng
            # Lưu ý: key trong report có thể là string '0', '1' hoặc int tuỳ version sklearn
            for label_key in ['0', '1', '2']:
                if label_key in report:
                    score = report[label_key]['f1-score']
                    label_name = "Bình thường" if label_key == '0' else ("Mỉa mai" if label_key == '1' else "Xúc phạm")
                    print(f"   - {label_name} (Class {label_key}): {score:.4f}")
    
    # 5. Save Model
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{args.model_type}.pkl")
    
    model.save(save_path)
    
    print("="*50)
    print(f"HOÀN TẤT! Model đã lưu tại: {save_path}")
    print("="*50)


if __name__ == "__main__":
    main()