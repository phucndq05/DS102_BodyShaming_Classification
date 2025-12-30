import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Import models
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
from models import BaselineModel, PhoBERTClassifier

# Cấu hình đường dẫn
DATA_DIR = os.path.join('data', 'processed')
OUTPUT_DIR = os.path.join('demo', 'artifacts')

def main():
    # 1. Cấu hình tham số
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Body Shaming.")
    
    parser.add_argument("--model_type", type=str, required=True, 
                        choices=["phobert", "svm", "naive_bayes", "logreg"],
                        help="Chọn loại mô hình để huấn luyện")
    
    parser.add_argument("--epochs", type=int, default=5, 
                        help="Số epochs (chỉ dành cho PhoBERT)")

    args = parser.parse_args()

    print(f"--- Bat dau huan luyen: {args.model_type.upper()} ---")

    # 2. Chọn file dữ liệu tương ứng
    if args.model_type == 'phobert':
        train_path = os.path.join(DATA_DIR, 'train_dl.csv')
        val_path = os.path.join(DATA_DIR, 'val_dl.csv')
    else:
        train_path = os.path.join(DATA_DIR, 'train_stat.csv')
        val_path = os.path.join(DATA_DIR, 'val_stat.csv')

    # 3. Load Data
    print(f"Dang doc du lieu tu: {train_path}")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # Ép kiểu string và lấy values
    X_train = df_train['text'].astype(str).values
    y_train = df_train['label'].values
    X_val = df_val['text'].astype(str).values
    y_val = df_val['label'].values

    # 4. Thực thi Training
    if args.model_type == 'phobert':
        # --- Deep Learning (PhoBERT) ---
        # Tính trọng số
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        print(f"Class Weights tu dong tinh: {weights}")

        classifier = PhoBERTClassifier(model_name="vinai/phobert-base")
        classifier.train(
            train_texts=X_train, 
            train_labels=y_train, 
            val_texts=X_val, 
            val_labels=y_val, 
            output_dir="./results_temp",
            class_weights=weights,
            epochs=args.epochs
        )
        
        # Lưu model
        save_path = os.path.join(OUTPUT_DIR, "phobert_final")
        classifier.save(save_path)
        
    else:
        # ---  Machine Learning (SVM, Naive Bayes, Logistic Regression) ---
        model = BaselineModel(model_type=args.model_type)
        model.train(X_train, y_train)
        
        # Đánh giá
        print("\nKet qua tren tap Validation:")
        model.evaluate(X_val, y_val)
        
        # Lưu model
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, f"{args.model_type}.pkl")
        model.save(save_path)

    print(f"\nHuan luyen hoan tat. Model da luu tai: {save_path}")

if __name__ == "__main__":
    main()