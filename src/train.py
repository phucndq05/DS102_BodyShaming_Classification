
import argparse
import pandas as pd
# from src.preprocessing import DataPreprocessor
# from src.models import BaselineModel

def main():
    print("--- Khởi tạo Training Script ---")
    
    parser = argparse.ArgumentParser(description="Script huấn luyện mô hình Body Shaming Detection")
    parser.add_argument("--model_type", type=str, default="svm", help="Loại mô hình: svm, naive_bayes, logreg")
    parser.add_argument("--data_path", type=str, default="data/processed/dummy_data.csv", help="Đường dẫn file dữ liệu train")
    parser.add_argument("--output_dir", type=str, default="models", help="Thư mục lưu model")
    
    args = parser.parse_args()
    
    print(f"Cấu hình: Model={args.model_type}, Data={args.data_path}")
    
    # 1. Load Data
    # df = pd.read_csv(args.data_path)
    # print("Đã tải dữ liệu...")
    
    # 2. Preprocess
    # processor = DataPreprocessor()
    # df['clean_text'] = df['cmt_text'].apply(processor.process)
    # print("Đã xử lý dữ liệu...")
    
    # 3. Train
    # model = BaselineModel(model_type=args.model_type)
    # model.train(df['clean_text'], df['label'])
    # print("Đã huấn luyện xong...")
    
    # 4. Save
    # model.save(f"{args.output_dir}/{args.model_type}.pkl")
    # print("Đã lưu model...")

if __name__ == "__main__":
    main()
