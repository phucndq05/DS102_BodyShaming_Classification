"""
Module định nghĩa các lớp mô hình (Model Classes).
Chứa các class wrapper cho Scikit-learn Pipeline.
"""

import os
import joblib
from typing import Dict, Any, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class BaselineModel:
    """
    Class mô hình học máy thống kê cơ bản.
    Sử dụng Scikit-learn Pipeline (TfidfVectorizer + Classifier).
    """
    
    def __init__(self, model_type: str = 'svm'):
        """
        Khởi tạo mô hình.
        Args:
            model_type (str): Loại thuật toán ('svm', 'naive_bayes', 'logreg').
        """
        self.model_type = model_type
        self.pipeline = None
        
    def build_pipeline(self) -> Pipeline:
        """
        Xây dựng pipeline xử lý: Tfidf -> Model.
        Cấu hình tham số dựa trên thiết kế kỹ thuật.
        """
        # [cite_start]1. Cấu hình TF-IDF: ngram_range=(1, 3) để bắt cụm từ 
        tfidf = TfidfVectorizer(ngram_range=(1, 3))
        
        # 2. Cấu hình Classifier
        if self.model_type == 'naive_bayes':
            # [cite_start]Naive Bayes: Baseline nhanh 
            clf = MultinomialNB()
            
        elif self.model_type == 'svm':
            # [cite_start]SVM: Kernel linear, class_weight balanced
            # probability=True để Demo hiển thị được thanh "Độ tin cậy"
            clf = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42)
            
        elif self.model_type == 'logreg':
            # [cite_start]Logistic Regression: Solver lbfgs 
            clf = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000, random_state=42)
            
        else:
            raise ValueError(f"Model type '{self.model_type}' chưa được hỗ trợ.")
            
        # Tạo Pipeline hoàn chỉnh
        self.pipeline = Pipeline([
            ('tfidf', tfidf),
            ('clf', clf)
        ])
        
        return self.pipeline
    
    def train(self, X_train, y_train) -> None:
        """
        Huấn luyện mô hình trên tập dataset.
        """
        if self.pipeline is None:
            self.build_pipeline()
            
        print(f"-> Đang huấn luyện mô hình: {self.model_type}...")
        self.pipeline.fit(X_train, y_train)
        print("-> Huấn luyện hoàn tất.")
        
    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        """
        Đánh giá độ chính xác của mô hình.
        Returns:
            Dict: Chứa báo cáo (report), ma trận nhầm lẫn (confusion_matrix).
        """
        if self.pipeline is None:
            raise Exception("Mô hình chưa được huấn luyện!")
            
        y_pred = self.pipeline.predict(X_test)
        
        # In báo cáo chi tiết ra màn hình
        print(f"\n--- Báo cáo Đánh giá ({self.model_type}) ---")
        print(classification_report(y_test, y_pred, target_names=['Bình thường', 'Mỉa mai', 'Xúc phạm']))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def save(self, model_path: str) -> None:
        """
        Lưu mô hình đã huấn luyện ra file (.pkl).
        """
        if self.pipeline is None:
            raise Exception("Không có model để lưu!")
        
        # [cite_start]Tạo thư mục nếu chưa tồn tại
        directory = os.path.dirname(model_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        joblib.dump(self.pipeline, model_path)
        print(f"-> Đã lưu model tại: {model_path}")