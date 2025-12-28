
class BaselineModel:
    """
    Class mô hình học máy thống kê cơ bản.
    Sử dụng Scikit-learn Pipeline (TfidfVectorizer + Classifier).
    """
    
    def __init__(self, model_type='svm'):
        """
        Khởi tạo mô hình.
        Args:
            model_type (str): Loại thuật toán ('svm', 'naive_bayes', 'logistic_regression').
        """
        pass
    
    def build_pipeline(self):
        """
        Xây dựng pipeline xử lý: Tfidf -> Model.
        """
        pass
    
    def train(self, X_train, y_train):
        """
        Huấn luyện mô hình trên tập dataset đã làm sạch.
        """
        pass
    
    def evaluate(self, X_test, y_test):
        """
        Đánh giá độ chính xác của mô hình.
        """
        pass
    
    def save(self, model_path):
        """
        Lưu mô hình đã huấn luyện ra file (.pkl hoặc .joblib).
        """
        pass
