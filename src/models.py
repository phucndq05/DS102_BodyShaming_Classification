"""
Module định nghĩa các lớp mô hình (Model Classes).
Hỗ trợ cả Deep Learning (PhoBERT) và Machine Learning cơ bản (SVM, NaiveBayes, LogReg).
"""

import os
import joblib
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Union

# Thư viện cho Baseline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Thư viện cho PhoBERT
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset

# ==========================================
# PHẦN 1: CÁC CLASS HỖ TRỢ PHOBERT
# ==========================================

class TextDataset(Dataset):
    """Dataset wrapper cho PyTorch/HuggingFace"""
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

class WeightedTrainer(Trainer):
    """Trainer tùy chỉnh hỗ trợ Weighted Cross Entropy Loss"""
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
        return (loss, outputs) if return_outputs else loss

# ==========================================
# PHẦN 2: PHOBERT CLASSIFIER
# ==========================================

class PhoBERTClassifier:
    """Wrapper cho mô hình PhoBERT"""
    def __init__(self, model_name="vinai/phobert-base", num_labels=3, load_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Nếu load_path có giá trị -> Load model đã train để predict
        path = load_path if load_path else model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=num_labels).to(self.device)
        self.trainer = None

    def train(self, train_texts, train_labels, val_texts, val_labels, output_dir, class_weights=None, epochs=5):
        # 1. Tokenize
        train_encodings = self.tokenizer(list(train_texts), truncation=True, padding=True, max_length=256)
        val_encodings = self.tokenizer(list(val_texts), truncation=True, padding=True, max_length=256)
        
        # 2. Tạo Dataset
        train_ds = TextDataset(train_encodings, train_labels)
        val_ds = TextDataset(val_encodings, val_labels)

        # 3. Cấu hình Training
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            eval_strategy="epoch",
            save_strategy="no",             # Không lưu checkpoint rác
            learning_rate=2e-5,
            logging_steps=50,
            report_to="none"                # Tắt wandb/mlflow cho gọn
        )

        # 4. Khởi tạo Trainer
        self.trainer = WeightedTrainer(
            class_weights=class_weights,
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=self._compute_metrics
        )
        
        print(f"[PhoBERT] Bắt đầu huấn luyện trên thiết bị: {self.device}")
        self.trainer.train()

    def evaluate(self, texts, labels):
        # Hàm thống nhất interface với BaselineModel
        if self.trainer is None: self.trainer = Trainer(model=self.model)
        
        encodings = self.tokenizer(list(texts), truncation=True, padding=True, max_length=256)
        dataset = TextDataset(encodings, labels)
        
        preds = self.trainer.predict(dataset)
        y_pred = np.argmax(preds.predictions, axis=-1)
        
        print("\n--- Báo cáo Đánh giá (PhoBERT) ---")
        print(classification_report(labels, y_pred))
        return {
            'accuracy': accuracy_score(labels, y_pred),
            'report': classification_report(labels, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(labels, y_pred)
        }

    def predict(self, texts):
        # Hàm dùng riêng cho dự đoán không nhãn
        if self.trainer is None: self.trainer = Trainer(model=self.model)
        encodings = self.tokenizer(list(texts), truncation=True, padding=True, max_length=256)
        dataset = TextDataset(encodings)
        preds = self.trainer.predict(dataset)
        return np.argmax(preds.predictions, axis=-1)

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[PhoBERT] Đã lưu model tại: {path}")

    @staticmethod
    def _compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return {'f1_macro': f1_score(labels, preds, average='macro')}

# ==========================================
# PHẦN 3: BASELINE MODEL (SVM, NB, LogReg)
# ==========================================
import os
import joblib
from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import os
import joblib
from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class BaselineModel:
    """
    Class mô hình học máy thống kê cơ bản: SVM, Naive Bayes, Logistic Regression.
    Sử dụng Scikit-learn Pipeline (TfidfVectorizer + Classifier).
    Tuning tham số bằng GridSearchCV.
    """
    
    def __init__(self, model_type: str = 'svm'):
        self.model_type = model_type
        self.pipeline = None
        
    def build_pipeline(self) -> Pipeline:
        # 1. Cấu hình TF-IDF
        tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
        
        # 2. Cấu hình Classifier
        if self.model_type == 'naive_bayes':
            clf = MultinomialNB()
        elif self.model_type == 'svm':
            clf = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42)
        elif self.model_type == 'logreg':
            clf = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Model type '{self.model_type}' chưa được hỗ trợ.")
            
        self.pipeline = Pipeline([('tfidf', tfidf), ('clf', clf)])
        return self.pipeline
    
    def train(self, X_train, y_train, **kwargs) -> None:
        """
        Huấn luyện mô hình với chế độ tự động Tuning (Macro-F1).
        """
        if self.pipeline is None: self.build_pipeline()
        
        print(f"\n[Baseline] Đang Tuning & Training ({self.model_type})")
        
        # Định nghĩa tham số
        param_grid = {}
        if self.model_type == 'naive_bayes':
            param_grid = {'clf__alpha': [0.1, 0.5, 1.0, 2.0]}
        elif self.model_type == 'svm':
            param_grid = {'clf__C': [0.1, 1, 10]}
        elif self.model_type == 'logreg':
            param_grid = {'clf__C': [0.1, 1, 10, 100]}

        # Chạy GridSearch
        grid = GridSearchCV(
            self.pipeline, 
            param_grid, 
            cv=3, 
            scoring='f1_macro', 
            n_jobs=-1, 
            verbose=1
        )
        
        grid.fit(X_train, y_train)
        
        # Lưu kết quả tốt nhất
        print(f"Best Params: {grid.best_params_}")
        print(f"Best CV Macro F1: {grid.best_score_:.4f}")
        
        self.pipeline = grid.best_estimator_
        print("-> [Baseline] Huấn luyện hoàn tất.")
        
    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        y_pred = self.pipeline.predict(X_test)
        
        print(f"\n--- Báo cáo Đánh giá ({self.model_type}) ---")
        print(classification_report(y_test, y_pred))
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

    def predict(self, texts):
        return self.pipeline.predict(texts)
    
    def save(self, model_path: str) -> None:
        directory = os.path.dirname(model_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        joblib.dump(self.pipeline, model_path)
        print(f"[Baseline] Đã lưu model tại: {model_path}")

    def load(self, model_path: str) -> None:
        self.pipeline = joblib.load(model_path)
        print(f"[Baseline] Đã load model từ: {model_path}")