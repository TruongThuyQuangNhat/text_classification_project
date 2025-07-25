import pandas as pd
import re
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- B1: Tiền xử lý ---
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ViTokenizer.tokenize(text)
    return text

# --- B2: Đọc dữ liệu ---
df = pd.read_csv('data/data.csv')
df['clean_text'] = df['text'].apply(preprocess)

# --- B3: Vector hoá ---
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# --- B4: Chia dữ liệu ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- B5: Huấn luyện ---
model = MultinomialNB()
model.fit(X_train, y_train)

# --- B6: Đánh giá ---
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# --- B7: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Confusion Matrix')
plt.show()

# --- B8: Lưu mô hình ---
joblib.dump(model, 'model/model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')
