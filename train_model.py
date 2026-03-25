from pathlib import Path
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "clean_spam_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.joblib"


def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_text_and_label_columns(df: pd.DataFrame):
    possible_text_cols = ["text", "message", "sms", "content", "emailtext", "v2"]
    possible_label_cols = ["label", "target", "class", "category", "v1"]

    text_col = None
    label_col = None

    for col in df.columns:
        if col.lower() in possible_text_cols:
            text_col = col
            break

    for col in df.columns:
        if col.lower() in possible_label_cols:
            label_col = col
            break

    if text_col is None or label_col is None:
        raise ValueError(
            f"Không tìm thấy cột text/label phù hợp. Các cột hiện có: {list(df.columns)}"
        )

    return text_col, label_col


def encode_label(value):
    value = str(value).strip().lower()
    if value in ["spam", "1"]:
        return 1
    return 0


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {DATA_PATH}")

    print(f"Đang đọc dữ liệu từ: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    text_col, label_col = find_text_and_label_columns(df)
    print(f"Cột text: {text_col}")
    print(f"Cột label: {label_col}")

    df = df[[text_col, label_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str).apply(preprocess)
    df[label_col] = df[label_col].apply(encode_label)

    X = df[text_col]
    y = df[label_col]

    print(f"Số mẫu: {len(df)}")
    print("Phân bố nhãn:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print("\n===== KẾT QUẢ TRAIN =====")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("\n===== ĐÃ LƯU FILE =====")
    print(f"Model: {MODEL_PATH}")
    print(f"Vectorizer: {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()