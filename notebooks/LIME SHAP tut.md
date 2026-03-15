# Hướng dẫn sử dụng LIME & SHAP — Spam Detection Project
**Thành viên:** D — XAI + Demo  
**Mục đích:** Tài liệu tham khảo thực hành cho Tuần 3 (tích hợp vào Streamlit)

---

## 1. LIME — Giải thích từng email cụ thể

### Nguyên lý hoạt động
LIME tạo ra nhiều biến thể của email đầu vào (xáo trộn, xóa từ), quan sát cách model thay đổi dự đoán, rồi xây dựng một mô hình tuyến tính đơn giản để xấp xỉ hành vi đó. Kết quả là **danh sách từ nào đẩy email về phía spam, từ nào về phía ham** — giải thích cho đúng 1 email cụ thể.

### Khi nào dùng
Khi người dùng nhập 1 email vào app và muốn biết **"tại sao model phân loại vậy?"**

### Code mẫu — dùng trong project này

```python
from lime.lime_text import LimeTextExplainer
import joblib

# Load model và vectorizer từ C và B
model      = joblib.load("best_model.pkl")       # từ C
vectorizer = joblib.load("tfidf_vectorizer.pkl") # từ B

# Bước 1: Khởi tạo explainer
explainer = LimeTextExplainer(class_names=["ham", "spam"])

# Bước 2: Hàm predict_proba — LIME cần xác suất, không phải nhãn
def predict_proba_fn(texts):
    X = vectorizer.transform(texts)
    return model.predict_proba(X)

# Bước 3: Giải thích 1 email cụ thể
email = "Free entry! Win a cash prize now, call immediately"
exp = explainer.explain_instance(email, predict_proba_fn, num_features=10)

# Xem kết quả
print(exp.as_list())           # danh sách (từ, trọng số)
exp.as_pyplot_figure()         # biểu đồ bar ngang
```

### Tích hợp vào Streamlit (Tuần 3)

```python
import streamlit as st
import matplotlib.pyplot as plt

email_input = st.text_area("Nhập nội dung email:")

if st.button("Phân tích"):
    exp = explainer.explain_instance(email_input, predict_proba_fn, num_features=10)
    
    # Hiển thị biểu đồ LIME
    fig = exp.as_pyplot_figure()
    st.pyplot(fig)
    plt.close(fig)
    
    # Lấy top từ quan trọng
    top_words = exp.as_list()
    for word, weight in top_words:
        direction = "→ SPAM" if weight > 0 else "→ HAM"
        st.write(f"`{word}` ({weight:+.3f}) {direction}")
```

### Lưu ý quan trọng
- **Không tạo lại TF-IDF mới** — phải dùng đúng `tfidf_vectorizer.pkl` của B, nếu không LIME sẽ dùng vocabulary khác với model.
- `num_features=10` là lấy top 10 từ — có thể tăng lên nếu muốn.
- `predict_proba_fn` nhận **list các string** (chưa vector hóa) — LIME tự xáo trộn text bên trong.

---

## 2. SHAP — Phân tích tầm quan trọng tổng thể

### Nguyên lý hoạt động
SHAP dựa trên lý thuyết trò chơi (Shapley values): mỗi từ được tính điểm đóng góp trung bình vào kết quả dự đoán, trên **toàn bộ dataset**. Khác với LIME (1 email), SHAP cho thấy **từ nào quan trọng nhất với cả model**.

### Khi nào dùng
Khi muốn vẽ **summary plot** cho báo cáo hoặc hiển thị trong Streamlit phần "Tổng quan mô hình".

### Code mẫu — dùng trong project này

```python
import shap
import joblib
import pandas as pd

# Load model, vectorizer và tập mẫu từ B (Tuần 3)
model      = joblib.load("best_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
df_sample  = pd.read_csv("shap_sample.csv")  # file mẫu 500-1000 dòng do B chuẩn bị

# Vector hóa tập mẫu
X_sample = vectorizer.transform(df_sample["clean_text"]).toarray()
feature_names = vectorizer.get_feature_names_out()

# Khởi tạo SHAP explainer (dùng LinearExplainer cho Logistic Regression / Naive Bayes)
shap_explainer = shap.LinearExplainer(model, X_sample, feature_names=feature_names)
shap_values    = shap_explainer(X_sample)

# Vẽ summary plot (beeswarm)
shap.plots.beeswarm(shap_values, max_display=15)

# Vẽ bar plot — dễ đọc hơn cho báo cáo
shap.plots.bar(shap_values, max_display=15)
```

### Tích hợp vào Streamlit (Tuần 3)

```python
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.subheader("Tầm quan trọng của từ (SHAP)")

fig, ax = plt.subplots()
shap.plots.bar(shap_values, max_display=15, show=False)
st.pyplot(fig)
plt.close(fig)
```

### Lưu ý quan trọng
- `shap.LinearExplainer` dùng cho **Logistic Regression** — nếu model tốt nhất của C là Naive Bayes thì dùng `shap.Explainer(model, X_sample)`.
- SHAP chạy chậm trên toàn bộ 5.000+ dòng — dùng **tập mẫu 500–1000 dòng** do B chuẩn bị (file `shap_sample.csv`), không chạy trên toàn bộ dataset.
- `X_sample` phải là **dense array** (`.toarray()`) — không phải sparse matrix.

---

## 3. So sánh nhanh LIME vs SHAP

| | LIME | SHAP |
|---|---|---|
| Phạm vi | 1 email cụ thể | Toàn bộ dataset |
| Tốc độ | Nhanh (~1–2s/email) | Chậm hơn (cần tính trước) |
| Dùng trong app | Giải thích kết quả real-time | Summary plot tĩnh |
| Đầu ra | Danh sách từ + biểu đồ | Beeswarm/bar plot |
| Tuần tích hợp | Tuần 3 — inline trong app | Tuần 3 — tab "Tổng quan" |