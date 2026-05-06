# Spam Detection XAI Dashboard

Hệ thống phân loại Email/SMS Spam sử dụng Machine Learning kết hợp **Explainable AI (XAI)** — giúp người dùng không chỉ biết kết quả phân loại mà còn hiểu **vì sao** mô hình đưa ra quyết định đó.

---

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Tính năng](#tính-năng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Mô hình ML](#mô-hình-ml)
- [Giải thích XAI](#giải-thích-xai)

---

## Giới thiệu

Dự án xây dựng một **dashboard tương tác** cho bài toán phân loại tin nhắn Spam / Ham (không spam), tích hợp hai kỹ thuật Explainable AI phổ biến:

- **LIME** (Local Interpretable Model-agnostic Explanations): Giải thích từng dự đoán cụ thể.
- **SHAP** (SHapley Additive exPlanations): Phân tích mức độ ảnh hưởng của từng đặc trưng trên toàn bộ tập dữ liệu.

Mục tiêu nghiên cứu hướng đến sự **minh bạch** và **tin cậy** trong các hệ thống AI ứng dụng thực tế.

---

## Tính năng

| Tab | Chức năng |
|-----|-----------|
| **Dashboard** | Thống kê tổng quan dữ liệu, biểu đồ phân bố Spam/Ham, WordCloud, histogram độ dài văn bản, đánh giá model |
| **Phân loại** | Nhập nội dung email/SMS và nhận kết quả dự đoán kèm xác suất, biểu đồ trực quan |
| **LIME** | Giải thích chi tiết lý do mô hình phân loại từng tin nhắn cụ thể |
| **SHAP** | Xem mức độ ảnh hưởng tổng quan của các từ đặc trưng trên toàn tập dữ liệu |
| **Phân tích nâng cao** | Mức độ rủi ro spam, highlight từ quan trọng, giải thích dễ hiểu bằng ngôn ngữ tự nhiên |

---

## Cấu trúc dự án

```
XAI-SpamDetection/
│
├── app.py                      # Ứng dụng Streamlit chính
│
├── notebooks/
│   └── Trainning_AI.ipynb      # Notebook huấn luyện và thử nghiệm mô hình
│
├── models/
│   ├── best_model.joblib       # Mô hình đã huấn luyện (LogisticRegression)
│   └── vectorizer.joblib       # TF-IDF Vectorizer đã fit
│
├── data/
│   └── clean_spam_dataset.csv  # Tập dữ liệu SMS Spam đã tiền xử lý
│
├── requirements.txt            # Danh sách thư viện Python
└── README.md                   # Tài liệu hướng dẫn (file này)
```

---

## Cài đặt

### Yêu cầu hệ thống

- Python **3.8+**
- pip hoặc conda

### Bước 1: Clone repository

```bash
git clone https://github.com/your-username/XAI-SpamDetection.git
cd XAI-SpamDetection
```

### Bước 2: Tạo môi trường ảo (khuyến nghị)

```bash
# Với venv
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Hoặc với conda
conda create -n spam-xai python=3.10
conda activate spam-xai
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Huấn luyện mô hình (nếu chưa có file `.joblib`)

Mở và chạy toàn bộ notebook:

```bash
jupyter notebook notebooks/Trainning_AI.ipynb
```

Sau khi chạy xong, thư mục `models/` sẽ chứa:
- `best_model.joblib`
- `vectorizer.joblib`

---

## Hướng dẫn sử dụng

### Khởi động ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ mở tại: **http://localhost:8501**

### Luồng sử dụng đề xuất

```
1. Xem Tab Dashboard  →  Nắm tổng quan dữ liệu và hiệu năng mô hình
2. Sang Tab Phân loại →  Nhập nội dung email/SMS để kiểm tra
3. Dùng Tab LIME      →  Hiểu tại sao mô hình đưa ra quyết định đó
4. Dùng Tab SHAP      →  Phân tích đặc trưng quan trọng toàn cục
```

### Ví dụ nội dung thử nghiệm

**Spam:**
```
Congratulations! You have won a free iPhone. Click here now!
```

**Ham:**
```
Hi, can we meet tomorrow at 9am to discuss the report?
```

### Sidebar Controls

| Điều khiển | Mô tả |
|------------|-------|
| Ví dụ mẫu | Nạp nhanh các mẫu tin nhắn có sẵn |
| Số từ LIME | Điều chỉnh số từ hiển thị trong giải thích LIME (5–20) |
| Số mẫu SHAP | Số mẫu dùng để tính SHAP (50–500) |
| Lọc Dashboard | Xem thống kê theo nhãn Ham / Spam / Tất cả |
| WordCloud | Chọn nhãn và số từ tối đa cho WordCloud |
| Độ dài văn bản | Lọc dữ liệu theo số từ |

---

## Công nghệ sử dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Web App | [Streamlit](https://streamlit.io/) |
| ML Pipeline | [scikit-learn](https://scikit-learn.org/) |
| XAI - Local | [LIME](https://github.com/marcotcr/lime) |
| XAI - Global | [SHAP](https://shap.readthedocs.io/) |
| Trực quan hóa | [Matplotlib](https://matplotlib.org/), [WordCloud](https://github.com/amueller/word_cloud) |
| Xử lý dữ liệu | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| Lưu model | [Joblib](https://joblib.readthedocs.io/) |

---

## Mô hình ML

### Pipeline

```
Văn bản thô
    │
    ▼
Tiền xử lý (lowercase, loại ký tự đặc biệt, chuẩn hóa khoảng trắng)
    │
    ▼
TF-IDF Vectorizer (Term Frequency - Inverse Document Frequency)
    │
    ▼
Logistic Regression Classifier
    │
    ▼
Nhãn: Ham (0) / Spam (1) + Xác suất
```

### Huấn luyện

- **Dữ liệu**: SMS Spam Collection Dataset
- **Mô hình chính**: `LogisticRegression` (max_iter=1000)
- **Vectorizer**: `TfidfVectorizer` (lowercase=True, stop_words="english")
- **Đánh giá**: Accuracy, F1-score, Confusion Matrix

---

## Giải thích XAI

### LIME (Local Explanations)

LIME giải thích **từng dự đoán** bằng cách:
1. Tạo nhiều biến thể của văn bản đầu vào (ẩn một số từ)
2. Quan sát sự thay đổi trong xác suất dự đoán
3. Học một mô hình tuyến tính cục bộ để xác định từ nào **đẩy về Spam** hoặc **đẩy về Ham**

### SHAP (Global Explanations)

SHAP sử dụng lý thuyết game theory (Shapley values) để:
1. Đánh giá **đóng góp công bằng** của từng từ/đặc trưng
2. Tạo `Summary Plot` cho thấy các đặc trưng quan trọng nhất trên toàn tập dữ liệu
3. Hỗ trợ `LinearExplainer` tối ưu cho Logistic Regression

---

## License

MIT License — Tự do sử dụng cho mục đích học thuật và nghiên cứu.

---

> **Hệ thống Spam Detection** sử dụng Machine Learning kết hợp Explainable AI (LIME, SHAP) giúp người dùng hiểu rõ quyết định của mô hình.
