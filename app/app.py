from pathlib import Path
import re

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.joblib"
DATA_PATH = BASE_DIR / "data" / "clean_spam_dataset.csv"

CLASS_NAMES = ["Ham", "Spam"]


# =========================
# Helpers
# =========================
def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(value):
    value = str(value).strip().lower()
    if value in ["spam", "1", "yes", "junk"]:
        return 1
    return 0


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model: {MODEL_PATH}. Hãy chạy train_model.py trước."
        )
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy vectorizer: {VECTORIZER_PATH}. Hãy chạy train_model.py trước."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

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

    if text_col is None:
        raise ValueError(
            f"Không tìm thấy cột văn bản trong {DATA_PATH}. Các cột hiện có: {list(df.columns)}"
        )

    df[text_col] = df[text_col].astype(str)
    df["clean_text"] = df[text_col].apply(preprocess)

    if label_col is not None:
        df["label_num"] = df[label_col].apply(normalize_label)
        df["label_name"] = df["label_num"].map({0: "Ham", 1: "Spam"})
    else:
        df["label_num"] = None
        df["label_name"] = "Unknown"

    df["text_len"] = df["clean_text"].apply(lambda x: len(str(x).split()))
    return df, text_col, label_col


def predict_text(text: str, model, vectorizer):
    clean_text = preprocess(text)
    text_vector = vectorizer.transform([clean_text])
    prediction = model.predict(text_vector)[0]

    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(text_vector)[0]

    return clean_text, text_vector, int(prediction), probs


def predict_proba_for_lime(texts, model, vectorizer):
    cleaned_texts = [preprocess(t) for t in texts]
    X = vectorizer.transform(cleaned_texts)
    return model.predict_proba(X)


def explain_with_lime(text: str, model, vectorizer, num_features: int = 10):
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model hiện tại không hỗ trợ predict_proba(), không thể dùng LIME.")

    explainer = LimeTextExplainer(class_names=CLASS_NAMES)
    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=lambda texts: predict_proba_for_lime(texts, model, vectorizer),
        num_features=num_features
    )
    return explanation


def build_shap_summary(model, vectorizer, texts):
    if len(texts) == 0:
        raise ValueError("Không có dữ liệu để vẽ SHAP.")

    X_sample = vectorizer.transform(texts)

    try:
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        raise RuntimeError(
            "Không thể dùng SHAP LinearExplainer với model hiện tại. "
            "Bản này phù hợp nhất khi model là LogisticRegression."
        ) from e

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=vectorizer.get_feature_names_out(),
        show=False
    )
    fig = plt.gcf()
    return fig


def make_label_chart(df_plot):
    counts = df_plot["label_name"].value_counts().reindex(["Ham", "Spam"], fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values)
    ax.set_title("Tỷ lệ Spam vs Ham")
    ax.set_xlabel("Nhãn")
    ax.set_ylabel("Số lượng")

    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01 if max(counts.values) > 0 else 0.1, str(v), ha="center")

    plt.tight_layout()
    return fig


def make_wordcloud(df_plot, label_filter="Spam", max_words=100):
    if "label_name" in df_plot.columns:
        if label_filter != "Tất cả":
            wc_df = df_plot[df_plot["label_name"] == label_filter]
        else:
            wc_df = df_plot
    else:
        wc_df = df_plot

    text_blob = " ".join(wc_df["clean_text"].dropna().astype(str).tolist()).strip()

    if not text_blob:
        return None

    wordcloud = WordCloud(
        width=1200,
        height=700,
        background_color="white",
        max_words=max_words,
        collocations=False
    ).generate(text_blob)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"WordCloud - {label_filter}")
    plt.tight_layout()
    return fig


def evaluate_on_dataset(df, text_col, model, vectorizer):
    eval_df = df.dropna(subset=["clean_text", "label_num"]).copy()

    if eval_df.empty:
        return None, None, None

    X = vectorizer.transform(eval_df["clean_text"].tolist())
    y_true = eval_df["label_num"].tolist()
    y_pred = model.predict(X)

    report = classification_report(
        y_true,
        y_pred,
        target_names=["Ham", "Spam"],
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return report, cm, len(eval_df)

def make_input_prob_chart(ham_prob: float, spam_prob: float):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Ham", "Spam"]
    values = [ham_prob, spam_prob]

    bars = ax.bar(labels, values)
    ax.set_title("Tỷ lệ Ham vs Spam của nội dung đang kiểm tra")
    ax.set_xlabel("Nhãn")
    ax.set_ylabel("Xác suất (%)")
    ax.set_ylim(0, 100)

    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.2f}%", ha="center")

    plt.tight_layout()
    return fig


def make_input_wordcloud(text: str, max_words: int = 100):
    clean_text = preprocess(text)

    if not clean_text.strip():
        return None

    wordcloud = WordCloud(
        width=1200,
        height=700,
        background_color="white",
        max_words=max_words,
        collocations=False
    ).generate(clean_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("WordCloud của nội dung đang kiểm tra")
    plt.tight_layout()
    return fig


# =========================
# Streamlit Config
# =========================
st.set_page_config(
    page_title="Spam Detection XAI Dashboard",
    page_icon="📩",
    layout="wide"
)

try:
    model, vectorizer = load_artifacts()
    df, text_col, label_col = load_data()
except Exception as e:
    st.exception(e)
    st.stop()

# =========================
# Session state
# =========================
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# =========================
# Sidebar
# =========================
st.sidebar.title("⚙️ Điều khiển")

sample_option = st.sidebar.selectbox(
    "Ví dụ mẫu",
    [
        "Tự nhập",
        "Spam: Congratulations! You have won a free iPhone. Click here now!",
        "Ham: Hi, can we meet tomorrow at 9am to discuss the report?",
        "Spam: Urgent! Your account is blocked. Verify now to avoid suspension.",
        "Ham: Please send me the assignment before 5pm."
    ]
)

if st.sidebar.button("Nạp ví dụ", use_container_width=True):
    if sample_option == "Tự nhập":
        st.session_state.user_input = ""
    else:
        st.session_state.user_input = sample_option.split(": ", 1)[1]

st.sidebar.markdown("---")

num_lime_features = st.sidebar.slider(
    "Số từ hiển thị trong LIME",
    min_value=5,
    max_value=20,
    value=10,
    step=1
)

shap_samples = st.sidebar.slider(
    "Số mẫu dùng cho SHAP",
    min_value=50,
    max_value=min(500, len(df)),
    value=min(300, len(df)),
    step=50 if len(df) >= 50 else 1
)

eda_label_filter = st.sidebar.selectbox(
    "Lọc dữ liệu cho Dashboard",
    ["Tất cả", "Ham", "Spam"]
)

wordcloud_label = st.sidebar.selectbox(
    "WordCloud theo nhãn",
    ["Spam", "Ham", "Tất cả"]
)

wordcloud_max_words = st.sidebar.slider(
    "Số từ tối đa trong WordCloud",
    min_value=30,
    max_value=200,
    value=100,
    step=10
)

min_len, max_len = int(df["text_len"].min()), int(df["text_len"].max())
text_len_range = st.sidebar.slider(
    "Lọc theo độ dài văn bản (số từ)",
    min_value=min_len,
    max_value=max_len,
    value=(min_len, max_len)
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Luồng đề xuất:\n"
    "1. Xem Dashboard\n"
    "2. Sang tab Phân loại để test\n"
    "3. Dùng LIME để giải thích từng email\n"
    "4. Dùng SHAP để xem ảnh hưởng tổng quan"
)

# =========================
# Filtered dataframe
# =========================
df_filtered = df[(df["text_len"] >= text_len_range[0]) & (df["text_len"] <= text_len_range[1])].copy()

if eda_label_filter != "Tất cả":
    df_filtered = df_filtered[df_filtered["label_name"] == eda_label_filter].copy()

total_samples = len(df_filtered)
ham_count = int((df_filtered["label_name"] == "Ham").sum()) if "label_name" in df_filtered.columns else 0
spam_count = int((df_filtered["label_name"] == "Spam").sum()) if "label_name" in df_filtered.columns else 0
spam_rate = (spam_count / total_samples * 100) if total_samples > 0 else 0

# =========================
# Header
# =========================
st.title("📩 Spam Detection XAI Dashboard")
st.caption("Phân loại email/tin nhắn và giải thích mô hình bằng LIME + SHAP")

st.markdown(
    """
Ứng dụng gồm 4 phần:
- **Dashboard**: thống kê và biểu đồ dữ liệu
- **Phân loại**: dự đoán nội dung là Ham hay Spam
- **LIME**: giải thích cho từng email/tin nhắn
- **SHAP**: xem ảnh hưởng đặc trưng trên tập dữ liệu
"""
)

# =========================
# Input box
# =========================
st.markdown("### Nhập nội dung cần kiểm tra")
st.text_area(
    "Nội dung email/tin nhắn",
    key="user_input",
    height=180,
    placeholder="Ví dụ: Congratulations! You have won a free iPhone. Click here now!"
)

# =========================
# Tabs
# =========================
tab0, tab1, tab2, tab3 = st.tabs(["Dashboard", "Phân loại", "LIME", "SHAP"])

# =========================
# Dashboard
# =========================
with tab0:
    st.subheader("Tổng quan dữ liệu")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng mẫu", total_samples)
    c2.metric("Ham", ham_count)
    c3.metric("Spam", spam_count)
    c4.metric("Tỷ lệ Spam", f"{spam_rate:.2f}%")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Biểu đồ phân bố Spam vs Ham")
        if total_samples > 0:
            fig_label = make_label_chart(df_filtered)
            st.pyplot(fig_label)
            plt.clf()
        else:
            st.warning("Không có dữ liệu sau khi lọc.")

    with right:
        st.markdown("### WordCloud")
        fig_wc = make_wordcloud(df_filtered, label_filter=wordcloud_label, max_words=wordcloud_max_words)
        if fig_wc is not None:
            st.pyplot(fig_wc)
            plt.clf()
        else:
            st.warning("Không có đủ dữ liệu để tạo WordCloud.")

    st.markdown("### Phân bố độ dài văn bản")
    if total_samples > 0:
        fig_len, ax_len = plt.subplots(figsize=(8, 4))
        ax_len.hist(df_filtered["text_len"], bins=30)
        ax_len.set_title("Phân bố số từ trong văn bản")
        ax_len.set_xlabel("Số từ")
        ax_len.set_ylabel("Số lượng")
        st.pyplot(fig_len)
        plt.clf()
    else:
        st.warning("Không có dữ liệu để hiển thị phân bố độ dài.")

    st.markdown("### Xem nhanh dữ liệu")
    show_cols = [text_col, "clean_text", "label_name", "text_len"]
    available_cols = [col for col in show_cols if col in df_filtered.columns]
    st.dataframe(df_filtered[available_cols].head(15), use_container_width=True)

    st.markdown("### Đánh giá nhanh model trên dataset")
    report, cm, n_eval = evaluate_on_dataset(df, text_col, model, vectorizer)

    if report is not None and cm is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric("Số mẫu đánh giá", n_eval)
        m2.metric("Accuracy", f"{report['accuracy']*100:.2f}%")
        m3.metric("Spam F1-score", f"{report['Spam']['f1-score']*100:.2f}%")

        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        im = ax_cm.imshow(cm)
        ax_cm.set_title("Confusion Matrix")
        ax_cm.set_xticks([0, 1], labels=["Ham", "Spam"])
        ax_cm.set_yticks([0, 1], labels=["Ham", "Spam"])
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center")

        st.pyplot(fig_cm)
        plt.clf()

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
    else:
        st.info("Dataset hiện chưa có nhãn nên không thể đánh giá model.")

# =========================
# Classification
# =========================
with tab1:
    st.subheader("Kết quả phân loại")

    if st.button("Phân loại ngay", use_container_width=True, key="predict_btn"):
        if not st.session_state.user_input.strip():
            st.warning("Vui lòng nhập nội dung trước khi phân loại.")
        else:
            try:
                clean_text, _, prediction, probs = predict_text(
                    st.session_state.user_input,
                    model,
                    vectorizer
                )

                result = {0: "Ham", 1: "Spam"}.get(int(prediction), str(prediction))

                k1, k2 = st.columns([1, 1])

                with k1:
                    if result == "Spam":
                        st.error("Kết luận: Đây là nội dung Spam.")
                        st.markdown(
                            """
Nội dung này có xu hướng giống thư rác, quảng cáo không mong muốn,
hoặc thông điệp có dấu hiệu lừa người dùng bấm link hoặc cung cấp thông tin.
"""
                        )
                    else:
                        st.success("Kết luận: Đây là nội dung Ham.")
                        st.markdown(
                            """
Nội dung này có xu hướng là email hoặc tin nhắn bình thường, hợp lệ,
không mang nhiều dấu hiệu của thư rác hoặc lừa đảo.
"""
                        )

                with k2:
                    st.markdown("### Nội dung sau tiền xử lý")
                    st.code(clean_text, language="text")

                    word_count = len(clean_text.split())
                    char_count = len(clean_text)

                    m1, m2 = st.columns(2)
                    m1.metric("Số từ", word_count)
                    m2.metric("Số ký tự", char_count)

                if probs is not None:
                    ham_prob = float(probs[0]) * 100
                    spam_prob = float(probs[1]) * 100

                    st.markdown("### Xác suất dự đoán")
                    p1, p2 = st.columns(2)
                    p1.metric("Ham", f"{ham_prob:.2f}%")
                    p2.metric("Spam", f"{spam_prob:.2f}%")

                    st.progress(min(max(spam_prob / 100, 0.0), 1.0))

                    if spam_prob >= 80:
                        st.warning("Mức độ nghi ngờ spam cao.")
                    elif spam_prob >= 60:
                        st.info("Nội dung có khá nhiều dấu hiệu spam.")
                    else:
                        st.info("Mức độ nghi ngờ spam chưa cao.")

                    st.markdown("---")
                    st.subheader("Biểu đồ của nội dung đang kiểm tra")

                    c1, c2 = st.columns(2)

                    with c1:
                        st.markdown("### Tỷ lệ Ham vs Spam")
                        fig_prob = make_input_prob_chart(ham_prob, spam_prob)
                        st.pyplot(fig_prob)
                        plt.clf()

                    with c2:
                        st.markdown("### WordCloud nội dung nhập")
                        fig_wc = make_input_wordcloud(
                            st.session_state.user_input,
                            max_words=wordcloud_max_words
                        )
                        if fig_wc is not None:
                            st.pyplot(fig_wc)
                            plt.clf()
                        else:
                            st.warning("Không thể tạo WordCloud từ nội dung rỗng.")

                else:
                    st.warning("Model hiện tại không hỗ trợ predict_proba(), nên không thể vẽ tỷ lệ Ham/Spam.")

                    st.markdown("---")
                    st.subheader("WordCloud của nội dung đang kiểm tra")
                    fig_wc = make_input_wordcloud(
                        st.session_state.user_input,
                        max_words=wordcloud_max_words
                    )
                    if fig_wc is not None:
                        st.pyplot(fig_wc)
                        plt.clf()
                    else:
                        st.warning("Không thể tạo WordCloud từ nội dung rỗng.")

            except Exception as e:
                st.error(f"Lỗi khi phân loại: {e}")

# =========================
# LIME
# =========================
with tab2:
    st.subheader("Giải thích bằng LIME")
    st.write("LIME giúp giải thích vì sao mô hình dự đoán email hoặc tin nhắn là Spam hoặc Ham.")

    if st.button("Chạy LIME", use_container_width=True, key="lime_btn"):
        if not st.session_state.user_input.strip():
            st.warning("Vui lòng nhập nội dung trước khi chạy LIME.")
        else:
            try:
                if not hasattr(model, "predict_proba"):
                    st.warning("Model hiện tại không hỗ trợ predict_proba(), nên không thể hiển thị LIME.")
                else:
                    clean_text, _, prediction, probs = predict_text(
                        st.session_state.user_input,
                        model,
                        vectorizer
                    )
                    result_label = int(prediction)

                    explanation = explain_with_lime(
                        text=st.session_state.user_input,
                        model=model,
                        vectorizer=vectorizer,
                        num_features=num_lime_features
                    )

                    left, right = st.columns([1.2, 1])

                    with left:
                        st.markdown("### Biểu đồ LIME")
                        fig_lime = explanation.as_pyplot_figure(label=result_label)
                        st.pyplot(fig_lime)

                    with right:
                        st.markdown("### Top từ quan trọng")
                        lime_words = explanation.as_list(label=result_label)
                        lime_df = pd.DataFrame(lime_words, columns=["Từ/Cụm từ", "Trọng số"])
                        st.dataframe(lime_df, use_container_width=True)

                    st.markdown("### Nội dung đã phân tích")
                    st.code(clean_text, language="text")

                    if probs is not None:
                        st.caption(
                            f"Xác suất dự đoán: Ham = {probs[0]*100:.2f}% | Spam = {probs[1]*100:.2f}%"
                        )

            except Exception as e:
                st.error(f"Lỗi khi chạy LIME: {e}")

# =========================
# SHAP
# =========================
with tab3:
    st.subheader("SHAP Summary Plot")
    st.write("SHAP cho biết các từ đặc trưng nào ảnh hưởng mạnh đến mô hình trên toàn bộ dữ liệu mẫu.")

    shap_source = df["clean_text"].dropna().astype(str).tolist()[:shap_samples]

    if st.button("Hiển thị SHAP Summary Plot", use_container_width=True, key="shap_btn"):
        try:
            fig_shap = build_shap_summary(model, vectorizer, shap_source)
            st.pyplot(fig_shap)
            plt.clf()
            st.caption("Biểu đồ cho thấy mức độ ảnh hưởng của các đặc trưng văn bản đến dự đoán của mô hình.")
        except Exception as e:
            st.error(f"Lỗi khi vẽ SHAP: {e}")

st.markdown("---")
st.markdown(
    """
### Gợi ý trình bày trong báo cáo
- Dashboard cho thấy phân bố dữ liệu và đặc trưng văn bản
- Tab Phân loại dùng để kiểm tra dự đoán trên từng email
- Tab LIME giải thích cục bộ cho từng mẫu đầu vào
- Tab SHAP giải thích tổng quan ở mức toàn cục
"""
)