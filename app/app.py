from pathlib import Path
import streamlit as st
import joblib
import re

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.joblib"


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy vectorizer: {VECTORIZER_PATH}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


st.set_page_config(
    page_title="Spam Detection App",
    page_icon="📩",
    layout="centered"
)

st.title("📩 Spam Detection App")
st.caption("Ứng dụng phân loại nội dung email/tin nhắn thành Spam hoặc Ham")

st.markdown(
    """
### Ứng dụng này làm gì?
Ứng dụng sử dụng mô hình học máy để phân loại nội dung văn bản thành:

- **Ham**: email/tin nhắn bình thường, hợp lệ, không có dấu hiệu rác hoặc lừa đảo.
- **Spam**: email/tin nhắn rác, quảng cáo không mong muốn, hoặc có dấu hiệu lừa đảo.

Bạn chỉ cần nhập nội dung vào ô bên dưới rồi bấm **Phân loại**.
"""
)

with st.expander("Giải thích Ham và Spam"):
    st.markdown(
        """
**Ham** thường là:
- thư công việc bình thường
- tin nhắn trao đổi cá nhân
- nội dung học tập, nhắc lịch, yêu cầu gửi tài liệu

**Spam** thường là:
- quảng cáo quá mức
- tin nhắn trúng thưởng giả
- nội dung dụ bấm link
- cảnh báo tài khoản giả mạo
- yêu cầu cung cấp mật khẩu, OTP, thông tin cá nhân

**Ví dụ Ham**
- "Hi, can we meet tomorrow at 9am to discuss the report?"
- "Please send me the assignment before 5pm."

**Ví dụ Spam**
- "Congratulations! You have won a free iPhone. Click here now!"
- "Urgent! Your account is blocked. Verify now to avoid suspension."
"""
    )

try:
    model, vectorizer = load_artifacts()
except Exception as e:
    st.error(f"Lỗi khi load model/vectorizer: {e}")
    st.stop()

st.success("Đã tải model và vectorizer thành công.")

st.markdown("### Nhập nội dung cần kiểm tra")
user_input = st.text_area(
    "Nội dung email/tin nhắn",
    height=220,
    placeholder="Ví dụ: Congratulations! You have won a free iPhone. Click here now!"
)

col1, col2 = st.columns([1, 1])

with col1:
    predict_btn = st.button("Phân loại", use_container_width=True)

with col2:
    clear_btn = st.button("Ví dụ mẫu", use_container_width=True)

if clear_btn:
    st.info("Ví dụ Spam: Congratulations! You have won a free iPhone. Click here now!")

if predict_btn:
    if not user_input.strip():
        st.warning("Vui lòng nhập nội dung trước khi phân loại.")
    else:
        clean_text = preprocess(user_input)
        text_vector = vectorizer.transform([clean_text])

        prediction = model.predict(text_vector)[0]
        label_map = {0: "Ham", 1: "Spam"}
        result = label_map.get(prediction, str(prediction))

        ham_prob = None
        spam_prob = None

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(text_vector)[0]
            ham_prob = probs[0] * 100
            spam_prob = probs[1] * 100

        st.markdown("---")
        st.markdown("## Kết quả phân loại")

        if result == "Spam":
            st.error("Kết luận: Đây là nội dung **Spam**.")
            st.markdown(
                """
Nội dung này có xu hướng giống thư rác, quảng cáo không mong muốn,
hoặc thông điệp có dấu hiệu lừa người dùng bấm link / cung cấp thông tin.
"""
            )
        else:
            st.success("Kết luận: Đây là nội dung **Ham**.")
            st.markdown(
                """
Nội dung này có xu hướng là email/tin nhắn bình thường, hợp lệ,
không mang nhiều dấu hiệu của thư rác hoặc lừa đảo.
"""
            )

        if ham_prob is not None and spam_prob is not None:
            st.markdown("### Xác suất dự đoán")
            c1, c2 = st.columns(2)

            with c1:
                st.metric(label="Ham", value=f"{ham_prob:.2f}%")

            with c2:
                st.metric(label="Spam", value=f"{spam_prob:.2f}%")

            st.progress(min(max(spam_prob / 100, 0.0), 1.0))

            if spam_prob >= 80:
                st.warning("Mức độ nghi ngờ spam cao.")
            elif spam_prob >= 60:
                st.info("Nội dung có khá nhiều dấu hiệu spam.")
            else:
                st.info("Mức độ nghi ngờ spam chưa cao.")

        st.markdown("### Nội dung sau tiền xử lý")
        st.code(clean_text, language="text")

st.markdown("---")
st.markdown("### Gợi ý sử dụng")
st.markdown(
    """
- Dùng các nội dung email/tin nhắn ngắn để test nhanh.
- Nếu kết quả chưa đúng như kỳ vọng, hãy kiểm tra lại dữ liệu huấn luyện và mapping nhãn.
- Ở tuần sau, bạn có thể bổ sung phần giải thích bằng LIME/SHAP.
"""
)