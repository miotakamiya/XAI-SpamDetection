1. LIME (Local Interpretable Model-agnostic Explanations)

LIME là một phương pháp giải thích mô hình ở mức cục bộ, tức là giải thích cho một dự đoán cụ thể của mô hình. Phương pháp này hoạt động bằng cách tạo ra nhiều biến thể của dữ liệu đầu vào, sau đó quan sát cách mô hình dự đoán trên các dữ liệu này. Từ đó, LIME xây dựng một mô hình tuyến tính đơn giản để xấp xỉ hành vi của mô hình gốc trong khu vực lân cận của dữ liệu cần giải thích.

Trong bài toán Spam Detection, LIME giúp xác định các từ quan trọng nhất ảnh hưởng đến quyết định của mô hình. Ví dụ, nếu một email chứa các từ như free, cash, hoặc prize, LIME có thể chỉ ra rằng những từ này có đóng góp lớn vào việc mô hình dự đoán email là spam.

Cách sử dụng LIME trong code gồm ba bước chính:

- Khởi tạo LimeTextExplainer.

- Tạo hàm dự đoán của mô hình (predict_proba).

- Gọi explain_instance() để giải thích một email cụ thể.

Kết quả của LIME thường được hiển thị dưới dạng danh sách các từ quan trọng hoặc biểu đồ, giúp người dùng dễ dàng hiểu được lý do của dự đoán.

2. SHAP (SHapley Additive exPlanations)

SHAP là phương pháp giải thích mô hình dựa trên lý thuyết trò chơi (Game Theory). SHAP sử dụng giá trị Shapley để đo lường mức độ đóng góp của từng feature vào kết quả dự đoán của mô hình. Không giống như LIME chỉ tập trung vào một dự đoán cụ thể, SHAP có thể giải thích mức độ ảnh hưởng của các feature trên toàn bộ mô hình.

Trong bài toán Spam Detection, SHAP giúp xác định các từ có ảnh hưởng lớn nhất đến việc phân loại spam trong toàn bộ dataset. Ví dụ, các từ như free, win, cash, hoặc offer thường có giá trị SHAP cao vì chúng thường xuất hiện trong spam messages.

Các bước sử dụng SHAP trong code gồm:

- Khởi tạo SHAP explainer cho mô hình.

- Tính toán giá trị SHAP cho tập dữ liệu.

- Trực quan hóa kết quả bằng các biểu đồ như beeswarm plot hoặc summary plot.

Những biểu đồ này cho phép người dùng nhìn thấy feature nào quan trọng nhất trong quá trình dự đoán của mô hình.

3. Tổng quan

Tóm lại, LIME phù hợp để giải thích từng email cụ thể, trong khi SHAP phù hợp để phân tích tầm quan trọng của các feature trong toàn bộ mô hình. Việc kết hợp hai phương pháp này giúp hệ thống Spam Detection trở nên minh bạch và dễ hiểu hơn, đồng thời hỗ trợ việc tích hợp giải thích mô hình vào ứng dụng demo trong các giai đoạn tiếp theo của project.