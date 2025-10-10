import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
from sklearn import metrics
import os  # <-- cần cho os.path.exists
from textwrap import dedent  # <-- (NEW) cho builder prompt
from typing import List, Dict, Optional, Union  # <-- (NEW)

# ===================== GEMINI INTEGRATION (NEW) =====================
try:
    from google import genai
    from google.genai.errors import APIError
    _GEMINI_OK = True
except Exception:
    _GEMINI_OK = False


def _get_gemini_api_key():
    """Lấy API Key từ st.secrets hoặc biến môi trường."""
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.environ.get("GEMINI_API_KEY", None)
    return key


def gemini_generate_text(system_prompt: str,
                         user_prompt: str,
                         model_name: str = "gemini-2.5-flash"):
    """Gọi Gemini tạo phân tích văn bản."""
    if not _GEMINI_OK:
        return None, "⚠️ Chưa cài 'google-genai'. Vui lòng chạy: pip install google-genai"

    api_key = _get_gemini_api_key()
    if not api_key:
        return None, "⚠️ Không tìm thấy GEMINI_API_KEY. Hãy đặt vào st.secrets hoặc biến môi trường."

    try:
        client = genai.Client(api_key=api_key)
        prompt = f"{system_prompt.strip()}\n\n---\n\n{user_prompt.strip()}"
        resp = client.models.generate_content(model=model_name, contents=prompt)
        return resp.text, None
    except APIError as e:
        return None, f"Lỗi gọi Gemini API: {e}"
    except Exception as e:
        return None, f"Đã xảy ra lỗi khi gọi Gemini: {e}"
# ===================================================================

# ===================== PROMPT BUILDER =====================
SYS_PROMPT_LITE = dedent("""
Bạn là Trợ lý AI Đánh giá rủi ro tín dụng KHCN của Agribank.  
Mục tiêu: tạo bản phân tích chi tiết, rõ ràng, đáng tin cậy để cán bộ tín dụng hiểu và ra quyết định chính xác.  

Hãy trả lời theo 4 mục sau, mỗi mục trình bày cụ thể, dễ hiểu và có luận cứ rõ ràng:

1️⃣ **Kết luận ngắn gọn:** Cho vay / Cho vay có điều kiện / Không cho vay.  
   Giải thích ngắn lý do chính, dựa trên xác suất vỡ nợ và kết quả dự báo.

2️⃣ **Giải trình chi tiết, có dẫn công thức:**  
   - Giải thích ý nghĩa các chỉ số:
       • **Kết quả dự đoán của mô hình:** cho biết khách hàng được đánh giá là an toàn hay rủi ro.  
       • **Xác suất rủi ro tín dụng:** là khả năng khách hàng không trả được nợ, càng thấp càng tốt.  
       • **Độ tin cậy của mô hình:** thể hiện độ chính xác khi mô hình kiểm tra với dữ liệu thực tế.  
   - Công thức Logistic Regression:
     ```
     P(vỡ nợ) = 1 / (1 + e^-(β0 + β1*x1 + β2*x2 + ... + βn*xn))
     ```
     Trong đó: các biến x_i đại diện cho đặc điểm khách hàng (thu nhập, nợ, độ tuổi, nghề nghiệp, v.v.).  
   - Phân tích tại sao các chỉ số này dẫn đến kết luận ở mục (1).

3️⃣ **Khuyến nghị thao tác tiếp theo:**  
   - Đưa ra 3–5 bước cụ thể: xác minh thu nhập, đối chiếu giấy tờ, thẩm định mục đích vay, điều kiện giải ngân, theo dõi sau vay.  
   - Đề xuất thêm điều kiện ràng buộc nếu có rủi ro trung bình.

4️⃣ **Tổng kết:**  
   - Nhấn mạnh độ tin cậy, mức độ an toàn, và logic của kết luận.  
   - Giọng văn trung lập, rõ ràng, thuyết phục, có thể >200 từ nếu cần thiết.
""").strip()


def build_gemini_prompt_lite(
    input_row: Dict[str, Union[str, float, int]],
    y_hat: int,
    pd_vector: list,
    score_test: float,
    explain_style: str = "Dễ hiểu – dành cho cán bộ tín dụng",
    note: str = ""
) -> str:
    pd_default = None
    try:
        if isinstance(pd_vector, (list, tuple)) and len(pd_vector) == 2:
            pd_default = float(pd_vector[1])
    except Exception:
        pd_default = None

    compact_items = list(input_row.items())[:5]
    compact_str = ", ".join([f"{k}={v}" for k, v in compact_items])

    prompt = dedent(f"""
    [PHONG CÁCH]: {explain_style}
    [DỮ LIỆU TÓM TẮT]: {compact_str} {'...(rút gọn)' if len(input_row) > 5 else ''}
    [KẾT QUẢ DỰ BÁO]: {y_hat}
    [XÁC SUẤT RỦI RO]: {pd_default if pd_default is not None else 'N/A'}
    [ĐỘ CHÍNH XÁC MÔ HÌNH]: {round(float(score_test), 4)}
    [GHI CHÚ]: {note}

    Trả lời chi tiết, logic, dễ hiểu, theo đúng cấu trúc SYS_PROMPT_LITE.
    """).strip()
    return prompt
# ===================================================================

# ===================== STREAMLIT CONFIG =====================
st.set_page_config(page_title="ỨNG DỤNG ĐÁNH GIÁ RỦI RO TÍN DỤNG KHCN", page_icon="🏦", layout="wide")

# Ẩn khung upload file CSV mặc định
st.markdown("""
<style>
[data-testid="stFileUploader"] { display: none; }
</style>
""", unsafe_allow_html=True)

# === CSS GIAO DIỆN & BANNER PHÓNG TO ===
st.markdown("""
<style>
    :root {
        --agri-red: #7A0019;
        --agri-soft-red: #FFF2F2;
        --agri-dark: #2b2b2b;
        --agri-white: #ffffff;
    }
    body, .main, .stApp { background-color: var(--agri-soft-red); }
    .agri-header {
        width: 100%;
        background: linear-gradient(90deg, #7A0019 0%, #9a2740 100%);
        padding: 25px 40px;               /* cao gấp đôi */
        color: var(--agri-white);
        border-radius: 16px; 
        margin-bottom: 24px;
        text-align: center;
        transform: scale(1.05);           /* phóng nhẹ banner */
    }
    .agri-title { font-size: 34px; font-weight: 800; margin: 0; line-height: 1.3; }
    .agri-subtitle { font-size: 16px; margin-top: 6px; opacity: 0.95; }
</style>
""", unsafe_allow_html=True)

# ===================== GIAO DIỆN HEADER =====================
LOGO_URL = "https://www.inlogo.vn/wp-content/uploads/2023/04/logo-agribank-300x295.png"
BANNER_URL = "https://drive.google.com/uc?export=view&id=1Rq9kOp6caGUU1kttdOk0oaWlfO15_xb2"

col_logo, col_title = st.columns([1, 6])
with col_logo:
    try:
        st.image(LOGO_URL, width=80)
    except Exception:
        st.warning("⚠️ Không tải được logo.")
with col_title:
    st.markdown(
        '<div class="agri-header">'
        '<div class="agri-title">ỨNG DỤNG ĐÁNH GIÁ RỦI RO TÍN DỤNG KHCN</div>'
        '<div class="agri-subtitle">Dự báo xác suất xảy ra rủi ro tín dụng của KHCN & Trợ lý AI cho phân tích</div>'
        '</div>',
        unsafe_allow_html=True
    )

try:
    st.image(BANNER_URL, use_container_width=True)
except Exception:
    st.info("ℹ️ Không tải được banner (kiểm tra quyền truy cập).")

# ===================== DỮ LIỆU & MÔ HÌNH =====================
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

df = pd.read_csv('credit access.csv', encoding='latin-1')

st.title("ĐÁNH GIÁ RỦI RO TÍN DỤNG KHCN")
st.write("## Tính toán xác suất xảy ra rủi ro tín dụng của khách hàng")

X = df.drop(columns=['y'])
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

model = LogisticRegression()
model.fit(X_train, y_train)

yhat_test = model.predict(X_test)
score_train = model.score(X_train, y_train)
score_test = model.score(X_test, y_test)

confusion_matrix = pd.crosstab(y_test, yhat_test, rownames=['Actual'], colnames=['Predicted'])

# ===================== MENU ỨNG DỤNG =====================
menu = ["Mục tiêu của ứng dụng", "Phương pháp sử dụng", "Bắt đầu dự báo"]
choice = st.sidebar.selectbox('Danh mục tính năng', menu)

if choice == 'Mục tiêu của ứng dụng':
    st.write("""
    ###### ❤️ ĐIỂM TỰA CỦA NGƯỜI CÁN BỘ TÍN DỤNG KHCN ❤️
💭 Làm tín dụng đâu phải dễ.
Mỗi hồ sơ là một câu chuyện, mỗi quyết định cho vay là một lần bạn phải cân não giữa rủi ro và cơ hội.

🤝 Ứng dụng này giúp bạn có thêm một góc nhìn dữ liệu, một “bản đồ rủi ro” rõ ràng hơn, 
để mỗi quyết định của bạn vừa an toàn cho ngân hàng, vừa đong đầy sự chia sẻ với khách hàng.
    """)
    image_path = "FARMER.jpg"
    if os.path.exists(image_path):
        st.image(image_path)
    else:
        st.warning("⚠️ Ảnh FARMER.jpg chưa được tải lên hoặc sai đường dẫn.")


elif choice == 'Phương pháp sử dụng':
    st.subheader("PHƯƠNG PHÁP SỬ DỤNG")
    st.markdown("""
    **Random Forest:**  
    Mô hình dựa trên nhiều cây quyết định, giúp nhận diện mẫu hành vi phức tạp và giảm sai lệch khi dự đoán khả năng vỡ nợ.  

    **Logistic Regression:**  
    Mô hình thống kê dự đoán xác suất một khách hàng không trả được nợ.  
    Dễ giải thích, rõ ràng, và phù hợp cho đánh giá rủi ro tín dụng.
    """)
    st.image("Random-Forest.png", caption="Mô hình Random Forest", use_container_width=True)
    st.image("LOGISTIC.jpg", caption="Mô hình Logistic Regression", use_container_width=True)

elif choice == 'Bắt đầu dự báo':
    st.subheader("Bắt đầu dự báo")
    uploaded_file_1 = st.file_uploader("Tải dữ liệu khách hàng", type=['csv', 'txt'])
    if uploaded_file_1 is not None:
        lines = pd.read_csv(uploaded_file_1)
        st.dataframe(lines)

        X_1 = lines.drop(columns=['y'])
        y_pred_new = model.predict(X_1)
        pd_pred = model.predict_proba(X_1)

        st.code("Giá trị dự báo: " + str(y_pred_new))

        risky_prob = pd_pred[0][1] * 100
        safe_prob = pd_pred[0][0] * 100

        st.write(f"**Xác suất KHÁCH HÀNG AN TOÀN:** {safe_prob:.2f}%")
        st.write(f"**Xác suất CÓ RỦI RO TÍN DỤNG:** {risky_prob:.2f}%")

        if risky_prob > 50:
            st.error("⚠️ Khách hàng có nguy cơ RỦI RO TÍN DỤNG CAO.")
        else:
            st.success("✅ Khách hàng có khả năng trả nợ tốt.")

        st.session_state.last_prediction = {
            "input_row": lines.to_dict(orient="records")[0],
            "y_hat": int(y_pred_new[0]),
            "pd_vector": pd_pred[0].tolist(),
            "score_train": float(score_train),
            "score_test": float(score_test),
            "note": "LogisticRegression – train/test split 80/20, random_state=12"
        }

        st.markdown("---")
        st.subheader("🤖 Phân tích kết quả dự báo bằng Gemini (AI – Nhanh)")

        explain_style = st.selectbox(
            "Chọn phong cách giải thích",
            ["Dễ hiểu – dành cho cán bộ tín dụng", "Ngắn gọn – bullet", "Rõ ràng – kỹ thuật"]
        )

        user_prompt_lite = build_gemini_prompt_lite(
            input_row=st.session_state.last_prediction.get("input_row", {}),
            y_hat=st.session_state.last_prediction.get("y_hat"),
            pd_vector=st.session_state.last_prediction.get("pd_vector"),
            score_test=st.session_state.last_prediction.get("score_test"),
            explain_style=explain_style,
            note=st.session_state.last_prediction.get("note", "")
        )

        if st.button("🧠 Phân tích nhanh (Lite)", use_container_width=True):
            text, err = gemini_generate_text(SYS_PROMPT_LITE, user_prompt_lite)
            if err:
                st.error(err)
            else:
                st.markdown(f"**Kết quả phân tích của Gemini:**\n\n{text}")
