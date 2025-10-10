import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
from sklearn import metrics
import os
from textwrap import dedent
from typing import List, Dict, Optional, Union

# ===================== GEMINI INTEGRATION =====================
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

# ===================== PROMPT NÂNG CẤP (FULL EXPLAIN) =====================
SYS_PROMPT_LITE = dedent("""
Bạn là Trợ lý AI của Agribank, chuyên hỗ trợ cán bộ tín dụng KHCN ra quyết định cho vay.

YÊU CẦU TRẢ LỜI CHI TIẾT, CỤ THỂ, RÕ RÀNG VÀ MINH BẠCH, gồm 5 phần sau:

1️⃣ **Kết luận:** Cho vay / Cho vay có điều kiện / Không cho vay.  
   Giải thích ngắn gọn lý do chính (ví dụ: PD thấp, khách hàng an toàn).

2️⃣ **Giải trình & Công thức tính toán:**  
   - Nêu rõ ý nghĩa và giá trị của từng chỉ số:
     • `y_hat`: kết quả mô hình dự đoán (0 = an toàn, 1 = rủi ro).  
     • `PD[default]`: xác suất khách hàng vỡ nợ.  
     • `score_test`: độ chính xác mô hình.
   - Giải thích công thức Logistic Regression:  
     ```
     P(default) = 1 / (1 + e^-(β0 + β1*x1 + β2*x2 + ... + βn*xn))
     ```
     Trong đó: các biến x_i là đặc điểm khách hàng (thu nhập, nợ, tuổi, nghề nghiệp...).
   - Mô tả cách mô hình dùng công thức trên để ước tính xác suất vỡ nợ.

3️⃣ **Phân tích định lượng:**  
   - Nhận xét PD, độ tin cậy của mô hình, so sánh với ngưỡng an toàn (ví dụ <5% là tốt).  
   - Giải thích vì sao khách hàng có/không đủ điều kiện tín dụng.

4️⃣ **Khuyến nghị thao tác tiếp theo (chi tiết):**  
   - Các giấy tờ, kiểm chứng, biện pháp bổ sung.  
   - Gợi ý điều kiện ràng buộc khi giải ngân, nếu có rủi ro trung bình.  
   - Nêu thêm cách giám sát sau giải ngân (ví dụ: tần suất theo dõi, dòng tiền kiểm soát).

5️⃣ **Tổng kết cho cán bộ tín dụng:**  
   - Tóm tắt logic ra quyết định, nhấn mạnh mức độ an toàn và minh chứng bằng số liệu.

Giọng văn thân thiện, khách quan, tránh dùng từ kỹ thuật phức tạp, không chèn bảng.
Trình bày rõ ràng bằng bullet hoặc đoạn ngắn. 
""").strip()
# ===================================================================

def build_gemini_prompt_lite(
    input_row: Dict[str, Union[str, float, int]],
    y_hat: int,
    pd_vector: list,
    score_test: float,
    explain_style: str = "Phân tích chi tiết cho cán bộ tín dụng",
    note: str = ""
) -> str:
    """Xây dựng prompt chi tiết để gửi đến Gemini."""
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
    [DỮ LIỆU KHÁCH HÀNG]: {compact_str} {'...(rút gọn)' if len(input_row) > 5 else ''}
    [DỰ BÁO NHÃN y_hat]: {y_hat}
    [PD(default)]: {pd_default if pd_default is not None else 'N/A'}
    [ĐỘ CHÍNH XÁC MÔ HÌNH (score_test)]: {round(float(score_test), 4)}
    [GHI CHÚ]: {note}

    Hãy trả lời chi tiết theo đúng 5 phần trong SYS_PROMPT_LITE, sử dụng ngôn ngữ tự nhiên, dễ hiểu và có tính thuyết phục.
    """).strip()
    return prompt
# ===================================================================

# ===================== STREAMLIT APP =====================
st.set_page_config(page_title="ĐÁNH GIÁ RỦI RO TÍN DỤNG AGRIBANK", page_icon="🏦", layout="wide")

st.title("🏦 ỨNG DỤNG ĐÁNH GIÁ RỦI RO TÍN DỤNG KHCN")
st.write("Phân tích chi tiết khả năng trả nợ của khách hàng cá nhân sử dụng mô hình Logistic Regression và AI Gemini.")

df = pd.read_csv('credit access.csv', encoding='latin-1')

uploaded_file = st.file_uploader("Tải file dữ liệu khách hàng (.csv)", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    df.to_csv("data.csv", index=False)

X = df.drop(columns=['y'])
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
model = LogisticRegression()
model.fit(X_train, y_train)

yhat_test = model.predict(X_test)
score_train = model.score(X_train, y_train)
score_test = model.score(X_test, y_test)

menu = ["Giới thiệu", "Phương pháp", "Phân tích chi tiết"]
choice = st.sidebar.selectbox("Danh mục", menu)

if choice == "Phân tích chi tiết":
    uploaded_file_1 = st.file_uploader("Chọn file dữ liệu cần dự báo", type=['csv', 'txt'])
    if uploaded_file_1 is not None:
        lines = pd.read_csv(uploaded_file_1)
        st.dataframe(lines)
        X_1 = lines.drop(columns=['y'])
        y_pred_new = model.predict(X_1)
        pd_pred = model.predict_proba(X_1)

        st.write(f"**Kết quả dự báo:** {y_pred_new}")
        risky_prob = pd_pred[0][1] * 100
        safe_prob = pd_pred[0][0] * 100

        st.metric("Xác suất khách hàng AN TOÀN", f"{safe_prob:.2f}%")
        st.metric("Xác suất CÓ RỦI RO TÍN DỤNG", f"{risky_prob:.2f}%")

        st.markdown("---")
        st.subheader("📊 Phân tích chi tiết bằng Gemini AI")

        explain_style = st.selectbox(
            "Chọn phong cách giải thích:",
            ["Phân tích chi tiết cho cán bộ tín dụng", "Ngắn gọn – kỹ thuật", "Diễn giải thân thiện"]
        )

        user_prompt_lite = build_gemini_prompt_lite(
            input_row=lines.to_dict(orient="records")[0],
            y_hat=int(y_pred_new[0]),
            pd_vector=pd_pred[0].tolist(),
            score_test=float(score_test),
            explain_style=explain_style,
            note="Phân tích Logistic Regression – 80/20 split, random_state=12"
        )

        if st.button("🧠 Phân tích chuyên sâu"):
            text, err = gemini_generate_text(SYS_PROMPT_LITE, user_prompt_lite)
            if err:
                st.error(err)
            else:
                st.markdown(f"### Kết quả phân tích chi tiết của Gemini:\n\n{text}")
