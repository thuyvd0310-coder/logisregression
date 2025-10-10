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
# Tham khảo cách tích hợp từ file đính kèm: dùng google-genai, đọc API key từ st.secrets / env
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
    """
    Gọi Gemini tạo phân tích văn bản.
    Trả về (text, error). Nếu lỗi, text=None và error là chuỗi thông báo.
    """
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

# ===================== PROMPT BUILDER (BẢN NHẸ – LITE) =====================
SYS_PROMPT_LITE = dedent("""
Bạn là Trợ lý AI Đánh giá rủi ro tín dụng KHCN của Agribank.  
Mục tiêu: tạo bản phân tích chi tiết, rõ ràng, đáng tin cậy để cán bộ tín dụng hiểu và ra quyết định chính xác.  

Hãy trả lời theo 4 mục sau, mỗi mục trình bày cụ thể, dễ hiểu và có luận cứ rõ ràng:

1️⃣ **Kết luận ngắn gọn:** Cho vay / Cho vay có điều kiện / Không cho vay.  
   Giải thích ngắn lý do chính, dựa trên xác suất vỡ nợ và kết quả dự báo.

2️⃣ **Giải trình chi tiết, có dẫn công thức:**  
   - Diễn giải ý nghĩa các chỉ số: 
       • `y_hat`: kết quả mô hình (0 = khách hàng có khả năng trả nợ tốt, 1 = rủi ro cao).  
       • `PD[default]`: xác suất vỡ nợ mà mô hình ước lượng.  
       • `score_test`: độ chính xác của mô hình trên tập kiểm tra.  
   - Giải thích ngắn gọn công thức Logistic Regression:
     ```
     P(default) = 1 / (1 + e^-(β0 + β1*x1 + β2*x2 + ... + βn*xn))
     ```
     Trong đó: mỗi biến x_i biểu thị đặc điểm khách hàng (thu nhập, nợ, độ tuổi, nghề nghiệp,…).  
   - Phân tích vì sao kết quả dự báo và xác suất vỡ nợ dẫn đến kết luận ở mục (1).

3️⃣ **Khuyến nghị thao tác tiếp theo:**  
   - Đề xuất 3–5 bước cụ thể để đảm bảo khoản vay an toàn (kiểm chứng thu nhập, xác nhận mục đích vay, điều kiện tài sản bảo đảm, kế hoạch trả nợ…).  
   - Nêu rõ nếu cần thêm điều kiện ràng buộc hoặc theo dõi sau giải ngân.

4️⃣ **Tổng kết cho cán bộ tín dụng:**  
   - Đưa nhận định tổng quan, nhấn mạnh mức độ an toàn / rủi ro, độ tin cậy của mô hình.  
   - Viết giọng văn thân thiện, trung lập, dùng số liệu và luận cứ thay vì cảm tính.  
   - Trình bày tự nhiên, có thể dài hơn 200 từ nếu cần để giải thích thuyết phục.

Không chèn bảng, không dùng ký hiệu kỹ thuật khó hiểu, tập trung vào tính minh bạch và logic.
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
    [DỰ BÁO NHÃN y_hat]: {y_hat}
    [PD(default)]: {pd_default if pd_default is not None else 'N/A'}
    [ĐỘ CHÍNH XÁC test]: {round(float(score_test), 4)}
    [GHI CHÚ]: {note}

    YÊU CẦU: Trả lời đúng 4 mục như SYS_PROMPT, không vượt quá ~200 từ, không lập bảng.
    """).strip()
    return prompt
# ===================================================================

st.set_page_config(page_title="ỨNG DỤNG ĐÁNH GIÁ RỦI RO TÍN DỤNG KHCN", page_icon="🏦", layout="wide")

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
        padding: 10px 16px; color: var(--agri-white);
        border-radius: 10px; margin-bottom: 12px;
    }
    .agri-title { font-size: 20px; font-weight: 700; margin: 0; line-height: 1.2; }
    .agri-subtitle { font-size: 13px; margin: 0; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

LOGO_URL = "https://www.inlogo.vn/wp-content/uploads/2023/04/logo-agribank-300x295.png"
BANNER_URL = "https://drive.google.com/uc?export=view&id=1Rq9kOp6caGUU1kttdOk0oaWlfO15_xb2"

# ===================== HEADER CÂN ĐỐI =====================
LOGO_URL = "https://www.inlogo.vn/wp-content/uploads/2023/04/logo-agribank-300x295.png"
BANNER_URL = "https://drive.google.com/uc?export=view&id=1Rq9kOp6caGUU1kttdOk0oaWlfO15_xb2"

st.markdown("""
<style>
.agri-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(90deg, #7A0019 0%, #9a2740 100%);
    border-radius: 10px;
    padding: 10px 20px;
    color: white;
}
.agri-header-left {
    display: flex;
    align-items: center;
}
.agri-logo {
    width: 70px;
    height: 70px;
    border-radius: 8px;
    margin-right: 15px;
}
.agri-title-block {
    display: flex;
    flex-direction: column;
}
.agri-title {
    font-size: 22px;
    font-weight: 700;
    margin: 0;
}
.agri-subtitle {
    font-size: 14px;
    margin: 0;
    opacity: 0.9;
}
.agri-banner {
    height: 70px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="agri-header">
    <div class="agri-header-left">
        <img src="{LOGO_URL}" class="agri-logo">
        <div class="agri-title-block">
            <div class="agri-title">ỨNG DỤNG ĐÁNH GIÁ RỦI RO TÍN DỤNG KHCN</div>
            <div class="agri-subtitle">Dự báo xác suất xảy ra rủi ro tín dụng của KHCN & Trợ lý AI cho phân tích</div>
        </div>
    </div>
    <img src="{BANNER_URL}" class="agri-banner">
</div>
""", unsafe_allow_html=True)
# ===========================================================


if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

df = pd.read_csv('credit access.csv', encoding='latin-1')

st.title("ĐÁNH GIÁ RỦI RO TÍN DỤNG KHCN")
st.write("##Tính toán xác suất xảy ra rủi ro tín dụng của khách hàng")

uploaded_file = st.file_uploader("Choose a file", type=['csv'])
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

confusion_matrix = pd.crosstab(y_test, yhat_test, rownames=['Actual'], colnames=['Predicted'])

menu = ["Mục tiêu của ứng dụng", "Phương pháp sử dụng", "Bắt đầu dự báo"]
choice = st.sidebar.selectbox('Danh mục tính năng', menu)

if choice == 'Mục tiêu của ứng dụng':
    st.write("""
    ###### ❤️ ĐIỂM TỰA CỦA NGƯỜI CÁN BỘ TÍN DỤNG KHCN ❤️
💭 Làm tín dụng đâu phải dễ.
Mỗi hồ sơ là một câu chuyện, mỗi quyết định cho vay là một lần bạn phải cân não giữa rủi ro và cơ hội, giữa niềm tin và nỗi lo.

📊 Có khi bạn mất cả buổi chỉ để rà lại vài con số, rồi vẫn trăn trở:

“Nếu cho vay, liệu có an toàn?
Nếu không cho vay, liệu có phải mình vừa khép lại một cánh cửa hi vọng của ai đó đang khao khát vươn lên?”

😔 Đó là áp lực mà chỉ những người làm tín dụng mới thấu.

Bạn không chỉ tính toán con số, mà còn cân nhắc giữa niềm tin và rủi ro, đưa ra những quyết định ảnh hưởng trực tiếp đến một cuộc đời.

🤝 Chính vì thế, ứng dụng này ra đời — như một người bạn đồng hành, giúp bạn có thêm một góc nhìn dữ liệu, một “bản đồ rủi ro” rõ ràng hơn, 
để mỗi quyết định của bạn vừa an toàn cho ngân hàng, vừa đong đầy sự chia sẻ, đồng hành với khách hàng.

❤️ Vì AGRIBANK tin rằng:

Khi người cán bộ tín dụng có trong tay công cụ tốt, họ sẽ tự tin hơn trong mỗi quyết định —
vừa bảo vệ an toàn cho ngân hàng và chính mình, vừa mở ra thêm nhiều cơ hội phát triển cho khách hàng, thắp lên hi vọng cho cuộc đời ❤️

    """)
    video_path = "3E423C63-17F9-4822-A73C-CE6D42B1D005.mov"
    if os.path.exists(video_path):
        st.video(video_path, autoplay=True)
    else:
        st.warning("⚠️ Video 3E423C63-17F9-4822-A73C-CE6D42B1D005.mov chưa được tải lên hoặc sai đường dẫn.")


elif choice == 'Phương pháp sử dụng':
    st.subheader("PHƯƠNG PHÁP SỬ DỤNG ĐỂ ĐÁNH GIÁ")
    st.write("""###### Mô hình sử dụng các thuật toán Random Forest và Logistic Regression""")
    st.markdown("""
    **Random Forest** là một thuật toán học máy dựa trên tập hợp nhiều cây quyết định (Decision Trees) để dự đoán kết quả.  
    Mỗi cây học từ một phần ngẫu nhiên của dữ liệu và bỏ phiếu để ra kết quả cuối cùng.  
    Trong đánh giá rủi ro tín dụng, Random Forest giúp mô hình nhận diện các mẫu hành vi tín dụng phức tạp và giảm nguy cơ sai lệch khi dự đoán khả năng vỡ nợ của khách hàng.  
    Nhờ tính ổn định và khả năng xử lý dữ liệu phi tuyến tốt, nó thường được dùng để xếp hạng rủi ro khách hàng.
    """)
    st.image("Random-Forest.png", caption="Mô hình Random Forest", use_container_width=True)
    st.markdown("""
    **Logistic Regression** là thuật toán thống kê dự đoán xác suất một sự kiện xảy ra, thường dùng cho bài toán phân loại nhị phân.  
    Trong đánh giá rủi ro tín dụng, nó giúp ước lượng xác suất khách hàng không trả được nợ (default probability).  
    Mô hình này dễ giải thích, cho phép cán bộ tín dụng hiểu rõ ảnh hưởng của từng yếu tố đến rủi ro tín dụng.
    """)
    st.image("LOGISTIC.jpg", caption="Mô hình Logistic Regression", use_container_width=True)

elif choice == 'Bắt đầu dự báo':
    st.subheader("Bắt đầu dự báo")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type == "Upload":
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1)
            st.dataframe(lines)
            flag = True
    if type == "Input":
        git = st.number_input('Insert y')
        DT = st.number_input('Insert DT')
        TN = st.number_input('Insert TN')
        SPT = st.number_input('Insert SPT')
        GTC = st.number_input('Insert GTC')
        GD = st.number_input('Insert GD')
        TCH = st.number_input('Insert TCH')
        GT = st.number_input('Insert GT')
        DV = st.number_input('Insert DV')
        VPCT = st.number_input('Insert VPCT')
        LS = st.number_input('Insert LS')
        lines = {'y': [git], 'DT': [DT], 'TN': [TN], 'SPT': [SPT], 'GTC': [GTC], 'GD': [GD],
                 'TCH': [TCH], 'GT': [GT], 'DV': [DV], 'VPCT': [VPCT], 'LS': [LS]}
        lines = pd.DataFrame(lines)
        st.dataframe(lines)
        flag = True

    if flag:
        st.write("Content:")
        if len(lines) > 0:
            st.code(lines)
            X_1 = lines.drop(columns=['y'])
            y_pred_new = model.predict(X_1)
            pd_pred = model.predict_proba(X_1)
            st.code("giá trị dự báo: " + str(y_pred_new))

            # === Hiển thị xác suất rủi ro (căn chỉnh đúng indent) ===
            risky_prob = pd_pred[0][1] * 100
            safe_prob = pd_pred[0][0] * 100

            st.write(f"**Xác suất KHÁCH HÀNG AN TOÀN:** {safe_prob:.2f}%")
            st.write(f"**Xác suất CÓ RỦI RO TÍN DỤNG:** {risky_prob:.2f}%")

            if risky_prob > 50:
                st.error("⚠️ Khách hàng có nguy cơ RỦI RO TÍN DỤNG CAO. Cần xem xét kỹ trước khi phê duyệt khoản vay.")
            else:
                st.success("✅ Khách hàng có khả năng TỐT trong việc trả nợ. Có thể xem xét phê duyệt khoản vay.")

            # ============ LƯU KẾT QUẢ VÀ PHÂN TÍCH BẰNG GEMINI (LITE) ============
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
            # ====================================================================
