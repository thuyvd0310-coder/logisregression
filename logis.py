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

# PHẢI đặt đầu tiên
st.set_page_config(page_title="ỨNG DỤNG ĐÁNH GIÁ RỦI RO TÍN DỤNG KHCN", page_icon="🏦", layout="wide")

# CSS
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

# Logo & banner (dùng link ảnh trực tiếp)
LOGO_URL   = "https://www.inlogo.vn/wp-content/uploads/2023/04/logo-agribank-300x295.png"
BANNER_URL = "https://drive.google.com/uc?export=view&id=1Rq9kOp6caGUU1kttdOk0oaWlfO15_xb2"  # đổi sang uc?export=view&id=

# Header trên cùng (KHÔNG dùng vertical_alignment)
col_logo, col_title = st.columns([1, 6])
with col_logo:
    try:
        st.image(LOGO_URL, width=80)
    except Exception:
        st.warning("⚠️ Không tải được logo.")
with col_title:
    st.markdown(
        '<div class="agri-header"><div class="agri-title">ỨNG DỤNG ĐÁNH GIÁ RỦI RO TÍN DỤNG KHCN</div>'
        '<div class="agri-subtitle">Dự báo xác suất vỡ nợ & Trợ lý AI cho phân tích</div></div>',
        unsafe_allow_html=True
    )
# Banner
try:
    st.image(BANNER_URL, use_container_width=True)
except Exception:
    st.info("ℹ️ Không tải được banner (kiểm tra quyền truy cập).")

# ===================== SESSION STATE (NEW – cho Gemini) =====================
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
# ===========================================================================

df = pd.read_csv('credit access.csv', encoding='latin-1')

st.title("ĐÁNH GIÁ RỦI RO TÍN DỤNG KHCN")
st.write("##Tính toán xác suất xảy ra rủi ro tín dụng của khách hàng")

uploaded_file = st.file_uploader("Choose a file", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    df.to_csv("data.csv", index = False)

X = df.drop(columns=['y'])
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 12)

model = LogisticRegression()

model.fit(X_train, y_train)

yhat_test = model.predict(X_test)

score_train=model.score(X_train, y_train)
score_test=model.score(X_test, y_test)

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
    image_path = "FARMER.jpg"
    if os.path.exists(image_path):
        st.image(image_path)
    else:
        st.warning("⚠️ Ảnh FARMER.jpg chưa được tải lên hoặc sai đường dẫn.")

elif choice == 'Phương pháp sử dụng':
    st.subheader("Phương pháp sử dụng")
    st.write("""###### Mô hình sử dụng các thuật toán Random Forest, LogisticRegression""")
    st.image("Random-Forest.jpg")
    st.image("LOGISTIC.jpg")

elif choice == 'Bắt đầu dự báo':
    st.subheader("Bắt đầu dự báo")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1)
            st.dataframe(lines)
            # st.write(lines.columns)
            flag = True       
    if type=="Input":        
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
        lines={'y':[git],'DT':[DT],'TN':[TN],'SPT':[SPT],'GTC':[GTC],'GD':[GD],'TCH':[TCH],'GT':[GT],'DV':[DV],'VPCT':[VPCT],'LS':[LS]}
        lines=pd.DataFrame(lines)
        st.dataframe(lines)
        flag = True
    
    if flag:
        st.write("Content:")
        if len(lines)>0:
            st.code(lines)
            X_1 = lines.drop(columns=['y'])   
            y_pred_new = model.predict(X_1)
            # Lưu ý: tránh đặt tên biến 'pd' vì sẽ đè lên pandas. Dùng 'pd_pred' an toàn hơn:
            pd_pred = model.predict_proba(X_1)   # shape (n, 2) với lớp 0/1
            st.code("giá trị dự báo: " + str(y_pred_new))
            st.code("xác suất vỡ nợ của hộ là: " + str(pd_pred))

            # ============ LƯU KẾT QUẢ VÀ PHÂN TÍCH BẰNG GEMINI (NEW) ============
            # Lưu vào session_state để Gemini dùng làm ngữ cảnh
            st.session_state.last_prediction = {
                "input_row": lines.to_dict(orient="records")[0],
                "y_hat": int(y_pred_new[0]),
                "pd_vector": pd_pred[0].tolist(),     # [P(class=0), P(class=1)]
                "score_train": float(score_train),
                "score_test": float(score_test),
                "note": "LogisticRegression – train/test split 80/20, random_state=12"
            }

            st.markdown("---")
            st.subheader("🤖 Phân tích kết quả dự báo bằng Gemini (AI)")

            explain_style = st.selectbox(
                "Chọn phong cách giải thích",
                ["Rõ ràng – kỹ thuật", "Dễ hiểu – dành cho cán bộ tín dụng", "Ngắn gọn – bullet"]
            )

            sys_prompt = """Bạn là Trợ lý AI của Agribank, chuyên phân tích rủi ro tín dụng KHCN.
Hãy giải thích kết quả dự báo theo phong cách được yêu cầu, gồm:
1) Kết luận ngắn gọn: nguy cơ vỡ nợ cao/thấp?
2) Nêu các chỉ số chính và ý nghĩa.
3) Khuyến nghị hành động tiếp theo cho cán bộ tín dụng (giấy tờ, xác minh, phương án trả nợ).
4) Giọng điệu thân thiện, hỗ trợ, đồng hành. Trả lời bằng tiếng Việt."""

            user_prompt = f"""
[PHONG CÁCH]: {explain_style}
[ĐẦU VÀO KHÁCH HÀNG]: {st.session_state.last_prediction.get("input_row")}
[DỰ BÁO NHÃN Y_HAT]: {st.session_state.last_prediction.get("y_hat")}
[XÁC SUẤT PD] = [P(no default), P(default)] = {st.session_state.last_prediction.get("pd_vector")}
[ĐỘ CHÍNH XÁC]: train={st.session_state.last_prediction.get("score_train")}, test={st.session_state.last_prediction.get("score_test")}
[GHI CHÚ MÔ HÌNH]: {st.session_state.last_prediction.get("note")}
"""

            if st.button("🧠 Phân tích bằng Gemini", use_container_width=True):
                text, err = gemini_generate_text(sys_prompt, user_prompt)
                if err:
                    st.error(err)
                else:
                    st.markdown(f"**Kết quả phân tích của Gemini:**\n\n{text}")
            # ====================================================================
