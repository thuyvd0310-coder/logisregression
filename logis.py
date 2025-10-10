import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
from sklearn import metrics


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
            pd=model.predict_proba(X_1)
            st.code("giá trị dự báo: " + str(y_pred_new))
            st.code("xác suất vỡ nợ của hộ là: " + str(pd))


# ====== (NEW) GIAO DIỆN – MÀU SẮC, BANNER & LOGO =============================
st.set_page_config(page_title="ỨNG DỤNG ĐÁNH GIÁ RỦI RO TÍN DỤNG KHCN", page_icon="🏦", layout="wide")

# CSS: nền đỏ nhạt, vùng nội dung dịu mắt, font & các thành phần cơ bản
st.markdown("""
<style>
    :root {
        --agri-red: #7A0019;       /* Bordeaux Agribank */
        --agri-soft-red: #FFF2F2;  /* Đỏ nhạt nền */
        --agri-dark: #2b2b2b;
        --agri-white: #ffffff;
    }
    body, .main, .stApp {
        background-color: var(--agri-soft-red);
    }
    .agri-header {
        width: 100%;
        background: linear-gradient(90deg, #7A0019 0%, #9a2740 100%);
        padding: 10px 16px;
        color: var(--agri-white);
        border-radius: 10px;
        margin-bottom: 12px;
    }
    .agri-title {
        font-size: 20px;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .agri-subtitle {
        font-size: 13px;
        margin: 0;
        opacity: 0.9;
    }
    .agri-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #f0d6db;
        box-shadow: 0 1px 6px rgba(122,0,25,0.07);
    }
    .agri-chip {
        display: inline-block;
        padding: 4px 10px;
        background: #fde7ec;
        color: #7A0019;
        border: 1px solid #f5c3cf;
        border-radius: 999px;
        font-size: 12px;
        margin-right: 8px;
    }
    .stRadio > label, .stSelectbox > label, .stFileUploader > label, .stTextInput > label, .stNumberInput > label {
        color: #7A0019; 
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Banner & logo (Google Drive link – nếu không tải được sẽ hiện cảnh báo)
BANNER_URL = "https://drive.google.com/file/d/1Rq9kOp6caGUU1kttdOk0oaWlfO15_xb2/view?usp=sharing"
LOGO_URL   = "https://www.inlogo.vn/wp-content/uploads/2023/04/logo-agribank-300x295.png"

col_logo, col_title = st.columns([1, 6], vertical_alignment="center")
with col_logo:
    try:
        st.image(LOGO_URL, caption=None, width=80)
    except Exception:
        st.warning("⚠️ Không tải được logo từ Google Drive. Vui lòng kiểm tra quyền chia sẻ.")
with col_title:
    st.markdown('<div class="agri-header"><div class="agri-title">ỨNG DỤNG ĐÁNH GIÁ RỦI RO TÍN DỤNG KHCN</div><div class="agri-subtitle">Dự báo xác suất vỡ nợ & Trợ lý AI Gemini cho phân tích & hỏi đáp</div></div>', unsafe_allow_html=True)

# Banner hình
try:
    st.image(BANNER_URL, use_container_width=True)
except Exception:
    st.info("ℹ️ Không tải được banner từ Google Drive (kiểm tra quyền truy cập).")
