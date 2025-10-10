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

# ===================== PROMPT BUILDER TÍCH HỢP (NEW) =====================
# 1) Hằng số: “vai trò hệ thống”
SYS_PROMPT_STRUCTURED = dedent("""
Bạn là Trợ lý Đánh giá rủi ro tín dụng KHCN của Agribank, am hiểu các quy định, quy trình nội bộ về cho vay của Agribank.
Hãy giải thích kết quả dự báo theo phong cách được yêu cầu, gồm:
1) Kết luận ngắn gọn: Không cho vay/Cho vay? Cho vay (kèm điều kiện nhận nợ)
2) Giải trình lý do đưa ra kết quả trên, ưu tiên nêu rõ các chỉ số định lượng (nếu tính toán được)
3) Khuyến nghị hành động tiếp theo cho cán bộ tín dụng (giấy tờ, xác minh, điều kiện nhận nợ, phương án trả nợ, các biện pháp giám sát khoản vay...).
4) Giọng điệu thân thiện, hỗ trợ, đồng hành. Trả lời bằng tiếng Việt.
""").strip()

# 2) Khối hướng dẫn nghiệp vụ: ép nêu công thức -> áp số liệu -> kết luận
FORMULA_BLOCK = dedent("""
BẮT BUỘC CÁCH TRÌNH BÀY SỐ LIỆU (theo thứ tự):
- Mỗi chỉ số: (a) VIẾT RÕ CÔNG THỨC, (b) THAY SỐ LIỆU ĐẦY ĐỦ, (c) KẾT QUẢ, (d) NGƯỠNG/DIỄN GIẢI.
- Nếu thiếu dữ liệu: ghi rõ "THIẾU DỮ LIỆU: <tên biến>", kèm hướng dẫn thu thập.
- Ưu tiên số liệu bình quân 3–12 tháng nếu có; nếu không, nêu rõ kỳ tham chiếu.

CÁC CHỈ SỐ CỐT LÕI CẦN TÍNH (nếu đủ dữ liệu):
1) Tỷ lệ gánh nợ (DSR) theo kỳ trả nợ:
   DSR = Tổng trả nợ kỳ (gốc+lãi) / Thu nhập ròng kỳ
2) Hệ số bảo đảm trả nợ (DSCR):
   DSCR = Dòng tiền thuần hoạt động kỳ / Tổng nghĩa vụ nợ kỳ
3) Hệ số khả năng chi trả lãi (ICR):
   ICR = Thu nhập trước lãi & thuế (EBIT) / Chi phí lãi kỳ
4) Tỷ lệ cho vay trên giá trị TSBĐ (LTV):
   LTV = Dư nợ dự kiến / Giá trị định giá TSBĐ
5) Khả dụng thu nhập ròng:
   NDI = Thu nhập (ổn định) – Chi phí sinh hoạt – Thuế/phí – Nghĩa vụ nợ hiện có
6) Chu kỳ chuyển đổi tiền mặt (đối với hộ SXKD):
   CCC = DIO + DSO – DPO
   (DIO = Hàng tồn kho bình quân / Giá vốn * 365; DSO = Phải thu / Doanh thu * 365; DPO = Phải trả / Giá vốn * 365)
7) Vốn tự có tham gia phương án & tỷ lệ LTC/LTV đối với phương án SXKD có đầu tư:
   LTC = Dư nợ đề nghị / (Tổng vốn đầu tư)

LÃI SUẤT THAM CHIẾU (nếu bạn nhập): ví dụ 5%/năm → lãi kỳ (tháng) = 5%/12.
Với khoản trả góp đều, gợi ý công thức annuity để minh họa: Kỳ trả = P * r / (1 - (1+r)^(-n))

RỦI RO MÔI TRƯỜNG – XÃ HỘI (MTXH) CẦN KIỂM:
- Tài liệu pháp lý MTXH: ĐTM/ĐG tác động MT sơ bộ, Giấy phép/Đăng ký môi trường, báo cáo định kỳ, biên bản thanh tra/kiểm tra.
- Tuân thủ lao động – an toàn (BHXH, ATVSLĐ), phản ánh cộng đồng, vi phạm/biện pháp khắc phục.
- Nếu thiếu/không hợp lệ: nêu rõ điều kiện tiên quyết giải ngân hoặc điều kiện duy trì hạn mức.

RA QUYẾT ĐỊNH:
- “Không cho vay” khi chỉ số không đạt ngưỡng an toàn (ví dụ: DSCR<1; DSR>50–60% theo khẩu vị; LTV vượt trần; hồ sơ MTXH thiếu/vi phạm).
- “Cho vay (kèm điều kiện)” khi rủi ro có thể giảm thiểu bằng điều kiện nhận nợ/giải ngân từng phần/TSBĐ bổ sung/giấy tờ MTXH hợp lệ.
""").strip()

# 3) Ngưỡng tham chiếu (có thể hiệu chỉnh theo đơn vị)
RISK_GUARDRAILS = dedent("""
THAM CHIẾU NGƯỠNG (điều chỉnh theo chính sách đơn vị nếu có):
- DSR: ≤ 40–50% với KHCN; có thể nới đến 60% khi thu nhập rất ổn định & có TSBĐ tốt.
- DSCR: ≥ 1,0; an toàn ≥ 1,2 cho phương án SXKD.
- LTV (TSBĐ nhà/đất ở): tuỳ quy định từng phân khúc; minh bạch phần định giá & hệ số haircut.
- ICR: > 2 là khỏe; < 1 là cảnh báo.
- Hồ sơ MTXH: bắt buộc đầy đủ & còn hiệu lực với ngành nghề thuộc diện quản lý môi trường.
""").strip()

def _chunk(text: str, max_chars: int = 8000) -> str:
    """Cắt ngắn nội dung file dài để tránh tràn ngữ cảnh."""
    text = text.strip()
    return text[:max_chars] + (" ...[đã cắt]" if len(text) > max_chars else "")

def build_gemini_prompt(
    customer_profile: Dict[str, Union[str, float, int]],
    financials: Dict[str, Union[float, int]],
    loan_terms: Dict[str, Union[float, int, str]],
    business_params: Optional[Dict[str, Union[float, int]]] = None,
    attached_file_names: Optional[List[str]] = None,
    embedded_reg_texts: Optional[List[str]] = None,
    explain_style: str = "Rõ ràng – kỹ thuật",
    model_meta: Optional[Dict[str, Union[str, float, int]]] = None,
) -> str:
    """Tạo prompt hoàn chỉnh cho Gemini theo khuôn đã thống nhất."""
    regs_part = ""
    if attached_file_names:
        regs_part += "TÀI LIỆU QUY ĐỊNH/QUY TRÌNH (đã đính kèm qua API, vui lòng đọc trực tiếp tệp):\n"
        for fn in attached_file_names:
            regs_part += f"- {fn}\n"
    if embedded_reg_texts:
        regs_part += "\nTRÍCH YẾU QUY ĐỊNH/QUY TRÌNH (nhúng vào prompt, đã cắt ngắn):\n"
        for i, txt in enumerate(embedded_reg_texts, 1):
            regs_part += f"\n--- [Văn bản #{i}] ---\n{_chunk(txt)}\n"

    output_spec = dedent("""
    YÊU CẦU ĐẦU RA (bắt buộc theo 4 phần, viết bằng tiếng Việt, giọng hỗ trợ):
    1) Kết luận ngắn gọn: Không cho vay / Cho vay / Cho vay (kèm điều kiện nhận nợ). Nêu rõ cơ sở.
    2) Giải trình định lượng: Lập bảng chỉ số. Với MỖI chỉ số, ghi (a) công thức, (b) phép thay số, (c) kết quả, (d) ngưỡng/diễn giải.
    3) Khuyến nghị tác nghiệp: 
       - Hồ sơ cần bổ sung/xác minh (CIC, chứng từ thu nhập, chứng từ mục đích, hồ sơ MTXH…)
       - Điều kiện nhận nợ/giải ngân (CP/DP), phương án trả nợ (lịch trả, nguồn trả), kiểm soát sau vay (soát chứng từ, dòng tiền về tài khoản, rà soát TSBĐ).
       - Biện pháp giảm thiểu rủi ro (giới hạn DSR, yêu cầu TSBĐ bổ sung, bảo hiểm, bảo lãnh…).
    4) Tóm lược rủi ro nổi bật & cảnh báo sớm, gắn trách nhiệm giám sát và tần suất theo dõi.
    """).strip()

    policy_clause = dedent("""
    CHỈ SỬ DỤNG CÁC QUY ĐỊNH/QUY TRÌNH ĐÍNH KÈM HOẶC NHÚNG TRONG PROMPT LÀM NGUỒN THAM CHIẾU.
    KHÔNG ĐƯỢC SUY DIỄN TỪ NGUỒN BÊN NGOÀI.
    Nếu phát hiện mâu thuẫn giữa các văn bản, hãy nêu rõ mâu thuẫn và ưu tiên văn bản mới hơn/đặc thù hơn nếu có.
    """).strip()

    data_block = f"[PHONG CÁCH]: {explain_style}\n"
    # customer
    data_block += "\nDỮ LIỆU ĐẦU VÀO KHÁCH HÀNG:\n"
    for k, v in customer_profile.items():
        data_block += f"- KH.{k}: {v}\n"
    # financials
    data_block += "\nSỐ LIỆU TÀI CHÍNH/DỰ BÁO:\n"
    for k, v in financials.items():
        data_block += f"- Tài chính.{k}: {v}\n"
    # loan terms
    if loan_terms:
        data_block += "\nTHÔNG SỐ KHOẢN VAY (nếu có):\n"
        for k, v in loan_terms.items():
            data_block += f"- Khoản vay.{k}: {v}\n"
    # sxkd
    if business_params:
        data_block += "\nCHỈ SỐ SXKD (nếu có):\n"
        for k, v in business_params.items():
            data_block += f"- SXKD.{k}: {v}\n"
    # meta
    if model_meta:
        data_block += "\nTHÔNG TIN MÔ HÌNH / KẾT QUẢ DỰ BÁO:\n"
        for k, v in model_meta.items():
            data_block += f"- Mô hình.{k}: {v}\n"

    prompt = "\n\n".join(
        s for s in [
            policy_clause,
            RISK_GUARDRAILS,
            FORMULA_BLOCK,
            output_spec,
            regs_part.strip(),
            data_block.strip(),
        ] if s
    )
    return prompt
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

            # (NEW) Danh sách file quy định/quy trình để mô hình ưu tiên tham chiếu
            # Nếu bạn có cơ chế attach file trực tiếp cho Gemini API, hãy upload các file này
            # và giữ đúng tên ở đây để mô hình "nhớ đọc" file đính kèm.
            attached_files = [
                "3439-QyD-NHNo-RRTD.pdf",
                "PL 06 - Hướng dẫn nhận diện rủi ro.txt",
                "2268-QyĐ-NHNo-TD...txt",
                "4466-QyĐ-NHNo-KHCN.txt",
            ]

            # Chuẩn bị dữ liệu vào prompt có cấu trúc:
            # - customer_profile: có thể để trống/ghi chú nguồn
            customer_profile = {"nguon_du_lieu": "Upload/Input tại màn hình dự báo"}
            # - financials: đẩy toàn bộ cặp key/value người dùng nhập (giữ nguyên tên cột)
            financials = st.session_state.last_prediction.get("input_row", {})
            # - loan_terms: chưa có, để trống {}
            loan_terms = {}
            # - model_meta: nhúng kết quả dự báo để GEMINI trình bày theo công thức trước → áp số liệu
            model_meta = {
                "y_hat": st.session_state.last_prediction.get("y_hat"),
                "pd_vector_[P(no default),P(default)]": st.session_state.last_prediction.get("pd_vector"),
                "score_train": st.session_state.last_prediction.get("score_train"),
                "score_test": st.session_state.last_prediction.get("score_test"),
                "note": st.session_state.last_prediction.get("note"),
            }

            # Xây prompt người dùng theo khuôn (ép nêu công thức → thay số → kết luận)
            user_prompt_structured = build_gemini_prompt(
                customer_profile=customer_profile,
                financials=financials,
                loan_terms=loan_terms,
                business_params=None,
                attached_file_names=attached_files,     # khuyến nghị attach thật qua API
                embedded_reg_texts=None,               # có thể nhúng trích yếu nếu cần
                explain_style=explain_style,
                model_meta=model_meta
            )

            # Gọi Gemini với SYSTEM = vai trò cố định + USER = prompt đã build
            if st.button("🧠 Phân tích bằng Gemini", use_container_width=True):
                text, err = gemini_generate_text(SYS_PROMPT_STRUCTURED, user_prompt_structured)
                if err:
                    st.error(err)
                else:
                    st.markdown(f"**Kết quả phân tích của Gemini:**\n\n{text}")
            # ====================================================================
