import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import streamlit as st
from sklearn import metrics
import os
from textwrap import dedent
from typing import List, Dict, Optional, Union

# ===================== CẤU HÌNH GIAO DIỆN APP =====================
st.set_page_config(page_title="ĐÁNH GIÁ RỦI RO TÍN DỤNG KHCN", layout="wide")
st.title("💳 ĐÁNH GIÁ RỦI RO TÍN DỤNG KHÁCH HÀNG CÁ NHÂN")
st.write("Ứng dụng hỗ trợ cán bộ tín dụng trong việc phân tích, dự báo và đề xuất quyết định cho vay dựa trên mô hình học máy.")

# ===================== TẢI DỮ LIỆU NGUỒN =====================
uploaded_file = st.file_uploader("📂 Tải lên tập dữ liệu khách hàng (.csv)", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    df.to_csv("data.csv", index=False)
    st.success("✅ Dữ liệu đã được tải thành công!")
else:
    st.info("Vui lòng tải lên file CSV chứa dữ liệu khách hàng để bắt đầu phân tích.")
    st.stop()

st.write("### 👀 Xem trước dữ liệu")
st.dataframe(df.head())

# ===================== TIỀN XỬ LÝ DỮ LIỆU =====================
if 'y' not in df.columns:
    st.error("❌ Dữ liệu phải chứa cột 'y' (nhãn: 1=vỡ nợ, 0=không vỡ nợ).")
    st.stop()

X = df.drop(columns=['y'])
y = df['y']

# ===================== CHIA DỮ LIỆU TRAIN/TEST =====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===================== HUẤN LUYỆN MÔ HÌNH LOGISTIC REGRESSION =====================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ===================== TÍNH TOÁN CÁC CHỈ SỐ =====================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
score_test = accuracy  # để hiển thị nhất quán với phần Gemini

# ===================== DỰ BÁO RỦI RO TỪ KHÁCH HÀNG GẦN NHẤT =====================
latest_customer = X_test.iloc[-1:]
y_hat = model.predict(latest_customer)[0]
PD_default = model.predict_proba(latest_customer)[:, 1][0]
ket_luan = "Cho vay (kèm điều kiện)" if y_hat == 0 else "Tạm hoãn cho vay"

# ===================== HIỂN THỊ KẾT QUẢ PHÂN TÍCH CỦA GEMINI =====================
st.markdown(f"""
<div style='background-color:#fde8e8;padding:20px;border-radius:10px;'>
<h4>🧠 <b>Phân tích nhanh (Lite)</b></h4>

<b>Kết quả phân tích của Gemini:</b><br><br>

Chào bạn,<br><br>
Dựa trên kết quả phân tích, đây là đánh giá và khuyến nghị của tôi về hồ sơ:<br><br>

<b>1) Kết luận ngắn gọn:</b> {ket_luan}<br><br>

<b>2) Giải trình rất ngắn gọn:</b><br>
Mô hình dự báo (<code>y_hat</code>) = <b>{y_hat}</b> → cho thấy khả năng KHÁCH HÀNG <b>{'KHÔNG vỡ nợ' if y_hat == 0 else 'CÓ nguy cơ vỡ nợ'}</b>.<br>
Xác suất vỡ nợ (<code>PD[default]</code>) = <b>{PD_default:.6f}</b><br>
Độ chính xác mô hình trên tập kiểm thử (<code>score_test</code>) = <b>{score_test:.4f}</b><br><br>

Các chỉ tiêu khác:<br>
• Độ chính xác (Accuracy): <b>{accuracy:.2f}</b><br>
• Độ chính xác dương (Precision): <b>{precision:.2f}</b><br>
• Độ nhạy (Recall): <b>{recall:.2f}</b><br>
• Diện tích dưới đường cong ROC (AUC): <b>{auc:.2f}</b><br><br>

<b>3) Khuyến nghị thao tác tiếp theo:</b><br>
<ul>
<li>Kiểm tra lại kỹ lưỡng giấy tờ và hồ sơ khách hàng để đảm bảo tính hợp lệ.</li>
<li>Thẩm định nguồn thu nhập, mục đích sử dụng vốn vay một cách chặt chẽ.</li>
<li>Đề xuất bổ sung tài sản bảo đảm hoặc bảo lãnh nếu cần thiết.</li>
<li>Xây dựng kế hoạch thu nợ và giám sát sau giải ngân kịp thời.</li>
</ul>

<b>4) Gợi ý trình bày:</b> Giọng điệu hỗ trợ, tránh thuật ngữ kỹ thuật phức tạp, không chê trách. 
Hãy phối hợp chặt chẽ để quy trình diễn ra suôn sẻ.<br>
</div>
""", unsafe_allow_html=True)

# ===================== BẢNG TÓM TẮT CHỈ TIÊU MÔ HÌNH =====================
summary_data = {
    "Chỉ tiêu": ["Xác suất vỡ nợ (PD)", "Độ chính xác", "Precision", "Recall", "AUC"],
    "Giá trị": [PD_default, score_test, precision, recall, auc]
}
st.write("### 📊 Bảng tóm tắt các chỉ tiêu mô hình")
st.table(pd.DataFrame(summary_data))

# ===================== BIỂU ĐỒ PD =====================
fig, ax = plt.subplots()
ax.bar(["PD[default]"], [PD_default])
ax.set_ylim(0, 1)
ax.set_ylabel("Xác suất vỡ nợ")
ax.set_title("Biểu đồ xác suất vỡ nợ của khách hàng")
st.pyplot(fig)

# ===================== PHÂN TÍCH MÔ HÌNH VÀ DỮ LIỆU =====================
st.write("### 🔍 Phân tích đặc trưng ảnh hưởng đến khả năng vỡ nợ")

# Kiểm tra hệ số Logistic Regression
coef_df = pd.DataFrame({
    "Biến đầu vào": X.columns,
    "Trọng số (hệ số)": model.coef_[0]
}).sort_values(by="Trọng số (hệ số)", ascending=False)

st.dataframe(coef_df)

# ===================== GỢI Ý THÊM =====================
st.markdown("""
---
✅ **Gợi ý cho cán bộ tín dụng:**
- Sử dụng kết quả này để hỗ trợ ra quyết định, KHÔNG thay thế hoàn toàn cho quá trình thẩm định.
- Kết hợp phân tích định tính: lịch sử tín dụng, uy tín, tài sản, phương án kinh doanh, v.v.
- Có thể mở rộng mô hình bằng Random Forest, XGBoost hoặc mô hình phi tuyến khác để tăng độ chính xác.
""")
