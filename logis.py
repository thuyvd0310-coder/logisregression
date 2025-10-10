# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from docx import Document
import io
import os

# =========================
# 1) ĐỌC DỮ LIỆU MẶC ĐỊNH
# =========================
# Ưu tiên data.xlsx theo đề bài; nếu không có sẽ fallback sang csv cũ (để không vỡ app).
if os.path.exists('data.xlsx'):
    df = pd.read_excel('data.xlsx')
elif os.path.exists('credit access.csv'):
    df = pd.read_csv('credit access.csv', encoding='latin-1')
else:
    # tạo khung rỗng với đúng cột để app vẫn chạy và chờ user upload
    df = pd.DataFrame(columns=["y","DT","TN","TCH","GD","GT","DV","SPT","LS","GTC","VPCT"])

# =========================
# 2) TIÊU ĐỀ ỨNG DỤNG
# =========================
st.title("ĐÁNH GIÁ RỦI RO TÍN DỤNG KHÁCH HÀNG CÁ NHÂN")
st.write("## Mô hình Logistic Regression ước lượng xác suất rủi ro (y = 1)")

# =========================
# 3) UPLOADER DỮ LIỆU HUẤN LUYỆN
# =========================
uploaded_file = st.file_uploader("Tải dữ liệu huấn luyện (khuyến nghị **data.xlsx**)", type=['xlsx', 'csv'])
if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        df.to_excel("data.xlsx", index=False)
    else:
        df = pd.read_csv(uploaded_file, encoding='latin-1')
        df.to_csv("data.csv", index=False)

# Chuẩn hóa cột đúng như đề bài
EXPECTED = ["y","DT","TN","TCH","GD","GT","DV","SPT","LS","GTC","VPCT"]
missing_cols = [c for c in EXPECTED if c not in df.columns]
if len(missing_cols)==0 and len(df)>0:
    X = df.drop(columns=['y'])
    y = df['y'].astype(int)
else:
    X = pd.DataFrame(columns=[c for c in EXPECTED if c!="y"])
    y = pd.Series(dtype=int)

# =========================
# 4) CHIA TẬP & HUẤN LUYỆN
# =========================
if len(df) > 0 and len(missing_cols) == 0:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state= 12, stratify=y
    )

    # Pipeline có chuẩn hóa để ổn định hệ số
    model = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs"))
    ])
    model.fit(X_train, y_train)
    yhat_test = model.predict(X_test)

    score_train = model.score(X_train, y_train)
    score_test  = model.score(X_test,  y_test)

    cm = pd.crosstab(y_test, yhat_test, rownames=['Actual'], colnames=['Predicted'])
    proba_test = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba_test)
else:
    model = None
    score_train = score_test = 0.0
    cm = pd.DataFrame()
    auc = np.nan

# =========================
# 5) GIAO DIỆN CHÍNH (MENU)
# =========================
menu = ["Mục tiêu của mô hình", "Xây dựng mô hình", "Sử dụng mô hình để dự báo"]
choice = st.sidebar.selectbox('Danh mục tính năng', menu)

# ====== MỤC TIÊU ======
if choice == 'Mục tiêu của mô hình':    
    st.subheader("Mục tiêu của mô hình")
    st.write("""
    ###### Dự báo **xác suất rủi ro tín dụng (y=1)** của khách hàng cá nhân dựa trên các biến:
    **DT, TN, TCH, GD, GT, DV, SPT, LS, GTC, VPCT**.
    """)  
    st.write("""###### Thuật toán sử dụng: **Logistic Regression** (chuẩn hóa đặc trưng bằng StandardScaler).""")
    st.info("Gợi ý dữ liệu: cột **y** là 0/1; các cột khác là số (GT/DV/LS/VPCT là biến nhị phân).")

# ====== XÂY DỰNG MÔ HÌNH ======
elif choice == 'Xây dựng mô hình':
    st.subheader("Xây dựng mô hình")
    if len(missing_cols) > 0:
        st.error(f"Thiếu cột {missing_cols}. Vui lòng kiểm tra file dữ liệu.")
    else:
        st.write("##### 1. Hiển thị dữ liệu")
        st.dataframe(df.head(5))
        st.dataframe(df.tail(5))  

        st.write("##### 2. Trực quan hóa dữ liệu")
        u = st.text_input('Nhập tên biến muốn vẽ (ví dụ: TN, GTC, TCH...)')
        if u and u in df.columns:
            fig1 = sns.regplot(data=df, x=u, y='y', logistic=True, ci=None)
            st.pyplot(fig1.figure)
        else:
            st.caption("Nhập đúng tên một cột để hiển thị đồ thị quan hệ với y.")

        st.write("##### 3. Huấn luyện")
        st.code("Đã huấn luyện Logistic Regression với chuẩn hóa đặc trưng (StandardScaler).")

        st.write("##### 4. Đánh giá")
        st.code("Score train: " + str(round(score_train,3)) + 
                " | Score test: " + str(round(score_test,3)) + 
                " | AUC: " + ( "NA" if np.isnan(auc) else str(round(auc,3)) ))
        if not cm.empty:
            fig2 = sns.heatmap(cm, annot=True, fmt="d")
            st.pyplot(fig2.figure)
        # hệ số
        if model is not None:
            coef = model.named_steps["clf"].coef_[0]
            cols = X.columns.tolist()
            st.write("**Hệ số mô hình (sau chuẩn hóa):**")
            st.dataframe(pd.DataFrame({"feature": cols, "coef": coef}).sort_values("coef", key=np.abs, ascending=False))
            st.code("he so chan trong mo hinh: " + str(model.named_steps["clf"].intercept_))

# ====== DỰ BÁO ======
elif choice == 'Sử dụng mô hình để dự báo':
    st.subheader("Sử dụng mô hình để dự báo")
    st.caption("Chọn cách nhập: tải file dữ liệu khách hàng hoặc nhập tay. Có thể upload **ho_so_kh.docx** để tự điền trước.")
    flag = False
    lines = None
    threshold = st.sidebar.slider("Ngưỡng phê duyệt (Approve nếu PD < ngưỡng)", 0.05, 0.80, 0.35, 0.01)

    type = st.radio("Cách nhập dữ liệu khách hàng?", options=("Upload CSV/TXT", "Nhập tay", "Upload ho_so_kh.docx"))
    if type == "Upload CSV/TXT":
        uploaded_file_1 = st.file_uploader("Chọn file (.txt, .csv) với cột: DT,TN,TCH,GD,GT,DV,SPT,LS,GTC,VPCT", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1)
            st.dataframe(lines)
            flag = True

    if type == "Upload ho_so_kh.docx":
        docx_file = st.file_uploader("Tải hồ sơ khách hàng (.docx)", type=['docx'])
        if docx_file is not None:
            # Rất đơn giản: tìm số theo từ khóa; có thể chỉnh Regex theo mẫu hồ sơ thực tế
            doc = Document(io.BytesIO(docx_file.read()))
            text = "\n".join(p.text for p in doc.paragraphs)

            def _find_num(pattern, default=None):
                import re
                m = re.search(pattern, text, flags=re.IGNORECASE)
                return float(m.group(1)) if m else default

            init = dict(
                DT=_find_num(r"(\d+)\s*m2"),   # nếu hồ sơ ghi m2; chuyển sang đơn vị 100m2 dưới đây
                TN=_find_num(r"thu\s*nhập.*?(\d+)", None),
                TCH=_find_num(r"tuổi.*?(\d+)", None),
                GD=_find_num(r"(?:lớp|năm\s*đến\s*trường).*?(\d+)", None),
                SPT=_find_num(r"phụ\s*thuộc.*?(\d+)", None),
                GTC=_find_num(r"tài\s*sản.*?(\d+)", None),
            )
            # Chuẩn hóa đơn vị DT: m2 -> đơn vị 100m2
            if init.get("DT") is not None:
                init["DT"] = round(init["DT"]/100.0, 2)

            # Các biến nhị phân (heuristic)
            lower = text.lower()
            init["GT"]   = 1 if "nữ" in lower else 0
            init["DV"]   = 1 if ("tổ trưởng" in lower or "chức vụ" in lower) else 0
            init["LS"]   = 1 if ("đã từng vay" in lower or "lịch sử tín dụng tốt" in lower) else 0
            init["VPCT"] = 0 if "không vay phi chính thức" in lower else 1

            lines = pd.DataFrame([{k:init.get(k, 0.0) for k in ["DT","TN","TCH","GD","GT","DV","SPT","LS","GTC","VPCT"]}])
            st.success("Đã tự trích vài trường từ hồ sơ. Vui lòng kiểm tra và chỉnh nếu cần.")
            st.dataframe(lines)
            flag = True

    if type == "Nhập tay":        
        DT  = st.number_input('DT – Diện tích đất (đv: 100 m²)', min_value=0.0, value=10.0, step=0.1)
        TN  = st.number_input('TN – Thu nhập năm (triệu đồng)', min_value=0.0, value=250.0, step=1.0)
        SPT = st.number_input('SPT – Số người phụ thuộc', min_value=0.0, value=2.0, step=1.0)
        GTC = st.number_input('GTC – Giá trị tài sản thế chấp (triệu đồng)', min_value=0.0, value=500.0, step=10.0)
        GD  = st.number_input('GD – Số năm đến trường', min_value=0.0, value=9.0, step=1.0)
        TCH = st.number_input('TCH – Tuổi chủ hộ', min_value=18.0, value=40.0, step=1.0)
        GT  = st.selectbox('GT – Giới tính (1: Nữ, 0: Nam)', options=[0,1], index=0)
        DV  = st.selectbox('DV – Chức vụ địa phương (1: Có, 0: Không)', options=[0,1], index=0)
        VPCT= st.selectbox('VPCT – Vay phi chính thức (1: Có, 0: Không)', options=[0,1], index=1)
        LS  = st.selectbox('LS – Lịch sử tín dụng (1: Đã từng vay, 0: Chưa)', options=[0,1], index=1)

        lines = pd.DataFrame([{
            'DT':DT,'TN':TN,'SPT':SPT,'GTC':GTC,'GD':GD,'TCH':TCH,'GT':int(GT),'DV':int(DV),'VPCT':int(VPCT),'LS':int(LS)
        }])
        st.dataframe(lines)
        flag = True
    
    # =========== SUY DIỄN ===========
    if flag:
        if model is None:
            st.error("Chưa có mô hình. Hãy upload dữ liệu huấn luyện hợp lệ ở tab 'Xây dựng mô hình'.")
        else:
            st.write("Content:")
            if len(lines)>0:
                st.code(lines)
                # Dự báo xác suất
                X_1 = lines[["DT","TN","TCH","GD","GT","DV","SPT","LS","GTC","VPCT"]]
                pd_prob = model.predict_proba(X_1)[:,1]
                y_pred_new = (pd_prob >= 0.5).astype(int)

                st.code("Giá trị dự báo (nhãn 0/1 ở ngưỡng 0.5): " + str(y_pred_new.tolist()))
                st.code("Xác suất rủi ro (PD) của khách hàng: " + str(pd_prob.tolist()))

                # Khuyến nghị theo ngưỡng quản trị rủi ro
                rec = ["KHÔNG CHO VAY" if p >= threshold else "CHO VAY" for p in pd_prob]
                st.success(f"Đề xuất theo ngưỡng {threshold:.2f}: {rec}")

                # Biểu đồ ROC (nếu có tập test)
                if isinstance(proba_test, np.ndarray) and not np.isnan(auc):
                    fpr, tpr, thr = roc_curve(y_test, proba_test)
                    fig = plt.figure()
                    plt.plot(fpr, tpr, label=f'ROC (AUC={auc:.3f})')
                    plt.plot([0,1],[0,1],'--')
                    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (trên tập kiểm định)")
                    plt.legend()
                    st.pyplot(fig)
