import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- CONFIG ---
st.set_page_config(page_title="Laptop Price Predictor", layout="wide")

# --- FUNCTIONS ---
def load_data(train_file, val_file, test_file):
    try:
        df_train = pd.read_csv(train_file)
        df_val = pd.read_csv(val_file)
        df_test = pd.read_csv(test_file)
        return df_train, df_val, df_test
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        return None, None, None

def align_columns(df_train, df_val, df_test, target='price_base'):
    # Loại bỏ 'title' nếu có
    for df in [df_train, df_val, df_test]:
        if 'title' in df.columns:
            df.drop(columns=['title'], inplace=True)
            
    # Lấy các cột chung (intersection) trừ cột target
    common_cols = list(set(df_train.columns) & set(df_val.columns) & set(df_test.columns))
    if target in common_cols:
        common_cols.remove(target)
    
    # Sắp xếp để đảm bảo thứ tự cột cố định
    common_cols.sort()
    
    X_train = df_train[common_cols]
    y_train = df_train[target]
    
    X_val = df_val[common_cols]
    y_val = df_val[target]
    
    X_test = df_test[common_cols]
    y_test = df_test[target]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, common_cols

def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Xử lý an toàn cho MAPE (tránh chia cho 0)
    y_true_safe = np.where(y_true == 0, 1e-9, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    # Accuracy: % dự đoán trong khoảng +- 10%
    diff_ratio = np.abs((y_true - y_pred) / y_true_safe)
    accuracy = np.mean(diff_ratio <= 0.10) * 100
    
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "Accuracy (±10%)": accuracy}

def build_model(model_type, params):
    if model_type == "Random Forest":
        base_model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=42
        )
    elif model_type == "XGBoost":
        base_model = XGBRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            random_state=42
        )
    else: # LightGBM
        base_model = LGBMRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            num_leaves=params['num_leaves'],
            random_state=42
        )
    
    # Sử dụng Pipeline để đóng gói Imputer và Model
    # Quan trọng: Imputer sẽ fit trên Train và transform lên Val/Test để tránh rò rỉ thông tin
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', base_model)
    ])
    return pipeline

# --- UI SIDEBAR ---
st.sidebar.header("1. Cấu hình Dữ liệu")
train_up = st.sidebar.file_uploader("Upload data_train.csv", type="csv")
val_up = st.sidebar.file_uploader("Upload data_validation.csv", type="csv")
test_up = st.sidebar.file_uploader("Upload data_test.csv", type="csv")

# Mặc định lấy file cục bộ nếu không upload
train_path = train_up if train_up else "data_train.csv"
val_path = val_up if val_up else "data_validation.csv"
test_path = test_up if test_up else "data_test.csv"

st.sidebar.header("2. Chọn Mô hình & Hyperparams")
model_choice = st.sidebar.selectbox("Mô hình", ["Random Forest", "XGBoost", "LightGBM"])

params = {}
params['n_estimators'] = st.sidebar.slider("n_estimators", 10, 500, 100)
params['max_depth'] = st.sidebar.slider("max_depth", 1, 20, 10)

if model_choice in ["XGBoost", "LightGBM"]:
    params['learning_rate'] = st.sidebar.number_input("learning_rate", 0.001, 1.0, 0.1, step=0.01)

if model_choice == "XGBoost":
    params['subsample'] = st.sidebar.slider("subsample", 0.5, 1.0, 0.8)
    params['colsample_bytree'] = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.8)
elif model_choice == "LightGBM":
    params['num_leaves'] = st.sidebar.slider("num_leaves", 10, 150, 31)

# --- MAIN APP ---
st.title("💻 Laptop Price Prediction App")

if all(os.path.exists(str(p)) or hasattr(p, 'read') for p in [train_path, val_path, test_path]):
    df_tr, df_vl, df_ts = load_data(train_path, val_path, test_path)
    
    if df_tr is not None:
        X_tr, y_tr, X_vl, y_vl, X_ts, y_ts, features = align_columns(df_tr, df_vl, df_ts)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Preview Data", "Training & Eval", "Predict", "Export"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Train Set**", df_tr.shape)
                st.dataframe(df_tr.head(5))
                st.write("NaN per column:", df_tr.isnull().sum().sum())
            with col2:
                st.write("**Val Set**", df_vl.shape)
                st.dataframe(df_vl.head(5))
                st.write("NaN per column:", df_vl.isnull().sum().sum())
            with col3:
                st.write("**Test Set**", df_ts.shape)
                st.dataframe(df_ts.head(5))
                st.write("NaN per column:", df_ts.isnull().sum().sum())

        with tab2:
            if st.button("🚀 Start Training"):
                with st.spinner("Đang huấn luyện..."):
                    model_pipeline = build_model(model_choice, params)
                    model_pipeline.fit(X_tr, y_tr)
                    
                    # Lưu model vào session_state
                    st.session_state['trained_model'] = model_pipeline
                    st.session_state['features'] = features
                    
                    # Eval
                    y_pred_vl = model_pipeline.predict(X_vl)
                    metrics_vl = calculate_metrics(y_vl, y_pred_vl)
                    
                    st.success("Huấn luyện hoàn tất!")
                    
                    # Hiển thị Metrics
                    m_cols = st.columns(5)
                    for idx, (m_name, m_val) in enumerate(metrics_vl.items()):
                        m_cols[idx].metric(m_name, f"{m_val:,.2f}")
                    
                    # Visualization
                    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
                    
                    # 1. Scatter
                    ax[0].scatter(y_vl, y_pred_vl, alpha=0.5, color='teal')
                    ax[0].plot([y_vl.min(), y_vl.max()], [y_vl.min(), y_vl.max()], 'r--')
                    ax[0].set_title("True vs Predicted (Validation)")
                    ax[0].set_xlabel("True Price")
                    ax[0].set_ylabel("Pred Price")
                    
                    # 2. Residuals
                    residuals = y_vl - y_pred_vl
                    ax[1].hist(residuals, bins=30, color='orange', edgecolor='black')
                    ax[1].set_title("Residuals Distribution")
                    
                    # 3. Feature Importance
                    raw_model = model_pipeline.named_steps['model']
                    if hasattr(raw_model, 'feature_importances_'):
                        importances = raw_model.feature_importances_
                        indices = np.argsort(importances)[-10:] # Top 10
                        ax[2].barh([features[i] for i in indices], importances[indices], color='skyblue')
                        ax[2].set_title("Top 10 Feature Importance")
                    else:
                        ax[2].text(0.5, 0.5, "Không hỗ trợ cho model này", ha='center')
                    
                    st.pyplot(fig)

            if st.button("🔍 Đánh giá trên Test Set"):
                if 'trained_model' in st.session_state:
                    y_pred_ts = st.session_state['trained_model'].predict(X_ts)
                    metrics_ts = calculate_metrics(y_ts, y_pred_ts)
                    st.write("**Kết quả trên Test Set:**")
                    st.json(metrics_ts)
                else:
                    st.warning("Vui lòng huấn luyện model trước.")

        with tab3:
            st.subheader("Dự đoán giá trị mới")
            if 'trained_model' in st.session_state:
                method = st.radio("Cách nhập dữ liệu:", ["Nhập tay", "Upload CSV (1 dòng)"])
                
                if method == "Nhập tay":
                    with st.form("predict_form"):
                        input_data = {}
                        # Tự động sinh input field cho các feature
                        cols = st.columns(3)
                        for i, f in enumerate(features):
                            # Lấy median làm default value
                            default_val = float(X_tr[f].median())
                            input_data[f] = cols[i % 3].number_input(f, value=default_val)
                        
                        submit = st.form_submit_button("Dự đoán")
                        if submit:
                            input_df = pd.DataFrame([input_data])[features]
                            pred = st.session_state['trained_model'].predict(input_df)[0]
                            st.success(f"💰 Giá dự đoán: {pred:,.0f} VND")
                
                else:
                    pred_file = st.file_uploader("Upload CSV để dự đoán", type="csv")
                    if pred_file:
                        p_df = pd.read_csv(pred_file)
                        # Align columns giống hệt lúc train
                        p_df_clean = p_df.reindex(columns=features)
                        pred = st.session_state['trained_model'].predict(p_df_clean)[0]
                        st.success(f"💰 Giá dự đoán từ file: {pred:,.0f} VND")
            else:
                st.info("Hãy huấn luyện model ở tab Training trước.")

        with tab4:
            if 'trained_model' in st.session_state:
                st.subheader("Xuất dữ liệu")
                
                # Download model
                joblib.dump(st.session_state['trained_model'], "model.joblib")
                with open("model.joblib", "rb") as f:
                    st.download_button("💾 Tải model.joblib", f, "model.joblib")
                
                # Download test predictions
                y_pred_ts = st.session_state['trained_model'].predict(X_ts)
                test_results = X_ts.copy()
                test_results['true_price'] = y_ts
                test_results['predicted_price'] = y_pred_ts
                csv = test_results.to_csv(index=False).encode('utf-8')
                st.download_button("📊 Tải test_predictions.csv", csv, "test_predictions.csv", "text/csv")
            else:
                st.info("Chưa có model để export.")
    else:
        st.warning("Vui lòng kiểm tra lại file CSV đầu vào.")
else:
    st.info("Chờ dữ liệu... Vui lòng đảm bảo các file data_train.csv, data_validation.csv, data_test.csv có sẵn trong thư mục hoặc được upload.")