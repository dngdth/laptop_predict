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


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Ứng dụng Dự đoán Giá Laptop", layout="wide")


# =========================
# FUNCTIONS
# =========================
def load_data(train_file, val_file, test_file):
    """Đọc 3 file CSV (upload hoặc file local)."""
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)
    return df_train, df_val, df_test


def align_columns(df_train, df_val, df_test, target="price_base"):
    """
    - Drop title nếu có
    - Lấy intersection columns (tránh lệch schema vì test có thể thừa cột)
    - Sort cột để ổn định
    """
    def _drop_title(df):
        if "title" in df.columns:
            return df.drop(columns=["title"])
        return df

    df_train = _drop_title(df_train)
    df_val = _drop_title(df_val)
    df_test = _drop_title(df_test)

    # intersection
    common_cols = list(set(df_train.columns) & set(df_val.columns) & set(df_test.columns))
    if target not in df_train.columns or target not in df_val.columns or target not in df_test.columns:
        raise ValueError(f"Thiếu cột target '{target}' trong 1 trong 3 tập dữ liệu.")

    if target in common_cols:
        common_cols.remove(target)

    common_cols.sort()

    X_train = df_train[common_cols].copy()
    y_train = df_train[target].copy()

    X_val = df_val[common_cols].copy()
    y_val = df_val[target].copy()

    X_test = df_test[common_cols].copy()
    y_test = df_test[target].copy()

    return X_train, y_train, X_val, y_val, X_test, y_test, common_cols


def calculate_metrics(y_true, y_pred):
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Safe MAPE (tránh chia 0 hoặc quá nhỏ)
    eps = 1e-8
    denom = np.maximum(np.abs(y_true), eps)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

    # Accuracy: % dự đoán trong ±10%
    diff_ratio = np.abs(y_true - y_pred) / denom
    accuracy = float(np.mean(diff_ratio <= 0.10) * 100)

    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "Accuracy (±10%)": accuracy
    }


def build_model(model_type, params):
    """Khởi tạo model theo loại + tham số."""
    if model_type == "Random Forest":
        base_model = RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if int(params["max_depth"]) == 0 else int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            n_jobs=-1,
            random_state=42
        )
        use_early_stop = False

    elif model_type == "XGBoost":
        base_model = XGBRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42
        )
        use_early_stop = True

    else:  # LightGBM
        base_model = LGBMRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=-1 if int(params["max_depth"]) == 0 else int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            num_leaves=int(params["num_leaves"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            n_jobs=-1,
            random_state=42
        )
        use_early_stop = True

    # Pipeline: Imputer fit trên train -> transform val/test (không rò rỉ)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", base_model)
    ])

    return pipeline, use_early_stop


def train_and_eval(model_pipeline, use_early_stop, model_type, X_tr, y_tr, X_vl, y_vl, early_rounds=50):
    """
    Train trên train, eval trên validation.
    XGB/LGBM dùng early stopping để giảm thời gian train.
    """
    if use_early_stop:
        # Fit imputer trước (fit trên train)
        X_tr_imp = model_pipeline.named_steps["imputer"].fit_transform(X_tr)
        X_vl_imp = model_pipeline.named_steps["imputer"].transform(X_vl)

        model = model_pipeline.named_steps["model"]

        if model_type == "XGBoost":
            model.fit(
                X_tr_imp, y_tr,
                eval_set=[(X_vl_imp, y_vl)],
                verbose=False,
                early_stopping_rounds=int(early_rounds)
            )
        else:  # LightGBM
            model.fit(
                X_tr_imp, y_tr,
                eval_set=[(X_vl_imp, y_vl)],
                eval_metric="l2",
                callbacks=[],
            )
            # LightGBM native early stopping:
            # Nếu bạn muốn chặt hơn, có thể bật callback early_stopping
            # nhưng nhiều môi trường Streamlit Cloud hạn chế log callback -> giữ đơn giản.

        # Predict validation
        y_pred_vl = model.predict(X_vl_imp)
        return model_pipeline, y_pred_vl

    # RF: fit pipeline bình thường
    model_pipeline.fit(X_tr, y_tr)
    y_pred_vl = model_pipeline.predict(X_vl)
    return model_pipeline, y_pred_vl


def plot_scatter(y_true, y_pred):
    fig = plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.5)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.title("So sánh Giá Thật vs Giá Dự đoán (Validation)")
    plt.xlabel("Giá thật")
    plt.ylabel("Giá dự đoán")
    st.pyplot(fig)


def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig = plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=30)
    plt.title("Phân phối Sai số (Residuals) - Validation")
    plt.xlabel("Sai số (giá thật - giá dự đoán)")
    plt.ylabel("Số lượng")
    st.pyplot(fig)


def plot_feature_importance(model_pipeline, feature_names, top_k=15):
    raw_model = model_pipeline.named_steps["model"]
    if not hasattr(raw_model, "feature_importances_"):
        st.info("Model này không hỗ trợ Feature Importance.")
        return

    importances = raw_model.feature_importances_
    idx = np.argsort(importances)[-top_k:]  # top_k
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig = plt.figure(figsize=(7, 6))
    plt.barh(names, vals)
    plt.title(f"Top {top_k} Feature Importance")
    plt.xlabel("Mức độ quan trọng")
    st.pyplot(fig)


def predict_from_csv(trained_model, features, csv_file):
    """Predict từ file CSV: tự align cột theo features, thiếu -> NaN (imputer sẽ xử lý)."""
    df_in = pd.read_csv(csv_file)
    X_in = df_in.reindex(columns=features)  # thiếu cột => NaN, thừa cột => drop
    preds = trained_model.predict(X_in)
    out = df_in.copy()
    out["predicted_price_base"] = preds
    return out


# =========================
# SIDEBAR (VN)
# =========================
st.sidebar.header("1) Dữ liệu đầu vào")

train_up = st.sidebar.file_uploader("Upload data_train.csv", type="csv")
val_up = st.sidebar.file_uploader("Upload data_validation.csv", type="csv")
test_up = st.sidebar.file_uploader("Upload data_test.csv", type="csv")

train_path = train_up if train_up else "data_train.csv"
val_path = val_up if val_up else "data_validation.csv"
test_path = test_up if test_up else "data_test.csv"

st.sidebar.header("2) Chọn mô hình & tham số")

model_choice = st.sidebar.selectbox("Mô hình", ["Random Forest", "XGBoost", "LightGBM"])

fast_mode = st.sidebar.checkbox("⚡ Huấn luyện nhanh (khuyến nghị)", value=True)
early_rounds = st.sidebar.slider("Early stopping (XGB)", 10, 200, 50, 10)

params = {}

if model_choice == "Random Forest":
    # giảm range để train nhanh
    params["n_estimators"] = st.sidebar.slider("n_estimators", 50, 600, 200 if fast_mode else 400, 50)
    params["max_depth"] = st.sidebar.slider("max_depth (0 = None)", 0, 30, 0 if fast_mode else 12, 1)
    params["min_samples_split"] = st.sidebar.slider("min_samples_split", 2, 20, 2, 1)
    params["min_samples_leaf"] = st.sidebar.slider("min_samples_leaf", 1, 20, 1, 1)

elif model_choice == "XGBoost":
    params["n_estimators"] = st.sidebar.slider("n_estimators", 200, 2500, 600 if fast_mode else 1500, 100)
    params["max_depth"] = st.sidebar.slider("max_depth", 2, 12, 6 if fast_mode else 8, 1)
    params["learning_rate"] = st.sidebar.number_input("learning_rate", 0.005, 0.3, 0.05 if fast_mode else 0.03, step=0.005)
    params["subsample"] = st.sidebar.slider("subsample", 0.5, 1.0, 0.9, 0.05)
    params["colsample_bytree"] = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.9, 0.05)
    params["reg_alpha"] = st.sidebar.number_input("reg_alpha", 0.0, 10.0, 0.0, step=0.1)
    params["reg_lambda"] = st.sidebar.number_input("reg_lambda", 0.0, 10.0, 1.0, step=0.1)

else:  # LightGBM
    params["n_estimators"] = st.sidebar.slider("n_estimators", 200, 5000, 800 if fast_mode else 2500, 100)
    params["max_depth"] = st.sidebar.slider("max_depth (0 = -1)", 0, 30, 0 if fast_mode else 10, 1)
    params["learning_rate"] = st.sidebar.number_input("learning_rate", 0.005, 0.3, 0.05 if fast_mode else 0.03, step=0.005)
    params["num_leaves"] = st.sidebar.slider("num_leaves", 15, 127, 31 if fast_mode else 63, 2)
    params["subsample"] = st.sidebar.slider("subsample", 0.5, 1.0, 0.9, 0.05)
    params["colsample_bytree"] = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.9, 0.05)
    params["reg_alpha"] = st.sidebar.number_input("reg_alpha", 0.0, 10.0, 0.0, step=0.1)
    params["reg_lambda"] = st.sidebar.number_input("reg_lambda", 0.0, 10.0, 0.0, step=0.1)


# =========================
# MAIN
# =========================
st.title("💻 Ứng dụng Dự đoán Giá Laptop")

# Check file exist (local) or is uploader-like
def _available(p):
    return (hasattr(p, "read")) or os.path.exists(str(p))

if not all(_available(p) for p in [train_path, val_path, test_path]):
    st.info("Vui lòng đặt đủ 3 file: data_train.csv, data_validation.csv, data_test.csv (hoặc upload ở sidebar).")
    st.stop()

try:
    df_tr, df_vl, df_ts = load_data(train_path, val_path, test_path)
    X_tr, y_tr, X_vl, y_vl, X_ts, y_ts, features = align_columns(df_tr, df_vl, df_ts)
except Exception as e:
    st.error(f"Lỗi xử lý dữ liệu: {e}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📌 Xem dữ liệu", "🧠 Huấn luyện & Đánh giá", "🎯 Dự đoán", "📤 Xuất file"])


with tab1:
    st.subheader("Xem nhanh dữ liệu (head) + thống kê NaN")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Train")
        st.write("Kích thước:", df_tr.shape)
        st.dataframe(df_tr.head(10), use_container_width=True)
        st.write("Tổng NaN:", int(df_tr.isna().sum().sum()))
    with c2:
        st.markdown("### Validation")
        st.write("Kích thước:", df_vl.shape)
        st.dataframe(df_vl.head(10), use_container_width=True)
        st.write("Tổng NaN:", int(df_vl.isna().sum().sum()))
    with c3:
        st.markdown("### Test")
        st.write("Kích thước:", df_ts.shape)
        st.dataframe(df_ts.head(10), use_container_width=True)
        st.write("Tổng NaN:", int(df_ts.isna().sum().sum()))

    st.info(f"Đã lấy **cột chung (intersection)** giữa train/val/test: **{len(features)} features** (đã loại 'title' và target).")


with tab2:
    st.subheader("Huấn luyện mô hình")

    start_train = st.button("🚀 Bắt đầu huấn luyện", type="primary")

    if start_train:
        with st.spinner("Đang huấn luyện..."):
            model_pipeline, use_es = build_model(model_choice, params)
            model_pipeline, y_pred_vl = train_and_eval(
                model_pipeline, use_es, model_choice, X_tr, y_tr, X_vl, y_vl, early_rounds=early_rounds
            )

            # Save state
            st.session_state["trained_model"] = model_pipeline
            st.session_state["features"] = features
            st.session_state["y_pred_vl"] = y_pred_vl

            metrics_vl = calculate_metrics(y_vl.values, y_pred_vl)
            st.session_state["metrics_vl"] = metrics_vl

        st.success("✅ Huấn luyện hoàn tất!")

    if "trained_model" in st.session_state:
        st.markdown("### Kết quả trên Validation")
        metrics_vl = st.session_state["metrics_vl"]

        m_cols = st.columns(5)
        keys = list(metrics_vl.keys())
        for i, k in enumerate(keys):
            m_cols[i].metric(k, f"{metrics_vl[k]:,.2f}")

        st.markdown("### Biểu đồ (1 ảnh / 1 hàng)")

        # 1) Scatter
        plot_scatter(y_vl.values, st.session_state["y_pred_vl"])

        # 2) Residuals
        plot_residuals(y_vl.values, st.session_state["y_pred_vl"])

        # 3) Feature importance
        st.markdown("#### Feature Importance")
        plot_feature_importance(st.session_state["trained_model"], features, top_k=15)

        st.divider()
        if st.button("🔍 Đánh giá thêm trên Test"):
            y_pred_ts = st.session_state["trained_model"].predict(X_ts)
            metrics_ts = calculate_metrics(y_ts.values, y_pred_ts)
            st.markdown("### Kết quả trên Test")
            st.json(metrics_ts)

            # store for export
            st.session_state["test_pred"] = y_pred_ts
    else:
        st.info("Bấm **Bắt đầu huấn luyện** để train model.")


with tab3:
    st.subheader("Dự đoán bằng Upload CSV (không nhập tay)")

    if "trained_model" not in st.session_state:
        st.info("Bạn cần huấn luyện model trước.")
    else:
        st.write("✅ Upload 1 file CSV để dự đoán. File có thể có **1 dòng hoặc nhiều dòng**.")
        st.write("App sẽ tự **align cột** theo đúng features lúc train (thừa cột sẽ bỏ, thiếu cột sẽ để NaN và imputer xử lý).")

        pred_file = st.file_uploader("Upload CSV để dự đoán", type="csv", key="pred_csv")
        if pred_file:
            out_df = predict_from_csv(st.session_state["trained_model"], st.session_state["features"], pred_file)
            st.success("✅ Dự đoán xong!")
            st.dataframe(out_df.head(20), use_container_width=True)

            # download
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Tải file dự đoán", csv_bytes, "predictions.csv", "text/csv")


with tab4:
    st.subheader("Xuất model và dự đoán test")

    if "trained_model" not in st.session_state:
        st.info("Bạn cần huấn luyện model trước.")
    else:
        # Export model
        buf = joblib.dump(st.session_state["trained_model"], "model.joblib")
        with open("model.joblib", "rb") as f:
            st.download_button("💾 Tải model.joblib", f, "model.joblib")

        st.divider()

        # Export test predictions (nếu đã chạy)
        if "test_pred" not in st.session_state:
            st.info("Hãy bấm **Đánh giá thêm trên Test** ở tab Huấn luyện để tạo file dự đoán test.")
        else:
            test_results = df_ts.copy()
            test_results["predicted_price_base"] = st.session_state["test_pred"]
            csv = test_results.to_csv(index=False).encode("utf-8")
            st.download_button("📊 Tải test_predictions.csv", csv, "test_predictions.csv", "text/csv")
