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
from matplotlib.ticker import FuncFormatter


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Ứng dụng Dự đoán Giá Laptop", layout="wide")


# =========================
# FUNCTIONS
# =========================
def load_data(train_file, val_file, test_file):
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)
    return df_train, df_val, df_test


def align_columns(df_train, df_val, df_test, target="price_base"):
    # Drop title nếu có
    def _drop_title(df):
        if "title" in df.columns:
            return df.drop(columns=["title"])
        return df

    df_train = _drop_title(df_train)
    df_val = _drop_title(df_val)
    df_test = _drop_title(df_test)

    # intersection columns (tránh test thừa cột)
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

    # Ép toàn bộ feature sang numeric để XGB/LGBM không bị crash nếu có cột object
    # (nếu là object -> NaN -> imputer median sẽ xử lý)
    for c in common_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_val[c] = pd.to_numeric(X_val[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    return X_train, y_train, X_val, y_val, X_test, y_test, common_cols


def calculate_metrics(y_true, y_pred):
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Safe MAPE (tránh chia 0 hoặc quá nhỏ)
    eps = 1e-8
    denom = np.maximum(np.abs(y_true), eps)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def build_model(model_type, params):
    """
    Trả về pipeline + cờ early_stop (đúng flow bạn đang dùng)
    """
    if model_type == "Random Forest":
        base_model = RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if int(params["max_depth"]) == 0 else int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=float(params["max_features"]),
            n_jobs=-1,
            random_state=42
        )
        use_early_stop = False

    elif model_type == "XGBoost":
        # Default tối ưu hơn (gần kiểu notebook hay dùng để lên R2)
        base_model = XGBRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            min_child_weight=float(params["min_child_weight"]),
            gamma=float(params["gamma"]),
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",   # nhanh và ổn trên cloud
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
            min_child_samples=int(params["min_child_samples"]),
            random_state=42,
            n_jobs=-1
        )
        use_early_stop = True

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", base_model)
    ])
    return pipeline, use_early_stop


def train_and_eval(model_pipeline, use_early_stop, model_type, X_tr, y_tr, X_vl, y_vl, early_rounds=50):
    """
    FIX XGBoost: chạy được cả khi version xgboost trên Streamlit Cloud khác nhau.
    - Ưu tiên early_stopping_rounds
    - Nếu TypeError -> fallback callbacks
    - Nếu vẫn fail -> fit bình thường (không crash)
    """
    if use_early_stop:
        # Fit imputer trên train -> transform val (chống rò rỉ)
        X_tr_imp = model_pipeline.named_steps["imputer"].fit_transform(X_tr)
        X_vl_imp = model_pipeline.named_steps["imputer"].transform(X_vl)
        model = model_pipeline.named_steps["model"]

        if model_type == "XGBoost":
            try:
                model.fit(
                    X_tr_imp, y_tr,
                    eval_set=[(X_vl_imp, y_vl)],
                    verbose=False,
                    early_stopping_rounds=int(early_rounds)
                )
            except TypeError:
                # Một số bản xgboost thay đổi API -> dùng callback
                try:
                    from xgboost.callback import EarlyStopping
                    cb = EarlyStopping(rounds=int(early_rounds), save_best=True, maximize=False)
                    model.fit(
                        X_tr_imp, y_tr,
                        eval_set=[(X_vl_imp, y_vl)],
                        verbose=False,
                        callbacks=[cb]
                    )
                except Exception:
                    # fallback cuối: train bình thường để app không chết
                    model.fit(X_tr_imp, y_tr)

            y_pred_vl = model.predict(X_vl_imp)
            return model_pipeline, y_pred_vl

        # LightGBM
        try:
            model.fit(
                X_tr_imp, y_tr,
                eval_set=[(X_vl_imp, y_vl)],
                eval_metric="l2",
            )
        except TypeError:
            # có môi trường bị khác chữ ký hàm
            model.fit(X_tr_imp, y_tr)

        y_pred_vl = model.predict(X_vl_imp)
        return model_pipeline, y_pred_vl

    # RandomForest
    model_pipeline.fit(X_tr, y_tr)
    y_pred_vl = model_pipeline.predict(X_vl)
    return model_pipeline, y_pred_vl


def _plain_int_formatter():
    # hiển thị 15690000 thay vì 1.569e7
    return FuncFormatter(lambda x, pos: f"{int(x):d}")


def plot_scatter(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.scatter(y_true, y_pred, alpha=0.5)

    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([mn, mx], [mn, mx], "r--")

    ax.set_title("Giá thật vs Giá dự đoán (Validation)")
    ax.set_xlabel("Giá thật (VND)")
    ax.set_ylabel("Giá dự đoán (VND)")

    ax.xaxis.set_major_formatter(_plain_int_formatter())
    ax.yaxis.set_major_formatter(_plain_int_formatter())
    st.pyplot(fig)


def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(residuals, bins=30)
    ax.set_title("Phân phối sai số (Residuals) - Validation")
    ax.set_xlabel("Sai số (giá thật - giá dự đoán)")
    ax.set_ylabel("Số lượng")

    ax.xaxis.set_major_formatter(_plain_int_formatter())
    st.pyplot(fig)


def plot_feature_importance(model_pipeline, feature_names, top_k=15):
    raw_model = model_pipeline.named_steps["model"]
    if not hasattr(raw_model, "feature_importances_"):
        st.info("Model này không hỗ trợ Feature Importance.")
        return

    importances = raw_model.feature_importances_
    idx = np.argsort(importances)[-top_k:]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    ax.barh(names, vals)
    ax.set_title(f"Top {top_k} Feature quan trọng nhất")
    ax.set_xlabel("Mức độ quan trọng")
    st.pyplot(fig)


def predict_from_csv(trained_model, features, csv_file):
    df_in = pd.read_csv(csv_file)
    X_in = df_in.reindex(columns=features)

    # ép numeric để tránh crash nếu upload có cột object
    for c in features:
        X_in[c] = pd.to_numeric(X_in[c], errors="coerce")

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

early_rounds = None
if model_choice == "XGBoost":
    early_rounds = st.sidebar.slider("Dừng sớm (Early stopping)", 10, 200, 50, 10)
    st.sidebar.caption("Dừng nếu Validation không cải thiện sau N vòng. Giúp nhanh hơn và giảm overfit.")

params = {}

# ===== Random Forest =====
if model_choice == "Random Forest":
    params["n_estimators"] = st.sidebar.slider("Số lượng cây (n_estimators)", 100, 800, 500, 50)
    st.sidebar.caption("Số cây càng nhiều → thường tốt hơn nhưng train lâu hơn. Gợi ý: 400–800.")

    params["max_depth"] = st.sidebar.slider("Độ sâu tối đa (max_depth) - 0 = không giới hạn", 0, 40, 0, 1)
    st.sidebar.caption("Depth lớn → mô hình phức tạp hơn, dễ overfit. Gợi ý: 10–20 hoặc 0 nếu muốn thử.")

    params["min_samples_split"] = st.sidebar.slider("Số mẫu tối thiểu để tách nhánh (min_samples_split)", 2, 20, 2, 1)
    st.sidebar.caption("Tăng lên → giảm overfit (cây ít tách nhánh hơn).")

    params["min_samples_leaf"] = st.sidebar.slider("Số mẫu tối thiểu tại lá (min_samples_leaf)", 1, 20, 1, 1)
    st.sidebar.caption("Tăng lên → ổn định hơn nhưng có thể giảm độ khớp.")

    params["max_features"] = st.sidebar.slider("Tỉ lệ feature mỗi cây (max_features)", 0.2, 1.0, 0.7, 0.05)
    st.sidebar.caption("Giảm xuống giúp chống overfit. Gợi ý: 0.5–0.8.")

# ===== XGBoost =====
elif model_choice == "XGBoost":
    # default “mạnh” hơn để bạn dễ lên R2
    params["n_estimators"] = st.sidebar.slider("Số vòng boosting (n_estimators)", 300, 4000, 2000, 100)
    st.sidebar.caption("Nhiều vòng → mô hình mạnh hơn nhưng chậm hơn. Nên dùng dừng sớm để tự ngắt.")

    params["max_depth"] = st.sidebar.slider("Độ sâu cây (max_depth)", 2, 12, 6, 1)
    st.sidebar.caption("Depth lớn → mạnh hơn nhưng dễ overfit. Gợi ý: 4–8.")

    params["learning_rate"] = st.sidebar.number_input("Tốc độ học (learning_rate)", 0.005, 0.3, 0.03, step=0.005)
    st.sidebar.caption("Learning rate nhỏ → ổn định hơn nhưng cần nhiều vòng hơn. Gợi ý: 0.02–0.08.")

    params["subsample"] = st.sidebar.slider("Tỉ lệ lấy mẫu dữ liệu (subsample)", 0.5, 1.0, 0.9, 0.05)
    st.sidebar.caption("Giảm <1.0 giúp chống overfit.")

    params["colsample_bytree"] = st.sidebar.slider("Tỉ lệ lấy mẫu feature (colsample_bytree)", 0.5, 1.0, 0.9, 0.05)
    st.sidebar.caption("Giảm <1.0 giúp chống overfit.")

    params["min_child_weight"] = st.sidebar.number_input("Min child weight", 0.0, 50.0, 1.0, step=0.5)
    st.sidebar.caption("Tăng lên nếu overfit (yêu cầu node phải đủ ‘nặng’ mới tách).")

    params["gamma"] = st.sidebar.number_input("Gamma", 0.0, 20.0, 0.0, step=0.1)
    st.sidebar.caption("Tăng gamma → khó tách nhánh hơn → giảm overfit.")

    params["reg_alpha"] = st.sidebar.number_input("Phạt L1 (reg_alpha)", 0.0, 10.0, 0.0, step=0.1)
    st.sidebar.caption("Tăng nếu feature nhiễu/overfit.")

    params["reg_lambda"] = st.sidebar.number_input("Phạt L2 (reg_lambda)", 0.0, 10.0, 2.0, step=0.1)
    st.sidebar.caption("Tăng để mô hình ‘mượt’ hơn và giảm overfit.")

# ===== LightGBM =====
else:
    params["n_estimators"] = st.sidebar.slider("Số vòng boosting (n_estimators)", 300, 8000, 3000, 100)
    st.sidebar.caption("Nhiều vòng → có thể tốt hơn nhưng chậm hơn. Gợi ý: 1500–4000.")

    params["max_depth"] = st.sidebar.slider("Độ sâu tối đa (max_depth) - 0 = không giới hạn", 0, 30, 0, 1)
    st.sidebar.caption("Giới hạn depth để tránh overfit. Gợi ý: 6–12 hoặc 0 nếu muốn thử.")

    params["learning_rate"] = st.sidebar.number_input("Tốc độ học (learning_rate)", 0.005, 0.3, 0.03, step=0.005)
    st.sidebar.caption("Nhỏ hơn → ổn định hơn nhưng cần nhiều vòng hơn. Gợi ý: 0.02–0.08.")

    params["num_leaves"] = st.sidebar.slider("Số lá tối đa (num_leaves)", 15, 255, 63, 2)
    st.sidebar.caption("num_leaves lớn → mô hình mạnh hơn nhưng dễ overfit. Gợi ý: 31–127.")

    params["subsample"] = st.sidebar.slider("Tỉ lệ lấy mẫu dữ liệu (subsample)", 0.5, 1.0, 0.9, 0.05)
    st.sidebar.caption("Giảm <1.0 giúp chống overfit.")

    params["colsample_bytree"] = st.sidebar.slider("Tỉ lệ lấy mẫu feature (colsample_bytree)", 0.5, 1.0, 0.9, 0.05)
    st.sidebar.caption("Giảm <1.0 giúp chống overfit.")

    params["min_child_samples"] = st.sidebar.slider("Min child samples", 5, 200, 20, 5)
    st.sidebar.caption("Tăng lên nếu overfit (lá phải có đủ mẫu mới tách).")

    params["reg_alpha"] = st.sidebar.number_input("Phạt L1 (reg_alpha)", 0.0, 10.0, 0.0, step=0.1)
    st.sidebar.caption("Tăng nếu dữ liệu nhiễu/overfit.")

    params["reg_lambda"] = st.sidebar.number_input("Phạt L2 (reg_lambda)", 0.0, 10.0, 0.0, step=0.1)
    st.sidebar.caption("Tăng nếu muốn mô hình ổn định hơn.")


# =========================
# MAIN
# =========================
st.title("💻 Ứng dụng Dự đoán Giá Laptop")

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
    st.subheader("Xem nhanh dữ liệu (head)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Train")
        st.write("Kích thước:", df_tr.shape)
        st.dataframe(df_tr.head(10), use_container_width=True)
    with c2:
        st.markdown("### Validation")
        st.write("Kích thước:", df_vl.shape)
        st.dataframe(df_vl.head(10), use_container_width=True)
    with c3:
        st.markdown("### Test")
        st.write("Kích thước:", df_ts.shape)
        st.dataframe(df_ts.head(10), use_container_width=True)

    st.info(f"Đã lấy **cột chung (intersection)** giữa train/val/test: **{len(features)} features** (đã loại 'title' và target).")


with tab2:
    st.subheader("Huấn luyện mô hình")
    start_train = st.button("🚀 Bắt đầu huấn luyện", type="primary")

    if start_train:
        with st.spinner("Đang huấn luyện..."):
            model_pipeline, use_es = build_model(model_choice, params)
            model_pipeline, y_pred_vl = train_and_eval(
                model_pipeline, use_es, model_choice, X_tr, y_tr, X_vl, y_vl,
                early_rounds=(early_rounds if early_rounds is not None else 50)
            )

            st.session_state["trained_model"] = model_pipeline
            st.session_state["features"] = features
            st.session_state["y_pred_vl"] = y_pred_vl

            metrics_vl = calculate_metrics(y_vl.values, y_pred_vl)
            st.session_state["metrics_vl"] = metrics_vl

        st.success("✅ Huấn luyện hoàn tất!")

    if "trained_model" in st.session_state:
        st.markdown("### Kết quả trên Validation")
        metrics_vl = st.session_state["metrics_vl"]

        m_cols = st.columns(4)
        keys = list(metrics_vl.keys())
        for i, k in enumerate(keys):
            m_cols[i].metric(k, f"{metrics_vl[k]:,.4f}" if k == "R2" else f"{metrics_vl[k]:,.0f}" if k in ["MAE","RMSE"] else f"{metrics_vl[k]:,.2f}")

        # chú thích metric
        st.caption("**R2**: càng gần 1 càng tốt (mô hình giải thích được biến động giá).")
        st.caption("**MAE**: sai số tuyệt đối trung bình (VND) — càng nhỏ càng tốt.")
        st.caption("**RMSE**: giống MAE nhưng phạt nặng lỗi lớn (VND) — càng nhỏ càng tốt.")
        st.caption("**MAPE**: % sai số trung bình so với giá thật — càng nhỏ càng tốt.")

        st.markdown("### Biểu đồ (1 ảnh / 1 hàng)")
        plot_scatter(y_vl.values, st.session_state["y_pred_vl"])
        plot_residuals(y_vl.values, st.session_state["y_pred_vl"])
        st.markdown("#### Feature Importance")
        plot_feature_importance(st.session_state["trained_model"], features, top_k=15)

        st.divider()
        if st.button("🔍 Đánh giá thêm trên Test"):
            y_pred_ts = st.session_state["trained_model"].predict(X_ts)
            metrics_ts = calculate_metrics(y_ts.values, y_pred_ts)
            st.markdown("### Kết quả trên Test")
            st.json(metrics_ts)
            st.session_state["test_pred"] = y_pred_ts
    else:
        st.info("Bấm **Bắt đầu huấn luyện** để train model.")


with tab3:
    st.subheader("Dự đoán bằng Upload CSV (không nhập tay)")
    if "trained_model" not in st.session_state:
        st.info("Bạn cần huấn luyện model trước.")
    else:
        st.write("✅ Upload 1 file CSV để dự đoán (1 dòng hoặc nhiều dòng).")
        st.write("App sẽ tự **align cột** theo features lúc train (thừa cột bỏ, thiếu cột → NaN và imputer xử lý).")

        pred_file = st.file_uploader("Upload CSV để dự đoán", type="csv", key="pred_csv")
        if pred_file:
            out_df = predict_from_csv(st.session_state["trained_model"], st.session_state["features"], pred_file)
            st.success("✅ Dự đoán xong!")
            st.dataframe(out_df.head(20), use_container_width=True)

            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Tải file dự đoán", csv_bytes, "predictions.csv", "text/csv")


with tab4:
    st.subheader("Xuất model và dự đoán test")
    if "trained_model" not in st.session_state:
        st.info("Bạn cần huấn luyện model trước.")
    else:
        joblib.dump(st.session_state["trained_model"], "model.joblib")
        with open("model.joblib", "rb") as f:
            st.download_button("💾 Tải model.joblib", f, "model.joblib")

        st.divider()

        if "test_pred" not in st.session_state:
            st.info("Hãy bấm **Đánh giá thêm trên Test** ở tab Huấn luyện để tạo file dự đoán test.")
        else:
            test_results = df_ts.copy()
            test_results["predicted_price_base"] = st.session_state["test_pred"]
            csv = test_results.to_csv(index=False).encode("utf-8")
            st.download_button("📊 Tải test_predictions.csv", csv, "test_predictions.csv", "text/csv")
