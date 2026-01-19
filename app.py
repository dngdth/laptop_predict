import streamlit as st
import inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Ứng dụng Dự đoán Giá Laptop", layout="wide")

# =========================
# UTILS UI
# =========================
def muted(text: str):
    st.markdown(f"<span style='color:#8a8a8a; font-size: 0.9rem'>{text}</span>", unsafe_allow_html=True)

def fmt_money(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)

# =========================
# FUNCTIONS
# =========================
def read_csv_safely(file_or_path):
    try:
        return pd.read_csv(file_or_path, encoding="utf-8")
    except Exception:
        return pd.read_csv(file_or_path, encoding="latin-1")


def load_data(train_file, val_file, test_file):
    df_train = read_csv_safely(train_file)
    df_val = read_csv_safely(val_file)
    df_test = read_csv_safely(test_file)
    return df_train, df_val, df_test


def align_columns(df_train, df_val, df_test, target="price_base"):
    def _drop_title(df):
        if "title" in df.columns:
            return df.drop(columns=["title"])
        return df

    df_train = _drop_title(df_train)
    df_val = _drop_title(df_val)
    df_test = _drop_title(df_test)

    if target not in df_train.columns or target not in df_val.columns or target not in df_test.columns:
        raise ValueError(f"Thiếu cột target '{target}' trong 1 trong 3 tập dữ liệu.")

    common_cols = list(set(df_train.columns) & set(df_val.columns) & set(df_test.columns))
    if target in common_cols:
        common_cols.remove(target)
    common_cols.sort()

    X_train = df_train[common_cols].copy()
    y_train = df_train[target].copy()

    X_val = df_val[common_cols].copy()
    y_val = df_val[target].copy()

    X_test = df_test[common_cols].copy()
    y_test = df_test[target].copy()

    for c in common_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_val[c] = pd.to_numeric(X_val[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # target to numeric (if string)
    y_train = pd.to_numeric(y_train, errors="coerce")
    y_val = pd.to_numeric(y_val, errors="coerce")
    y_test = pd.to_numeric(y_test, errors="coerce")

    # drop rows where y is NaN
    tr_mask = ~y_train.isna()
    vl_mask = ~y_val.isna()
    ts_mask = ~y_test.isna()

    X_train, y_train = X_train.loc[tr_mask], y_train.loc[tr_mask]
    X_val, y_val = X_val.loc[vl_mask], y_val.loc[vl_mask]
    X_test, y_test = X_test.loc[ts_mask], y_test.loc[ts_mask]

    return X_train, y_train, X_val, y_val, X_test, y_test, common_cols


def calculate_metrics_row(name, y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    eps = 1e-8
    denom = np.maximum(np.abs(y_true), eps)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    return {"Dataset": name, "R2": r2, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def build_model(model_type, params):
    """
    - RandomForest: train trực tiếp.
    - XGBoost/LightGBM: train theo notebook -> dùng log1p(price_base), predict -> expm1.
    """
    if model_type == "Random Forest":
        base_model = RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if int(params["max_depth"]) == 0 else int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=float(params["max_features"]),
            n_jobs=-1,
            random_state=42,
        )
        use_early_stop = False
        use_log_target = False

    elif model_type == "XGBoost":
        # theo notebook model_XGBoost.ipynb: log1p target + giảm overfit mạnh
        base_model = XGBRegressor(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]),
            gamma=float(params["gamma"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        use_early_stop = True
        use_log_target = True

    else:  # LightGBM
        # theo ý tưởng notebook Tu_LightGBM.ipynb: log1p target + early stopping
        base_model = LGBMRegressor(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=-1 if int(params["max_depth"]) == 0 else int(params["max_depth"]),
            num_leaves=int(params["num_leaves"]),
            min_child_samples=int(params["min_child_samples"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            random_state=42,
            n_jobs=-1,
        )
        use_early_stop = True
        use_log_target = True

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", base_model),
        ]
    )
    return pipeline, use_early_stop, use_log_target


def _safe_log1p(y):
    y = np.asarray(y, dtype=float)
    y = np.where(y < 0, 0.0, y)
    return np.log1p(y)


def predict_price(model_pipeline, X, use_log_target: bool):
    pred = model_pipeline.predict(X)
    if use_log_target:
        pred = np.expm1(pred)
    return pred


def train_model(pipeline, use_early_stop, use_log_target, model_type,
                X_tr, y_tr, X_vl, y_vl, early_rounds=100):

    y_tr_fit = _safe_log1p(y_tr) if use_log_target else np.asarray(y_tr, dtype=float)
    y_vl_fit = _safe_log1p(y_vl) if use_log_target else np.asarray(y_vl, dtype=float)

    # ép y về 1D numpy (xgboost rất hay kén shape)
    y_tr_fit = np.asarray(y_tr_fit, dtype=np.float32).ravel()
    y_vl_fit = np.asarray(y_vl_fit, dtype=np.float32).ravel()

    if not use_early_stop:
        pipeline.fit(X_tr, y_tr_fit)
        return pipeline

    # Early stopping: fit imputer trước
    X_tr_imp = pipeline.named_steps["imputer"].fit_transform(X_tr)
    X_vl_imp = pipeline.named_steps["imputer"].transform(X_vl)

    # ép dtype float32 (xgboost trên cloud hay kén float64/object)
    X_tr_imp = np.asarray(X_tr_imp, dtype=np.float32)
    X_vl_imp = np.asarray(X_vl_imp, dtype=np.float32)

    model = pipeline.named_steps["model"]

    # ========= XGBOOST (fix compat) =========
    if model_type == "XGBoost":
        fit_params = inspect.signature(model.fit).parameters

        # Case A: hỗ trợ early_stopping_rounds (phổ biến)
        if "early_stopping_rounds" in fit_params:
            try:
                model.fit(
                    X_tr_imp, y_tr_fit,
                    eval_set=[(X_vl_imp, y_vl_fit)],
                    verbose=False,
                    early_stopping_rounds=int(early_rounds),
                )
                return pipeline
            except TypeError:
                pass
            except Exception:
                # nếu fail vì lý do khác thì thử phương án khác bên dưới
                pass

        # Case B: hỗ trợ callbacks (một số version)
        if "callbacks" in fit_params:
            try:
                from xgboost.callback import EarlyStopping
                cb = EarlyStopping(rounds=int(early_rounds), save_best=True)
                model.fit(
                    X_tr_imp, y_tr_fit,
                    eval_set=[(X_vl_imp, y_vl_fit)],
                    verbose=False,
                    callbacks=[cb],
                )
                return pipeline
            except Exception:
                pass

        # Case C: fallback cuối cùng - train thường (không early stopping) để app không crash
        model.fit(X_tr_imp, y_tr_fit)
        return pipeline

    # ========= LIGHTGBM =========
    try:
        model.fit(
            X_tr_imp, y_tr_fit,
            eval_set=[(X_vl_imp, y_vl_fit)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(int(early_rounds), verbose=False)],
        )
    except Exception:
        model.fit(X_tr_imp, y_tr_fit)

    return pipeline


def plot_feature_importance(model_pipeline, feature_names, top_k=15):
    raw_model = model_pipeline.named_steps["model"]
    if not hasattr(raw_model, "feature_importances_"):
        st.info("Model này không hỗ trợ feature_importances_.")
        return

    importances = np.asarray(raw_model.feature_importances_, dtype=float)
    if importances.size != len(feature_names):
        st.info("Không khớp số lượng feature_importances_ với feature_names.")
        return

    top_k = int(min(top_k, len(feature_names)))
    idx = np.argsort(importances)[-top_k:]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(names, vals)
    ax.set_title(f"Top {top_k} Feature Importances")
    ax.set_xlabel("Relative Importance")
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    st.pyplot(fig)


def plot_residuals(y_true, y_pred):
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.hist(err, bins=40)
    ax.axvline(0, linestyle="--")
    ax.set_title("Residuals Distribution (Sai số)")
    ax.set_xlabel("Error Amount")
    ax.set_ylabel("Count")
    fig.tight_layout()
    st.pyplot(fig)


def plot_actual_vs_pred(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4.9))
    ax.scatter(y_true, y_pred, alpha=0.35)
    mn = float(np.min([y_true.min(), y_pred.min()]))
    mx = float(np.max([y_true.max(), y_pred.max()]))
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=2)
    ax.set_title("Actual vs Predicted Prices")
    ax.set_xlabel("Actual Price (VND)")
    ax.set_ylabel("Predicted Price (VND)")
    fig.tight_layout()
    st.pyplot(fig)


def predict_from_csv(trained_model, features, csv_file, use_log_target):
    df_in = read_csv_safely(csv_file)
    if df_in.shape[0] == 0:
        raise ValueError("File CSV rỗng (0 dòng).")

    X_in = df_in.reindex(columns=features)
    for c in features:
        X_in[c] = pd.to_numeric(X_in[c], errors="coerce")
    X_in = X_in.replace([np.inf, -np.inf], np.nan)

    preds = predict_price(trained_model, X_in, use_log_target)
    out = df_in.copy()
    out["predicted_price_base"] = preds
    return out


# =========================
# SIDEBAR
# =========================
st.sidebar.header("1) Dữ liệu đầu vào")
train_up = st.sidebar.file_uploader("Upload data_train.csv", type="csv")
val_up = st.sidebar.file_uploader("Upload data_validation.csv", type="csv")
test_up = st.sidebar.file_uploader("Upload data_test.csv", type="csv")

train_path = train_up if train_up else "data_train.csv"
val_path = val_up if val_up else "data_validation.csv"
test_path = test_up if test_up else "data_test.csv"

st.sidebar.header("2) Chọn mô hình")
model_choice = st.sidebar.selectbox("Mô hình", ["Random Forest", "XGBoost", "LightGBM"])

early_rounds = None
if model_choice in ["XGBoost", "LightGBM"]:
    early_rounds = st.sidebar.slider("Early stopping rounds", 20, 300, 100, 10)
    st.sidebar.caption("Tăng lên → model “kiên nhẫn” hơn; giảm → dừng sớm hơn để tránh overfit.")

params = {}

st.sidebar.divider()
st.sidebar.subheader("3) Tham số (có gợi ý)")

if model_choice == "Random Forest":
    params["n_estimators"] = st.sidebar.slider("n_estimators", 200, 1200, 600, 50)
    st.sidebar.caption("Số cây. Tăng → ổn định hơn nhưng chậm. Thường 400–900 là hợp lý.")

    params["max_depth"] = st.sidebar.slider("max_depth (0 = None)", 0, 40, 20, 1)
    st.sidebar.caption("Độ sâu tối đa. Tăng → dễ overfit. Với kết quả Train cao/Test thấp → nên đặt 12–25.")

    params["min_samples_split"] = st.sidebar.slider("min_samples_split", 2, 30, 4, 1)
    st.sidebar.caption("Tăng → khó tách nhánh hơn → giảm overfit. Gợi ý: 3–10.")

    params["min_samples_leaf"] = st.sidebar.slider("min_samples_leaf", 1, 30, 2, 1)
    st.sidebar.caption("Tăng → mỗi lá cần nhiều mẫu hơn → mượt hơn. Gợi ý: 1–6.")

    params["max_features"] = st.sidebar.slider("max_features", 0.2, 1.0, 0.7, 0.05)
    st.sidebar.caption("Tỉ lệ feature mỗi cây. 0.6–0.9 thường cho Test R2 tốt hơn.")

elif model_choice == "XGBoost":
    params["n_estimators"] = st.sidebar.slider("n_estimators", 500, 6000, 2000, 100)
    st.sidebar.caption("Kết hợp với learning_rate nhỏ. Early stopping sẽ tự chọn best iteration.")

    params["learning_rate"] = st.sidebar.number_input("learning_rate", 0.005, 0.3, 0.01, step=0.005)
    st.sidebar.caption("Giảm → học chậm nhưng bền, hay cho Test tốt hơn (0.01–0.05).")

    params["max_depth"] = st.sidebar.slider("max_depth", 2, 12, 4, 1)
    st.sidebar.caption("Giảm → chống overfit mạnh. Notebook của bạn dùng 4.")

    params["min_child_weight"] = st.sidebar.number_input("min_child_weight", 1.0, 50.0, 5.0, step=0.5)
    st.sidebar.caption("Tăng → khó tách nhánh → giảm overfit. Notebook dùng 5.")

    params["gamma"] = st.sidebar.number_input("gamma", 0.0, 20.0, 0.2, step=0.1)
    st.sidebar.caption("Tăng → cần giảm loss đủ lớn mới split → bớt overfit. Notebook dùng 0.2.")

    params["subsample"] = st.sidebar.slider("subsample", 0.5, 1.0, 0.6, 0.05)
    st.sidebar.caption("Giảm (0.6–0.9) → chống overfit. Notebook dùng 0.6.")

    params["colsample_bytree"] = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.6, 0.05)
    st.sidebar.caption("Giảm → mỗi cây nhìn ít feature hơn → bớt overfit. Notebook dùng 0.6.")

    params["reg_alpha"] = st.sidebar.number_input("reg_alpha", 0.0, 10.0, 1.0, step=0.1)
    st.sidebar.caption("L1 regularization. Tăng nhẹ → lọc feature nhiễu. Notebook dùng 1.0.")

    params["reg_lambda"] = st.sidebar.number_input("reg_lambda", 0.0, 20.0, 2.0, step=0.1)
    st.sidebar.caption("L2 regularization. Tăng → model mượt hơn. Notebook dùng 2.0.")

else:  # LightGBM
    params["n_estimators"] = st.sidebar.slider("n_estimators", 500, 12000, 4000, 100)
    st.sidebar.caption("Early stopping sẽ chọn best iteration, bạn có thể để lớn.")

    params["learning_rate"] = st.sidebar.number_input("learning_rate", 0.005, 0.3, 0.03, step=0.005)
    st.sidebar.caption("0.01–0.05 thường ổn. Nhỏ hơn → cần nhiều cây hơn.")

    params["max_depth"] = st.sidebar.slider("max_depth (0 = -1)", 0, 30, 0, 1)
    st.sidebar.caption("0 = không giới hạn. Nếu overfit → thử 6–16.")

    params["num_leaves"] = st.sidebar.slider("num_leaves", 15, 255, 63, 2)
    st.sidebar.caption("Tăng → mạnh hơn nhưng dễ overfit. Nếu Test thấp → giảm (31–127).")

    params["min_child_samples"] = st.sidebar.slider("min_child_samples", 5, 200, 20, 5)
    st.sidebar.caption("Tăng → mỗi lá cần nhiều mẫu hơn → giảm overfit (20–80).")

    params["subsample"] = st.sidebar.slider("subsample", 0.5, 1.0, 0.9, 0.05)
    st.sidebar.caption("Giảm nhẹ (0.7–0.9) giúp chống overfit.")

    params["colsample_bytree"] = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.9, 0.05)
    st.sidebar.caption("Giảm nhẹ (0.7–0.9) giúp generalize tốt hơn.")

    params["reg_alpha"] = st.sidebar.number_input("reg_alpha", 0.0, 10.0, 0.0, step=0.1)
    st.sidebar.caption("L1. Tăng nhẹ nếu feature nhiễu.")

    params["reg_lambda"] = st.sidebar.number_input("reg_lambda", 0.0, 10.0, 0.0, step=0.1)
    st.sidebar.caption("L2. Tăng nhẹ để mượt và bớt overfit.")

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

    st.info(f"Đã lấy intersection features: {len(features)} cột (đã loại 'title' và target).")

with tab2:
    st.subheader("Huấn luyện mô hình")
    muted("Gợi ý: Nếu Train R2 rất cao nhưng Test R2 thấp → mô hình đang overfit. Hãy giảm độ phức tạp (max_depth ↓, min_samples_leaf ↑, subsample/colsample ↓, regularization ↑).")

    start_train = st.button("🚀 Bắt đầu huấn luyện", type="primary")

    if start_train:
        with st.spinner("Đang huấn luyện..."):
            model_pipeline, use_es, use_log_target = build_model(model_choice, params)
            model_pipeline = train_model(
                model_pipeline,
                use_es,
                use_log_target,
                model_choice,
                X_tr, y_tr, X_vl, y_vl,
                early_rounds=(early_rounds if early_rounds is not None else 100),
            )

            st.session_state["trained_model"] = model_pipeline
            st.session_state["features"] = features
            st.session_state["use_log_target"] = use_log_target

            # predict all (inverse if needed)
            y_pred_tr = predict_price(model_pipeline, X_tr, use_log_target)
            y_pred_vl = predict_price(model_pipeline, X_vl, use_log_target)
            y_pred_ts = predict_price(model_pipeline, X_ts, use_log_target)

            rows = [
                calculate_metrics_row("Train", y_tr.values, y_pred_tr),
                calculate_metrics_row("Validation", y_vl.values, y_pred_vl),
                calculate_metrics_row("Test", y_ts.values, y_pred_ts),
            ]
            metrics_df = pd.DataFrame(rows)
            st.session_state["metrics_df"] = metrics_df
            st.session_state["test_pred"] = y_pred_ts

        st.success("✅ Huấn luyện hoàn tất!")

    if "trained_model" in st.session_state:
        st.subheader("Bảng kết quả")
        muted("R2: độ phù hợp mô hình (gần 1 là tốt).  MAE: sai số tuyệt đối trung bình (VND).  RMSE: phạt sai số lớn mạnh hơn (VND).  MAPE: % sai số trung bình.")
        st.dataframe(st.session_state["metrics_df"], use_container_width=True)

        st.divider()

        # 3 VISUALS — 3 ROWS (giống yêu cầu)
        st.subheader("1) Feature Importance")
        top_k = st.slider("Top K", 5, min(50, len(features)), 15, 1)
        plot_feature_importance(st.session_state["trained_model"], features, top_k=top_k)

        st.subheader("2) Residuals Distribution (Sai số)")
        plot_residuals(y_ts.values, st.session_state["test_pred"])

        st.subheader("3) Actual vs Predicted (Test)")
        plot_actual_vs_pred(y_ts.values, st.session_state["test_pred"])

    else:
        st.info("Bấm **Bắt đầu huấn luyện** để train model.")

with tab3:
    st.subheader("Dự đoán")
    if "trained_model" not in st.session_state:
        st.info("Bạn cần huấn luyện model trước.")
    else:
        st.markdown("### A) Nhập tay (để test khi deploy/commit)")
        feats = st.session_state["features"]

        # chọn “vài feature chính” (ưu tiên các feature hay quan trọng)
        priority = [
            "ram_size", "storage_size", "screen_size", "cpu_cores", "cpu_threads",
            "gpu_vram", "res_width", "res_height", "battery_wh", "brand_score"
        ]
        default_pick = [f for f in priority if f in feats]
        if len(default_pick) == 0:
            default_pick = feats[:6] if len(feats) >= 6 else feats

        with st.form("manual_form"):
            picked = st.multiselect("Chọn feature muốn nhập", options=feats, default=default_pick)

            cols = st.columns(2)
            values = {}
            for i, f in enumerate(picked):
                with cols[i % 2]:
                    values[f] = st.number_input(f, value=0.0, step=1.0)

            submit_manual = st.form_submit_button("🎯 Dự đoán")

        if submit_manual:
            x = {c: np.nan for c in feats}
            for k, v in values.items():
                x[k] = float(v)

            X_one = pd.DataFrame([x], columns=feats)
            pred = float(predict_price(st.session_state["trained_model"], X_one, st.session_state["use_log_target"])[0])
            st.success(f"✅ Giá dự đoán: {pred:,.0f} VND")

        st.divider()

        st.markdown("### B) Upload CSV để dự đoán")
        pred_file = st.file_uploader("Upload CSV", type="csv", key="pred_csv")
        if pred_file:
            try:
                out_df = predict_from_csv(
                    st.session_state["trained_model"],
                    st.session_state["features"],
                    pred_file,
                    st.session_state["use_log_target"],
                )
                st.success("✅ Dự đoán xong!")
                st.dataframe(out_df.head(20), use_container_width=True)

                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Tải file dự đoán", csv_bytes, "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Lỗi dự đoán: {e}")

with tab4:
    st.subheader("Xuất file")
    if "trained_model" not in st.session_state:
        st.info("Bạn cần huấn luyện model trước.")
    else:
        joblib.dump(st.session_state["trained_model"], "model.joblib")
        with open("model.joblib", "rb") as f:
            st.download_button("💾 Tải model.joblib", f, "model.joblib")

        st.divider()

        test_results = df_ts.copy()
        preds = predict_price(st.session_state["trained_model"], X_ts, st.session_state["use_log_target"])
        test_results["predicted_price_base"] = preds

        csv = test_results.to_csv(index=False).encode("utf-8")
        st.download_button("📊 Tải test_predictions.csv", csv, "test_predictions.csv", "text/csv")

