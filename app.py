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
def read_csv_safely(file_or_path):
    # UploadedFile hoặc path
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
    # Drop title nếu có
    def _drop_title(df):
        if "title" in df.columns:
            return df.drop(columns=["title"])
        return df

    df_train = _drop_title(df_train)
    df_val = _drop_title(df_val)
    df_test = _drop_title(df_test)

    # intersection columns
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

    # Ép numeric (object -> NaN), imputer median sẽ xử lý
    for c in common_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_val[c] = pd.to_numeric(X_val[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    # tránh crash vì inf
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    return X_train, y_train, X_val, y_val, X_test, y_test, common_cols


def acc_within_threshold(y_true, y_pred, threshold=5_000_000):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = np.abs(y_true - y_pred)
    return float(np.mean(diff <= threshold) * 100.0)


def calculate_metrics_row(name, y_true, y_pred):
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    eps = 1e-8
    denom = np.maximum(np.abs(np.asarray(y_true, dtype=float)), eps)
    mape = float(np.mean(np.abs((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) / denom)) * 100.0)

    acc5 = acc_within_threshold(y_true, y_pred, threshold=5_000_000)

    return {
        "Dataset": name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "Acc<=5Tr(%)": acc5
    }


def build_model(model_type, params):
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
            tree_method="hist",
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


def train_model(pipeline, use_early_stop, model_type, X_tr, y_tr, X_vl, y_vl, early_rounds=50):
    if use_early_stop:
        X_tr_imp = pipeline.named_steps["imputer"].fit_transform(X_tr)
        X_vl_imp = pipeline.named_steps["imputer"].transform(X_vl)
        model = pipeline.named_steps["model"]

        if model_type == "XGBoost":
            try:
                model.fit(
                    X_tr_imp, y_tr,
                    eval_set=[(X_vl_imp, y_vl)],
                    verbose=False,
                    early_stopping_rounds=int(early_rounds)
                )
            except TypeError:
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
                    model.fit(X_tr_imp, y_tr)

        else:  # LightGBM
            try:
                model.fit(
                    X_tr_imp, y_tr,
                    eval_set=[(X_vl_imp, y_vl)],
                    eval_metric="l2",
                )
            except TypeError:
                model.fit(X_tr_imp, y_tr)

        return pipeline

    pipeline.fit(X_tr, y_tr)
    return pipeline


def plot_feature_importance(model_pipeline, feature_names, top_k=15):
    raw_model = model_pipeline.named_steps["model"]

    if not hasattr(raw_model, "feature_importances_"):
        st.info("Model này không hỗ trợ Feature Importance.")
        return

    importances = np.asarray(raw_model.feature_importances_, dtype=float)
    if importances.size != len(feature_names):
        st.info("Không khớp số lượng feature_importances_ với feature_names.")
        return

    top_k = int(min(top_k, len(feature_names)))
    idx = np.argsort(importances)[-top_k:]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.barh(names, vals)
    ax.set_title(f"Top {top_k} Feature Importance")
    ax.set_xlabel("Importance")
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    st.pyplot(fig)


def predict_from_csv(trained_model, features, csv_file):
    df_in = read_csv_safely(csv_file)

    if df_in.shape[0] == 0:
        raise ValueError("File CSV rỗng (0 dòng).")

    # thiếu cột -> NaN (imputer xử lý), thừa cột -> bỏ
    X_in = df_in.reindex(columns=features)

    for c in features:
        X_in[c] = pd.to_numeric(X_in[c], errors="coerce")

    X_in = X_in.replace([np.inf, -np.inf], np.nan)

    preds = trained_model.predict(X_in)
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
if model_choice == "XGBoost":
    early_rounds = st.sidebar.slider("Early stopping rounds", 10, 200, 50, 10)

params = {}

if model_choice == "Random Forest":
    params["n_estimators"] = st.sidebar.slider("n_estimators", 100, 800, 500, 50)
    params["max_depth"] = st.sidebar.slider("max_depth (0 = None)", 0, 40, 0, 1)
    params["min_samples_split"] = st.sidebar.slider("min_samples_split", 2, 20, 2, 1)
    params["min_samples_leaf"] = st.sidebar.slider("min_samples_leaf", 1, 20, 1, 1)
    params["max_features"] = st.sidebar.slider("max_features", 0.2, 1.0, 0.7, 0.05)

elif model_choice == "XGBoost":
    params["n_estimators"] = st.sidebar.slider("n_estimators", 300, 4000, 2000, 100)
    params["max_depth"] = st.sidebar.slider("max_depth", 2, 12, 6, 1)
    params["learning_rate"] = st.sidebar.number_input("learning_rate", 0.005, 0.3, 0.03, step=0.005)
    params["subsample"] = st.sidebar.slider("subsample", 0.5, 1.0, 0.9, 0.05)
    params["colsample_bytree"] = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.9, 0.05)
    params["min_child_weight"] = st.sidebar.number_input("min_child_weight", 0.0, 50.0, 1.0, step=0.5)
    params["gamma"] = st.sidebar.number_input("gamma", 0.0, 20.0, 0.0, step=0.1)
    params["reg_alpha"] = st.sidebar.number_input("reg_alpha", 0.0, 10.0, 0.0, step=0.1)
    params["reg_lambda"] = st.sidebar.number_input("reg_lambda", 0.0, 10.0, 2.0, step=0.1)

else:  # LightGBM
    params["n_estimators"] = st.sidebar.slider("n_estimators", 300, 8000, 3000, 100)
    params["max_depth"] = st.sidebar.slider("max_depth (0 = -1)", 0, 30, 0, 1)
    params["learning_rate"] = st.sidebar.number_input("learning_rate", 0.005, 0.3, 0.03, step=0.005)
    params["num_leaves"] = st.sidebar.slider("num_leaves", 15, 255, 63, 2)
    params["subsample"] = st.sidebar.slider("subsample", 0.5, 1.0, 0.9, 0.05)
    params["colsample_bytree"] = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.9, 0.05)
    params["min_child_samples"] = st.sidebar.slider("min_child_samples", 5, 200, 20, 5)
    params["reg_alpha"] = st.sidebar.number_input("reg_alpha", 0.0, 10.0, 0.0, step=0.1)
    params["reg_lambda"] = st.sidebar.number_input("reg_lambda", 0.0, 10.0, 0.0, step=0.1)


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
    start_train = st.button("🚀 Bắt đầu huấn luyện", type="primary")

    if start_train:
        with st.spinner("Đang huấn luyện..."):
            model_pipeline, use_es = build_model(model_choice, params)
            model_pipeline = train_model(
                model_pipeline,
                use_es,
                model_choice,
                X_tr, y_tr, X_vl, y_vl,
                early_rounds=(early_rounds if early_rounds is not None else 50)
            )

            st.session_state["trained_model"] = model_pipeline
            st.session_state["features"] = features

            # predict all
            y_pred_tr = model_pipeline.predict(X_tr)
            y_pred_vl = model_pipeline.predict(X_vl)
            y_pred_ts = model_pipeline.predict(X_ts)

            # metrics table like image
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
        st.dataframe(st.session_state["metrics_df"], use_container_width=True)

        st.subheader("Feature Importance")
        top_k = st.slider("Top K", 5, min(50, len(features)), 15, 1)
        plot_feature_importance(st.session_state["trained_model"], features, top_k=top_k)
    else:
        st.info("Bấm **Bắt đầu huấn luyện** để train model.")


with tab3:
    st.subheader("Dự đoán")
    if "trained_model" not in st.session_state:
        st.info("Bạn cần huấn luyện model trước.")
    else:
        st.markdown("### A) Nhập tay (để test khi deploy/commit)")
        feats = st.session_state["features"]

        # mặc định lấy vài feature đầu (bạn muốn đổi list này cũng được)
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
            pred = float(st.session_state["trained_model"].predict(X_one)[0])
            st.success(f"✅ Giá dự đoán: {pred:,.0f} VND")

        st.divider()

        st.markdown("### B) Upload CSV để dự đoán")
        pred_file = st.file_uploader("Upload CSV", type="csv", key="pred_csv")
        if pred_file:
            try:
                out_df = predict_from_csv(st.session_state["trained_model"], st.session_state["features"], pred_file)
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
        test_results["predicted_price_base"] = st.session_state.get("test_pred", st.session_state["trained_model"].predict(X_ts))
        csv = test_results.to_csv(index=False).encode("utf-8")
        st.download_button("📊 Tải test_predictions.csv", csv, "test_predictions.csv", "text/csv")
