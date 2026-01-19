import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from matplotlib.ticker import FuncFormatter

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Ứng dụng Dự đoán Giá Laptop", layout="wide")


# =========================
# NOTEBOOK-IDENTICAL HELPERS
# =========================
def clean_currency(x):
    # y hệt notebook: lấy hết chữ số trong chuỗi
    if isinstance(x, str):
        clean_str = "".join(filter(str.isdigit, x))
        try:
            return float(clean_str)
        except ValueError:
            return np.nan
    return x


def read_csv_safely(file_or_path):
    # UploadedFile hoặc path
    try:
        return pd.read_csv(file_or_path, encoding="utf-8")
    except Exception:
        return pd.read_csv(file_or_path, encoding="latin-1")


def prepare_like_notebook(df_train, df_val, df_test):
    # --- clean target price_base ---
    for df in [df_train, df_test, df_val]:
        if "price_base" in df.columns:
            df["price_base"] = df["price_base"].apply(clean_currency)

    # --- remove too small price ---
    df_train = df_train[df_train["price_base"] > 1_000_000].copy()
    df_val = df_val[df_val["price_base"] > 1_000_000].copy()
    df_test = df_test[df_test["price_base"] > 1_000_000].copy()

    exclude_cols = ["title", "price_base", "price_sale"]

    # notebook: numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()

    # notebook: intersection train/val/test + not in exclude
    feature_cols = [
        c for c in numeric_cols
        if (c in df_test.columns) and (c in df_val.columns) and (c not in exclude_cols)
    ]

    # notebook: pd.to_numeric(...).fillna(0)
    for col in feature_cols:
        df_train[col] = pd.to_numeric(df_train[col], errors="coerce").fillna(0)
        df_val[col] = pd.to_numeric(df_val[col], errors="coerce").fillna(0)
        df_test[col] = pd.to_numeric(df_test[col], errors="coerce").fillna(0)

    X_train, y_train = df_train[feature_cols], df_train["price_base"]
    X_val, y_val = df_val[feature_cols], df_val["price_base"]
    X_test, y_test = df_test[feature_cols], df_test["price_base"]

    return df_train, df_val, df_test, X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def custom_accuracy(y_true, y_pred, threshold=5_000_000):
    # notebook: Acc (<=5Tr)
    diff = np.abs(y_true - y_pred)
    return (np.sum(diff <= threshold) / len(y_true)) * 100


# =========================
# VISUAL HELPERS (FIX OVERLAP)
# =========================
def _money_million_formatter():
    # hiển thị theo "triệu" để ngắn -> đỡ đè chữ
    return FuncFormatter(lambda x, pos: f"{x/1e6:.0f}M")


def _beautify_axis(ax):
    # giảm size + xoay nhãn để khỏi đè
    ax.tick_params(axis="both", labelsize=9)
    ax.tick_params(axis="x", labelrotation=25)


def plot_scatter(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.scatter(y_true, y_pred, alpha=0.5)

    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([mn, mx], [mn, mx], "r--")

    ax.set_title("Giá thật vs Giá dự đoán")
    ax.set_xlabel("Giá thật (triệu VND)")
    ax.set_ylabel("Giá dự đoán (triệu VND)")

    ax.xaxis.set_major_formatter(_money_million_formatter())
    ax.yaxis.set_major_formatter(_money_million_formatter())
    _beautify_axis(ax)

    fig.tight_layout()
    st.pyplot(fig)


def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.hist(residuals, bins=30)

    ax.set_title("Phân phối sai số (Residuals)")
    ax.set_xlabel("Sai số (triệu VND)")
    ax.set_ylabel("Số lượng")

    ax.xaxis.set_major_formatter(_money_million_formatter())
    _beautify_axis(ax)

    fig.tight_layout()
    st.pyplot(fig)


def plot_learning_curve(epoch_list, val_rmse_list, val_r2_list):
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.plot(epoch_list, val_rmse_list, marker="o", linewidth=1)

    ax.set_title("Val RMSE theo Epoch (warm_start)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val RMSE (VND)")
    _beautify_axis(ax)

    fig.tight_layout()
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(6.6, 4.3))
    ax2.plot(epoch_list, val_r2_list, marker="o", linewidth=1)

    ax2.set_title("Val R2 theo Epoch (warm_start)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val R2")
    _beautify_axis(ax2)

    fig2.tight_layout()
    st.pyplot(fig2)


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

st.sidebar.header("2) Huấn luyện (GIỐNG NOTEBOOK)")
epochs = st.sidebar.slider("Epoch (max_iter tăng dần)", 10, 300, 100, 10)
learning_rate = st.sidebar.number_input("learning_rate", 0.01, 0.5, 0.1, step=0.01)

# =========================
# MAIN
# =========================
st.title("💻 Ứng dụng Dự đoán Giá Laptop")

def _available(p):
    return (hasattr(p, "read")) or os.path.exists(str(p))

if not all(_available(p) for p in [train_path, val_path, test_path]):
    st.info("Vui lòng upload đủ 3 file: data_train.csv, data_validation.csv, data_test.csv (hoặc đặt sẵn trong repo).")
    st.stop()

try:
    df_tr = read_csv_safely(train_path)
    df_vl = read_csv_safely(val_path)
    df_ts = read_csv_safely(test_path)

    df_tr, df_vl, df_ts, X_tr, y_tr, X_vl, y_vl, X_ts, y_ts, features = prepare_like_notebook(df_tr, df_vl, df_ts)
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

    st.info(f"Đang dùng **{len(features)}** đặc trưng numeric (intersection) — đã loại title/price_base/price_sale.")


with tab2:
    st.subheader("Huấn luyện mô hình (y hệt notebook)")

    start_train = st.button("🚀 Bắt đầu huấn luyện", type="primary")

    if start_train:
        # notebook: est = HistGradientBoostingRegressor(learning_rate=0.1, max_iter=1, warm_start=True, random_state=42)
        est = HistGradientBoostingRegressor(
            learning_rate=float(learning_rate),
            max_iter=1,
            warm_start=True,
            random_state=42
        )

        epoch_list = []
        val_rmse_history = []
        val_r2_history = []

        with st.spinner("Đang huấn luyện warm_start..."):
            for i in range(1, int(epochs) + 1):
                est.set_params(max_iter=i)
                est.fit(X_tr, y_tr)

                y_val_pred_step = est.predict(X_vl)
                val_rmse = float(np.sqrt(mean_squared_error(y_vl, y_val_pred_step)))
                val_r2 = float(r2_score(y_vl, y_val_pred_step))

                epoch_list.append(i)
                val_rmse_history.append(val_rmse)
                val_r2_history.append(val_r2)

        # lưu model + features
        st.session_state["trained_model"] = est
        st.session_state["features"] = features
        st.session_state["history"] = (epoch_list, val_rmse_history, val_r2_history)

        # dự đoán cuối
        y_train_pred = est.predict(X_tr)
        y_val_pred = est.predict(X_vl)
        y_test_pred = est.predict(X_ts)

        metrics_data = []
        for name, y_true, y_pred in [
            ("Train", y_tr, y_train_pred),
            ("Validation", y_vl, y_val_pred),
            ("Test", y_ts, y_test_pred)
        ]:
            metrics_data.append({
                "Dataset": name,
                "R2": float(r2_score(y_true, y_pred)),
                "MAE": float(mean_absolute_error(y_true, y_pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
                "Acc<=5Tr(%)": float(custom_accuracy(y_true.values, y_pred))
            })

        metrics_df = pd.DataFrame(metrics_data)
        st.session_state["metrics_df"] = metrics_df
        st.session_state["val_pred"] = y_val_pred
        st.session_state["test_pred"] = y_test_pred

        st.success("✅ Huấn luyện hoàn tất!")

    if "trained_model" in st.session_state:
        st.markdown("### Bảng kết quả (giống notebook)")
        st.dataframe(st.session_state["metrics_df"], use_container_width=True)

        st.markdown("### Learning curve")
        epoch_list, val_rmse_history, val_r2_history = st.session_state["history"]
        plot_learning_curve(epoch_list, val_rmse_history, val_r2_history)

        st.markdown("### Biểu đồ (đã sửa đỡ đè số)")
        plot_scatter(y_vl.values, st.session_state["val_pred"])
        plot_residuals(y_vl.values, st.session_state["val_pred"])

    else:
        st.info("Bấm **Bắt đầu huấn luyện** để train model.")


with tab3:
    st.subheader("Dự đoán")

    if "trained_model" not in st.session_state:
        st.info("Bạn cần huấn luyện model trước.")
    else:
        st.markdown("## A) Nhập tay (khuyến nghị để test khi deploy/commit)")
        st.caption("Bạn chọn **một vài feature chính** để nhập. Các feature còn lại tự set = 0 (đúng kiểu notebook fillna(0)).")

        feats = st.session_state["features"]

        # default: lấy 6 feature đầu (hoặc ít hơn)
        default_pick = feats[:6] if len(feats) >= 6 else feats

        with st.form("manual_form"):
            picked = st.multiselect("Chọn feature muốn nhập", options=feats, default=default_pick)

            cols = st.columns(2)
            values = {}
            for i, f in enumerate(picked):
                with cols[i % 2]:
                    values[f] = st.number_input(f, value=0.0, step=1.0)

            submit_manual = st.form_submit_button("🎯 Dự đoán từ dữ liệu nhập tay")

        if submit_manual:
            x = {c: 0.0 for c in feats}
            for k, v in values.items():
                x[k] = float(v)

            X_one = pd.DataFrame([x], columns=feats)
            pred = float(st.session_state["trained_model"].predict(X_one)[0])
            st.success(f"✅ Giá dự đoán: **{pred:,.0f} VND**")

        st.divider()

        st.markdown("## B) Upload CSV để dự đoán")
        st.write("App sẽ align theo features lúc train: thiếu cột -> 0, thừa cột -> bỏ.")

        pred_file = st.file_uploader("Upload CSV để dự đoán", type="csv", key="pred_csv")
        if pred_file:
            df_in = read_csv_safely(pred_file)

            if df_in.shape[0] == 0:
                st.error("File CSV rỗng (0 dòng) nên không thể dự đoán.")
            else:
                X_in = df_in.reindex(columns=feats, fill_value=0)

                # ép numeric, lỗi -> NaN rồi fill 0 (giống notebook tinh thần fill 0)
                for c in feats:
                    X_in[c] = pd.to_numeric(X_in[c], errors="coerce").fillna(0)

                preds = st.session_state["trained_model"].predict(X_in)
                out = df_in.copy()
                out["predicted_price_base"] = preds

                st.success("✅ Dự đoán xong!")
                st.dataframe(out.head(20), use_container_width=True)

                csv_bytes = out.to_csv(index=False).encode("utf-8")
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

        # xuất dự đoán test
        test_results = df_ts.copy()
        test_results["predicted_price_base"] = st.session_state["test_pred"]
        csv = test_results.to_csv(index=False).encode("utf-8")
        st.download_button("📊 Tải test_predictions.csv", csv, "test_predictions.csv", "text/csv")
