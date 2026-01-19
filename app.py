import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# XGBoost notebook
from xgboost import XGBRegressor
from sklearn.preprocessing import OrdinalEncoder

# "LightGBM" notebook thực ra dùng HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Ứng dụng Dự đoán Giá Laptop", layout="wide")

# =========================
# HELPERS
# =========================
def read_csv_fallback(file_or_path):
    """Read CSV with encoding fallback."""
    if file_or_path is None:
        return None
    try:
        return pd.read_csv(file_or_path, encoding="latin-1")
    except Exception:
        return pd.read_csv(file_or_path, encoding="utf-8")


def clean_currency_xgb(x):
    """Notebook model_XGBoost.ipynb"""
    if isinstance(x, str):
        clean_str = "".join(filter(str.isdigit, x))
        try:
            return float(clean_str)
        except ValueError:
            return np.nan
    return x


def clean_price_hgb(x):
    """Notebook Tu_LightGBM.ipynb"""
    if pd.isna(x):
        return np.nan
    s = str(x).replace(".", "").replace(",", "").replace("đ", "").strip()
    try:
        val = float(s)
        return val if 1e6 < val < 500e6 else np.nan
    except Exception:
        return np.nan


def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    # MAPE (%)
    eps = 1e-9
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)
    return r2, mae, rmse, mape


def make_results_df(rows):
    return pd.DataFrame(rows, columns=["Dataset", "R2", "MAE", "RMSE", "MAPE (%)"])


def style_small_ticks(ax):
    ax.tick_params(axis="both", labelsize=8)


# =========================
# PREPROCESS: RANDOM FOREST (GIỮ NGUYÊN TƯ DUY PIPELINE)
# =========================
def prepare_rf_data(df_train, df_val, df_test, target="price_base"):
    # drop title nếu có
    for df in (df_train, df_val, df_test):
        if df is not None and "title" in df.columns:
            df.drop(columns=["title"], inplace=True, errors="ignore")

    # chỉ lấy cột chung
    common_cols = set(df_train.columns) & set(df_val.columns) & set(df_test.columns)
    common_cols = list(common_cols)

    if target not in common_cols:
        # target có thể tồn tại nhưng không nằm trong common_cols do df nào đó thiếu -> ép đảm bảo
        if target in df_train.columns and target in df_val.columns and target in df_test.columns:
            pass
        else:
            raise ValueError("Không tìm thấy cột target 'price_base' trong đủ 3 file.")

    # features: tất cả trừ target
    feature_cols = [c for c in common_cols if c != target]
    # loại thêm vài cột định danh nếu có
    for bad in ["link", "url", "id", "price_sale"]:
        if bad in feature_cols:
            feature_cols.remove(bad)

    X_train = df_train[feature_cols].copy()
    y_train = pd.to_numeric(df_train[target], errors="coerce")

    X_val = df_val[feature_cols].copy()
    y_val = pd.to_numeric(df_val[target], errors="coerce")

    X_test = df_test[feature_cols].copy()
    y_test = pd.to_numeric(df_test[target], errors="coerce")

    # keep only numeric for RF pipeline (nếu dữ liệu bạn toàn numeric thì OK)
    # Nếu có object lẫn vào -> cố ép numeric
    for X in (X_train, X_val, X_test):
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # drop nan y
    mask_tr = y_train.notna()
    mask_vl = y_val.notna()
    mask_ts = y_test.notna()

    return (
        X_train.loc[mask_tr], y_train.loc[mask_tr].astype(float).values,
        X_val.loc[mask_vl], y_val.loc[mask_vl].astype(float).values,
        X_test.loc[mask_ts], y_test.loc[mask_ts].astype(float).values,
        feature_cols
    )


# =========================
# PREPROCESS: XGBOOST (THEO model_XGBoost.ipynb)
# =========================
def prepare_xgb_data(df_train, df_val, df_test):
    # clean price_base
    for df in (df_train, df_val, df_test):
        if "price_base" in df.columns:
            df["price_base"] = df["price_base"].apply(clean_currency_xgb)

    # filter price too small
    df_train = df_train[df_train["price_base"] > 1_000_000].copy()
    df_val = df_val[df_val["price_base"] > 1_000_000].copy()
    df_test = df_test[df_test["price_base"] > 1_000_000].copy()

    exclude_cols = ["title", "price_base", "price_sale", "link", "url", "id"]
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    # align columns for val/test
    for col in feature_cols:
        if col not in df_val.columns:
            df_val[col] = 0
        if col not in df_test.columns:
            df_test[col] = 0

    # numeric/categorical
    numeric_cols = df_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_train[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

    # categorical preprocess like notebook
    ord_enc = None
    if len(categorical_cols) > 0:
        for df in (df_train, df_val, df_test):
            df[categorical_cols] = df[categorical_cols].fillna("Unknown")

        ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df_train[categorical_cols] = ord_enc.fit_transform(df_train[categorical_cols].astype(str))
        df_val[categorical_cols] = ord_enc.transform(df_val[categorical_cols].astype(str))
        df_test[categorical_cols] = ord_enc.transform(df_test[categorical_cols].astype(str))

        # cast int -> category (xgb enable_categorical)
        for col in categorical_cols:
            df_train[col] = df_train[col].astype(int).astype("category")
            df_val[col] = df_val[col].astype(int).astype("category")
            df_test[col] = df_test[col].astype(int).astype("category")

    # numeric impute median (để giống pipeline)
    for df in (df_train, df_val, df_test):
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            med = df_train[c].median()
            df[c] = df[c].fillna(med)

    X_train = df_train[feature_cols]
    y_train = np.log1p(df_train["price_base"].values)

    X_val = df_val[feature_cols]
    y_val = np.log1p(df_val["price_base"].values)

    X_test = df_test[feature_cols]
    y_test = df_test["price_base"].values.astype(float)

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


# =========================
# PREPROCESS: HGB (THEO Tu_LightGBM.ipynb)
# =========================
def prepare_hgb_data(df_train, df_val, df_test):
    for df in (df_train, df_val, df_test):
        df["price_base"] = df["price_base"].apply(clean_price_hgb)
        df.dropna(subset=["price_base"], inplace=True)

    # collect numeric cols across all
    all_numeric_cols = set()
    for df in (df_train, df_val, df_test):
        all_numeric_cols.update(df.select_dtypes(include=[np.number]).columns)

    target_related = {"price_base", "price_sale"}
    feats = sorted(list(all_numeric_cols - target_related))

    # storage_type -> is_ssd
    def map_ssd(x):
        s = str(x).upper()
        if s == "SSD":
            return 1.0
        if s == "HDD":
            return 0.0
        return 0.5

    for df in (df_train, df_val, df_test):
        if "storage_type" in df.columns:
            df["is_ssd"] = df["storage_type"].apply(map_ssd)
        else:
            df["is_ssd"] = 0.5

    feats = sorted(list(set(feats + ["is_ssd"])))

    def finalize(df, train_ref=None):
        X = pd.DataFrame(index=df.index)
        for c in feats:
            if c in df.columns:
                X[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                X[c] = 0.0

        for c in feats:
            ref = train_ref if train_ref is not None else df
            if c in ref.columns:
                m = pd.to_numeric(ref[c], errors="coerce").median()
            else:
                m = 0.0
            if pd.isna(m):
                m = 0.0
            X[c] = X[c].fillna(m)

        y_log = np.log1p(df["price_base"].values.astype(float))
        y_raw = df["price_base"].values.astype(float)
        return X, y_log, y_raw

    X_tr, y_tr_l, y_tr = finalize(df_train, None)
    X_vl, y_vl_l, y_vl = finalize(df_val, df_train)
    X_ts, y_ts_l, y_ts = finalize(df_test, df_train)

    return X_tr, y_tr_l, y_tr, X_vl, y_vl_l, y_vl, X_ts, y_ts, feats


# =========================
# UI
# =========================
st.title("💻 Ứng dụng Dự đoán Giá Laptop")

with st.sidebar:
    st.header("1) Upload dữ liệu")
    up_train = st.file_uploader("Upload data_train.csv", type=["csv"])
    up_val = st.file_uploader("Upload data_validation.csv", type=["csv"])
    up_test = st.file_uploader("Upload data_test.csv", type=["csv"])

    st.header("2) Chọn mô hình")
    model_name = st.selectbox("Mô hình", ["Random Forest", "XGBoost", "LightGBM (theo notebook)"])

    # ===== Random Forest sliders =====
    rf_params = {}
    if model_name == "Random Forest":
        rf_params["n_estimators"] = st.slider(
            "n_estimators",
            50, 2000, 500, 50,
            help="Số cây. Tăng thường ổn định hơn nhưng train chậm hơn."
        )
        rf_params["max_depth"] = st.slider(
            "max_depth (0 = None)",
            0, 40, 0, 1,
            help="Độ sâu tối đa của cây. Tăng -> fit mạnh hơn nhưng dễ overfit. Giảm -> tổng quát tốt hơn."
        )
        rf_params["min_samples_split"] = st.slider(
            "min_samples_split",
            2, 30, 2, 1,
            help="Số mẫu tối thiểu để tách node. Tăng -> giảm overfit, nhưng có thể underfit."
        )
        rf_params["min_samples_leaf"] = st.slider(
            "min_samples_leaf",
            1, 30, 1, 1,
            help="Số mẫu tối thiểu ở lá. Tăng -> mô hình mượt hơn, thường giúp Test R2 tốt hơn nếu đang overfit."
        )
        rf_params["max_features"] = st.slider(
            "max_features",
            0.10, 1.0, 0.70, 0.05,
            help="Tỉ lệ feature dùng mỗi lần split. Giảm -> tăng đa dạng cây, giảm overfit (thường tốt cho test)."
        )
        rf_params["random_state"] = 42
        rf_params["n_jobs"] = -1

    st.divider()
    st.caption(
        "Gợi ý nhanh: nếu Train R2 cao nhưng Test R2 thấp → đang **overfit**. "
        "Hãy **giảm max_depth**, **tăng min_samples_leaf**, và **giảm max_features**."
    )

train_btn = st.button("🚀 Bắt đầu huấn luyện")


# =========================
# LOAD DATA
# =========================
def load_default_or_upload(uploaded, default_name):
    if uploaded is not None:
        return read_csv_fallback(uploaded)
    if os.path.exists(default_name):
        return read_csv_fallback(default_name)
    return None


df_train = load_default_or_upload(up_train, "data_train.csv")
df_val = load_default_or_upload(up_val, "data_validation.csv")
df_test = load_default_or_upload(up_test, "data_test.csv")

if df_train is None or df_val is None or df_test is None:
    st.info("Hãy upload đủ 3 file hoặc đặt 3 file cùng thư mục với app.py: data_train.csv, data_validation.csv, data_test.csv")
    st.stop()


# =========================
# TRAIN
# =========================
if train_btn:
    st.success("✅ Đang huấn luyện...")

    results_rows = []
    fig_scatter = None
    fig_resid = None
    fig_imp = None

    # -------------------------
    # RANDOM FOREST
    # -------------------------
    if model_name == "Random Forest":
        X_tr, y_tr, X_vl, y_vl, X_ts, y_ts, feature_cols = prepare_rf_data(df_train.copy(), df_val.copy(), df_test.copy())

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=rf_params["n_estimators"],
                max_depth=None if rf_params["max_depth"] == 0 else rf_params["max_depth"],
                min_samples_split=rf_params["min_samples_split"],
                min_samples_leaf=rf_params["min_samples_leaf"],
                max_features=rf_params["max_features"],
                random_state=rf_params["random_state"],
                n_jobs=rf_params["n_jobs"],
            ))
        ])

        pipeline.fit(X_tr, y_tr)

        # Predict
        pred_tr = pipeline.predict(X_tr)
        pred_vl = pipeline.predict(X_vl)
        pred_ts = pipeline.predict(X_ts)

        # Metrics
        for name, yt, yp in [
            ("Train", y_tr, pred_tr),
            ("Validation", y_vl, pred_vl),
            ("Test", y_ts, pred_ts),
        ]:
            r2, mae, rmse, mape = calc_metrics(yt, yp)
            results_rows.append([name, r2, mae, rmse, mape])

        # Visualize: use TEST
        y_true = y_ts
        y_pred = pred_ts
        resid = y_true - y_pred

        # Scatter
        fig_scatter, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(y_true, y_pred, alpha=0.35)
        mn = float(min(y_true.min(), y_pred.min()))
        mx = float(max(y_true.max(), y_pred.max()))
        ax.plot([mn, mx], [mn, mx], linestyle="--")
        ax.set_title("Actual vs Predicted Prices (Test)")
        ax.set_xlabel("Actual Price (VND)")
        ax.set_ylabel("Predicted Price (VND)")
        style_small_ticks(ax)

        # Residuals
        fig_resid, ax2 = plt.subplots(figsize=(7, 4))
        ax2.hist(resid, bins=40, alpha=0.8)
        ax2.axvline(0, linestyle="--")
        ax2.set_title("Residuals Distribution (Sai số - Test)")
        ax2.set_xlabel("Error Amount (VND)")
        ax2.set_ylabel("Count")
        style_small_ticks(ax2)

        # Feature importance (RF)
        model = pipeline.named_steps["model"]
        importances = getattr(model, "feature_importances_", None)

        top_k = 15
        if importances is not None:
            imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(top_k)[::-1]
            fig_imp, ax3 = plt.subplots(figsize=(7, 4.5))
            ax3.barh(imp.index, imp.values)
            ax3.set_title(f"Top {top_k} Feature Importances (Random Forest)")
            ax3.set_xlabel("Relative Importance")
            style_small_ticks(ax3)

        # Gợi ý params (dựa gap)
        r2_train = results_rows[0][1]
        r2_test = results_rows[2][1]
        gap = r2_train - r2_test

        with st.expander("📌 Gợi ý thông số để Test R2 tốt hơn"):
            if gap > 0.12:
                st.warning(
                    f"Train R2 ({r2_train:.3f}) cao hơn Test R2 ({r2_test:.3f}) khá nhiều → **overfit**."
                )
                st.markdown(
                    "- Giảm `max_depth` (ví dụ 12–20)\n"
                    "- Tăng `min_samples_leaf` (ví dụ 3–8)\n"
                    "- Tăng `min_samples_split` (ví dụ 6–15)\n"
                    "- Giảm `max_features` (ví dụ 0.4–0.7)\n"
                    "- (Nếu bạn thêm được) `max_samples` ~ 0.7–0.9 để giảm overfit\n"
                )
            else:
                st.success("Gap Train–Test không quá lớn. Bạn có thể thử tăng nhẹ n_estimators và tối ưu max_depth.")

            st.caption("Preset hay ổn định cho laptop price (RF): n_estimators=800–1200, max_depth=16, min_samples_leaf=4, min_samples_split=10, max_features=0.5–0.7")

    # -------------------------
    # XGBOOST (theo model_XGBoost.ipynb)
    # -------------------------
    elif model_name == "XGBoost":
        X_tr, y_tr_l, X_vl, y_vl_l, X_ts, y_ts, feature_cols = prepare_xgb_data(df_train.copy(), df_val.copy(), df_test.copy())

        model = XGBRegressor(
            n_estimators=2000,
            learning_rate=0.01,

            max_depth=4,
            min_child_weight=5,
            gamma=0.2,

            subsample=0.6,
            colsample_bytree=0.6,

            reg_alpha=1.0,
            reg_lambda=2.0,

            objective="reg:squarederror",
            tree_method="hist",
            enable_categorical=True,
            n_jobs=-1,
            random_state=42,
            early_stopping_rounds=100,
            eval_metric="rmse"
        )

        model.fit(
            X_tr, y_tr_l,
            eval_set=[(X_tr, y_tr_l), (X_vl, y_vl_l)],
            verbose=False
        )

        # preds: train/val trên log → convert về VND để tính metrics giống bảng bạn
        pred_tr = np.expm1(model.predict(X_tr))
        pred_vl = np.expm1(model.predict(X_vl))
        pred_ts = np.expm1(model.predict(X_ts))

        y_tr = np.expm1(y_tr_l)
        y_vl = np.expm1(y_vl_l)

        for name, yt, yp in [
            ("Train", y_tr, pred_tr),
            ("Validation", y_vl, pred_vl),
            ("Test", y_ts, pred_ts),
        ]:
            r2, mae, rmse, mape = calc_metrics(yt, yp)
            results_rows.append([name, r2, mae, rmse, mape])

        # Visualize: TEST
        y_true = y_ts
        y_pred = pred_ts
        resid = y_true - y_pred

        fig_scatter, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(y_true, y_pred, alpha=0.35)
        mn = float(min(y_true.min(), y_pred.min()))
        mx = float(max(y_true.max(), y_pred.max()))
        ax.plot([mn, mx], [mn, mx], linestyle="--")
        ax.set_title("Actual vs Predicted Prices (Test)")
        ax.set_xlabel("Actual Price (VND)")
        ax.set_ylabel("Predicted Price (VND)")
        style_small_ticks(ax)

        fig_resid, ax2 = plt.subplots(figsize=(7, 4))
        ax2.hist(resid, bins=40, alpha=0.8)
        ax2.axvline(0, linestyle="--")
        ax2.set_title("Residuals Distribution (Sai số - Test)")
        ax2.set_xlabel("Error Amount (VND)")
        ax2.set_ylabel("Count")
        style_small_ticks(ax2)

        # Feature importance (XGB built-in)
        top_k = 15
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(top_k)[::-1]
            fig_imp, ax3 = plt.subplots(figsize=(7, 4.5))
            ax3.barh(imp.index, imp.values)
            ax3.set_title(f"Top {top_k} Feature Importances (XGBoost)")
            ax3.set_xlabel("Relative Importance")
            style_small_ticks(ax3)

        with st.expander("📌 Gợi ý thông số để Test R2 tốt hơn (XGBoost)"):
            st.markdown(
                "- Nếu overfit: giảm `max_depth` (3–4), tăng `min_child_weight` (5–10), tăng `reg_lambda` (2–5)\n"
                "- Nếu underfit: tăng `max_depth` (5–6) hoặc tăng `n_estimators` (nhưng giữ `learning_rate` nhỏ)\n"
                "- Thường ổn định: `subsample/colsample_bytree` trong 0.6–0.9\n"
            )
            st.caption("Notebook preset bạn gửi đang thiên về chống overfit, khá hợp nếu dữ liệu nhiều nhiễu.")

    # -------------------------
    # "LIGHTGBM" theo Tu_LightGBM.ipynb (HGB)
    # -------------------------
    else:
        X_tr, y_tr_l, y_tr, X_vl, y_vl_l, y_vl, X_ts, y_ts, feats = prepare_hgb_data(df_train.copy(), df_val.copy(), df_test.copy())

        model = HistGradientBoostingRegressor(
            max_iter=1,
            learning_rate=0.04,
            max_leaf_nodes=127,
            min_samples_leaf=5,
            l2_regularization=0.1,
            warm_start=True,
            random_state=42
        )

        # warm_start training loop + early stopping
        best_val_rmse = float("inf")
        patience = 50
        no_improve = 0
        best_iter = 1

        for epoch in range(1, 2001):
            model.max_iter = epoch
            model.fit(X_tr, y_tr_l)

            p_vl = model.predict(X_vl)
            rmse_vl = float(np.sqrt(mean_squared_error(y_vl_l, p_vl)))

            if rmse_vl < best_val_rmse:
                best_val_rmse = rmse_vl
                best_iter = epoch
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        # fit lại best_iter
        model.max_iter = best_iter
        model.fit(X_tr, y_tr_l)

        pred_tr = np.expm1(model.predict(X_tr))
        pred_vl = np.expm1(model.predict(X_vl))
        pred_ts = np.expm1(model.predict(X_ts))

        for name, yt, yp in [
            ("Train", y_tr, pred_tr),
            ("Validation", y_vl, pred_vl),
            ("Test", y_ts, pred_ts),
        ]:
            r2, mae, rmse, mape = calc_metrics(yt, yp)
            results_rows.append([name, r2, mae, rmse, mape])

        # Visualize: TEST
        y_true = y_ts
        y_pred = pred_ts
        resid = y_true - y_pred

        fig_scatter, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(y_true, y_pred, alpha=0.35)
        mn = float(min(y_true.min(), y_pred.min()))
        mx = float(max(y_true.max(), y_pred.max()))
        ax.plot([mn, mx], [mn, mx], linestyle="--")
        ax.set_title("Actual vs Predicted Prices (Test)")
        ax.set_xlabel("Actual Price (VND)")
        ax.set_ylabel("Predicted Price (VND)")
        style_small_ticks(ax)

        fig_resid, ax2 = plt.subplots(figsize=(7, 4))
        ax2.hist(resid, bins=40, alpha=0.8)
        ax2.axvline(0, linestyle="--")
        ax2.set_title("Residuals Distribution (Sai số - Test)")
        ax2.set_xlabel("Error Amount (VND)")
        ax2.set_ylabel("Count")
        style_small_ticks(ax2)

        # Feature importance: HGB không có built-in -> dùng permutation importance (nhanh, lấy top nhỏ)
        top_k = 12
        try:
            perm = permutation_importance(model, X_vl, y_vl_l, n_repeats=5, random_state=42)
            imp = pd.Series(perm.importances_mean, index=feats).sort_values(ascending=False).head(top_k)[::-1]
            fig_imp, ax3 = plt.subplots(figsize=(7, 4.5))
            ax3.barh(imp.index, imp.values)
            ax3.set_title(f"Top {top_k} Feature Importances (Permutation - HGB)")
            ax3.set_xlabel("Importance (mean decrease)")
            style_small_ticks(ax3)
        except Exception:
            fig_imp = None

        with st.expander("📌 Gợi ý thông số để Test R2 tốt hơn (HGB)"):
            st.markdown(
                "- `max_leaf_nodes` tăng → fit mạnh hơn nhưng dễ overfit\n"
                "- `min_samples_leaf` tăng → mượt hơn, thường giúp test tốt hơn nếu overfit\n"
                "- `learning_rate` nhỏ + `max_iter` lớn → học chậm, ổn định hơn\n"
                f"- Best_iter (theo early stopping) hiện tại: **{best_iter}**"
            )

    # =========================
    # OUTPUT: TABLE + CAPTIONS
    # =========================
    st.success("✅ Huấn luyện hoàn tất!")

    st.subheader("Bảng kết quả")
    st.caption("R2: mức độ mô hình giải thích biến thiên giá (gần 1 là tốt). MAE: sai số tuyệt đối trung bình (VND). RMSE: phạt sai số lớn mạnh hơn (VND). MAPE: % sai số trung bình.")
    res_df = make_results_df(results_rows)
    st.dataframe(res_df, use_container_width=True)

    # =========================
    # VISUALIZE (3 HÀNG RIÊNG)
    # =========================
    st.subheader("Visualize (3 phần)")

    st.markdown("### 1) Actual vs Predicted (Test)")
    st.pyplot(fig_scatter, clear_figure=False)

    st.markdown("### 2) Residuals Distribution (Test)")
    st.pyplot(fig_resid, clear_figure=False)

    st.markdown("### 3) Feature Importance")
    if fig_imp is not None:
        st.pyplot(fig_imp, clear_figure=False)
    else:
        st.info("Model này không có feature importance trực tiếp (hoặc không tính được trong lần chạy này).")
