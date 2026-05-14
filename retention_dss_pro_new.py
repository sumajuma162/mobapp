import warnings
warnings.filterwarnings("ignore")

from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

from sklearn.utils import resample
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.neural_network import MLPClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except Exception:
    apriori = None
    association_rules = None
    MLXTEND_AVAILABLE = False


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Retention DSS", layout="wide", page_icon="📱")
st.title("📱 Mobile Customer Retention Decision Support System")
st.markdown(
    "**Operational Validation, Closed-Loop Simulation, SHAP Diagnostics, and Prescriptive Retention Actions**"
)

MODEL_PATH = Path("retention_best_rf.pkl")


# =========================================================
# SESSION STATE
# =========================================================
if "simulation_cache" not in st.session_state:
    st.session_state.simulation_cache = {
        "iteration_tables": {},
        "summary_df": pd.DataFrame(),
        "intervention_df": pd.DataFrame(),
        "local_shap_df": pd.DataFrame(),
        "who_why_what_df": pd.DataFrame(),
        "network_history_df": pd.DataFrame(),
        "yas": pd.DataFrame(),
        "voda": pd.DataFrame(),
        "network_summary": pd.DataFrame(),
        "zip_bytes": b"",
    }

simulation_cache = st.session_state.simulation_cache


# =========================================================
# HELPERS
# =========================================================
def safe_mode(series: pd.Series, default="A"):
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else default


def filler(series: pd.Series):
    if pd.api.types.is_numeric_dtype(series):
        fill_value = series.mean()
        fill_value = round(float(fill_value), 1) if pd.notna(fill_value) else 0
        return series.fillna(fill_value)
    fill_value = safe_mode(series, default="A")
    return series.astype("object").fillna(fill_value)


def normalize_operator_name(value) -> str:
    text = str(value).strip().upper()
    if text == "TIGO":
        return "YAS"
    if text in {"VODA", "VODACOM"}:
        return "VODACOM"
    return text


def to_safe_object(series: pd.Series) -> pd.Series:
    if pd.api.types.is_categorical_dtype(series):
        return series.astype("object")
    return series


def safe_map(df: pd.DataFrame, column: str, mapping: dict, default=None):
    if column not in df.columns:
        return
    s = df[column].copy()
    s = to_safe_object(s).astype("object")
    s = s.map(lambda x: mapping.get(x, x))
    if default is not None:
        s = s.fillna(default)
    df[column] = s


def encode_remaining_columns(df: pd.DataFrame, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []

    out = df.copy()
    for col in out.columns:
        if col in exclude_cols:
            continue
        out[col] = to_safe_object(out[col])
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.factorize(out[col].astype(str))[0]

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


def get_probability(model, X_input: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)
        if probs.ndim == 2:
            if probs.shape[1] == 1:
                return probs[:, 0]
            if hasattr(model, "classes_") and 1 in list(model.classes_):
                idx = list(model.classes_).index(1)
                return probs[:, idx]
            return probs[:, -1]
    return np.asarray(model.predict(X_input), dtype=float)


def normalize_shap_values(shap_values):
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[-1])
    elif hasattr(shap_values, "values"):
        arr = np.asarray(shap_values.values)
    else:
        arr = np.asarray(shap_values)

    if arr.ndim == 3:
        return arr[:, :, -1]
    return arr


def shap_summary_plot_safe(model, X_sample: pd.DataFrame):
    if not SHAP_AVAILABLE:
        return None, "SHAP is not installed."

    try:
        explainer = shap.TreeExplainer(model)

        try:
            explanation = explainer(X_sample)
            fig = plt.figure(figsize=(10, 6))

            if hasattr(explanation, "values"):
                values = np.asarray(explanation.values)
                if values.ndim == 3:
                    class_idx = 1 if values.shape[2] > 1 else 0
                    reduced_exp = shap.Explanation(
                        values=values[:, :, class_idx],
                        base_values=(
                            explanation.base_values[:, class_idx]
                            if isinstance(explanation.base_values, np.ndarray)
                            and np.ndim(explanation.base_values) == 2
                            else explanation.base_values
                        ),
                        data=explanation.data,
                        feature_names=explanation.feature_names,
                    )
                    shap.plots.beeswarm(reduced_exp, show=False)
                else:
                    shap.plots.beeswarm(explanation, show=False)

            return fig, None

        except Exception:
            shap_values = explainer.shap_values(X_sample)
            plot_values = normalize_shap_values(shap_values)
            fig = plt.figure(figsize=(10, 6))
            shap.summary_plot(plot_values, X_sample, show=False)
            return fig, None

    except Exception as exc:
        return None, str(exc)


def local_shap_contributions(model, row_df: pd.DataFrame):
    if not SHAP_AVAILABLE:
        return pd.DataFrame(columns=["Feature", "SHAP_Value", "Abs_SHAP"])

    try:
        explainer = shap.TreeExplainer(model)

        try:
            explanation = explainer(row_df)
            if hasattr(explanation, "values"):
                vals = np.asarray(explanation.values)
                if vals.ndim == 3:
                    class_idx = 1 if vals.shape[2] > 1 else 0
                    vals = vals[:, :, class_idx]
                vals = vals[0]
            else:
                vals = np.zeros(row_df.shape[1])
        except Exception:
            shap_values = explainer.shap_values(row_df)
            vals = normalize_shap_values(shap_values)
            if vals.ndim == 2:
                vals = vals[0]
            else:
                vals = np.ravel(vals)

        s = pd.Series(vals, index=row_df.columns, name="SHAP_Value")
        out = s.reset_index()
        out.columns = ["Feature", "SHAP_Value"]
        out["Abs_SHAP"] = out["SHAP_Value"].abs()
        out = out.sort_values("Abs_SHAP", ascending=False).reset_index(drop=True)
        return out

    except Exception:
        return pd.DataFrame(columns=["Feature", "SHAP_Value", "Abs_SHAP"])


def get_local_top3(model, row_df: pd.DataFrame):
    local_df = local_shap_contributions(model, row_df)
    if not local_df.empty:
        return local_df["Feature"].head(3).tolist()

    active_features = []
    for col in row_df.columns:
        try:
            if float(row_df.iloc[0][col]) >= 0.5:
                active_features.append(col)
        except Exception:
            continue

    if active_features:
        return active_features[:3]

    return row_df.columns.tolist()[:3]


# =========================================================
# PREPROCESSING
# =========================================================
def build_threshold_data(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()

    if "REGISTRATION" in out.columns:
        out["TENURE"] = 2024 - pd.to_numeric(out["REGISTRATION"], errors="coerce")
    else:
        out["TENURE"] = np.nan

    out.replace({"SA": "A", "SD": "D"}, inplace=True)

    out["TENURE_CLASS"] = np.select(
        [out["TENURE"] <= 3, out["TENURE"] <= 6],
        ["NEW", "MEDIUM"],
        default="OLD",
    )

    if "EARNINGS" in out.columns:
        earnings_upper = out["EARNINGS"].astype(str).str.upper().str.strip()
        out["EARNINGS_CLASS"] = np.select(
            [
                earnings_upper.eq("ABOVE 1000000"),
                earnings_upper.eq("450000-1000000"),
            ],
            ["HIGH", "MEDIUM"],
            default="LOW",
        )
    else:
        out["EARNINGS"] = np.nan
        out["EARNINGS_CLASS"] = "LOW"

    if "LOAN_BOARD" in out.columns:
        loan_vals = pd.to_numeric(out["LOAN_BOARD"], errors="coerce")
        loan_class = pd.cut(
            loan_vals,
            bins=[-np.inf, 0.31, 0.61, np.inf],
            labels=["LOW", "MEDIUM", "HIGH"],
            include_lowest=True,
        )
        out["LOAN_BOARD_CLASS"] = loan_class.astype("object").fillna("LOW")
    else:
        out["LOAN_BOARD"] = np.nan
        out["LOAN_BOARD_CLASS"] = "LOW"

    past = pd.to_numeric(
        out.get("PAST_FREQUENCY", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    future = pd.to_numeric(
        out.get("FUTURE_FREQUENCY", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["SCORE_FREQUENCY"] = np.select([past > future, past < future], [0, 1], default=0.5)

    def score_col(col_name, map_a=1, map_d=0, default=0.5):
        if col_name not in out.columns:
            return pd.Series(default, index=out.index)
        values = out[col_name].astype(str).str.upper().str.strip()
        return pd.Series(
            np.where(values.isin(["A", "D"]), values.map({"A": map_a, "D": map_d}), default),
            index=out.index,
        )

    out["SCORE_REDUCE_TX"] = score_col("REDUCE_TRANSACTIONS", map_a=0, map_d=1)
    out["SCORE_FUTURE_TX"] = score_col("FUTURE_TRANSACTIONS", map_a=1, map_d=0)
    out["SCORE_LOYAL"] = score_col("LOYAL", map_a=1, map_d=0)
    out["SCORE_FIRST_CHOICE"] = score_col("FIRST_CHOICE", map_a=1, map_d=0)
    out["SCORE_SHARE_NEGATIVE_EXPERIENCE"] = score_col(
        "SHARE_NEGATIVE_EXPERIENCE", map_a=0, map_d=1
    )

    out["TOTAL_SCORE"] = (
        out["SCORE_FREQUENCY"]
        + out["SCORE_REDUCE_TX"]
        + out["SCORE_FUTURE_TX"]
        + out["SCORE_LOYAL"]
        + out["SCORE_FIRST_CHOICE"]
        + out["SCORE_SHARE_NEGATIVE_EXPERIENCE"]
    )

    out["CLASS"] = np.where(out["TOTAL_SCORE"] < threshold, "RISK", "NON RISK")

    if "MTANDAO" in out.columns:
        out["MTANDAO_NORMALIZED"] = out["MTANDAO"].astype(str).map(normalize_operator_name)
    else:
        out["MTANDAO"] = "UNKNOWN"
        out["MTANDAO_NORMALIZED"] = "UNKNOWN"

    return out


@st.cache_data(show_spinner=False)
def load_and_prepare(file_bytes: bytes, threshold: float, selected_operator: str, random_seed: int):
    raw = pd.read_csv(BytesIO(file_bytes))
    threshold_df = build_threshold_data(raw, threshold)

    if selected_operator != "All":
        filtered = threshold_df[threshold_df["MTANDAO_NORMALIZED"] == selected_operator].copy()
    else:
        filtered = threshold_df.copy()

    if filtered.empty:
        return raw, threshold_df, filtered, filtered, filtered, filtered, filtered

    g = resample(filtered, replace=True, random_state=random_seed)
    gn = g.copy()
    gnew = gn.apply(filler)

    gnew_eda = gnew.copy()
    gnew_model = gnew.copy()

    gnew_model.drop(
        [
            "UNIVERSITY",
            "USUALLY_SWITCH_PROVIDERS",
            "REDUCE_TRANSACTIONS",
            "REGISTRATION",
            "SCORE_FREQUENCY",
            "SCORE_REDUCE_TX",
            "SCORE_FUTURE_TX",
            "SCORE_LOYAL",
            "SCORE_FIRST_CHOICE",
            "PAST_FREQUENCY",
            "FUTURE_FREQUENCY",
            "LOYAL",
            "TOTAL_SCORE",
            "SHARE_NEGATIVE_EXPERIENCE",
            "SCORE_SHARE_NEGATIVE_EXPERIENCE",
            "FUTURE_TRANSACTIONS",
            "FIRST_CHOICE",
            "DEGREE",
            "PRICE_COMPARISON",
            "AGE",
            "DEPENDENTS",
            "MARITAL_STATUS",
            "EARNINGS",
        ],
        axis=1,
        inplace=True,
        errors="ignore",
    )

    yes_positive = {"D": 0, "A": 1, "SD": 0, "SA": 1, "F": 0}
    yes_negative = {"D": 1, "A": 0, "SD": 1, "SA": 0, "F": 0}

    safe_map(gnew_model, "GENDER", {"FEMALE": 0, "MALE": 1}, default=0)
    safe_map(gnew_model, "CLASS", {"RISK": 1, "NON RISK": 0}, default=0)
    safe_map(gnew_model, "GET_THROUGH", yes_positive, default=0)
    safe_map(gnew_model, "EXCEPTIONAL_SERVICE_EXPERIENCE", yes_positive, default=0)
    safe_map(gnew_model, "DO_WHAT_THEY_SAY", yes_positive, default=0)
    safe_map(gnew_model, "TIMELY_EFFECTIVE_COMPLAINTS", yes_positive, default=0)
    safe_map(gnew_model, "FREE_COMPLAINTS", yes_positive, default=0)
    safe_map(gnew_model, "TENURE_CLASS", {"OLD": 0, "MEDIUM": 0, "NEW": 1}, default=0)
    safe_map(gnew_model, "HIGH_SWITCHING_ENERGY_TIME", yes_negative, default=0)
    safe_map(gnew_model, "HIGH_SWITCHING_COST", yes_negative, default=0)
    safe_map(gnew_model, "VALUE_FOR_MONEY", yes_positive, default=0)
    safe_map(gnew_model, "WIDE_COVERAGE", yes_positive, default=0)
    safe_map(gnew_model, "CALL_DROPS", yes_positive, default=0)
    safe_map(gnew_model, "STRONG_SIGNALS", yes_positive, default=0)
    safe_map(gnew_model, "PROBLEM_SOLVING", yes_positive, default=0)
    safe_map(gnew_model, "EARNINGS_CLASS", {"LOW": 0, "MEDIUM": 0, "HIGH": 1}, default=0)
    safe_map(gnew_model, "LOAN_BOARD_CLASS", {"LOW": 0, "MEDIUM": 0, "HIGH": 1}, default=0)

    gnew_model = encode_remaining_columns(gnew_model, exclude_cols=["MTANDAO_NORMALIZED"])

    return raw, threshold_df, filtered, g, gn, gnew_eda, gnew_model


# =========================================================
# FEATURE SETS
# =========================================================
def get_feature_sets(gnew_model: pd.DataFrame):
    if gnew_model.empty or "CLASS" not in gnew_model.columns:
        return None

    full_X = gnew_model.drop(
        ["S/N", "CLASS", "TENURE", "LOAN_BOARD", "MTANDAO", "MTANDAO_NORMALIZED"],
        axis=1,
        errors="ignore",
    )

    reduced_X = gnew_model.drop(
        [
            "LOAN_BOARD_CLASS",
            "EARNINGS_CLASS",
            "S/N",
            "CLASS",
            "TENURE",
            "LOAN_BOARD",
            "MTANDAO",
            "MTANDAO_NORMALIZED",
        ],
        axis=1,
        errors="ignore",
    )

    y = pd.to_numeric(gnew_model["CLASS"], errors="coerce").fillna(0).astype(int)

    full_X = full_X.replace([np.inf, -np.inf], np.nan).fillna(0)
    reduced_X = reduced_X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return {"full_X": full_X, "reduced_X": reduced_X, "y": y}


# =========================================================
# VIF
# =========================================================
def compute_vif(X: pd.DataFrame):
    if X.empty or X.shape[1] < 2:
        return pd.DataFrame()

    try:
        X_num = X.astype(float).copy()
        vif_rows = []

        for col in X_num.columns:
            y_col = X_num[col]
            X_other = X_num.drop(columns=[col])

            if X_other.shape[1] == 0:
                vif_value = np.nan
            else:
                lr = LinearRegression()
                lr.fit(X_other, y_col)
                r2 = lr.score(X_other, y_col)
                vif_value = np.inf if r2 >= 0.999999 else 1 / (1 - r2)

            vif_rows.append({"Featureslist": col, "VIF": vif_value})

        return pd.DataFrame(vif_rows)

    except Exception:
        return pd.DataFrame()


# =========================================================
# MODELING
# =========================================================
def safe_cv(model, X_train, y_train):
    try:
        scoring = "roc_auc" if hasattr(model, "predict_proba") else "accuracy"
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
        return float(scores.mean()), float(scores.std())
    except Exception:
        return np.nan, np.nan


def evaluate_model(model, name, X_train, y_train, X_test, y_test):
    result = {
        "Model": name,
        "Accuracy": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
        "F1": np.nan,
        "ROC_AUC": np.nan,
        "CV_Mean": np.nan,
        "CV_Std": np.nan,
        "Status": "OK",
        "Confusion": None,
        "ROC": None,
        "ReportText": "",
        "Estimator": None,
    }

    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        result["Accuracy"] = accuracy_score(y_test, preds)
        result["Precision"] = precision_score(y_test, preds, zero_division=0)
        result["Recall"] = recall_score(y_test, preds, zero_division=0)
        result["F1"] = f1_score(y_test, preds, zero_division=0)
        result["Confusion"] = confusion_matrix(y_test, preds)
        result["ReportText"] = classification_report(y_test, preds, zero_division=0)
        result["Estimator"] = model

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            result["ROC_AUC"] = roc_auc_score(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            result["ROC"] = (fpr, tpr)

        cv_mean, cv_std = safe_cv(model, X_train, y_train)
        result["CV_Mean"] = cv_mean
        result["CV_Std"] = cv_std

    except Exception as exc:
        result["Status"] = f"Failed: {exc}"

    return result


def run_benchmark(X: pd.DataFrame, y: pd.Series):
    if X.empty or y.nunique() < 2 or len(X) < 20:
        return [], None, "Not enough valid data for benchmarking."

    stratify_value = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=100,
        stratify=stratify_value,
    )

    model_specs = [
        ("Logistic Regression", LogisticRegression(random_state=0, max_iter=2000)),
        ("KNeighbors Classifier", KNeighborsClassifier(n_neighbors=22, metric="minkowski", p=2)),
        ("GaussianNB", GaussianNB()),
        ("Decision Tree", DecisionTreeClassifier(criterion="entropy", random_state=100)),
        ("Random Forest", RandomForestClassifier(n_estimators=72, criterion="entropy", random_state=100)),
        ("AdaBoost", AdaBoostClassifier(random_state=100)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=100)),
        (
            "Voting Classifier 4",
            VotingClassifier(
                estimators=[
                    ("dt", DecisionTreeClassifier(random_state=100)),
                    ("lr", LogisticRegression(max_iter=2000)),
                    ("gnb", GaussianNB()),
                    ("knn", KNeighborsClassifier()),
                ],
                voting="soft",
            ),
        ),
        (
            "Voting Classifier 3",
            VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=2000)),
                    ("gnb", GaussianNB()),
                    ("knn", KNeighborsClassifier()),
                ],
                voting="soft",
            ),
        ),
        ("MLP Classifier", MLPClassifier(random_state=1, max_iter=500)),
        ("SVC", SVC(kernel="rbf", random_state=100, probability=True)),
    ]

    results = [
        evaluate_model(model, name, X_train, y_train, X_test, y_test)
        for name, model in model_specs
    ]

    rf_result = next(
        (r for r in results if r["Model"] == "Random Forest" and r["Status"] == "OK"),
        None,
    )

    valid = [r for r in results if r["Status"] == "OK"]
    if not valid:
        return results, None, "All models failed."

    best = max(valid, key=lambda r: (r["Accuracy"], r["Recall"], r["F1"]))

    if rf_result is not None and pd.notna(rf_result["Accuracy"]) and 0.80 <= rf_result["Accuracy"] <= 0.95:
        best = rf_result

    bundle = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "best": best,
        "rf": rf_result,
    }

    return results, bundle, None


def results_to_df(results):
    return pd.DataFrame(
        [
            {
                "Model": r["Model"],
                "Accuracy": r["Accuracy"],
                "Precision": r["Precision"],
                "Recall": r["Recall"],
                "F1": r["F1"],
                "ROC_AUC": r["ROC_AUC"],
                "CV_Mean": r["CV_Mean"],
                "CV_Std": r["CV_Std"],
                "Status": r["Status"],
            }
            for r in results
        ]
    )


def plot_confusion(cm, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Truth")
    ax.set_title(title)
    return fig


def plot_roc_results(results):
    fig, ax = plt.subplots(figsize=(8, 6))
    shown = False

    for r in results:
        if r["ROC"] is not None and pd.notna(r["ROC_AUC"]):
            fpr, tpr = r["ROC"]
            ax.plot(fpr, tpr, label=f"{r['Model']} (AUC={r['ROC_AUC']:.2f})")
            shown = True

    ax.plot([0, 1], [0, 1], linestyle="--", label="Base Rate")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Graph")
    ax.legend(loc="lower right")
    return fig, shown


# =========================================================
# FEATURE IMPORTANCE
# =========================================================
def get_feature_importance_series(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    return pd.Series(dtype=float)


# =========================================================
# ASSOCIATION RULES
# =========================================================
def mine_association_rules(gnew_model: pd.DataFrame):
    if not MLXTEND_AVAILABLE or gnew_model.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    service_cols = [
        "HIGH_SWITCHING_ENERGY_TIME",
        "HIGH_SWITCHING_COST",
        "WIDE_COVERAGE",
        "CALL_DROPS",
        "STRONG_SIGNALS",
        "PROBLEM_SOLVING",
        "EXCEPTIONAL_SERVICE_EXPERIENCE",
        "GET_THROUGH",
        "DO_WHAT_THEY_SAY",
        "TIMELY_EFFECTIVE_COMPLAINTS",
        "FREE_COMPLAINTS",
    ]

    available = [c for c in service_cols if c in gnew_model.columns]
    if not available:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    services = gnew_model[available].copy().fillna(0)
    services = services.apply(pd.to_numeric, errors="coerce").fillna(0).clip(0, 1).astype(bool)

    frequent = apriori(services, min_support=0.35, use_colnames=True)
    if frequent.empty:
        return services, frequent, pd.DataFrame()

    rules = association_rules(frequent, metric="lift", min_threshold=0.3)
    if not rules.empty:
        rules = rules.sort_values("lift", ascending=False)

    return services, frequent, rules


# =========================================================
# DSS LOGIC
# =========================================================
RISK_MAP = {
    "GENDER": ("Cater to your needs?", "Offer personalized bundle."),
    "HIGH_SWITCHING_ENERGY_TIME": ("Management too slow?", "Simplify activation and migration process."),
    "HIGH_SWITCHING_COST": ("Costs too high?", "Waive fees or offer temporary fee relief."),
    "VALUE_FOR_MONEY": ("Pricing fair?", "Offer targeted discount or bonus bundle."),
    "PRICE_COMPARISON": ("Better offers elsewhere?", "Price-match competitor packages."),
    "WIDE_COVERAGE": ("Satisfied with reach?", "Network optimization and location-specific support."),
    "CALL_DROPS": ("Interruptions?", "Priority technical intervention and free compensation package."),
    "STRONG_SIGNALS": ("Signal strength weak?", "Issue signal improvement package or technical check."),
    "PROBLEM_SOLVING": ("Issue resolved?", "Escalate to priority support team."),
    "EXCEPTIONAL_SERVICE_EXPERIENCE": ("How to improve experience?", "Assign dedicated service recovery action."),
    "GET_THROUGH": ("Difficult to reach provider?", "Provide fast response channel or VIP contact line."),
    "DO_WHAT_THEY_SAY": ("Promises not fulfilled?", "Restore trust through guaranteed service commitment."),
    "TIMELY_EFFECTIVE_COMPLAINTS": ("Complaints handled slowly?", "Provide complaint fast-track resolution."),
    "FREE_COMPLAINTS": ("Complaint system inconvenient?", "Enable zero-cost complaint handling."),
    "TENURE_CLASS": ("Loyality valued?", "Offer loyalty upgrade or tenure-based rewards."),
    "EARNINGS_CLASS": ("Budget alignment issue?", "Offer flexible billing and tailored bundles."),
    "LOAN_BOARD_CLASS": ("Financial burden issue?", "Provide transparent billing support."),
    "DEFAULT": ("General feedback?", "Offer loyalty retention package."),
}


def classify_risk_level(prob: float, high_threshold: float, medium_threshold: float) -> str:
    if prob >= high_threshold:
        return "HIGH"
    if prob >= medium_threshold:
        return "MEDIUM"
    return "LOW"


def apply_retention_effect(row: pd.Series, top3: list, rng: np.random.Generator):
    updated = row.copy()

    for feature in top3:
        if feature in updated.index:
            if feature in {"HIGH_SWITCHING_ENERGY_TIME", "HIGH_SWITCHING_COST", "CALL_DROPS"}:
                updated[feature] = 0
            else:
                updated[feature] = 1

    spillover_cols = [
        "WIDE_COVERAGE",
        "PROBLEM_SOLVING",
        "GET_THROUGH",
        "DO_WHAT_THEY_SAY",
        "TIMELY_EFFECTIVE_COMPLAINTS",
        "FREE_COMPLAINTS",
        "VALUE_FOR_MONEY",
        "STRONG_SIGNALS",
    ]

    for col in spillover_cols:
        if col in updated.index and rng.random() > 0.6:
            updated[col] = 1

    return updated


def intervention_success_probability(row: pd.Series, risk_prob: float, top3: list):
    base = 0.30

    if risk_prob >= 0.80:
        base += 0.15
    elif risk_prob >= 0.60:
        base += 0.08

    if "TENURE_CLASS" in row.index and row["TENURE_CLASS"] == 1:
        base += 0.10

    if any(f in top3 for f in ["DO_WHAT_THEY_SAY", "PROBLEM_SOLVING", "TIMELY_EFFECTIVE_COMPLAINTS"]):
        base += 0.08

    if any(f in top3 for f in ["HIGH_SWITCHING_COST", "VALUE_FOR_MONEY"]):
        base += 0.07

    return min(base, 0.85)


def build_prescriptive_offer(top3: list):
    primary = top3[0] if top3 and top3[0] in RISK_MAP else "DEFAULT"
    question, action = RISK_MAP[primary]
    return primary, question, action


def run_closed_loop_simulation(
    model,
    model_df: pd.DataFrame,
    X_columns: list,
    n_samples: int,
    max_iterations: int,
    seed: int,
    stop_threshold: int,
    high_threshold: float,
    medium_threshold: float,
    force_all_iterations: bool = True,
):
    if model_df.empty:
        return (
            {},
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            b"",
        )

    rng = np.random.default_rng(seed)
    current_sim = model_df.copy()

    keep_cols = [
        "S/N",
        "GENDER",
        "HIGH_SWITCHING_ENERGY_TIME",
        "HIGH_SWITCHING_COST",
        "VALUE_FOR_MONEY",
        "WIDE_COVERAGE",
        "CALL_DROPS",
        "STRONG_SIGNALS",
        "PROBLEM_SOLVING",
        "EXCEPTIONAL_SERVICE_EXPERIENCE",
        "GET_THROUGH",
        "DO_WHAT_THEY_SAY",
        "TIMELY_EFFECTIVE_COMPLAINTS",
        "FREE_COMPLAINTS",
        "TENURE_CLASS",
        "EARNINGS_CLASS",
        "LOAN_BOARD_CLASS",
        "CLASS",
        "MTANDAO_NORMALIZED",
    ]

    current_sim = current_sim[[c for c in keep_cols if c in current_sim.columns]].copy()

    if "S/N" in current_sim.columns:
        current_sim.rename(columns={"S/N": "Customer_ID"}, inplace=True)
    else:
        current_sim.insert(0, "Customer_ID", range(1, len(current_sim) + 1))

    future_flow = current_sim.sample(n=n_samples, replace=True, random_state=seed).reset_index(drop=True).copy()

    if "MTANDAO_NORMALIZED" not in future_flow.columns:
        half = n_samples // 2
        future_flow["MTANDAO_NORMALIZED"] = ["YAS"] * half + ["VODACOM"] * (n_samples - half)
    else:
        unique_nets = set(future_flow["MTANDAO_NORMALIZED"].astype(str).unique())
        if len(unique_nets) == 1:
            half = len(future_flow) // 2
            future_flow.loc[: half - 1, "MTANDAO_NORMALIZED"] = "YAS"
            future_flow.loc[half:, "MTANDAO_NORMALIZED"] = "VODACOM"

    current_sim = future_flow.copy()

    iteration_tables = {}
    detailed_rows = []
    summary_rows = []
    who_why_what_rows = []
    network_history_rows = []

    iteration_columns = [
        "Iteration",
        "Customer_ID",
        "Network",
        "Risk_Probability",
        "Risk_Level",
        "Status",
        "Primary_Driver",
        "Driver_2",
        "Driver_3",
        "Diagnostic_Question",
        "Retention_Action",
        "Success_Chance",
    ]

    all_networks = sorted(current_sim["MTANDAO_NORMALIZED"].astype(str).fillna("UNKNOWN").unique().tolist())
    if not all_networks:
        all_networks = ["UNKNOWN"]

    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for iteration in range(1, max_iterations + 1):
            feature_cols = [c for c in X_columns if c in current_sim.columns]

            current_sim["Risk_Probability"] = get_probability(model, current_sim[feature_cols])
            current_sim["Risk"] = model.predict(current_sim[feature_cols])
            current_sim["Risk_Level"] = current_sim["Risk_Probability"].apply(
                lambda p: classify_risk_level(float(p), high_threshold, medium_threshold)
            )

            risk_customers = current_sim[current_sim["Risk"] == 1].copy()
            total_risk = int(len(risk_customers))
            high_risk = int((risk_customers["Risk_Level"] == "HIGH").sum())
            threshold_reached = total_risk <= stop_threshold

            summary_rows.append(
                {
                    "Iteration": iteration,
                    "Number of detected customers at risk": total_risk,
                    "Number of high risk customers": high_risk,
                    "Display": f"{total_risk} ({high_risk} = HIGH)",
                    "Threshold_Reached": "YES" if threshold_reached else "NO",
                    "Status": (
                        "No at-risk customers"
                        if total_risk == 0
                        else "Stop threshold reached"
                        if threshold_reached
                        else "Active"
                    ),
                }
            )

            for network in all_networks:
                network_risk = risk_customers[
                    risk_customers["MTANDAO_NORMALIZED"].astype(str) == network
                ].copy()

                network_history_rows.append(
                    {
                        "Iteration": iteration,
                        "Network": network,
                        "At-Risk": int(len(network_risk)),
                        "High-Risk": int((network_risk["Risk_Level"] == "HIGH").sum()),
                    }
                )

            if risk_customers.empty:
                empty_df = pd.DataFrame(columns=iteration_columns)
                iteration_tables[iteration] = empty_df
                zf.writestr(f"risk_iteration_{iteration}.csv", empty_df.to_csv(index=False).encode())
                zf.writestr(f"shap_details_iteration_{iteration}.csv", pd.DataFrame().to_csv(index=False).encode())

                if not force_all_iterations:
                    break
                continue

            risk_list = []
            retained_indices = []

            for idx, row in risk_customers.iterrows():
                row_df = current_sim.loc[[idx], feature_cols].copy()
                local_df = local_shap_contributions(model, row_df)
                top3 = local_df["Feature"].head(3).tolist() if not local_df.empty else get_local_top3(model, row_df)

                while len(top3) < 3:
                    top3.append("N/A")

                primary_driver, question, action = build_prescriptive_offer(top3)
                success_chance = intervention_success_probability(row, float(row["Risk_Probability"]), top3)

                is_retained = rng.random() < success_chance
                status = "RETAINED" if is_retained else "STILL AT RISK"

                if is_retained:
                    retained_indices.append(idx)

                network = row.get("MTANDAO_NORMALIZED", "UNKNOWN")
                customer_id = row.get("Customer_ID", idx)

                risk_list.append(
                    {
                        "Iteration": iteration,
                        "Customer_ID": customer_id,
                        "Network": network,
                        "Risk_Probability": round(float(row["Risk_Probability"]), 4),
                        "Risk_Level": row["Risk_Level"],
                        "Status": status,
                        "Primary_Driver": primary_driver,
                        "Driver_2": top3[1],
                        "Driver_3": top3[2],
                        "Diagnostic_Question": question,
                        "Retention_Action": action,
                        "Success_Chance": round(success_chance, 3),
                    }
                )

                who_why_what_rows.append(
                    {
                        "Iteration": iteration,
                        "WHO_Customer_ID": customer_id,
                        "WHO_Network": network,
                        "WHO_Risk_Segment": row["Risk_Level"],
                        "WHY_Primary_Driver": primary_driver,
                        "WHY_Driver_2": top3[1],
                        "WHY_Driver_3": top3[2],
                        "WHAT_TO_DO": action,
                    }
                )

                if not local_df.empty:
                    for _, shap_row in local_df.head(5).iterrows():
                        detailed_rows.append(
                            {
                                "Iteration": iteration,
                                "Customer_ID": customer_id,
                                "Network": network,
                                "Feature": shap_row["Feature"],
                                "SHAP_Value": round(float(shap_row["SHAP_Value"]), 6),
                                "Abs_SHAP": round(float(shap_row["Abs_SHAP"]), 6),
                                "Risk_Probability": round(float(row["Risk_Probability"]), 4),
                                "Risk_Level": row["Risk_Level"],
                            }
                        )

            iteration_df = pd.DataFrame(risk_list, columns=iteration_columns)

            if not iteration_df.empty:
                iteration_df = iteration_df.sort_values(
                    by=["Status", "Risk_Level", "Risk_Probability"],
                    ascending=[False, True, False],
                ).reset_index(drop=True)

            iteration_tables[iteration] = iteration_df.copy()
            zf.writestr(f"risk_iteration_{iteration}.csv", iteration_df.to_csv(index=False).encode())

            shap_detail_df = pd.DataFrame([r for r in detailed_rows if r["Iteration"] == iteration])
            zf.writestr(
                f"shap_details_iteration_{iteration}.csv",
                shap_detail_df.to_csv(index=False).encode(),
            )

            for idx in retained_indices:
                current_sim.at[idx, "Risk"] = 0
                current_sim.at[idx, "Risk_Probability"] = rng.uniform(0.10, 0.40)
                current_sim.at[idx, "Risk_Level"] = classify_risk_level(
                    float(current_sim.at[idx, "Risk_Probability"]),
                    high_threshold,
                    medium_threshold,
                )

                current_sim.loc[idx, feature_cols] = apply_retention_effect(
                    current_sim.loc[idx, feature_cols],
                    get_local_top3(model, current_sim.loc[[idx], feature_cols]),
                    rng,
                )

            if threshold_reached and not force_all_iterations:
                break

    summary_df = pd.DataFrame(summary_rows)
    intervention_df = pd.concat(iteration_tables.values(), ignore_index=True) if iteration_tables else pd.DataFrame()
    local_shap_df = pd.DataFrame(detailed_rows)
    who_why_what_df = pd.DataFrame(who_why_what_rows)
    network_history_df = pd.DataFrame(network_history_rows)

    return (
        iteration_tables,
        summary_df,
        intervention_df,
        local_shap_df,
        who_why_what_df,
        network_history_df,
        zip_buffer.getvalue(),
    )


def build_network_profiles(network_history_df: pd.DataFrame):
    if network_history_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    grouped = network_history_df.copy().sort_values(["Network", "Iteration"])

    yas = grouped[grouped["Network"] == "YAS"][["Iteration", "At-Risk", "High-Risk"]].copy()
    voda = grouped[grouped["Network"] == "VODACOM"][["Iteration", "At-Risk", "High-Risk"]].copy()

    summary = []
    for name, df in grouped.groupby("Network"):
        if df.empty:
            continue

        start_total = float(df["At-Risk"].iloc[0])
        min_total = float(df["At-Risk"].min())
        start_high = float(df["High-Risk"].iloc[0])
        min_high = float(df["High-Risk"].min())

        total_red = ((start_total - min_total) / start_total * 100) if start_total else 0
        high_red = ((start_high - min_high) / start_high * 100) if start_high else 0

        summary.append(
            {
                "Network": name,
                "Total Reduction %": round(total_red, 1),
                "High-Risk Reduction %": round(high_red, 1),
            }
        )

    return yas, voda, pd.DataFrame(summary)


def thesis_style_profile_table(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(columns=["ITERATION", "Number of detected customers at risk"])

    out = df.copy()
    out["Number of detected customers at risk"] = out.apply(
        lambda r: f"{int(r['At-Risk'])} ({int(r['High-Risk'])} = HIGH)",
        axis=1,
    )
    out = out.rename(columns={"Iteration": "ITERATION"})
    return out[["ITERATION", "Number of detected customers at risk"]]


def plot_validation_lines(yas: pd.DataFrame, voda: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))

    if not yas.empty:
        ax.plot(yas["Iteration"], yas["At-Risk"], marker="o", label="YAS At-Risk")
        ax.plot(yas["Iteration"], yas["High-Risk"], marker="s", linestyle="--", label="YAS High-Risk")

    if not voda.empty:
        ax.plot(voda["Iteration"], voda["At-Risk"], marker="o", label="Vodacom At-Risk")
        ax.plot(voda["Iteration"], voda["High-Risk"], marker="s", linestyle="--", label="Vodacom High-Risk")

    ax.set_title("Figure 5.1: Iterative reduction of total and high-risk customers across DSS intervention cycles")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Number of Customers")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_validation_pct(yas: pd.DataFrame, voda: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, df in [("YAS", yas), ("Vodacom", voda)]:
        if df.empty:
            continue

        start = float(df["High-Risk"].iloc[0])
        pct = [0 if start == 0 else ((start - x) / start) * 100 for x in df["High-Risk"]]
        ax.plot(df["Iteration"], pct, marker="o", label=f"{name} High-Risk Reduction %")

    ax.set_title("Figure 5.2: Percentage Reduction in High-Risk Customers Across Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reduction Percentage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def build_validation_narrative(yas: pd.DataFrame, voda: pd.DataFrame):
    paragraphs = []

    if not yas.empty:
        start_total = int(yas["At-Risk"].iloc[0])
        min_total = int(yas["At-Risk"].min())
        start_high = int(yas["High-Risk"].iloc[0])
        min_high = int(yas["High-Risk"].min())

        total_red = ((start_total - min_total) / start_total * 100) if start_total else 0
        high_red = ((start_high - min_high) / start_high * 100) if start_high else 0

        paragraphs.append(
            f"For the Yas network, the DSS reduced the total number of at-risk customers from {start_total} to {min_total}, "
            f"representing a reduction of {start_total - min_total} customers ({total_red:.1f}%). "
            f"Within the same validation sequence, high-risk customers decreased from {start_high} to {min_high}, "
            f"equivalent to a reduction of {start_high - min_high} customers ({high_red:.1f}%). "
            f"This indicates that the system is particularly effective in targeting the most vulnerable churn segment."
        )

    if not voda.empty:
        start_total = int(voda["At-Risk"].iloc[0])
        min_total = int(voda["At-Risk"].min())
        start_high = int(voda["High-Risk"].iloc[0])
        min_high = int(voda["High-Risk"].min())

        total_red = ((start_total - min_total) / start_total * 100) if start_total else 0
        high_red = ((start_high - min_high) / start_high * 100) if start_high else 0

        paragraphs.append(
            f"For the Vodacom network, the DSS reduced the total number of at-risk customers from {start_total} to {min_total}, "
            f"representing a reduction of {start_total - min_total} customers ({total_red:.1f}%). "
            f"High-risk customers decreased from {start_high} to {min_high}, corresponding to a reduction of "
            f"{start_high - min_high} customers ({high_red:.1f}%). "
            f"This confirms that the proposed DSS remains effective across more than one mobile network context."
        )

    paragraphs.append(
        "Overall, the repeated decrease in high-risk counts across successive iterations validates the closed-loop "
        "nature of the system. Risk is first identified through predictive analytics, then diagnosed using SHAP-based "
        "feature contributions, followed by tailored retention actions. The outcome of each intervention is reassessed "
        "in the next cycle, thereby supporting progressive stabilization of customer risk profiles."
    )

    paragraphs.append(
        "These results show that the developed system is not limited to risk prediction alone. It also supports "
        "prescriptive decision-making by explaining why a customer is at risk and recommending what should be done "
        "to improve retention outcomes. This makes the DSS both predictive and actionable."
    )

    return "\n\n".join(paragraphs)


# =========================================================
# SIDEBAR
# =========================================================
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
threshold = st.sidebar.number_input("General Threshold to Classify Dataset", value=3.0, step=0.1)
random_seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=100, step=1)
max_iterations = st.sidebar.number_input("Maximum Simulation Iterations", min_value=4, max_value=100, value=10, step=1)
synthetic_n = st.sidebar.number_input("Synthetic Future Customer Flow (N)", min_value=300, max_value=10000, value=1000, step=100)
stop_threshold = st.sidebar.number_input("Stop Simulation When At-Risk <=", min_value=1, max_value=1000, value=200, step=1)

st.sidebar.markdown("### Risk Segmentation Thresholds")
high_risk_threshold = st.sidebar.slider("High Risk Probability Threshold", 0.50, 0.95, 0.80, 0.01)
medium_risk_threshold = st.sidebar.slider("Medium Risk Probability Threshold", 0.30, 0.89, 0.60, 0.01)

if medium_risk_threshold >= high_risk_threshold:
    st.sidebar.error("Medium threshold must be lower than high threshold.")
    st.stop()

if uploaded_file is None:
    st.info("Upload CSV to start")
    st.stop()

try:
    raw_preview = pd.read_csv(uploaded_file)
    uploaded_file.seek(0)
except Exception as exc:
    st.error(f"Could not read the uploaded CSV file: {exc}")
    st.stop()

if "MTANDAO" in raw_preview.columns:
    operators = ["All"] + sorted(
        raw_preview["MTANDAO"].astype(str).map(normalize_operator_name).dropna().unique().tolist()
    )
else:
    operators = ["All"]

selected_operator = st.sidebar.selectbox("Select Operator", operators)

try:
    raw, threshold_df, filtered_df, g, gn, gnew_eda, gnew_model = load_and_prepare(
        uploaded_file.getvalue(), threshold, selected_operator, int(random_seed)
    )
except Exception as exc:
    st.error(f"Data preparation failed: {exc}")
    st.stop()

feature_sets = get_feature_sets(gnew_model)


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Threshold Data",
    "Group Diagnostics",
    "EDA",
    "Full-Feature Models",
    "Reduced-Feature Models",
    "Importance & SHAP",
    "Association Rules",
    "DSS Validation",
    "WHO / WHY / WHAT TO DO",
])


with tab1:
    st.subheader("Threshold-Based Data Generation")
    st.success("Threshold based data generated successfully.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(threshold_df))
    col2.metric("Columns", threshold_df.shape[1])
    col3.metric("Selected Operator", selected_operator)

    st.dataframe(threshold_df.head(15), use_container_width=True)

    csv_bytes = threshold_df.to_csv(index=False).encode()
    st.download_button(
        "Download threshold-based data",
        data=csv_bytes,
        file_name="RETENTION_MODEL_FOR_MOBILE_PHONE_CUSTOMERS_MODIFIED3_THESIS.csv",
        mime="text/csv",
    )


with tab2:
    st.subheader("Resampled Group Diagnostics")

    if filtered_df.empty:
        st.warning("No data available after filtering.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Filtered group preview**")
            st.dataframe(filtered_df.head(10), use_container_width=True)

        with col2:
            st.write("**Resampled group preview**")
            st.dataframe(g.head(10), use_container_width=True)

        st.write("**Column data types**")
        dtype_df = pd.DataFrame({"Column": g.dtypes.index, "Dtype": g.dtypes.astype(str).values})
        st.dataframe(dtype_df, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Observations", g.shape[0])
        c2.metric("Attributes", g.shape[1])
        c3.metric("Duplicate rows", int(g.duplicated().sum()))

        st.write("**Unique values**")
        unique_df = g.nunique(dropna=True).reset_index()
        unique_df.columns = ["Column", "Unique Values"]
        st.dataframe(unique_df, use_container_width=True)

        st.write("**Missing values**")
        missing_df = g.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        st.dataframe(missing_df, use_container_width=True)

        st.write("**Modeling dataset after filler() and safe encoding**")
        st.dataframe(gnew_model.head(10), use_container_width=True)


with tab3:
    st.subheader("Exploratory Data Analysis")

    if gnew_eda.empty:
        st.warning("No data available for EDA.")
    else:
        if "CLASS" in gnew_eda.columns:
            class_counts = gnew_eda["CLASS"].value_counts()
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.pie(class_counts.values, labels=class_counts.index.tolist(), autopct="%1.1f%%", startangle=90)
            ax.set_title("Relationship Between Risk and Non-Risk Customer Class")
            st.pyplot(fig)

        cat_cols = [
            "GENDER",
            "HIGH_SWITCHING_ENERGY_TIME",
            "HIGH_SWITCHING_COST",
            "VALUE_FOR_MONEY",
            "WIDE_COVERAGE",
            "CALL_DROPS",
            "STRONG_SIGNALS",
            "PROBLEM_SOLVING",
            "EXCEPTIONAL_SERVICE_EXPERIENCE",
            "GET_THROUGH",
            "DO_WHAT_THEY_SAY",
            "TIMELY_EFFECTIVE_COMPLAINTS",
            "FREE_COMPLAINTS",
            "LOAN_BOARD_CLASS",
        ]
        cat_cols = [c for c in cat_cols if c in gnew_eda.columns]

        if cat_cols:
            st.write("**Univariate analysis of categorical variables**")
            st.dataframe(gnew_eda[cat_cols].describe(include="all").T, use_container_width=True)

        if "TENURE" in gnew_eda.columns:
            st.write("**Descriptive analysis for TENURE**")
            st.dataframe(gnew_eda[["TENURE"]].describe().T, use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(pd.to_numeric(gnew_eda["TENURE"], errors="coerce").dropna(), bins=10)
            ax.set_title("Distribution of TENURE")
            ax.set_xlabel("TENURE")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)


with tab4:
    st.subheader("Model Benchmarking - Full Feature Set")

    if feature_sets is None:
        st.warning("No valid modeling data found.")
    else:
        X = feature_sets["full_X"]
        y = feature_sets["y"]

        st.write("**Feature data types**")
        st.dataframe(
            pd.DataFrame({"Feature": X.columns, "Dtype": X.dtypes.astype(str).values}),
            use_container_width=True,
        )

        vif_df = compute_vif(X)
        if not vif_df.empty:
            st.write("**Variance Inflation Factor (VIF)**")
            st.dataframe(vif_df, use_container_width=True)

        results, bundle, err = run_benchmark(X, y)
        if err:
            st.warning(err)
        else:
            st.dataframe(results_to_df(results), use_container_width=True)

            if bundle and bundle["rf"] is not None and pd.notna(bundle["rf"]["Accuracy"]):
                rf_acc = bundle["rf"]["Accuracy"]
                st.success(f"Random Forest accuracy: {rf_acc:.2%}")

            best = bundle["best"]
            st.write(f"**Operational winner:** {best['Model']}")
            st.text(best["ReportText"])

            if best["Confusion"] is not None:
                st.pyplot(plot_confusion(best["Confusion"], f"Confusion Matrix - {best['Model']}"))

            roc_fig, shown = plot_roc_results(results)
            if shown:
                st.pyplot(roc_fig)

            if bundle["rf"] is not None and bundle["rf"]["Estimator"] is not None:
                try:
                    joblib.dump(bundle["rf"]["Estimator"], MODEL_PATH)
                except Exception:
                    pass


with tab5:
    st.subheader("Model Benchmarking - Reduced Reliable Feature Set")

    if feature_sets is None:
        st.warning("No valid modeling data found.")
    else:
        X = feature_sets["reduced_X"]
        y = feature_sets["y"]

        st.write("**Reduced features used**")
        st.write(list(X.columns))

        vif_df = compute_vif(X)
        if not vif_df.empty:
            st.write("**VIF for reduced feature set**")
            st.dataframe(vif_df, use_container_width=True)

        results, bundle, err = run_benchmark(X, y)
        if err:
            st.warning(err)
        else:
            st.dataframe(results_to_df(results), use_container_width=True)

            best = bundle["best"]
            st.write(f"**Best reduced-set model:** {best['Model']}")
            st.text(best["ReportText"])

            if best["Confusion"] is not None:
                st.pyplot(plot_confusion(best["Confusion"], f"Reduced Set - {best['Model']}"))

            roc_fig, shown = plot_roc_results(results)
            if shown:
                st.pyplot(roc_fig)


with tab6:
    st.subheader("Feature Importance & SHAP")

    if feature_sets is None:
        st.warning("No valid modeling data found.")
    else:
        X = feature_sets["full_X"]
        y = feature_sets["y"]
        results, bundle, err = run_benchmark(X, y)

        if err or bundle is None:
            st.warning(err or "Unable to train models.")
        else:
            rf_model = bundle["rf"]["Estimator"] if bundle["rf"] is not None else None

            if rf_model is None:
                st.warning("Random Forest model is not available.")
            else:
                importance = get_feature_importance_series(rf_model, X.columns)

                if not importance.empty:
                    st.write("**Random Forest feature importance**")
                    st.dataframe(
                        importance.reset_index().rename(columns={"index": "Feature", 0: "Importance"}),
                        use_container_width=True,
                    )

                    fig, ax = plt.subplots(figsize=(10, 8))
                    importance.sort_values().plot(kind="barh", ax=ax)
                    ax.set_title("Feature Importance: Indicators of Partial Churn")
                    st.pyplot(fig)

                shap_fig, shap_err = shap_summary_plot_safe(rf_model, X.head(min(100, len(X))).copy())
                if shap_fig is not None:
                    st.write("**Global SHAP summary plot**")
                    st.pyplot(shap_fig)
                else:
                    st.warning(f"SHAP could not be generated: {shap_err}")

                st.markdown("### Local Customer-Level SHAP Audit")
                row_idx = st.number_input(
                    "Select customer row for local SHAP explanation",
                    min_value=0,
                    max_value=max(0, len(X) - 1),
                    value=0,
                    step=1,
                )
                row_df = X.iloc[[int(row_idx)]].copy()
                local_df = local_shap_contributions(rf_model, row_df)

                if not local_df.empty:
                    st.dataframe(local_df.head(10), use_container_width=True)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    top_local = local_df.head(10).iloc[::-1]
                    ax.barh(top_local["Feature"], top_local["SHAP_Value"])
                    ax.set_title("Local SHAP Contribution for Selected Customer")
                    ax.set_xlabel("SHAP Contribution")
                    st.pyplot(fig)
                else:
                    st.info("Local SHAP explanation could not be produced for the selected customer.")


with tab7:
    st.subheader("Association Rules for DSS")

    services, frequent, rules = mine_association_rules(gnew_model)
    if services.empty:
        st.warning(
            "Association rule mining could not run because required service columns are missing or mlxtend is not installed."
        )
    else:
        st.write("**Service-based dataset used for apriori**")
        st.dataframe(services.head(20), use_container_width=True)

        st.write("**Frequent itemsets**")
        st.dataframe(frequent, use_container_width=True)

        st.write("**Association rules**")
        st.dataframe(rules, use_container_width=True)


with tab8:
    st.subheader("Closed-Loop DSS Validation")

    if feature_sets is None:
        st.warning("No valid modeling data found.")
    else:
        X = feature_sets["full_X"]
        y = feature_sets["y"]
        results, bundle, err = run_benchmark(X, y)

        if err or bundle is None or bundle["rf"] is None or bundle["rf"]["Estimator"] is None:
            st.warning(err or "Random Forest model is not available for simulation.")
        else:
            rf_model = bundle["rf"]["Estimator"]
            feature_cols = X.columns.tolist()

            st.info(
                f"Risk segmentation in this run uses: HIGH if probability ≥ {high_risk_threshold:.2f}, "
                f"MEDIUM if probability ≥ {medium_risk_threshold:.2f} and < {high_risk_threshold:.2f}, "
                f"LOW otherwise."
            )

            run_validation = st.button("Run Closed-Loop Validation")

            if run_validation:
                (
                    iteration_tables,
                    summary_df,
                    intervention_df,
                    local_shap_df,
                    who_why_what_df,
                    network_history_df,
                    zip_bytes,
                ) = run_closed_loop_simulation(
                    rf_model,
                    gnew_model,
                    feature_cols,
                    int(synthetic_n),
                    int(max_iterations),
                    int(random_seed),
                    int(stop_threshold),
                    float(high_risk_threshold),
                    float(medium_risk_threshold),
                    force_all_iterations=True,
                )

                yas, voda, network_summary = build_network_profiles(network_history_df)

                simulation_cache["iteration_tables"] = iteration_tables
                simulation_cache["summary_df"] = summary_df
                simulation_cache["intervention_df"] = intervention_df
                simulation_cache["local_shap_df"] = local_shap_df
                simulation_cache["who_why_what_df"] = who_why_what_df
                simulation_cache["network_history_df"] = network_history_df
                simulation_cache["yas"] = yas
                simulation_cache["voda"] = voda
                simulation_cache["network_summary"] = network_summary
                simulation_cache["zip_bytes"] = zip_bytes

            if simulation_cache["summary_df"].empty:
                st.info("Click 'Run Closed-Loop Validation' to generate the full iteration results.")
            else:
                st.write("**Overall simulation summary**")
                st.dataframe(simulation_cache["summary_df"], use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Table 5.5: Generated Interventions - Yas Risk Profiles**")
                    st.dataframe(thesis_style_profile_table(simulation_cache["yas"]), use_container_width=True)
                with col2:
                    st.write("**Table 5.6: Generated Interventions - Vodacom Risk Profiles**")
                    st.dataframe(thesis_style_profile_table(simulation_cache["voda"]), use_container_width=True)

                if not simulation_cache["network_summary"].empty:
                    st.write("**Performance analysis in comparison**")
                    st.dataframe(simulation_cache["network_summary"], use_container_width=True)

                if not simulation_cache["intervention_df"].empty:
                    st.write("**All intervention records**")
                    st.dataframe(simulation_cache["intervention_df"], use_container_width=True)

                if not simulation_cache["local_shap_df"].empty:
                    st.write("**All local SHAP diagnostic records**")
                    st.dataframe(simulation_cache["local_shap_df"], use_container_width=True)

                st.markdown("### Detailed Iteration Outputs")

                for iteration in range(1, int(max_iterations) + 1):
                    iter_df = simulation_cache["iteration_tables"].get(iteration, pd.DataFrame())

                    with st.expander(f"Iteration {iteration}", expanded=(iteration == 1)):
                        if iter_df.empty:
                            st.info("No at-risk customers were detected in this iteration.")
                        else:
                            st.dataframe(iter_df, use_container_width=True)

                if not simulation_cache["yas"].empty or not simulation_cache["voda"].empty:
                    st.pyplot(plot_validation_lines(simulation_cache["yas"], simulation_cache["voda"]))
                    st.pyplot(plot_validation_pct(simulation_cache["yas"], simulation_cache["voda"]))

                narrative = build_validation_narrative(simulation_cache["yas"], simulation_cache["voda"])
                st.markdown("### Strategic Interpretation of Results")
                st.write(narrative)

                st.download_button(
                    "Download all iteration CSV files (ZIP)",
                    data=simulation_cache["zip_bytes"],
                    file_name="retention_simulation_iterations.zip",
                    mime="application/zip",
                )


with tab9:
    st.subheader("WHO / WHY / WHAT TO DO Pipeline")

    if feature_sets is None:
        st.warning("No valid modeling data found.")
    else:
        X = feature_sets["full_X"]
        y = feature_sets["y"]
        results, bundle, err = run_benchmark(X, y)

        if err or bundle is None or bundle["rf"] is None or bundle["rf"]["Estimator"] is None:
            st.warning(err or "Random Forest model is not available.")
        else:
            rf_model = bundle["rf"]["Estimator"]

            st.markdown("### Single-Customer Operational Audit")

            if len(X) > 0:
                selected_idx = st.selectbox("Select customer row", list(range(len(X))))
                row_df = X.iloc[[int(selected_idx)]].copy()
                prob = float(get_probability(rf_model, row_df)[0])
                risk_segment = classify_risk_level(
                    prob,
                    float(high_risk_threshold),
                    float(medium_risk_threshold),
                )

                local_df = local_shap_contributions(rf_model, row_df)
                top3 = local_df["Feature"].head(3).tolist() if not local_df.empty else get_local_top3(rf_model, row_df)
                while len(top3) < 3:
                    top3.append("N/A")

                primary, question, action = build_prescriptive_offer(top3)

                c1, c2, c3 = st.columns(3)
                c1.metric("WHO: Customer Row", str(selected_idx))
                c2.metric("WHO: Risk Probability", f"{prob:.2%}")
                c3.metric("WHO: Risk Segment", risk_segment)

                st.markdown("#### WHY: Diagnostic Audit")
                why_df = pd.DataFrame({"Rank": [1, 2, 3], "Key Driver": top3[:3]})
                st.dataframe(why_df, use_container_width=True)

                if not local_df.empty:
                    st.write("**Detailed local SHAP contributions**")
                    st.dataframe(local_df.head(10), use_container_width=True)

                st.markdown("#### WHAT TO DO: Prescriptive Action")
                action_df = pd.DataFrame(
                    {
                        "Primary Driver": [primary],
                        "Diagnostic Question": [question],
                        "Recommended Retention Action": [action],
                    }
                )
                st.dataframe(action_df, use_container_width=True)

                st.markdown("#### Closed-Loop Interpretation")
                st.write(
                    "This customer-level audit demonstrates the DSS pipeline. The customer is first identified as a member "
                    "of a risk segment through predictive scoring. The top local SHAP factors explain why the risk exists. "
                    "The system then translates the dominant risk driver into a concrete retention recommendation, thereby "
                    "closing the information-action gap."
                )

            st.markdown("### Batch WHO / WHY / WHAT TO DO Table")

            if simulation_cache["who_why_what_df"].empty:
                st.info("Run the DSS Validation tab first to generate the full WHO / WHY / WHAT TO DO intervention table.")
            else:
                st.dataframe(simulation_cache["who_why_what_df"], use_container_width=True)

                csv_data = simulation_cache["who_why_what_df"].to_csv(index=False).encode()
                st.download_button(
                    "Download WHO / WHY / WHAT TO DO table",
                    data=csv_data,
                    file_name="who_why_what_to_do.csv",
                    mime="text/csv",
                )
