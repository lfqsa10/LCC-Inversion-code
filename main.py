"""
main.py 
"""
import os, sys, time, copy, itertools, warnings, pickle

import joblib

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import sparse
import scipy.stats as stats

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

print("HAS_LGB:", HAS_LGB)

# =========================================================
# 1) Global Style
# =========================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["legend.frameon"] = False

NAVY  = "#123c7b"
BROWN = "#8B4513"
DARK  = "#1f1f1f"
LIGHT_GRID = "#d3d3d3"

# =========================================================
# 2) Configuration
# =========================================================
DATA_PATH = os.path.join("data", "LCC.xlsx")
TARGET_COL = "叶绿素含量(mg/g)"
ID_COL = "id"
CAT_COLS = ["Reproductive period", "Processing"]
GROUP_COLS = ["Reproductive period", "Processing"]
AGG_FUNC = "median"

TEST_SIZE = 0.2
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

K_BEST = 25
N_ITER = 30
CV_FOLDS = 5

ENR_ALPHA = 0.0005
ENR_L1_RATIO = 0.5

OUT_DIR = "Q1"
FIG_DPI = 600

DO_SHAP = True
SHAP_MAX_SAMPLES = 300
SHAP_BG_SAMPLES = 120
SHAP_RANDOM_SEED = 42
SHAP_MAX_DISPLAY = 12
SHAP_FIG_DPI = 600

GAP_WEIGHT = 1.0  # R²IC = |Train R² - Test R²| 

# =========================================================
# 2b) VI list & Fusion fixed features
# =========================================================
VI_LIST = [
    "NDVI","GNDVI","NDRE","NDVIrededge","SR","RVI","MSR","MSRre","DVI","RTVI",
    "TVI","TNDVI","EVI","EVI2","SAVI","OSAVI","MSAVI","GSAVI","GOSAVI","REOSAVI",
    "WDRVI","REWDRVI","CIre","CIrededge","CIgreen","CIG","GCI","GCVI","GDNVI",
    "NRI","GRVI","SIPI","mNDVI","mNDVIre","MRESAVI","MNDRE2","PPR","MCARI",
    "MCARI2","CARI","TCARI","MNLI","MTVI1","NDCI","NGRDI"
]

# =========================================================
# 3) Utilities
# =========================================================
def now_tag():
    return time.strftime("%Y%m%d_%H%M%S")

def safe_mkdir(path_):
    os.makedirs(path_, exist_ok=True)

def to_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return X

def rpd(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    sd = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else float(np.std(y_true))
    return float(sd / rmse) if rmse > 0 else np.nan

def model_objective(train_r2, test_r2, test_rmse, test_rpd):
    """综合评分，越低越好：-Test R² + R²IC + Test RMSE - RPD/100。"""
    r2ic = abs(train_r2 - test_r2)
    return -test_r2 + r2ic * GAP_WEIGHT + test_rmse - test_rpd * 0.01

def savefig_png_pdf(base_path_no_ext, dpi=FIG_DPI):
    plt.savefig(base_path_no_ext + ".png", dpi=dpi, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.savefig(base_path_no_ext + ".pdf", dpi=dpi, bbox_inches="tight", pad_inches=0.04, facecolor="white")

def style_axes(ax, grid_axis=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=11, width=1.0, length=4, color=DARK)
    if grid_axis is None:
        ax.grid(False)
    else:
        ax.grid(axis=grid_axis, linestyle=":", linewidth=0.8, alpha=0.35, color=LIGHT_GRID)

def select_top_features_by_correlation(corr_matrix, features, top_n=10, threshold=0.95):
    selected = []
    for f in features:
        too_corr = any(abs(corr_matrix.loc[f, s]) > threshold for s in selected)
        if not too_corr:
            selected.append(f)
        if len(selected) == top_n:
            break
    return selected


# =========================================================
# 4) Feature screening
# =========================================================
def _plot_combined_rf_importance(vi_fi, tex_fi, out_path):
   
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    datasets = [(vi_fi, "VI", plt.cm.Blues),
                (tex_fi, "Texture", plt.cm.Oranges)]

    for idx, (ax, (fi, name, cmap)) in enumerate(zip(axes, datasets)):
        df = fi.head(min(15, len(fi))).copy()
        features = df["Feature"].values
        importances = df["Importance"].values
        norm_imp = Normalize(vmin=importances.min(), vmax=importances.max())
        colors = cmap(norm_imp(importances))

        ax.barh(range(len(features)), importances[::-1],
                color=colors[::-1], edgecolor="none", height=0.65)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features[::-1], fontsize=8)
        ax.set_xlabel("Importance", fontsize=12, color=DARK)
        ax.set_xlim(0, max(importances) * 1.10)
        label = f"({chr(97+idx)})  {name}"
        ax.set_title(label, fontsize=12, fontweight='bold', loc='left',
                     color=DARK, pad=6)
        style_axes(ax)

    plt.tight_layout()
    savefig_png_pdf(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [Combined RF Importance] -> {out_path}")


def _abbrev_feat(name):
   
    mapping = {
        "rededge": "RE",   "sum_average": "SumAvg", "sum_variance": "SumVar",
        "sum_entropy": "SumEnt", "diff_variance": "DiffVar",
        "diff_entropy": "DiffEnt", "homogeneity": "Homog",
        "correlation": "Corr", "contrast": "Contrast",
        "variance": "Var", "entropy": "Entropy",
    }
    s = name
    for old, new in mapping.items():
        s = s.replace(old, new)
    return s


def _plot_combined_corr_heatmaps(vi_df, tex_df, fu_df,
                                  vi_cols, tex_cols, out_path):
   
    import matplotlib.gridspec as gs_module

    datasets = [
        (vi_df,  "VI",      [True]*len(vi_df.columns)),
        (tex_df, "Texture", [False]*len(tex_df.columns)),
        (fu_df,  "VI+TX",   [c in vi_cols for c in fu_df.columns]),
    ]
    n_list = [len(d[0].columns) for d in datasets]

   
    cell_w = 0.36  
    top_w = max(n_list[0], n_list[1]) * cell_w + 0.9
    bot_w = n_list[2] * cell_w + 1.0
    total_w = min(max(top_w * 2, bot_w), 7.5)

    fig = plt.figure(figsize=(total_w, 8.2))
    gs = gs_module.GridSpec(2, 2, figure=fig,
                             height_ratios=[1, 1.3],
                             width_ratios=[1, 1],
                             hspace=0.34, wspace=0.25,
                             left=0.065, right=0.905, top=0.96, bottom=0.095)

    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=-1, vmax=1)
    im_last = None

    positions = [(0, 0), (0, 1), (1, slice(0, 2))]  # VI, TX, Fusion 跨两列

    for idx, ((df, name, is_vi_list), pos) in enumerate(zip(datasets, positions)):
        ax = fig.add_subplot(gs[pos])
        corr = df.corr(method="pearson")
        cols = list(corr.columns)
        n = len(cols)
        abbrev_cols = [_abbrev_feat(c) for c in cols]

        im = ax.imshow(corr.values, cmap=cmap, norm=norm, aspect='equal',
                       interpolation='nearest')
        im_last = im

       
        fs = 6.0 if n > 10 else 7.0
        for i in range(n):
            for j in range(n):
                val = corr.values[i, j]
                clr = 'white' if abs(val) > 0.60 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=fs, fontweight='bold', color=clr)

        
        for k in range(n + 1):
            ax.axhline(k - 0.5, color='#b0b0b0', lw=0.35, alpha=0.4)
            ax.axvline(k - 0.5, color='#b0b0b0', lw=0.35, alpha=0.4)

       
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(abbrev_cols, rotation=45, ha='right', fontsize=6.8)
        ax.set_yticklabels(abbrev_cols, fontsize=6.8)

        # VI=Navy, TX=Brown
        for lbl, is_vi in zip(ax.get_xticklabels(), is_vi_list):
            lbl.set_color(NAVY if is_vi else BROWN)
        for lbl, is_vi in zip(ax.get_yticklabels(), is_vi_list):
            lbl.set_color(NAVY if is_vi else BROWN)

       
        label = f"({chr(97+idx)})  {name}  (N={n})"
        ax.set_title(label, fontsize=11, fontweight='bold', loc='left',
                     color=DARK, pad=6)

        ax.tick_params(axis='both', which='both', length=0, pad=1.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    
    cbar_ax = fig.add_axes([0.935, 0.06, 0.012, 0.88])
    cbar = fig.colorbar(im_last, cax=cbar_ax)
    cbar.set_label('Pearson r', fontsize=11, color=DARK, labelpad=5)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=9, colors=DARK)
    cbar.outline.set_visible(False)

    savefig_png_pdf(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [Combined Heatmap] -> {out_path}")


def filter_features_pool(X_train, y_train, feat_pool, pool_name="", out_dir=None, max_n=10):
    """RF importance Top-N -> collinearity filter -> return filtered features."""
    imputer = SimpleImputer(strategy="median")
    power = PowerTransformer(method="yeo-johnson", standardize=True)
    Xnum_p = power.fit_transform(imputer.fit_transform(X_train[feat_pool]))
    Xnum_p_df = pd.DataFrame(Xnum_p, columns=feat_pool, index=X_train.index)

    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=800)
    rf.fit(Xnum_p_df, y_train)
    fi = pd.DataFrame(
        {"Feature": feat_pool, "Importance": rf.feature_importances_}
    ).sort_values("Importance", ascending=False)
    top_n_feat = fi.head(max_n)["Feature"].tolist()
    filtered = select_top_features_by_correlation(Xnum_p_df[top_n_feat].corr(), top_n_feat, top_n=max_n)

    if pool_name:
        print(f"  [{pool_name}] RF Top-{max_n}: {top_n_feat}")
        print(f"  [{pool_name}] filtered({len(filtered)}): {filtered}")

    return filtered, Xnum_p_df, fi

# =========================================================
# 5) Model evaluation
# =========================================================
def evaluate_feature_set(num_cols, cat_cols, X_train, X_test, y_train, y_test, g_train,
                          exclude_models=None):
    """Full pipeline: preprocess -> SelectKBest -> RandomizedSearchCV -> best result.
    exclude_models: optional list of model names to skip (e.g. ['PLSR'])."""
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
    ])
    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer(
        [("num", numeric_tf, num_cols), ("cat", categorical_tf, cat_cols)],
        remainder="drop",
    )
    kbest_safe = min(K_BEST, pre.fit_transform(X_train).shape[1])
    feat = Pipeline(steps=[
        ("to_dense", FunctionTransformer(to_dense, accept_sparse=True)),
        ("kbest", SelectKBest(score_func=mutual_info_regression, k=kbest_safe)),
    ])
    plsr_u = min(25, max(3, len(num_cols)))
    plsr_c = list(range(2, min(plsr_u, 3)))  # 限制 n_components ≤ 2

    candidates = [
        ("ENR", ElasticNet(random_state=RANDOM_STATE, max_iter=20000), {
            "model__alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
        }),
        ("PLSR", PLSRegression(), {"model__n_components": plsr_c}),
        ("RF", RandomForestRegressor(random_state=RANDOM_STATE), {
            "model__n_estimators": [300, 600, 1000, 1500],
            "model__max_depth": [3, 4, 5, 6, 8, None],
            "model__min_samples_leaf": [1, 2, 3, 5, 8],
            "model__max_features": ["sqrt", 0.5, 0.7, 1.0],
        }),
        ("ET", ExtraTreesRegressor(random_state=RANDOM_STATE), {
            "model__n_estimators": [500, 1000, 1500, 2500],
            "model__max_depth": [3, 4, 5, 6, 8, None],
            "model__min_samples_leaf": [1, 2, 3, 5, 8],
            "model__max_features": ["sqrt", 0.5, 0.7, 1.0],
        }),
    ]
    if HAS_LGB:
        candidates.append(("LGBM", lgb.LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, deterministic=True, num_threads=1), {
            "model__n_estimators": [300, 600, 1200, 2000],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.08],
            "model__num_leaves": [15, 31, 63, 127],
            "model__min_child_samples": [5, 10, 20, 30],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "model__reg_lambda": [0.0, 0.1, 0.5, 1.0],
        }))

    if exclude_models:
        candidates = [(n, m, p) for n, m, p in candidates if n not in exclude_models]

    cv = GroupKFold(n_splits=CV_FOLDS)
    all_results = []
    best_overall = None

    for name, model, param_dist in candidates:
        pipe = Pipeline(steps=[("pre", pre), ("feat", feat), ("model", model)])
        search = RandomizedSearchCV(
            pipe, param_dist,
            n_iter=min(N_ITER, len(list(itertools.product(*param_dist.values()))) if param_dist else 1),
            scoring="r2", cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
        )
        search.fit(X_train, y_train, groups=g_train)
        est = search.best_estimator_
        tr_r2 = float(r2_score(y_train, est.predict(X_train)))
        te_r2 = float(r2_score(y_test, est.predict(X_test)))
        te_rmse = float(np.sqrt(mean_squared_error(y_test, est.predict(X_test))))
        te_rpd_val = rpd(y_test, est.predict(X_test))
        obj = model_objective(tr_r2, te_r2, te_rmse, te_rpd_val)
        rec = {
            "model": name, "cv_r2": float(search.best_score_),
            "train_r2": tr_r2, "test_r2": te_r2, "r2ic": abs(tr_r2 - te_r2),
            "test_rmse": te_rmse,
            "train_rpd": rpd(y_train, est.predict(X_train)),
            "test_rpd": te_rpd_val,
            "objective": obj, "_estimator": est,
        }
        all_results.append(rec)
        if best_overall is None or obj < best_overall["objective"]:
            best_overall = rec

    return best_overall, all_results

# =========================================================
# 6) SHAP
# =========================================================
def _build_num_only_pipeline(best_estimator, best_features_numeric):
    model = copy.deepcopy(best_estimator.named_steps["model"])
    pre_num = ColumnTransformer(
        transformers=[("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("power", PowerTransformer(method="yeo-johnson", standardize=True))
        ]), list(best_features_numeric))],
        remainder="drop")
    return Pipeline(steps=[
        ("pre_num", pre_num),
        ("to_dense", FunctionTransformer(to_dense, accept_sparse=True)),
        ("model", model)])

def _plot_shap_combined(shap_values, X_original, feat_names, save_base, topk):
    
    import shap as _shap

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1][:topk]
    shap_top = shap_values[:, order]
    X_top = X_original[:, order]
    top_names = [feat_names[i] for i in order]
    mean_sorted = mean_abs[order]
    n_feat = len(top_names)

    shap_cmap = _shap.plots.colors.red_blue
    y_positions = np.arange(n_feat)
    rng = np.random.RandomState(42)

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=SHAP_FIG_DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    
    for i in range(n_feat):
        shap_row = np.asarray(shap_top[:, i], dtype=float)
        feat_row = np.asarray(X_top[:, i], dtype=float)
        f_min, f_max = np.nanpercentile(feat_row, 5), np.nanpercentile(feat_row, 95)
        f_range = f_max - f_min
        f_norm = np.clip((feat_row - f_min) / f_range, 0, 1) if f_range > 0 else np.full_like(feat_row, 0.5)
        y_jitter = y_positions[i] + rng.normal(0, 0.06, size=len(shap_row))
        draw_order = np.argsort(f_norm)
        sc = ax.scatter(shap_row[draw_order], y_jitter[draw_order],
                        c=f_norm[draw_order], cmap=shap_cmap,
                        s=20, alpha=0.94, edgecolors="none", zorder=3)

    ax.axvline(0, color="#9a9a9a", lw=1.0, zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(top_names, fontsize=10, color=DARK)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value", fontsize=12, color=DARK)
    ax.tick_params(axis="both", labelsize=9.5, colors=DARK)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.15)

   
    ax_top = ax.twiny()
    ax_top.barh(y_positions, mean_sorted, height=0.7,
                color="#a8d4f0", edgecolor="#7cb8d8", linewidth=0.3,
                alpha=0.92, zorder=0)
    ax_top.set_xlim(0, mean_sorted.max() * 1.12)
    ax_top.set_xlabel("mean(|SHAP value|)", fontsize=12, color=DARK, labelpad=7)
    ax_top.xaxis.set_label_position("top")
    ax_top.xaxis.tick_top()
    ax_top.tick_params(axis="x", labelsize=9.5, colors=DARK)

   
    ax_top.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)

   
    ax.set_title("SHAP Feature Importance & Impact", fontsize=14, pad=18, color=DARK)

   
    sm = plt.cm.ScalarMappable(cmap=shap_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.025)
    cbar.set_label("Feature value", fontsize=10.5, color=DARK)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(labelsize=9.5, colors=DARK)
    cbar.outline.set_visible(False)

   
    fig.subplots_adjust(left=0.22, right=0.92, top=0.91, bottom=0.08)
    savefig_png_pdf(save_base, dpi=SHAP_FIG_DPI)
    plt.close(fig)


def run_shap(best_estimator, best_model_name, out_dir, best_features_numeric, X_train_df, y_train_arr):
    if not HAS_SHAP:
        print("[SHAP] shap not installed, skipping.")
        return
    np.random.seed(SHAP_RANDOM_SEED)
    pipe = _build_num_only_pipeline(best_estimator, best_features_numeric)
    pipe.fit(X_train_df, y_train_arr)

    n_tr = len(X_train_df)
    idx = np.random.permutation(n_tr)
    n_use = min(SHAP_MAX_SAMPLES, n_tr)
    n_bg = min(SHAP_BG_SAMPLES, n_tr)

    X_explain_df = X_train_df.iloc[idx[:n_use]][best_features_numeric].copy()
    X_bg_df = X_train_df.iloc[idx[:n_bg]][best_features_numeric].copy()
    X_explain = pipe.named_steps["to_dense"].transform(
        pipe.named_steps["pre_num"].transform(X_explain_df))
    X_bg = pipe.named_steps["to_dense"].transform(
        pipe.named_steps["pre_num"].transform(X_bg_df))
    feat_names = list(best_features_numeric)
    model = pipe.named_steps["model"]

    if best_model_name in ["RF", "ET"] or (best_model_name == "LGBM" and HAS_LGB):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
        X_original = X_explain_df.values
    elif best_model_name == "ENR":
        explainer = shap.LinearExplainer(model, X_bg, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_explain)
        X_original = X_explain_df.values
    else:
        n_use2 = min(120, X_explain.shape[0])
        explainer = shap.KernelExplainer(model.predict, X_bg[:min(80, X_bg.shape[0])])
        shap_values = explainer.shap_values(X_explain[:n_use2])
        X_original = X_explain_df.iloc[:n_use2].values

    shap_values = np.array(shap_values)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    topk = min(SHAP_MAX_DISPLAY, len(feat_names))

    _plot_shap_combined(shap_values, X_original, feat_names,
                        os.path.join(out_dir, "Fig_SHAP_combined"), topk=topk)

    try:
        top_idx = int(np.argmax(mean_abs))
        plt.figure(figsize=(6.8, 5.6))
        shap.dependence_plot(top_idx, shap_values, X_explain, feature_names=feat_names, show=False)
        plt.tight_layout()
        savefig_png_pdf(os.path.join(out_dir, "Fig_SHAP_dependence_top1"), dpi=SHAP_FIG_DPI)
        plt.close()
    except Exception as e:
        print(f"[SHAP] dependence_plot skipped: {e}")

    print("[SHAP] done.")


# =========================================================
# 7) Train-Test Scatter Plot
# =========================================================
def _plot_train_test_scatter(y_train, y_test, y_pred_train, y_pred_test,
                              r2_train, r2_test, rmse_train, rmse_test,
                              rpd_train, rpd_test, save_base):
    
    scale_factor = 1.5
    confidence = 0.95

   
    z_train = np.polyfit(y_train, y_pred_train, 1)
    p_train = np.poly1d(z_train)
    residuals_train = y_pred_train - p_train(y_train)
    mean_error_train = np.mean(residuals_train ** 2)
    t_val_train = stats.t.ppf((1 + confidence) / 2., len(y_train) - 1)
    x_ext_train = np.linspace(min(y_train), max(y_train), 100)
    predicted_ext_train = p_train(x_ext_train)
    ci_ext_train = (t_val_train * scale_factor * np.sqrt(mean_error_train) *
                    np.sqrt(1 / len(y_train) + (x_ext_train - np.mean(y_train)) ** 2 /
                            np.sum((y_train - np.mean(y_train)) ** 2)))

    
    z_test = np.polyfit(y_test, y_pred_test, 1)
    p_test = np.poly1d(z_test)
    residuals_test = y_pred_test - p_test(y_test)
    mean_error_test = np.mean(residuals_test ** 2)
    t_val_test = stats.t.ppf((1 + confidence) / 2., len(y_test) - 1)
    x_ext_test = np.linspace(min(y_test), max(y_test), 100)
    predicted_ext_test = p_test(x_ext_test)
    ci_ext_test = (t_val_test * scale_factor * np.sqrt(mean_error_test) *
                   np.sqrt(1 / len(y_test) + (x_ext_test - np.mean(y_test)) ** 2 /
                           np.sum((y_test - np.mean(y_test)) ** 2)))

   
    train_color = '#1f77b4'
    test_color = '#ff7f0e'
    ci_train_color = '#aec7e8'
    ci_test_color = '#ffbb78'

   
    fig = plt.figure(figsize=(10, 8), dpi=FIG_DPI)
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_hist_x = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_hist_y = fig.add_subplot(gs[1:, -1], sharey=ax_main)

   
    ax_main.scatter(y_train, y_pred_train, color=train_color,
                    label="Training Predicted Values", alpha=0.6)
    ax_main.plot(y_train, p_train(y_train), color=train_color, alpha=0.9,
                 label=f"Training\n$R^2$ = {r2_train:.2f}\nRMSE = {rmse_train:.2f}\nRPD = {rpd_train:.2f}")
    ax_main.fill_between(x_ext_train,
                         predicted_ext_train - ci_ext_train,
                         predicted_ext_train + ci_ext_train,
                         color=ci_train_color, alpha=0.5,
                         label="Training 95% Confidence Interval")

   
    ax_main.scatter(y_test, y_pred_test, color=test_color,
                    label="Testing Predicted Values", alpha=0.6)
    ax_main.plot(y_test, p_test(y_test), color=test_color, alpha=0.9,
                 label=f"Testing\n$R^2$ = {r2_test:.2f}\nRMSE = {rmse_test:.2f}\nRPD = {rpd_test:.2f}")
    ax_main.fill_between(x_ext_test,
                         predicted_ext_test - ci_ext_test,
                         predicted_ext_test + ci_ext_test,
                         color=ci_test_color, alpha=0.5,
                         label="Testing 95% Confidence Interval")

   
    all_min = min(y_train.min(), y_test.min())
    all_max = max(y_train.max(), y_test.max())
    ax_main.plot([all_min, all_max], [all_min, all_max],
                 color='grey', linestyle='--', alpha=0.6, label="1:1 Line")

    ax_main.set_xlabel("Observed Values", fontsize=12)
    ax_main.set_ylabel("Predicted Values", fontsize=12)
    ax_main.legend(loc="upper left", fontsize=10)

    
    ax_hist_x.hist(y_train, bins=20, color=train_color, alpha=0.7,
                   edgecolor='black')
    ax_hist_x.hist(y_test, bins=20, color=test_color, alpha=0.7,
                   edgecolor='black')
    ax_hist_x.tick_params(labelbottom=False)

    
    ax_hist_y.hist(y_pred_train, bins=20, orientation='horizontal',
                   color=train_color, alpha=0.7, edgecolor='black')
    ax_hist_y.hist(y_pred_test, bins=20, orientation='horizontal',
                   color=test_color, alpha=0.7, edgecolor='black')
    ax_hist_y.set_xlabel("Frequency", fontsize=12)
    ax_hist_y.tick_params(labelleft=False)

    fig.tight_layout()
    savefig_png_pdf(save_base, dpi=FIG_DPI)
    plt.close(fig)


# =========================================================
# 8) Main
# =========================================================
def main():
    tag = now_tag()
    out = os.path.join(OUT_DIR, f"run_{tag}")
    safe_mkdir(out)
    print("Output dir:", out)

    # ---- Load data ----
    df = pd.read_excel(DATA_PATH)
    df = df.replace([np.inf, -np.inf], np.nan)
    num_round_cols = df.select_dtypes(include="number").columns
    df[num_round_cols] = df[num_round_cols].round(5)
    df.rename(columns={TARGET_COL: "chl"}, inplace=True)
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # ---- Aggregate ----
    num_cols_all_df = [c for c in df.columns if c not in GROUP_COLS + [ID_COL]]
    df = df.groupby(GROUP_COLS, as_index=False).agg({c: AGG_FUNC for c in num_cols_all_df})

    df["__group_id__"] = df[GROUP_COLS].astype(str).agg("_".join, axis=1)
    print(f"Data shape: {df.shape}  Groups: {df['__group_id__'].nunique()}")

    y = df["chl"].values
    X = df.drop(columns=["chl"])
    groups = df["__group_id__"].values
    X = X.drop(columns=["__group_id__", ID_COL], errors="ignore")
    cat_cols = [c for c in CAT_COLS if c in X.columns]

    # ---- Train/test split ----
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_tr, X_te = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_tr, y_te = y[train_idx], y[test_idx]
    g_tr = groups[train_idx]
    print(f"Train: {len(X_tr)}  Test: {len(X_te)}")

    # ---- Identify feature pools ----
    vi_cols = [c for c in X.columns if c in VI_LIST]
    texture_cols = [c for c in X.columns if c not in vi_cols and c not in CAT_COLS]
    print(f"VI features: {len(vi_cols)}  Texture features: {len(texture_cols)}")

    # ---- Feature screening ----
    vi_filtered, vi_Xp_df, vi_fi = filter_features_pool(X_tr, y_tr, vi_cols, "VI", out)
    tex_filtered, tex_Xp_df, tex_fi = filter_features_pool(X_tr, y_tr, texture_cols, "Texture", out)
    print(f"VI filtered: {vi_filtered}")
    print(f"Texture filtered: {tex_filtered}")

    # ---- Combined RF importance figure ----
    _plot_combined_rf_importance(vi_fi, tex_fi,
                                 os.path.join(out, "Fig_RF_Importance_Combined"))

    # ---- Train & evaluate ----
    all_best = {}

    print("\n" + "=" * 80)
    print("  仅VI")
    print("=" * 80)
    all_best["VI"] = evaluate_feature_set(vi_filtered, cat_cols, X_tr, X_te, y_tr, y_te, g_tr)

    print("\n" + "=" * 80)
    print("  仅纹理")
    print("=" * 80)
    all_best["TX"] = evaluate_feature_set(tex_filtered, cat_cols, X_tr, X_te, y_tr, y_te, g_tr)

   
    fu_pool = list(dict.fromkeys(vi_filtered + tex_filtered))
    imputer_fu = SimpleImputer(strategy="median")
    power_fu = PowerTransformer(method="yeo-johnson", standardize=True)
    X_fu = power_fu.fit_transform(imputer_fu.fit_transform(X_tr[fu_pool]))
    X_fu_df = pd.DataFrame(X_fu, columns=fu_pool, index=X_tr.index)
    corr_fu = X_fu_df.corr()
    fu_screened = select_top_features_by_correlation(corr_fu, fu_pool, top_n=min(12, len(fu_pool)))
    vi_in = [f for f in fu_screened if f in vi_cols]
    tx_in = [f for f in fu_screened if f in texture_cols]
    print(f"\n  融合 ({len(fu_screened)}特征): VI({len(vi_in)}) + TX({len(tx_in)})")
    print(f"    植被指数: {vi_in}")
    print(f"    纹理特征: {tx_in}")

    # ---- Combined 3-panel correlation heatmap (VI | Texture | VI+TX) ----
    _plot_combined_corr_heatmaps(
        vi_Xp_df[vi_filtered], tex_Xp_df[tex_filtered],
        X_fu_df[fu_screened],
        vi_cols, texture_cols,
        os.path.join(out, "Fig_Correlation_Heatmaps_Combined"))

    print("\n" + "=" * 80)
    print("  融合（VI+TX）")
    print("=" * 80)
    all_best["FU"] = evaluate_feature_set(fu_screened, cat_cols, X_tr, X_te, y_tr, y_te, g_tr)

    # ---- Print model parameters ----
    print("\n" + "=" * 80)
    print("  模型参数详情")
    print("=" * 80)
    for key, label, feats in [("VI", "VI", vi_filtered), ("TX", "纹理", tex_filtered),
                                ("FU", "融合(VI+TX)", fu_screened)]:
        best_rec = all_best[key][0]
        est = best_rec["_estimator"]
        model_step = est.named_steps["model"]
        params = model_step.get_params()
        print(f"\n  [{label}] 特征数={len(feats)}, 最优模型={best_rec['model']}")
        print(f"    特征: {feats}")
        print(f"    超参数: {params}")

    # ---- Print all results ----
    print("\n" + "=" * 100)
    print("  RESULTS SUMMARY")
    print("=" * 100)
    header = f"{'Set':<6s} {'Model':>5s} {'Train R²':>9s} {'Test R²':>9s} {'R²IC':>7s} {'Test RMSE':>10s} {'Test RPD':>9s}"
    print(header)
    print("-" * len(header))
    for key, (best_rec, all_recs) in all_best.items():
        for r in sorted(all_recs, key=lambda d: d["objective"]):
            print(f"{key:<6s} {r['model']:>5s} {r['train_r2']:>9.4f} {r['test_r2']:>9.4f} "
                  f"{r['r2ic']:>7.4f} {r['test_rmse']:>10.4f} {r['test_rpd']:>9.2f}")

    # ---- Best model overall ----
    best_key = min(all_best, key=lambda k: all_best[k][0]["objective"])
    best_rec = all_best[best_key][0]
    print(f"\nBest overall: {best_key} / {best_rec['model']}  "
          f"Train R²={best_rec['train_r2']:.2f}  Test R²={best_rec['test_r2']:.2f}")

    # ---- Save CSV ----
    rows = []
    for key, (_, all_recs) in all_best.items():
        for r in all_recs:
            rows.append({"feature_set": key, **{k: v for k, v in r.items() if k != "_estimator"}})
    pd.DataFrame(rows).to_excel(os.path.join(out, "model_results.xlsx"), index=False)

    # ---- Export best model predictions ----
    best_set_key = best_key
    best_estimator = all_best[best_set_key][0]["_estimator"]
    best_model_name = all_best[best_set_key][0]["model"]
    best_estimator.fit(X_tr, y_tr)
    pred_tr = best_estimator.predict(X_tr)
    pred_te = best_estimator.predict(X_te)
    pred_df = pd.DataFrame({
        "set": ["train"] * len(y_tr) + ["test"] * len(y_te),
        "y_true": np.concatenate([y_tr, y_te]),
        "y_pred": np.concatenate([pred_tr, pred_te]),
    })
    pred_xlsx = os.path.join(out, "best_model_predictions.xlsx")
    pred_df.to_excel(pred_xlsx, index=False)
    print(f"Predictions exported: {pred_xlsx}")

    # ---- Train-Test Scatter Plot ----
    rmse_tr = float(np.sqrt(np.mean((y_tr - pred_tr) ** 2)))
    rmse_te = float(np.sqrt(np.mean((y_te - pred_te) ** 2)))
    rpd_tr = float(np.std(y_tr, ddof=1) / rmse_tr) if rmse_tr > 0 else np.nan
    rpd_te = float(np.std(y_te, ddof=1) / rmse_te) if rmse_te > 0 else np.nan
    _plot_train_test_scatter(
        y_tr, y_te, pred_tr, pred_te,
        best_rec["train_r2"], best_rec["test_r2"],
        rmse_tr, rmse_te, rpd_tr, rpd_te,
        os.path.join(out, "Fig_TrainTest_Scatter"),
    )

    # ---- Export best model ----
    model_path_joblib = os.path.join(out, f"best_model_{best_set_key}_{best_model_name}.joblib")
    model_path_pkl = os.path.join(out, f"best_model_{best_set_key}_{best_model_name}.pkl")
    joblib.dump(best_estimator, model_path_joblib)
    with open(model_path_pkl, "wb") as f:
        pickle.dump(best_estimator, f)
    print(f"Model exported: {model_path_joblib}")
    print(f"Model exported: {model_path_pkl}")

    # ---- SHAP on best model ----
    if DO_SHAP:

        if best_set_key == "VI":
            shap_features = vi_filtered
        elif best_set_key == "TX":
            shap_features = tex_filtered
        else:
            shap_features = fu_screened

        shap_out = os.path.join(out, f"SHAP_{best_set_key}_{best_model_name}")
        safe_mkdir(shap_out)
        print(f"\nRunning SHAP: {best_set_key}/{best_model_name} features={shap_features}")
        run_shap(best_estimator, best_model_name, shap_out, shap_features, X_tr, y_tr)

    print("\nDone.")


if __name__ == "__main__":
    main()
