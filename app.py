import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              f1_score, precision_score, roc_auc_score, mean_absolute_error,
                              silhouette_score)
from scipy.stats import weibull_min
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="STATSIM v5",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

PALETTE = ["#0066cc","#e8284a","#1e8c00","#7c3aed","#cc7700",
           "#c2185b","#00796b","#e64a19","#0097a7","#558b2f"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#f4f7ff",
    "axes.edgecolor": "#d0daf0", "axes.labelcolor": "#1a2540",
    "xtick.color": "#8a9bbf", "ytick.color": "#8a9bbf",
    "text.color": "#1a2540", "grid.color": "#d0daf0",
    "grid.alpha": 0.5, "axes.spines.top": False, "axes.spines.right": False,
})

# ─────────────────────────────────────────────
#  CSS CUSTOM
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Rajdhani:wght@700&family=Share+Tech+Mono&display=swap");
html, body, [class*="css"] { font-family: "Share Tech Mono", monospace; }
.main-title {
    text-align: center; font-family: "Rajdhani", sans-serif;
    font-size: 48px; font-weight: 700; letter-spacing: 10px;
    color: #0066cc; padding: 10px 0 4px;
}
.main-title span { color: #e8284a; }
.sub-title {
    text-align: center; font-size: 11px; color: #8a9bbf;
    letter-spacing: 4px; margin-bottom: 20px;
}
.metric-card {
    background: #f4f7ff; border: 1px solid #d0daf0;
    border-radius: 12px; padding: 14px 18px; margin: 6px 0;
}
.metric-label { font-size: 9px; color: #8a9bbf; letter-spacing: 2px; }
.metric-value { font-family: "Rajdhani",sans-serif; font-size: 22px; font-weight: 700; }
.pred-result {
    background: #d6f5d6; border: 2px solid #1e8c00;
    border-radius: 12px; padding: 16px 20px; text-align: center; margin-top: 12px;
}
.pred-result-clf {
    background: #ede9fe; border: 2px solid #7c3aed;
    border-radius: 12px; padding: 16px 20px; margin-top: 12px;
}
.section-header {
    font-family: "Rajdhani",sans-serif; font-size: 18px;
    font-weight: 700; letter-spacing: 4px; color: #0066cc;
    border-bottom: 2px solid #d0daf0; padding-bottom: 6px; margin-bottom: 12px;
}
div[data-testid="stSidebar"] { background: #f4f7ff; border-right: 1px solid #d0daf0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">STAT<span>SIM</span> v5</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ANALYSE STATISTIQUE COMPLÈTE · v5 · RF · K-MEANS · MÉTRIQUES · FRÉQUENCES</div>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
#  SIDEBAR — NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 STATSIM v4")
    st.markdown("---")
    section = st.radio("Navigation", [
        "📂 Import Dataset",
        "🔍 Types de Variables",
        "📊 Statistiques Descriptives",
        "📋 Tableaux de Fréquence",
        "🔗 Corrélation",
        "📈 Régression",
        "⚗️ ANOVA",
        "🤖 Classification",
        "🧪 Tests d'Hypothèse",
        "🎨 Graphiques"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div style="font-size:9px;color:#8a9bbf;letter-spacing:2px">PROJET STAT APPLIQUÉ IA</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def detect_type(series):
    n_unique = series.nunique()
    n_total  = series.count()
    is_num   = pd.api.types.is_numeric_dtype(series)
    if is_num:
        if n_unique == 2:             return "Binaire",      "🟢 BIN"
        if n_unique <= 10:            return "Ordinale",     "🟣 ORD"
        if n_unique / n_total > 0.95: return "Identifiant",  "⚫ ID"
        return "Continue",   "🔵 NUM"
    else:
        if n_unique == 2:             return "Binaire",      "🟢 BIN"
        if n_unique <= 15:            return "Catégorielle", "🔴 CAT"
        return "Texte/ID",   "⚫ TXT"

def metric_card(label, value, color="#0066cc"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
    </div>""", unsafe_allow_html=True)

def fig_to_st(fig):
    st.pyplot(fig)
    plt.close(fig)

# ─────────────────────────────────────────────
#  SESSION STATE — stocker le DataFrame
# ─────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "last_reg" not in st.session_state:
    st.session_state.last_reg = None
if "last_clf" not in st.session_state:
    st.session_state.last_clf = None

df = st.session_state.df

# ══════════════════════════════════════════════
#  SECTION 0 — IMPORT
# ══════════════════════════════════════════════
if section == "📂 Import Dataset":
    st.markdown('<div class="section-header">📂 IMPORT DATASET</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        uploaded = st.file_uploader("Glisser / Déposer un fichier CSV", type=["csv","txt"])
        sep_choice = st.selectbox("Séparateur", [",", ";", "\\t", "|"], index=0)
        sep_map = {",":",", ";":";", "\\t":"\t", "|":"|"}

        if uploaded:
            try:
                df_loaded = pd.read_csv(uploaded, sep=sep_map[sep_choice])
                st.session_state.df = df_loaded
                df = df_loaded
                st.success(f"✅ Fichier chargé : {uploaded.name}")
            except Exception as e:
                st.error(f"❌ Erreur : {e}")

        st.markdown("---")
        if st.button("🎲 Générer un dataset exemple"):
            np.random.seed(42)
            n = 150
            df_ex = pd.DataFrame({
                "age":         np.random.randint(18, 65, n).astype(float),
                "salaire":     np.random.normal(45000, 12000, n).round(0),
                "score":       np.random.normal(75, 15, n).clip(0, 100).round(1),
                "anciennete":  np.random.exponential(5, n).round(2),
                "departement": np.random.choice(["RH","IT","Finance","Marketing"], n),
                "niveau":      np.random.choice(["Junior","Senior","Manager"], n),
                "satisfait":   np.random.choice([0, 1], n),
            })
            st.session_state.df = df_ex
            df = df_ex
            st.success("✅ Dataset exemple généré (150 lignes, 7 variables)")

    with col2:
        if df is not None:
            df_num = df.select_dtypes(include=[np.number])
            df_cat = df.select_dtypes(exclude=[np.number])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Lignes",     df.shape[0])
            c2.metric("Colonnes",   df.shape[1])
            c3.metric("Numériques", df_num.shape[1])
            c4.metric("Catégorielles", df_cat.shape[1])
            st.markdown("**Aperçu du dataset (5 premières lignes)**")
            st.dataframe(df.head(8), use_container_width=True)
            st.markdown("**Valeurs manquantes**")
            miss = df.isnull().sum()
            miss_df = miss[miss > 0].reset_index()
            if len(miss_df):
                miss_df.columns = ["Variable", "Valeurs manquantes"]
                st.dataframe(miss_df, use_container_width=True)
            else:
                st.success("✅ Aucune valeur manquante")
        else:
            st.info("⬅️ Importez un fichier CSV ou générez un dataset exemple")

# ══════════════════════════════════════════════
#  GUARD — dataset requis
# ══════════════════════════════════════════════
elif df is None:
    st.warning("⚠️ Importez d\'abord un dataset dans la section **📂 Import Dataset**")
    st.stop()

# ══════════════════════════════════════════════
#  SECTION 1 — TYPES
# ══════════════════════════════════════════════
elif section == "🔍 Types de Variables":
    st.markdown('<div class="section-header">🔍 TYPES DE VARIABLES</div>', unsafe_allow_html=True)

    type_rows = []
    counts = {"BIN":0, "ORD":0, "NUM":0, "CAT":0, "ID":0, "TXT":0}
    for col in df.columns:
        dtype_str, label = detect_type(df[col])
        code = label.split()[1]
        counts[code] = counts.get(code, 0) + 1
        type_rows.append({"Variable": col, "Type Python": str(df[col].dtype),
                           "Type Statistique": dtype_str, "Label": label,
                           "Valeurs uniques": df[col].nunique(),
                           "Valeurs manquantes": df[col].isnull().sum()})
    type_df = pd.DataFrame(type_rows)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(type_df, use_container_width=True, height=400)
    with col2:
        labels_c = [k for k, v in counts.items() if v > 0]
        vals_c   = [counts[k] for k in labels_c]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(vals_c, labels=labels_c, colors=PALETTE[:len(labels_c)],
               autopct="%1.0f%%", startangle=90,
               wedgeprops={"edgecolor": "white", "linewidth": 2})
        ax.set_title("Répartition des types", color="#0066cc", fontsize=11)
        fig_to_st(fig)

# ══════════════════════════════════════════════
#  SECTION 2 — STATISTIQUES DESCRIPTIVES
# ══════════════════════════════════════════════
elif section == "📊 Statistiques Descriptives":
    st.markdown('<div class="section-header">📊 STATISTIQUES DESCRIPTIVES</div>', unsafe_allow_html=True)
    df_num = df.select_dtypes(include=[np.number])
    if df_num.empty:
        st.warning("Aucune variable numérique détectée")
        st.stop()

    selected_var = st.selectbox("Sélectionner une variable", df_num.columns.tolist())
    data = df_num[selected_var].dropna().values

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Moyenne (μ)",  f"{np.mean(data):.4f}")
    col2.metric("Médiane",      f"{np.median(data):.4f}")
    col3.metric("Écart-type (σ)", f"{np.std(data):.4f}")
    col4.metric("Variance (σ²)", f"{np.var(data):.4f}")

    col5, col6, col7, col8 = st.columns(4)
    q1, q3 = np.percentile(data, 25), np.percentile(data, 75)
    col5.metric("Q1",  f"{q1:.3f}")
    col6.metric("Q3",  f"{q3:.3f}")
    col7.metric("IQR", f"{q3-q1:.3f}")
    col8.metric("Outliers (IQR)", str(int(np.sum((data < q1-1.5*(q3-q1)) | (data > q3+1.5*(q3-q1))))))

    col9, col10, col11, col12 = st.columns(4)
    col9.metric("Skewness",  f"{stats.skew(data):.4f}")
    col10.metric("Kurtosis", f"{stats.kurtosis(data):.4f}")
    col11.metric("Min", f"{data.min():.3f}")
    col12.metric("Max", f"{data.max():.3f}")

    st.markdown("---")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    color = PALETTE[df_num.columns.tolist().index(selected_var) % len(PALETTE)]

    # Histogramme + KDE
    axes[0].set_facecolor("#f4f7ff")
    axes[0].hist(data, bins=20, color=color, alpha=0.7, edgecolor="white", density=True)
    xk = np.linspace(data.min(), data.max(), 200)
    kde = stats.gaussian_kde(data)
    axes[0].plot(xk, kde(xk), color="#e8284a", lw=2, label="KDE")
    axes[0].axvline(data.mean(), color="#0066cc", lw=2, linestyle="--", label=f"μ={data.mean():.2f}")
    axes[0].axvline(np.median(data), color="#1e8c00", lw=2, linestyle=":", label=f"Méd={np.median(data):.2f}")
    axes[0].set_title(f"{selected_var} — Histogramme", color=color, fontsize=11)
    axes[0].legend(fontsize=8)

    # Box plot
    axes[1].set_facecolor("#f4f7ff")
    bp = axes[1].boxplot(data, patch_artist=True, vert=True,
                          medianprops={"color": "#e8284a", "linewidth": 2})
    for patch in bp["boxes"]: patch.set_facecolor(color + "55")
    axes[1].set_title(f"{selected_var} — Box Plot", color=color, fontsize=11)
    axes[1].set_xticks([])

    # QQ-Plot
    axes[2].set_facecolor("#f4f7ff")
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
    axes[2].scatter(osm, osr, color=color, alpha=0.6, s=20)
    axes[2].plot(osm, slope*np.array(osm)+intercept, color="#e8284a", lw=2)
    axes[2].set_title(f"{selected_var} — QQ-Plot  (r={r:.3f})", color=color, fontsize=11)
    axes[2].set_xlabel("Quantiles théoriques", fontsize=9)
    axes[2].set_ylabel("Quantiles observés", fontsize=9)

    plt.tight_layout()
    fig_to_st(fig)

    st.markdown("**📋 Résumé global de toutes les variables numériques**")
    desc = df_num.describe().T
    desc["skewness"] = df_num.apply(lambda c: stats.skew(c.dropna()))
    desc["kurtosis"] = df_num.apply(lambda c: stats.kurtosis(c.dropna()))
    st.dataframe(desc.style.format("{:.3f}"), use_container_width=True)

# ══════════════════════════════════════════════
#  SECTION — TABLEAUX DE FRÉQUENCE
# ══════════════════════════════════════════════
elif section == "📋 Tableaux de Fréquence":
    st.markdown('<div class="section-header">📋 TABLEAUX DE FRÉQUENCE & FRÉQUENCE CUMULÉE</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        var_freq = st.selectbox("Sélectionner une variable", df.columns.tolist())
        n_bins   = st.slider("Nombre de classes (variables numériques)", 5, 30, 10)
        run_freq = st.button("▶ CALCULER", use_container_width=True)

    with col2:
        if run_freq:
            serie = df[var_freq].dropna()
            is_num = pd.api.types.is_numeric_dtype(serie)

            if is_num:
                cuts = pd.cut(serie, bins=n_bins)
                freq_df = cuts.value_counts().sort_index().reset_index()
                freq_df.columns = ["Classe", "Effectif"]
                freq_df["Classe"] = freq_df["Classe"].astype(str)
                freq_df["Fréquence relative (%)"] = (freq_df["Effectif"] / len(serie) * 100).round(2)
                freq_df["Fréquence cumulée"]     = freq_df["Effectif"].cumsum()
                freq_df["Fréquence cumulée (%)"] = (freq_df["Fréquence cumulée"] / len(serie) * 100).round(2)

                st.dataframe(freq_df.style.format({"Fréquence relative (%)":"{:.2f}","Fréquence cumulée (%)":"{:.2f}"}),
                             use_container_width=True)

                fig, axes = plt.subplots(1, 3, figsize=(16, 4))
                axes[0].set_facecolor("#f4f7ff")
                axes[0].bar(range(len(freq_df)), freq_df["Effectif"],
                             color="#0066cc", alpha=0.8, edgecolor="white")
                axes[0].set_xticks(range(len(freq_df)))
                axes[0].set_xticklabels(freq_df["Classe"], rotation=45, ha="right", fontsize=7)
                axes[0].set_ylabel("Effectif")
                axes[0].set_title(f"Fréquence absolue — {var_freq}", color="#0066cc", fontsize=11)

                axes[1].set_facecolor("#f4f7ff")
                axes[1].bar(range(len(freq_df)), freq_df["Fréquence relative (%)"],
                             color="#7c3aed", alpha=0.8, edgecolor="white")
                axes[1].set_xticks(range(len(freq_df)))
                axes[1].set_xticklabels(freq_df["Classe"], rotation=45, ha="right", fontsize=7)
                axes[1].set_ylabel("Fréquence (%)")
                axes[1].set_title("Fréquence relative (%)", color="#7c3aed", fontsize=11)

                axes[2].set_facecolor("#f4f7ff")
                axes[2].plot(range(len(freq_df)), freq_df["Fréquence cumulée (%)"],
                              "o-", color="#e8284a", lw=2.5, label="Fréquence cumulée (%)")
                axes[2].fill_between(range(len(freq_df)), freq_df["Fréquence cumulée (%)"],
                                      alpha=0.15, color="#e8284a")
                axes[2].axhline(50,  color="#1e8c00", lw=1.5, linestyle="--", alpha=0.7, label="50%")
                axes[2].axhline(80,  color="#cc7700", lw=1.5, linestyle="--", alpha=0.7, label="80%")
                axes[2].axhline(100, color="#0066cc", lw=1,   linestyle=":",  alpha=0.5, label="100%")
                axes[2].set_xticks(range(len(freq_df)))
                axes[2].set_xticklabels(freq_df["Classe"], rotation=45, ha="right", fontsize=7)
                axes[2].set_ylabel("Fréquence cumulée (%)")
                axes[2].set_title("Ogive (Fréquence cumulée)", color="#e8284a", fontsize=11)
                axes[2].legend(fontsize=8)
                axes[2].set_ylim(0, 105)
                plt.tight_layout()
                fig_to_st(fig)

                st.markdown("### 📈 PP-Plot — Passage Fréquences → Probabilités")
                data_pp = serie.values
                sorted_data = np.sort(data_pp)
                n_pp = len(sorted_data)
                empirical_cdf = np.arange(1, n_pp + 1) / n_pp
                mu_pp, sigma_pp = np.mean(data_pp), np.std(data_pp)
                theoretical_cdf = stats.norm.cdf(sorted_data, mu_pp, sigma_pp)

                fig_pp, axes_pp = plt.subplots(1, 2, figsize=(12, 4))
                axes_pp[0].set_facecolor("#f4f7ff")
                axes_pp[0].scatter(theoretical_cdf, empirical_cdf, color="#0066cc", alpha=0.6, s=15)
                axes_pp[0].plot([0,1],[0,1], color="#e8284a", lw=2, linestyle="--", label="Référence")
                axes_pp[0].set_xlabel("Probabilités théoriques (Normale)")
                axes_pp[0].set_ylabel("Probabilités empiriques")
                axes_pp[0].set_title(f"PP-Plot — {var_freq}", color="#0066cc", fontsize=11)
                axes_pp[0].legend(fontsize=9)

                axes_pp[1].set_facecolor("#f4f7ff")
                xk_pp = np.linspace(sorted_data.min(), sorted_data.max(), 200)
                kde_pp = stats.gaussian_kde(data_pp)
                axes_pp[1].plot(xk_pp, kde_pp(xk_pp), color="#0066cc", lw=2.5, label="Densité estimée (KDE)")
                axes_pp[1].plot(xk_pp, stats.norm.pdf(xk_pp, mu_pp, sigma_pp),
                                 color="#e8284a", lw=2, linestyle="--", label=f"Normale N({mu_pp:.1f}, {sigma_pp:.1f})")
                axes_pp[1].set_title("Densité estimée vs Normale théorique", color="#7c3aed", fontsize=11)
                axes_pp[1].legend(fontsize=9)
                plt.tight_layout()
                fig_to_st(fig_pp)

                st.markdown("### 🔧 Estimation — Loi de Weibull")
                try:
                    shape_w, loc_w, scale_w = weibull_min.fit(serie[serie > 0].values, floc=0)
                    col_w1, col_w2, col_w3 = st.columns(3)
                    col_w1.metric("Paramètre de forme (k)", f"{shape_w:.4f}")
                    col_w2.metric("Paramètre d'échelle (λ)", f"{scale_w:.4f}")
                    col_w3.metric("Localisation", f"{loc_w:.4f}")
                    fig_w, ax_w = plt.subplots(figsize=(7, 3.5))
                    ax_w.set_facecolor("#f4f7ff")
                    xk_w = np.linspace(0.001, sorted_data.max(), 300)
                    ax_w.plot(xk_w, weibull_min.pdf(xk_w, shape_w, loc_w, scale_w),
                               color="#1e8c00", lw=2.5, label=f"Weibull(k={shape_w:.2f}, λ={scale_w:.2f})")
                    ax_w.hist(data_pp[data_pp>0], bins=20, density=True,
                               alpha=0.4, color="#0066cc", edgecolor="white", label="Données")
                    ax_w.set_title("Ajustement Loi de Weibull", color="#1e8c00", fontsize=11)
                    ax_w.legend(fontsize=9)
                    fig_to_st(fig_w)
                except Exception as e_w:
                    st.warning(f"Ajustement Weibull impossible : {e_w}")

                st.markdown("### 🔢 Loi de Benford (Premier Chiffre Significatif)")
                first_digits = serie[serie > 0].apply(
                    lambda x: int(str(abs(x)).replace(".", "").lstrip("0")[0])
                    if str(abs(x)).replace(".", "").lstrip("0") else None
                ).dropna().astype(int)
                first_digits = first_digits[first_digits.between(1, 9)]
                obs_freq = first_digits.value_counts().sort_index() / len(first_digits) * 100
                benford_freq = pd.Series({d: np.log10(1 + 1/d) * 100 for d in range(1, 10)})
                fig_b, ax_b = plt.subplots(figsize=(8, 4))
                ax_b.set_facecolor("#f4f7ff")
                x_b = np.arange(1, 10)
                ax_b.bar(x_b - 0.2, benford_freq.values, width=0.4, color="#e8284a", alpha=0.7,
                          edgecolor="white", label="Loi de Benford théorique")
                ax_b.bar(x_b + 0.2, [obs_freq.get(d, 0) for d in range(1, 10)],
                          width=0.4, color="#0066cc", alpha=0.7, edgecolor="white", label="Fréquences observées")
                ax_b.set_xlabel("Premier chiffre significatif")
                ax_b.set_ylabel("Fréquence (%)")
                ax_b.set_xticks(range(1, 10))
                ax_b.set_title("Loi de Benford vs Données", color="#0066cc", fontsize=11)
                ax_b.legend(fontsize=9)
                fig_to_st(fig_b)

            else:
                freq_cat = serie.value_counts().reset_index()
                freq_cat.columns = ["Modalité", "Effectif"]
                freq_cat["Fréquence relative (%)"] = (freq_cat["Effectif"] / len(serie) * 100).round(2)
                freq_cat["Fréquence cumulée"]     = freq_cat["Effectif"].cumsum()
                freq_cat["Fréquence cumulée (%)"] = (freq_cat["Fréquence cumulée"] / len(serie) * 100).round(2)

                st.dataframe(freq_cat.style.format({"Fréquence relative (%)":"{:.2f}","Fréquence cumulée (%)":"{:.2f}"}),
                             use_container_width=True)

                fig, axes = plt.subplots(1, 3, figsize=(16, 4))
                axes[0].set_facecolor("#f4f7ff")
                axes[0].bar(freq_cat["Modalité"], freq_cat["Effectif"],
                             color=PALETTE[:len(freq_cat)], alpha=0.8, edgecolor="white")
                axes[0].set_ylabel("Effectif")
                axes[0].set_title(f"Fréquence absolue — {var_freq}", color="#0066cc", fontsize=11)
                axes[0].tick_params(axis="x", rotation=30)

                axes[1].set_facecolor("#f4f7ff")
                axes[1].bar(freq_cat["Modalité"], freq_cat["Fréquence relative (%)"],
                             color=PALETTE[:len(freq_cat)], alpha=0.8, edgecolor="white")
                axes[1].set_ylabel("Fréquence (%)")
                axes[1].set_title("Fréquence relative (%)", color="#7c3aed", fontsize=11)
                axes[1].tick_params(axis="x", rotation=30)

                axes[2].set_facecolor("#f4f7ff")
                axes[2].plot(range(len(freq_cat)), freq_cat["Fréquence cumulée (%)"],
                              "o-", color="#e8284a", lw=2.5)
                axes[2].fill_between(range(len(freq_cat)), freq_cat["Fréquence cumulée (%)"],
                                      alpha=0.15, color="#e8284a")
                axes[2].axhline(50, color="#1e8c00", lw=1.5, linestyle="--", alpha=0.7, label="50%")
                axes[2].axhline(80, color="#cc7700", lw=1.5, linestyle="--", alpha=0.7, label="80%")
                axes[2].set_xticks(range(len(freq_cat)))
                axes[2].set_xticklabels(freq_cat["Modalité"], rotation=30, ha="right", fontsize=8)
                axes[2].set_ylabel("Fréquence cumulée (%)")
                axes[2].set_title("Fréquence cumulée", color="#e8284a", fontsize=11)
                axes[2].legend(fontsize=9)
                axes[2].set_ylim(0, 105)
                plt.tight_layout()
                fig_to_st(fig)


# ══════════════════════════════════════════════
#  SECTION 3 — CORRÉLATION
# ══════════════════════════════════════════════
elif section == "🔗 Corrélation":
    st.markdown('<div class="section-header">🔗 MATRICE DE CORRÉLATION</div>', unsafe_allow_html=True)
    df_num = df.select_dtypes(include=[np.number])
    if df_num.shape[1] < 2:
        st.warning("Il faut au moins 2 variables numériques")
        st.stop()

    method = st.selectbox("Méthode", ["pearson", "spearman", "kendall"])
    corr = df_num.corr(method=method)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = False
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", ax=ax,
                    linewidths=0.5, linecolor="white",
                    annot_kws={"size": 9}, vmin=-1, vmax=1)
        ax.set_title(f"Matrice de Corrélation ({method.capitalize()})",
                     color="#0066cc", fontsize=12)
        plt.xticks(rotation=30, ha="right", fontsize=9)
        plt.yticks(fontsize=9)
        fig_to_st(fig)

    with col2:
        st.markdown("**Paires les plus corrélées**")
        pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                r = corr.iloc[i, j]
                force = "Très forte" if abs(r)>0.8 else "Forte" if abs(r)>0.6 else "Modérée" if abs(r)>0.4 else "Faible" if abs(r)>0.2 else "Très faible"
                pairs.append({"Paire": f"{cols[i]} × {cols[j]}", "r": round(r,3), "Force": force})
        pairs_df = pd.DataFrame(pairs).sort_values("r", key=abs, ascending=False)
        st.dataframe(pairs_df.head(15), use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.set_facecolor("#f4f7ff")
        corr_flat = corr.values[np.tril_indices_from(corr.values, k=-1)]
        ax2.hist(corr_flat, bins=20, color="#0066cc", alpha=0.7, edgecolor="white")
        ax2.axvline(0, color="#e8284a", lw=2, linestyle="--")
        ax2.set_title("Distribution des corrélations", color="#0066cc", fontsize=10)
        ax2.set_xlabel("r"); ax2.set_ylabel("Fréquence")
        fig_to_st(fig2)

# ══════════════════════════════════════════════
#  SECTION 4 — RÉGRESSION
# ══════════════════════════════════════════════
elif section == "📈 Régression":
    st.markdown('<div class="section-header">📈 RÉGRESSION LINÉAIRE</div>', unsafe_allow_html=True)
    df_num = df.select_dtypes(include=[np.number])
    if df_num.shape[1] < 2:
        st.warning("Il faut au moins 2 variables numériques")
        st.stop()

    col1, col2 = st.columns([1, 2])
    with col1:
        y_var  = st.selectbox("Variable dépendante (Y)", df_num.columns.tolist())
        x_vars = st.multiselect("Variables indépendantes (X)",
                                 [c for c in df_num.columns if c != y_var],
                                 default=[c for c in df_num.columns if c != y_var][:1])
        run_reg = st.button("▶ CALCULER RÉGRESSION", use_container_width=True)

    with col2:
        if run_reg and x_vars:
            sub = df_num[[y_var] + x_vars].dropna()
            X = sub[x_vars].values
            y = sub[y_var].values
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            r2     = model.score(X, y)
            n, p   = len(y), len(x_vars)
            r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else 0
            mse    = np.mean((y - y_pred)**2)
            rmse   = np.sqrt(mse)

            # Stocker pour la prédiction
            st.session_state.last_reg = {
                "model": model, "x_vars": x_vars, "y_var": y_var,
                "r2": r2, "r2_adj": r2_adj, "rmse": rmse, "n": n
            }

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²",      f"{r2:.4f}",  delta=f"{r2*100:.1f}%")
            c2.metric("R² ajusté", f"{r2_adj:.4f}")
            c3.metric("RMSE",    f"{rmse:.4f}")
            c4.metric("N obs.",  n)

            # Équation
            eq_parts = [f"{model.intercept_:.3f}"]
            for coef, name in zip(model.coef_, x_vars):
                eq_parts.append(f"{coef:+.3f}·{name}")
            st.info(f"**Équation :** Ŷ = {'  '.join(eq_parts)}")

            # Graphiques
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            # Ajustement
            axes[0].set_facecolor("#f4f7ff")
            axes[0].scatter(y_pred, y, color="#0066cc", alpha=0.6, s=20)
            mn, mx = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
            axes[0].plot([mn, mx], [mn, mx], color="#e8284a", lw=2, linestyle="--")
            axes[0].set_xlabel("Ŷ prédites"); axes[0].set_ylabel("Y réelles")
            axes[0].set_title("Valeurs réelles vs prédites", color="#0066cc", fontsize=11)
            # Résidus
            resid = y - y_pred
            axes[1].set_facecolor("#f4f7ff")
            axes[1].scatter(y_pred, resid, color="#7c3aed", alpha=0.6, s=20)
            axes[1].axhline(0, color="#e8284a", lw=2, linestyle="--")
            axes[1].set_xlabel("Ŷ prédites"); axes[1].set_ylabel("Résidus")
            axes[1].set_title("Résidus vs Valeurs prédites", color="#7c3aed", fontsize=11)
            # Coefficients
            axes[2].set_facecolor("#f4f7ff")
            colors_coef = ["#1e8c00" if c >= 0 else "#e8284a" for c in model.coef_]
            axes[2].barh(x_vars, model.coef_, color=colors_coef, alpha=0.8, edgecolor="white")
            axes[2].axvline(0, color="#1a2540", lw=1)
            axes[2].set_title("Coefficients β", color="#1e8c00", fontsize=11)
            plt.tight_layout()
            fig_to_st(fig)

            # Tableau coefficients
            coef_df = pd.DataFrame({
                "Variable": ["Intercept"] + x_vars,
                "Coefficient": [model.intercept_] + list(model.coef_),
                "Interprétation": ["Valeur de Y quand tous X=0"] +
                    [f"+1 unité {n} → {c:+.3f} unités Y" for n, c in zip(x_vars, model.coef_)]
            })
            st.dataframe(coef_df, use_container_width=True)
        elif run_reg:
            st.warning("Sélectionnez au moins une variable X")

    # ─── PRÉDICTION RÉGRESSION ───
    if st.session_state.last_reg is not None:
        st.markdown("---")
        st.markdown("### 🔮 Prédiction en Temps Réel")
        st.markdown("Saisissez les valeurs des features pour obtenir une prédiction instantanée :")
        reg_info = st.session_state.last_reg
        pred_cols = st.columns(len(reg_info["x_vars"]))
        pred_inputs = {}
        for i, xv in enumerate(reg_info["x_vars"]):
            mean_v = float(df[xv].mean()) if xv in df.columns else 0.0
            pred_inputs[xv] = pred_cols[i].number_input(
                xv, value=round(mean_v, 2), key=f"reg_pred_{xv}"
            )
        if st.button("🔮 PRÉDIRE", key="btn_pred_reg"):
            X_new = np.array([[pred_inputs[xv] for xv in reg_info["x_vars"]]])
            y_hat = reg_info["model"].predict(X_new)[0]
            eq = " + ".join([f"{c:.3f}×{n}" for c, n in zip(reg_info["model"].coef_, reg_info["x_vars"])])
            st.markdown(f"""
            <div class="pred-result">
                <div style="font-size:11px;color:#5a6a8a;letter-spacing:2px;margin-bottom:4px">PRÉDICTION Ŷ ({reg_info['y_var']})</div>
                <div style="font-family:Rajdhani,sans-serif;font-size:36px;font-weight:700;color:#1e8c00">{y_hat:.4f}</div>
                <div style="font-size:10px;color:#8a9bbf;margin-top:6px">{reg_info['model'].intercept_:.3f} + {eq} = {y_hat:.4f}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  SECTION 5 — ANOVA
# ══════════════════════════════════════════════
elif section == "⚗️ ANOVA":
    st.markdown('<div class="section-header">⚗️ ANALYSE DE LA VARIANCE (ANOVA)</div>', unsafe_allow_html=True)
    df_num = df.select_dtypes(include=[np.number])

    col1, col2 = st.columns([1, 2])
    with col1:
        y_anova = st.selectbox("Variable numérique (Y)", df_num.columns.tolist())
        x_anova = st.selectbox("Variable de groupe (X catégorielle)", df.columns.tolist())
        run_anova = st.button("▶ CALCULER ANOVA", use_container_width=True)

    with col2:
        if run_anova:
            groups_data = {g: grp[y_anova].dropna().values
                           for g, grp in df.groupby(x_anova) if len(grp[y_anova].dropna()) > 0}
            if len(groups_data) < 2:
                st.error("Il faut au moins 2 groupes")
            else:
                F, p = f_oneway(*groups_data.values())
                all_vals = np.concatenate(list(groups_data.values()))
                grand_mean = np.mean(all_vals)
                N = len(all_vals); k = len(groups_data)
                SS_b = sum(len(v)*(np.mean(v)-grand_mean)**2 for v in groups_data.values())
                SS_w = sum(np.sum((v-np.mean(v))**2) for v in groups_data.values())
                SS_t = np.sum((all_vals - grand_mean)**2)
                eta2 = SS_b / SS_t if SS_t > 0 else 0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("F-statistic", f"{F:.3f}")
                c2.metric("p-value",     f"{p:.4f}")
                c3.metric("η² (effet)",  f"{eta2:.4f}")
                c4.metric("Groupes",     k)

                sig = p < 0.05
                if sig:
                    st.success(f"✅ Différence significative entre groupes (p = {p:.4f} < 0.05)")
                else:
                    st.warning(f"❌ Pas de différence significative (p = {p:.4f} > 0.05)")

                # Tableau ANOVA
                anova_tbl = pd.DataFrame({
                    "Source": ["Entre groupes", "Intra groupes", "Total"],
                    "SS":  [SS_b, SS_w, SS_t],
                    "df":  [k-1, N-k, N-1],
                    "MS":  [SS_b/(k-1), SS_w/(N-k), ""],
                    "F":   [F, "", ""],
                    "p":   [p, "", ""]
                })
                st.dataframe(anova_tbl, use_container_width=True)

                # Graphiques
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                # Box plot par groupe
                axes[0].set_facecolor("#f4f7ff")
                data_list  = list(groups_data.values())
                group_names = list(groups_data.keys())
                bp = axes[0].boxplot(data_list, patch_artist=True,
                                     medianprops={"color":"#e8284a","linewidth":2})
                for patch, color in zip(bp["boxes"], PALETTE):
                    patch.set_facecolor(color + "55")
                axes[0].set_xticklabels(group_names, rotation=20, ha="right", fontsize=9)
                axes[0].set_title(f"{y_anova} par {x_anova}", color="#0066cc", fontsize=11)
                # Moyennes par groupe
                means = [np.mean(v) for v in data_list]
                bars  = axes[1].bar(group_names, means, color=PALETTE[:k], alpha=0.8, edgecolor="white")
                axes[1].set_facecolor("#f4f7ff")
                for bar, m in zip(bars, means):
                    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01*max(means),
                                  f"{m:.2f}", ha="center", fontsize=9, fontweight="bold")
                axes[1].set_title("Moyennes par groupe", color="#1e8c00", fontsize=11)
                axes[1].tick_params(axis="x", rotation=20)
                plt.tight_layout()
                fig_to_st(fig)

# ══════════════════════════════════════════════
#  SECTION 6 — CLASSIFICATION
# ══════════════════════════════════════════════
elif section == "🤖 Classification":
    st.markdown('<div class="section-header">🤖 CLASSIFICATION</div>', unsafe_allow_html=True)
    df_num = df.select_dtypes(include=[np.number])

    col1, col2 = st.columns([1, 2])
    with col1:
        y_clf = st.selectbox("Variable cible (Classe)", df.columns.tolist())
        x_clf = st.multiselect("Features (variables explicatives)",
                                df_num.columns.tolist(),
                                default=df_num.columns.tolist()[:3])
        model_name = st.selectbox("Modèle", [
            "K-Nearest Neighbors (KNN)",
            "Naive Bayes (Gaussien)",
            "Régression Logistique",
            "SVM — Support Vector Machine",
            "Arbre de Décision (Decision Tree)",
            "Random Forest (RF)",
            "K-Means (Clustering)",
            "ACP — Analyse en Composantes Principales",
            "ACM — Analyse des Correspondances Multiples"
        ])
        # Paramètres selon le modèle
        if "KNN" in model_name:
            k_knn = st.slider("K (nombre de voisins)", 1, 20, 5)
        elif "SVM" in model_name:
            kernel = st.selectbox("Noyau SVM", ["rbf", "linear", "poly"])
            C_svm  = st.slider("Paramètre C", 0.1, 10.0, 1.0)
        elif "Random Forest" in model_name:
            n_trees = st.slider("Nombre d'arbres", 10, 200, 100)
            max_depth_rf = st.slider("Profondeur max", 2, 20, 5)
        elif "Arbre" in model_name:
            max_depth_dt = st.slider("Profondeur max", 2, 20, 5)
        elif "K-Means" in model_name:
            k_means = st.slider("Nombre de clusters K", 2, 10, 3)
        elif "ACP" in model_name:
            n_comp = st.slider("Nombre de composantes", 2, min(len(x_clf), 10) if x_clf else 2, 2)
        test_size = st.slider("Taille test (%)", 10, 40, 20)
        run_clf = st.button("▶ ENTRAÎNER LE MODÈLE", use_container_width=True)

    with col2:
        if run_clf and x_clf:
            sub = df[[y_clf] + x_clf].dropna()
            le  = LabelEncoder()
            y   = le.fit_transform(sub[y_clf].astype(str))
            X   = sub[x_clf].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            classes = le.classes_

            is_facto = "ACP" in model_name or "ACM" in model_name

            if is_facto:
                # ── ACP / ACM ──
                nc = n_comp if "ACP" in model_name else 2
                pca = PCA(n_components=min(nc, X_scaled.shape[1]))
                X_pca = pca.fit_transform(X_scaled)
                explained = pca.explained_variance_ratio_ * 100

                st.markdown(f"**{model_name}**")
                # Tableau inertie
                inertia_df = pd.DataFrame({
                    "Composante": [f"PC{i+1}" for i in range(len(explained))],
                    "Valeur propre": pca.explained_variance_,
                    "Variance (%)": explained,
                    "Cumul (%)": np.cumsum(explained)
                })
                st.dataframe(inertia_df.style.format({"Valeur propre":"{:.3f}","Variance (%)":"{:.1f}","Cumul (%)":"{:.1f}"}),
                             use_container_width=True)

                fig, axes = plt.subplots(1, 2, figsize=(13, 4))
                # Scree plot
                axes[0].set_facecolor("#f4f7ff")
                axes[0].bar(range(1, len(explained)+1), explained,
                             color=PALETTE[:len(explained)], alpha=0.8, edgecolor="white")
                axes[0].plot(range(1, len(explained)+1), np.cumsum(explained),
                              "o-", color="#e8284a", lw=2, label="Cumul")
                axes[0].axhline(70, color="#1e8c00", lw=1.5, linestyle="--", label="Seuil 70%")
                axes[0].set_xlabel("Composante"); axes[0].set_ylabel("Variance expliquée (%)")
                axes[0].set_title("Scree Plot", color="#0066cc", fontsize=11)
                axes[0].legend(fontsize=9)
                # Plan factoriel
                axes[1].set_facecolor("#f4f7ff")
                for i, cls in enumerate(np.unique(y)):
                    mask = y == cls
                    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1] if X_pca.shape[1]>1 else np.zeros(mask.sum()),
                                     color=PALETTE[i%len(PALETTE)], alpha=0.6, s=30, label=classes[cls])
                axes[1].set_xlabel(f"PC1 ({explained[0]:.1f}%)")
                axes[1].set_ylabel(f"PC2 ({explained[1]:.1f}%)" if len(explained)>1 else "PC2")
                axes[1].set_title("Plan factoriel PC1 × PC2", color="#7c3aed", fontsize=11)
                axes[1].legend(fontsize=9)
                plt.tight_layout()
                fig_to_st(fig)

                # Cercle des corrélations
                if X_scaled.shape[1] >= 2:
                    loadings = pca.components_.T
                    fig2, ax2 = plt.subplots(figsize=(5, 5))
                    ax2.set_facecolor("#f4f7ff")
                    circle = plt.Circle((0,0), 1, fill=False, color="#d0daf0", lw=1.5)
                    ax2.add_patch(circle)
                    for i, feat in enumerate(x_clf):
                        ax2.annotate("", xy=(loadings[i,0], loadings[i,1] if loadings.shape[1]>1 else 0),
                                      xytext=(0,0),
                                      arrowprops=dict(arrowstyle="->", color=PALETTE[i%len(PALETTE)], lw=2))
                        ax2.text(loadings[i,0]*1.12, (loadings[i,1] if loadings.shape[1]>1 else 0)*1.12,
                                  feat, fontsize=9, color=PALETTE[i%len(PALETTE)])
                    ax2.set_xlim(-1.2, 1.2); ax2.set_ylim(-1.2, 1.2)
                    ax2.axhline(0, color="#d0daf0", lw=0.8)
                    ax2.axvline(0, color="#d0daf0", lw=0.8)
                    ax2.set_xlabel(f"PC1 ({explained[0]:.1f}%)"); ax2.set_ylabel(f"PC2 ({explained[1]:.1f}%)" if len(explained)>1 else "PC2")
                    ax2.set_title("Cercle des corrélations", color="#7c3aed", fontsize=11)
                    fig_to_st(fig2)

                # Pas de prédiction pour ACP/ACM
                st.session_state.last_clf = None

            else:
                # ── Modèles supervisés ──
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_scaled, y, test_size=test_size/100, random_state=42, stratify=y)

                if "KNN" in model_name:
                    clf = KNeighborsClassifier(n_neighbors=k_knn)
                elif "Naive" in model_name:
                    clf = GaussianNB()
                elif "Logistique" in model_name:
                    clf = LogisticRegression(max_iter=500)
                elif "Random Forest" in model_name:
                    clf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth_rf, random_state=42)
                elif "Arbre" in model_name:
                    clf = DecisionTreeClassifier(max_depth=max_depth_dt, random_state=42)
                elif "K-Means" in model_name:
                    km = KMeans(n_clusters=k_means, random_state=42, n_init=10)
                    km.fit(X_scaled)
                    labels = km.labels_
                    sil = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0
                    inertia = km.inertia_
                    c_km1, c_km2 = st.columns(2)
                    c_km1.metric("Inertie (WSS)", f"{inertia:.2f}")
                    c_km2.metric("Score Silhouette", f"{sil:.4f}")
                    from sklearn.decomposition import PCA as PCA2
                    pca_km = PCA2(n_components=2)
                    X_2d = pca_km.fit_transform(X_scaled)
                    fig_km, axes_km = plt.subplots(1, 2, figsize=(12, 4))
                    axes_km[0].set_facecolor("#f4f7ff")
                    for k_i in range(k_means):
                        mask_k = labels == k_i
                        axes_km[0].scatter(X_2d[mask_k, 0], X_2d[mask_k, 1],
                                      color=PALETTE[k_i % len(PALETTE)], alpha=0.6, s=30,
                                      label=f"Cluster {k_i+1}")
                    axes_km[0].set_title(f"K-Means (k={k_means}) — Plan PCA", color="#0066cc", fontsize=12)
                    axes_km[0].legend(fontsize=9)
                    inertias = []
                    ks = range(2, min(11, len(X_scaled)))
                    for ki in ks:
                        inertias.append(KMeans(n_clusters=ki, random_state=42, n_init=10).fit(X_scaled).inertia_)
                    axes_km[1].set_facecolor("#f4f7ff")
                    axes_km[1].plot(list(ks), inertias, "o-", color="#0066cc", lw=2)
                    axes_km[1].axvline(k_means, color="#e8284a", lw=2, linestyle="--", label=f"k={k_means} choisi")
                    axes_km[1].set_xlabel("Nombre de clusters K"); axes_km[1].set_ylabel("Inertie")
                    axes_km[1].set_title("Méthode du coude (Elbow)", color="#0066cc", fontsize=11)
                    axes_km[1].legend()
                    plt.tight_layout(); fig_to_st(fig_km)
                    st.session_state.last_clf = None
                    clf = None
                else:  # SVM
                    clf = SVC(kernel=kernel, C=C_svm, probability=True)

                if clf is None:
                    st.stop()
                clf.fit(X_tr, y_tr)
                y_pred_te = clf.predict(X_te)
                acc  = accuracy_score(y_te, y_pred_te)
                f1   = f1_score(y_te, y_pred_te, average="weighted", zero_division=0)
                prec = precision_score(y_te, y_pred_te, average="weighted", zero_division=0)
                mae  = mean_absolute_error(y_te, y_pred_te)
                try:
                    auc_val = roc_auc_score(y_te, clf.predict_proba(X_te), multi_class="ovr", average="weighted") if hasattr(clf, "predict_proba") else 0.5
                    gini = 2 * auc_val - 1
                except:
                    auc_val = 0.5; gini = 0
                try:
                    sil_val = silhouette_score(X_scaled, y)
                except:
                    sil_val = 0
                cv   = cross_val_score(clf, X_scaled, y, cv=min(5, len(np.unique(y))), scoring="accuracy").mean()

                # Stocker pour prédiction
                st.session_state.last_clf = {
                    "clf": clf, "scaler": scaler, "le": le,
                    "x_clf": x_clf, "y_clf": y_clf,
                    "classes": classes, "model_name": model_name
                }

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy (test)", f"{acc:.4f}", delta=f"{acc*100:.1f}%")
                c2.metric("Accuracy (CV-5)", f"{cv:.4f}")
                c3.metric("Classes", len(classes))
                c4.metric("N test", len(y_te))
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("F1-Score (weighted)", f"{f1:.4f}")
                c6.metric("Precision (weighted)", f"{prec:.4f}")
                c7.metric("MAE", f"{mae:.4f}")
                c8.metric("Indice de Gini", f"{gini:.4f}")
                st.info(f"ROC-AUC : **{auc_val:.4f}** · Silhouette : **{sil_val:.4f}**")

                st.text(classification_report(y_te, y_pred_te, target_names=[str(c) for c in classes]))

                fig, axes = plt.subplots(1, 3, figsize=(16, 4))
                # Matrice de confusion
                cm = confusion_matrix(y_te, y_pred_te)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                             xticklabels=[str(c) for c in classes],
                             yticklabels=[str(c) for c in classes],
                             linewidths=0.5, linecolor="white", annot_kws={"size":11})
                axes[0].set_title(f"Matrice de Confusion\n{model_name}", color="#7c3aed", fontsize=11)
                axes[0].set_ylabel("Réel"); axes[0].set_xlabel("Prédit")
                # Accuracy bar
                axes[1].set_facecolor("#f4f7ff")
                axes[1].bar(["Test", "CV-5"], [acc, cv], color=[PALETTE[0], PALETTE[2]], alpha=0.85, edgecolor="white")
                axes[1].set_ylim(0, 1.1)
                for xi, val in enumerate([acc, cv]):
                    axes[1].text(xi, val+0.03, f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
                axes[1].axhline(0.5, color="#e8284a", lw=1.5, linestyle="--", alpha=0.6, label="50%")
                axes[1].set_title("Accuracy", color="#0066cc", fontsize=11)
                axes[1].legend(fontsize=9)
                # Importance features
                if hasattr(clf, "feature_importances_"):
                    importances = clf.feature_importances_
                    imp_label = "Importance (RF/DT Gini)"
                else:
                    importances = [abs(np.corrcoef(X_scaled[:,i], y)[0,1]) for i in range(len(x_clf))]
                    imp_label = "Importance des Features (|corr|)"
                axes[2].set_facecolor("#f4f7ff")
                bars2 = axes[2].barh(x_clf, importances, color=PALETTE[:len(x_clf)], alpha=0.85, edgecolor="white")
                for bar, val in zip(bars2, importances):
                    axes[2].text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                                  f"{val:.3f}", va="center", fontsize=9)
                axes[2].set_title(imp_label, color="#1e8c00", fontsize=11)
                plt.tight_layout()
                fig_to_st(fig)
        elif run_clf:
            st.warning("Sélectionnez au moins une feature")

    # ─── PRÉDICTION CLASSIFICATION ───
    if st.session_state.last_clf is not None:
        st.markdown("---")
        st.markdown("### 🔮 Prédiction sur Nouvelles Données")
        st.markdown("Saisissez les valeurs des features pour prédire la classe en temps réel :")
        clf_info = st.session_state.last_clf
        pred_cols2 = st.columns(len(clf_info["x_clf"]))
        pred_inputs2 = {}
        for i, xv in enumerate(clf_info["x_clf"]):
            mean_v = float(df[xv].mean()) if xv in df.columns else 0.0
            pred_inputs2[xv] = pred_cols2[i].number_input(
                xv, value=round(mean_v, 2), key=f"clf_pred_{xv}"
            )
        if st.button("🔮 PRÉDIRE LA CLASSE", key="btn_pred_clf"):
            X_new2 = np.array([[pred_inputs2[xv] for xv in clf_info["x_clf"]]])
            X_new2_scaled = clf_info["scaler"].transform(X_new2)
            pred_class_idx = clf_info["clf"].predict(X_new2_scaled)[0]
            pred_class = clf_info["le"].inverse_transform([pred_class_idx])[0]
            # Probabilités
            try:
                probs = clf_info["clf"].predict_proba(X_new2_scaled)[0]
                has_proba = True
            except:
                probs = np.zeros(len(clf_info["classes"]))
                probs[pred_class_idx] = 1.0
                has_proba = False

            st.markdown(f"""
            <div class="pred-result-clf">
                <div style="font-size:11px;color:#5a6a8a;letter-spacing:2px;margin-bottom:4px">CLASSE PRÉDITE</div>
                <div style="font-family:Rajdhani,sans-serif;font-size:36px;font-weight:700;color:#7c3aed">{pred_class}</div>
                <div style="font-size:10px;color:#8a9bbf;margin-top:4px">Confiance : {max(probs)*100:.1f}%  ·  Modèle : {clf_info["model_name"]}</div>
            </div>""", unsafe_allow_html=True)

            # Barres de probabilité
            st.markdown("**Probabilités par classe :**")
            for cls, prob in sorted(zip(clf_info["classes"], probs), key=lambda x: -x[1]):
                color = "#7c3aed" if cls == pred_class else "#d0daf0"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
                    <div style="font-size:11px;min-width:90px;font-weight:{'700' if cls==pred_class else '400'};color:{'#7c3aed' if cls==pred_class else '#5a6a8a'}">{cls}</div>
                    <div style="flex:1;height:12px;background:#f0f0f0;border-radius:6px;overflow:hidden">
                        <div style="width:{prob*100:.1f}%;height:100%;background:{color};border-radius:6px"></div>
                    </div>
                    <div style="font-size:11px;min-width:45px;font-weight:700;color:{'#7c3aed' if cls==pred_class else '#8a9bbf'}">{prob*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  SECTION — TESTS D'HYPOTHÈSE
# ══════════════════════════════════════════════
elif section == "🧪 Tests d'Hypothèse":
    st.markdown('<div class="section-header">🧪 TESTS D\'HYPOTHÈSE — PASSAGE STAT → PROBA</div>', unsafe_allow_html=True)
    df_num = df.select_dtypes(include=[np.number])

    test_type = st.selectbox("Type de test", [
        "Test t de Student (1 échantillon)",
        "Test t de Student (2 échantillons)",
        "Test de Shapiro-Wilk (Normalité)",
        "Test de Kolmogorov-Smirnov",
        "Test du Chi-Deux (indépendance)",
        "Test de Mann-Whitney (non-paramétrique)"
    ])
    st.markdown("---")

    if test_type == "Test t de Student (1 échantillon)":
        col1, col2 = st.columns(2)
        with col1:
            var_t = st.selectbox("Variable", df_num.columns.tolist())
            mu0   = st.number_input("Valeur de référence μ₀", value=float(df_num[var_t].mean()))
            alpha = st.selectbox("Niveau de signification α", [0.01, 0.05, 0.10], index=1)
            run_t = st.button("▶ CALCULER", use_container_width=True)
        with col2:
            if run_t:
                data_t = df_num[var_t].dropna()
                t_stat, p_val = stats.ttest_1samp(data_t, mu0)
                df_t = len(data_t) - 1
                t_crit = stats.t.ppf(1 - alpha/2, df=df_t)
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("t-statistique", f"{t_stat:.4f}")
                col_b.metric("p-valeur", f"{p_val:.4f}")
                col_c.metric("t critique (±α/2)", f"{t_crit:.4f}")
                if p_val < alpha:
                    st.success(f"✅ H₀ rejetée : différence significative à α={alpha} (p={p_val:.4f})")
                else:
                    st.warning(f"⚠️ H₀ non rejetée à α={alpha} (p={p_val:.4f})")
                st.info(f"H₀ : μ = {mu0}  |  H₁ : μ ≠ {mu0}  |  ddl = {df_t}")
                fig_t, ax_t = plt.subplots(figsize=(8, 4))
                ax_t.set_facecolor("#f4f7ff")
                x_range = np.linspace(-5, 5, 500)
                ax_t.plot(x_range, stats.t.pdf(x_range, df=df_t), color="#0066cc", lw=2.5, label=f"t(ddl={df_t})")
                ax_t.axvline(t_stat, color="#e8284a", lw=2.5, label=f"t obs = {t_stat:.3f}")
                ax_t.axvline( t_crit, color="#1e8c00", lw=2, linestyle="--", label=f"t crit = ±{t_crit:.3f}")
                ax_t.axvline(-t_crit, color="#1e8c00", lw=2, linestyle="--")
                x_rej_r = np.linspace(t_crit, 5, 200)
                x_rej_l = np.linspace(-5, -t_crit, 200)
                ax_t.fill_between(x_rej_r, stats.t.pdf(x_rej_r, df=df_t), alpha=0.25, color="#e8284a", label="Zone de rejet")
                ax_t.fill_between(x_rej_l, stats.t.pdf(x_rej_l, df=df_t), alpha=0.25, color="#e8284a")
                ax_t.set_xlabel("Valeur de t"); ax_t.set_ylabel("Densité")
                ax_t.set_title("Distribution t de Student — Zone de rejet", color="#0066cc", fontsize=11)
                ax_t.legend(fontsize=9)
                fig_to_st(fig_t)

    elif test_type == "Test t de Student (2 échantillons)":
        col1, col2 = st.columns(2)
        with col1:
            var_t2  = st.selectbox("Variable numérique", df_num.columns.tolist())
            grp_col = st.selectbox("Variable de groupe (2 groupes)", df.columns.tolist())
            alpha2  = st.selectbox("Niveau α", [0.01, 0.05, 0.10], index=1)
            run_t2  = st.button("▶ CALCULER", use_container_width=True)
        with col2:
            if run_t2:
                grps = df[grp_col].dropna().unique()
                if len(grps) < 2:
                    st.error("La variable de groupe doit avoir au moins 2 modalités")
                else:
                    g1 = df[df[grp_col] == grps[0]][var_t2].dropna()
                    g2 = df[df[grp_col] == grps[1]][var_t2].dropna()
                    t2, p2 = stats.ttest_ind(g1, g2)
                    col_t1, col_t2 = st.columns(2)
                    col_t1.metric("t-stat", f"{t2:.4f}")
                    col_t2.metric("p-valeur", f"{p2:.4f}")
                    if p2 < alpha2:
                        st.success(f"✅ H₀ rejetée (p={p2:.4f} < α={alpha2})")
                    else:
                        st.warning(f"⚠️ H₀ non rejetée (p={p2:.4f} ≥ α={alpha2})")
                    st.info(f"Groupe {grps[0]}: n={len(g1)}, μ={g1.mean():.3f}  |  Groupe {grps[1]}: n={len(g2)}, μ={g2.mean():.3f}")

    elif test_type == "Test de Shapiro-Wilk (Normalité)":
        var_sw = st.selectbox("Variable", df_num.columns.tolist())
        alpha_sw = st.selectbox("Niveau α", [0.01, 0.05, 0.10], index=1)
        if st.button("▶ CALCULER", use_container_width=True):
            data_sw = df_num[var_sw].dropna()
            if len(data_sw) > 5000:
                data_sw = data_sw.sample(5000, random_state=42)
                st.warning("⚠️ Échantillon limité à 5000 obs pour Shapiro-Wilk")
            w_stat, p_sw = stats.shapiro(data_sw)
            c_sw1, c_sw2 = st.columns(2)
            c_sw1.metric("W-statistique", f"{w_stat:.4f}")
            c_sw2.metric("p-valeur", f"{p_sw:.4f}")
            if p_sw < alpha_sw:
                st.error(f"❌ Non-normalité détectée (p={p_sw:.4f} < {alpha_sw})")
            else:
                st.success(f"✅ Normalité acceptée (p={p_sw:.4f} ≥ {alpha_sw})")

    elif test_type == "Test de Kolmogorov-Smirnov":
        var_ks = st.selectbox("Variable", df_num.columns.tolist())
        dist_ks = st.selectbox("Distribution de référence", ["norm", "expon", "uniform"])
        if st.button("▶ CALCULER", use_container_width=True):
            data_ks = df_num[var_ks].dropna()
            data_std = (data_ks - data_ks.mean()) / data_ks.std()
            ks_stat, p_ks = stats.kstest(data_std, dist_ks)
            c_ks1, c_ks2 = st.columns(2)
            c_ks1.metric("KS-statistique", f"{ks_stat:.4f}")
            c_ks2.metric("p-valeur", f"{p_ks:.4f}")
            if p_ks < 0.05:
                st.error(f"❌ Les données ne suivent pas une loi {dist_ks} (p={p_ks:.4f})")
            else:
                st.success(f"✅ Les données semblent suivre une loi {dist_ks} (p={p_ks:.4f})")

    elif test_type == "Test du Chi-Deux (indépendance)":
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if len(cat_cols) < 2:
            st.warning("Il faut au moins 2 variables catégorielles")
        else:
            col1, col2 = st.columns(2)
            with col1:
                var_x_chi = st.selectbox("Variable X", cat_cols)
                var_y_chi = st.selectbox("Variable Y", [c for c in cat_cols if c != var_x_chi])
                run_chi = st.button("▶ CALCULER", use_container_width=True)
            with col2:
                if run_chi:
                    ct = pd.crosstab(df[var_x_chi], df[var_y_chi])
                    chi2, p_chi, dof, expected = stats.chi2_contingency(ct)
                    col_c1, col_c2, col_c3 = st.columns(3)
                    col_c1.metric("χ²", f"{chi2:.4f}")
                    col_c2.metric("p-valeur", f"{p_chi:.4f}")
                    col_c3.metric("ddl", dof)
                    if p_chi < 0.05:
                        st.success("✅ Les deux variables sont liées (dépendance significative)")
                    else:
                        st.warning("⚠️ Pas d'association significative entre les variables")
                    st.markdown("**Tableau de contingence**")
                    st.dataframe(ct, use_container_width=True)

    elif test_type == "Test de Mann-Whitney (non-paramétrique)":
        col1, col2 = st.columns(2)
        with col1:
            var_mw   = st.selectbox("Variable numérique", df_num.columns.tolist())
            grp_mw   = st.selectbox("Variable de groupe", df.columns.tolist())
            alpha_mw = st.selectbox("Niveau α", [0.01, 0.05, 0.10], index=1)
            run_mw   = st.button("▶ CALCULER", use_container_width=True)
        with col2:
            if run_mw:
                grps_mw = df[grp_mw].dropna().unique()
                if len(grps_mw) < 2:
                    st.error("Il faut au moins 2 groupes")
                else:
                    g1 = df[df[grp_mw] == grps_mw[0]][var_mw].dropna()
                    g2 = df[df[grp_mw] == grps_mw[1]][var_mw].dropna()
                    u_stat, p_mw = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                    c_mw1, c_mw2 = st.columns(2)
                    c_mw1.metric("U-statistique", f"{u_stat:.4f}")
                    c_mw2.metric("p-valeur", f"{p_mw:.4f}")
                    if p_mw < alpha_mw:
                        st.success(f"✅ Différence significative (p={p_mw:.4f} < α={alpha_mw})")
                    else:
                        st.warning(f"⚠️ Pas de différence significative (p={p_mw:.4f})")


# ══════════════════════════════════════════════
#  SECTION 7 — GRAPHIQUES
# ══════════════════════════════════════════════
elif section == "🎨 Graphiques":
    st.markdown('<div class="section-header">🎨 GRAPHIQUES PERSONNALISÉS</div>', unsafe_allow_html=True)
    df_num = df.select_dtypes(include=[np.number])

    col1, col2 = st.columns([1, 3])
    with col1:
        graph_type = st.selectbox("Type de graphique", [
            "Nuage de Points (Scatter)",
            "Histogramme",
            "Box Plot Comparatif",
            "QQ-Plot (Normalité)",
            "Violin Plot",
            "Histogrammes Multiples",
            "Scatter Matrix"
        ])
        if graph_type in ["Nuage de Points (Scatter)"]:
            gx = st.selectbox("Axe X", df_num.columns.tolist())
            gy = st.selectbox("Axe Y", df_num.columns.tolist(),
                               index=min(1, len(df_num.columns)-1))
            hue_col = st.selectbox("Couleur par (optionnel)", [""] + df.columns.tolist())
        elif graph_type in ["Histogramme", "QQ-Plot (Normalité)", "Violin Plot"]:
            gx = st.selectbox("Variable", df_num.columns.tolist())
        elif graph_type == "Box Plot Comparatif":
            gx = st.selectbox("Variable numérique (Y)", df_num.columns.tolist())
            gy = st.selectbox("Variable de groupe (X)", df.columns.tolist())
        gen_btn = st.button("▶ GÉNÉRER", use_container_width=True)

    with col2:
        if gen_btn:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.set_facecolor("#f4f7ff")
            if graph_type == "Nuage de Points (Scatter)":
                if hue_col and hue_col in df.columns:
                    for i, (grp, sub) in enumerate(df.groupby(hue_col)):
                        ax.scatter(sub[gx], sub[gy], color=PALETTE[i%len(PALETTE)],
                                    alpha=0.6, s=25, label=str(grp))
                    ax.legend(fontsize=9)
                else:
                    ax.scatter(df[gx], df[gy], color="#0066cc", alpha=0.6, s=25)
                r_val = np.corrcoef(df[gx].dropna(), df[gy].dropna())[0,1]
                ax.set_xlabel(gx); ax.set_ylabel(gy)
                ax.set_title(f"{gx} × {gy}  (r={r_val:.3f})", color="#0066cc", fontsize=12)
            elif graph_type == "Histogramme":
                d = df_num[gx].dropna()
                ax.hist(d, bins=25, color="#0066cc", alpha=0.7, edgecolor="white")
                ax.axvline(d.mean(), color="#e8284a", lw=2, linestyle="--", label=f"μ={d.mean():.2f}")
                ax.set_xlabel(gx); ax.set_ylabel("Fréquence")
                ax.set_title(f"{gx} — Distribution", color="#0066cc", fontsize=12)
                ax.legend(fontsize=9)
            elif graph_type == "Box Plot Comparatif":
                groups = {g: sub[gx].dropna().values for g, sub in df.groupby(gy)}
                bp = ax.boxplot(list(groups.values()), patch_artist=True,
                                 medianprops={"color":"#e8284a","linewidth":2})
                for patch, color in zip(bp["boxes"], PALETTE):
                    patch.set_facecolor(color + "55")
                ax.set_xticklabels(list(groups.keys()), rotation=20, ha="right")
                ax.set_ylabel(gx)
                ax.set_title(f"{gx} par {gy}", color="#0066cc", fontsize=12)
            elif graph_type == "QQ-Plot (Normalité)":
                d = df_num[gx].dropna().values
                (osm, osr), (slope, intercept, r) = stats.probplot(d, dist="norm")
                ax.scatter(osm, osr, color="#0066cc", alpha=0.6, s=20)
                ax.plot(osm, slope*np.array(osm)+intercept, color="#e8284a", lw=2)
                ax.set_xlabel("Quantiles théoriques"); ax.set_ylabel("Quantiles observés")
                ax.set_title(f"QQ-Plot — {gx}  (r={r:.3f})", color="#0066cc", fontsize=12)
            elif graph_type == "Violin Plot":
                d = df_num[gx].dropna().values
                ax.violinplot([d], positions=[1], showmeans=True, showmedians=True)
                ax.set_xticks([1]); ax.set_xticklabels([gx])
                ax.set_title(f"Violin Plot — {gx}", color="#0066cc", fontsize=12)
            elif graph_type == "Histogrammes Multiples":
                plt.close(fig)
                ncols = min(3, len(df_num.columns))
                nrows = (len(df_num.columns) + ncols - 1) // ncols
                fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows), facecolor="white")
                axes_flat = np.array(axes2).flatten()
                for i, col in enumerate(df_num.columns):
                    axes_flat[i].set_facecolor("#f4f7ff")
                    axes_flat[i].hist(df_num[col].dropna(), bins=15,
                                       color=PALETTE[i%len(PALETTE)], alpha=0.7, edgecolor="white")
                    axes_flat[i].set_title(col, color=PALETTE[i%len(PALETTE)], fontsize=10, fontweight="bold")
                for j in range(i+1, len(axes_flat)): axes_flat[j].set_visible(False)
                plt.suptitle("Histogrammes — Toutes les variables numériques",
                              fontsize=12, fontweight="bold", color="#0066cc")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
                fig = None
            elif graph_type == "Scatter Matrix":
                plt.close(fig)
                cols_sc = df_num.columns[:min(4, len(df_num.columns))].tolist()
                fig3, axes3 = plt.subplots(len(cols_sc), len(cols_sc),
                                            figsize=(3*len(cols_sc), 3*len(cols_sc)), facecolor="white")
                for i, ci in enumerate(cols_sc):
                    for j, cj in enumerate(cols_sc):
                        aij = axes3[i][j]; aij.set_facecolor("#f4f7ff")
                        if i == j:
                            aij.hist(df_num[ci].dropna(), bins=12,
                                      color=PALETTE[i%len(PALETTE)], alpha=0.7, edgecolor="white")
                            aij.set_title(ci, fontsize=9, color=PALETTE[i%len(PALETTE)])
                        else:
                            xd = df_num[cj].dropna().values
                            yd = df_num[ci].dropna().values
                            n_pts = min(len(xd), len(yd))
                            aij.scatter(xd[:n_pts], yd[:n_pts],
                                         color=PALETTE[j%len(PALETTE)], alpha=0.4, s=10)
                            rv = np.corrcoef(xd[:n_pts], yd[:n_pts])[0,1]
                            aij.text(0.05, 0.95, f"r={rv:.2f}", transform=aij.transAxes,
                                      fontsize=8, va="top", color="#e8284a", fontweight="bold")
                        if j == 0: aij.set_ylabel(ci, fontsize=8)
                        if i == len(cols_sc)-1: aij.set_xlabel(cj, fontsize=8)
                        aij.tick_params(labelsize=7)
                plt.suptitle("Scatter Matrix", fontsize=12, fontweight="bold", color="#0066cc")
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)
                fig = None

            if fig is not None:
                plt.tight_layout()
                fig_to_st(fig)
'''

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

print('✅ Fichier app.py créé avec succès !')
print('📄 Taille :', len(app_code), 'caractères')
