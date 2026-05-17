"""
CRISPR scRNA-seq Differential Expression Analyzer
---------------------------------------------------
A Streamlit app for comparing gene expression between
CRISPR-edited and unedited single cells.

Requirements:
    pip install streamlit pandas numpy scipy plotly anthropic

Run:
    streamlit run crispr_scrna_analyzer.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import io
import anthropic

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CRISPR scRNA-seq Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px; padding: 6px 18px; }
    .metric-label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
    .gene-tag { font-family: monospace; font-size: 12px; color: #1a73e8; }
</style>
""", unsafe_allow_html=True)


# ─── Statistics ──────────────────────────────────────────────────────────────

def welch_ttest(a: np.ndarray, b: np.ndarray) -> float:
    """Welch's t-test, returns p-value."""
    if len(a) < 2 or len(b) < 2:
        return 1.0
    _, p = stats.ttest_ind(a, b, equal_var=False)
    return float(np.clip(p, 1e-300, 1.0))


def bh_correction(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    padj = np.ones(n)
    running_min = 1.0
    for k in range(n - 1, -1, -1):
        running_min = min(running_min, pvalues[order[k]] * n / (k + 1))
        padj[order[k]] = min(running_min, 1.0)
    return padj


# ─── Sample Data ─────────────────────────────────────────────────────────────

SAMPLE_GENES = [
    "TP53", "BRCA1", "EGFR", "MYC", "KRAS", "PTEN", "BCL2", "VEGFA", "CDK4", "RB1",
    "AKT1", "MTOR", "STAT3", "JAK2", "NOTCH1", "WNT5A", "FOXP3", "IRF4", "GATA3", "SOX2",
    "OCT4", "NANOG", "ALDH1A1", "CD44", "CD133", "ITGA6", "VIM", "CDH1", "ZEB1", "SNAI1",
    "TGFB1", "IL6", "TNF", "IFNG", "CXCL8", "CCL2", "HIF1A", "LDHA", "PKM2", "G6PD",
    "IDH1", "FTO", "DNMT3A", "TET2", "HDAC1", "EZH2", "BRD4", "MED12", "CTCF", "RAD51",
]
UP_IN_EDITED = {"TP53", "BRCA1", "PTEN", "RB1", "FOXP3", "CDH1", "IRF4", "RAD51"}
DOWN_IN_EDITED = {"MYC", "KRAS", "BCL2", "EGFR", "AKT1", "MTOR", "STAT3", "HIF1A", "EZH2", "BRD4"}


@st.cache_data
def generate_sample_data(n_edited: int = 100, n_unedited: int = 100, seed: int = 42):
    """Generate simulated scRNA-seq data with realistic CRISPR editing effects."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_edited + n_unedited):
        edited = i < n_edited
        row = {"cell_id": f"cell_{i+1}", "condition": "edited" if edited else "unedited"}
        for g in SAMPLE_GENES:
            base = 2 + rng.random() * 3
            if edited and g in UP_IN_EDITED:
                val = base * (2.8 + rng.random() * 1.5)
            elif edited and g in DOWN_IN_EDITED:
                val = base * (0.12 + rng.random() * 0.2)
            else:
                val = base + (rng.random() - 0.5) * 2
            row[g] = max(0.0, round(val, 4))
        rows.append(row)
    return pd.DataFrame(rows)


# ─── DE Analysis ─────────────────────────────────────────────────────────────

@st.cache_data
def run_de_analysis(df_hash: str, _df: pd.DataFrame):
    """
    Run differential expression analysis.
    Compares 'edited' vs 'unedited' cells for each gene.
    Returns a DataFrame sorted by adjusted p-value.
    """
    genes = [c for c in _df.columns if c not in ("cell_id", "condition")]
    edited = _df[_df["condition"] == "edited"]
    unedited = _df[_df["condition"] == "unedited"]

    results = []
    for gene in genes:
        e_vals = edited[gene].values
        u_vals = unedited[gene].values
        mean_e = e_vals.mean()
        mean_u = u_vals.mean()
        log2fc = np.log2((mean_e + 0.1) / (mean_u + 0.1))
        pval = welch_ttest(e_vals, u_vals)
        results.append({
            "gene": gene,
            "log2FC": round(log2fc, 4),
            "pval": pval,
            "mean_edited": round(mean_e, 4),
            "mean_unedited": round(mean_u, 4),
        })

    res_df = pd.DataFrame(results)
    res_df["padj"] = bh_correction(res_df["pval"].values)
    res_df["neg_log10_padj"] = -np.log10(np.clip(res_df["padj"], 1e-300, 1))
    res_df["significant"] = (res_df["padj"] < 0.05) & (res_df["log2FC"].abs() > 1)
    res_df["direction"] = res_df["log2FC"].apply(lambda x: "Up" if x > 0 else "Down")
    res_df["status"] = res_df.apply(
        lambda r: ("↑ Up" if r["direction"] == "Up" else "↓ Down") if r["significant"] else "NS",
        axis=1
    )
    return res_df.sort_values("padj").reset_index(drop=True)


# ─── Plots ───────────────────────────────────────────────────────────────────

def make_volcano(de_df: pd.DataFrame) -> go.Figure:
    color_map = {"Up": "#639922", "Down": "#D85A30", "NS": "#B4B2A9"}
    de_df = de_df.copy()
    de_df["color_group"] = de_df.apply(
        lambda r: r["direction"] if r["significant"] else "NS", axis=1
    )
    fig = go.Figure()
    for group, color in color_map.items():
        sub = de_df[de_df["color_group"] == group]
        fig.add_trace(go.Scatter(
            x=sub["log2FC"], y=sub["neg_log10_padj"],
            mode="markers",
            name={"Up": "Upregulated", "Down": "Downregulated", "NS": "Not significant"}[group],
            marker=dict(color=color, size=7 if group != "NS" else 5, opacity=0.75),
            text=sub["gene"],
            hovertemplate="<b>%{text}</b><br>log₂FC: %{x:.3f}<br>−log₁₀(p.adj): %{y:.3f}<extra></extra>",
        ))
    max_y = de_df["neg_log10_padj"].max() * 1.05
    p_line = -np.log10(0.05)
    fig.add_hline(y=p_line, line_dash="dash", line_color="#aaa", line_width=1)
    fig.add_vline(x=1, line_dash="dash", line_color="#aaa", line_width=1)
    fig.add_vline(x=-1, line_dash="dash", line_color="#aaa", line_width=1)
    fig.update_layout(
        xaxis_title="log₂(Fold Change)", yaxis_title="−log₁₀(adjusted p-value)",
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=50, r=20), height=480,
        font=dict(size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=True, zerolinecolor="#ddd")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


def make_top_genes_chart(de_df: pd.DataFrame, n: int = 20) -> go.Figure:
    sig = de_df[de_df["significant"]].copy()
    sig = sig.reindex(sig["log2FC"].abs().sort_values(ascending=False).index).head(n)
    sig = sig.sort_values("log2FC")
    colors = ["#639922" if d == "Up" else "#D85A30" for d in sig["direction"]]
    fig = go.Figure(go.Bar(
        x=sig["log2FC"], y=sig["gene"],
        orientation="h",
        marker_color=colors,
        text=sig["log2FC"].round(2),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>log₂FC: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="log₂(Fold Change)", yaxis_title="",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=40, l=20, r=60),
        height=max(350, len(sig) * 26 + 80),
        font=dict(size=12),
        yaxis=dict(tickfont=dict(family="monospace", size=11, color="#185FA5")),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=True, zerolinecolor="#ddd")
    return fig


def make_expression_violin(de_df: pd.DataFrame, df: pd.DataFrame, gene: str) -> go.Figure:
    fig = go.Figure()
    for condition, color in [("edited", "#185FA5"), ("unedited", "#D85A30")]:
        vals = df[df["condition"] == condition][gene].values
        fig.add_trace(go.Violin(
            y=vals, name=condition.capitalize(), box_visible=True,
            meanline_visible=True, fillcolor=color, opacity=0.6,
            line_color=color, points="outliers",
        ))
    fc = de_df[de_df["gene"] == gene]["log2FC"].values[0]
    padj = de_df[de_df["gene"] == gene]["padj"].values[0]
    fig.update_layout(
        title=f"{gene}  (log₂FC = {fc:.3f}, p.adj = {padj:.2e})",
        yaxis_title="Normalized expression",
        plot_bgcolor="white", paper_bgcolor="white",
        height=380, margin=dict(t=50, b=40, l=50, r=20),
        legend=dict(orientation="h", y=1.05),
    )
    return fig


# ─── AI Interpretation ───────────────────────────────────────────────────────

def get_ai_interpretation(de_df: pd.DataFrame, api_key: str) -> str:
    sig_genes = de_df[de_df["significant"]].head(15)
    gene_summary = "\n".join([
        f"{r['gene']}: log2FC={r['log2FC']:.2f}, padj={r['padj']:.2e}, "
        f"{'upregulated' if r['direction']=='Up' else 'downregulated'} in edited cells"
        for _, r in sig_genes.iterrows()
    ])
    prompt = f"""You are an expert bioinformatician analyzing single-cell RNA-seq data from a CRISPR experiment.

Top significant differentially expressed genes (edited vs unedited cells):
{gene_summary}

Provide a concise biological interpretation covering:
1. What pathways or biological processes are likely affected by the CRISPR edit?
2. What do the up/downregulated genes suggest about the cellular phenotype?
3. Any potential off-target effects suggested by the expression changes?
4. Recommended follow-up experiments to validate these findings.

Be specific, reference the gene names provided, and keep to approximately 350 words."""

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ─── UI ──────────────────────────────────────────────────────────────────────

def main():
    st.title("🧬 CRISPR scRNA-seq Analyzer")
    st.caption("Post-editing differential expression · Welch t-test · Benjamini-Hochberg FDR")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Data Input")
        data_source = st.radio("Data source", ["Use sample data", "Upload CSV"])

        df = None
        if data_source == "Use sample data":
            n_edited = st.slider("Edited cells", 50, 300, 100, 10)
            n_unedited = st.slider("Unedited cells", 50, 300, 100, 10)
            seed = st.number_input("Random seed", 0, 9999, 42)
            if st.button("Generate sample data", type="primary"):
                st.session_state["df"] = generate_sample_data(n_edited, n_unedited, int(seed))
                st.session_state["de"] = None
                st.success("Sample data generated!")
        else:
            st.markdown("""
**CSV format:**
- Rows = cells
- Columns = genes (numeric expression values)
- Required column: `condition` (values: `edited` / `unedited`)
- Optional column: `cell_id`
""")
            uploaded = st.file_uploader("Upload expression matrix", type="csv")
            if uploaded:
                df_up = pd.read_csv(uploaded)
                if "condition" not in df_up.columns:
                    st.error("CSV must have a 'condition' column with values 'edited' or 'unedited'.")
                else:
                    st.session_state["df"] = df_up
                    st.session_state["de"] = None
                    st.success(f"Loaded {len(df_up)} cells")

        st.divider()
        st.header("Analysis Parameters")
        padj_thresh = st.number_input("FDR threshold (p.adj)", 0.001, 0.1, 0.05, 0.005, format="%.3f")
        fc_thresh = st.number_input("|log₂FC| threshold", 0.5, 3.0, 1.0, 0.25)

        st.divider()
        st.header("AI Interpretation")
        api_key = st.text_input("Anthropic API key", type="password",
                                help="Optional. Needed for AI interpretation tab.")

    # ── Main Panel ────────────────────────────────────────────────────────────
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.info("👈 Generate sample data or upload a CSV in the sidebar to get started.")
        with st.expander("Expected CSV format"):
            st.markdown("""
| cell_id | condition | TP53 | MYC | EGFR | ... |
|---------|-----------|------|-----|------|-----|
| cell_1  | edited    | 8.2  | 0.4 | 1.1  | ... |
| cell_2  | unedited  | 2.1  | 5.6 | 4.3  | ... |
""")
        return

    df = st.session_state["df"]
    genes = [c for c in df.columns if c not in ("cell_id", "condition")]
    n_edited = (df["condition"] == "edited").sum()
    n_unedited = (df["condition"] == "unedited").sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total cells", len(df))
    col2.metric("Edited cells", n_edited)
    col3.metric("Unedited cells", n_unedited)

    if st.button("▶ Run differential expression analysis", type="primary"):
        with st.spinner("Running Welch t-test + Benjamini-Hochberg correction..."):
            df_hash = str(df.shape) + str(df.columns.tolist())
            de_df = run_de_analysis(df_hash, df)
            # Apply user thresholds
            de_df["significant"] = (de_df["padj"] < padj_thresh) & (de_df["log2FC"].abs() > fc_thresh)
            de_df["status"] = de_df.apply(
                lambda r: ("↑ Up" if r["direction"] == "Up" else "↓ Down") if r["significant"] else "NS",
                axis=1
            )
            st.session_state["de"] = de_df
        st.success("Analysis complete!")

    if "de" not in st.session_state or st.session_state["de"] is None:
        return

    de_df = st.session_state["de"]
    sig = de_df[de_df["significant"]]
    up = sig[sig["direction"] == "Up"]
    dn = sig[sig["direction"] == "Down"]

    # ── Results Tabs ──────────────────────────────────────────────────────────
    tab_summary, tab_volcano, tab_genes, tab_violin, tab_ai = st.tabs([
        "📋 Summary", "🌋 Volcano plot", "📊 Top genes", "🎻 Gene explorer", "🤖 AI interpretation"
    ])

    # Summary Tab
    with tab_summary:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Genes tested", len(de_df))
        m2.metric("Significant", len(sig), help=f"padj < {padj_thresh}, |log₂FC| > {fc_thresh}")
        m3.metric("Upregulated", len(up))
        m4.metric("Downregulated", len(dn))

        st.subheader("Top differentially expressed genes")
        display_df = de_df.head(20)[["gene", "log2FC", "padj", "mean_edited", "mean_unedited", "status"]].copy()
        display_df.columns = ["Gene", "log₂FC", "p.adj", "Mean (edited)", "Mean (unedited)", "Status"]
        display_df["p.adj"] = display_df["p.adj"].apply(lambda x: f"{x:.2e}")
        display_df["log₂FC"] = display_df["log₂FC"].round(3)

        def color_status(val):
            if "Up" in str(val): return "background-color: #EAF3DE; color: #27500A"
            if "Down" in str(val): return "background-color: #FAECE7; color: #712B13"
            return "color: #888"

        st.dataframe(
            display_df.style.applymap(color_status, subset=["Status"]),
            use_container_width=True, height=480,
        )

        csv_buf = io.StringIO()
        de_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇ Download full results (CSV)",
            csv_buf.getvalue(), "de_results.csv", "text/csv",
        )

    # Volcano Tab
    with tab_volcano:
        st.plotly_chart(make_volcano(de_df), use_container_width=True)
        st.caption(f"Dashed lines: |log₂FC| = {fc_thresh}, p.adj = {padj_thresh}  ·  "
                   f"{len(up)} upregulated (green)  ·  {len(dn)} downregulated (coral)  ·  "
                   f"{len(de_df) - len(sig)} not significant (gray)")

    # Top Genes Tab
    with tab_genes:
        n_show = st.slider("Number of genes to show", 5, 30, 20)
        if len(sig) == 0:
            st.warning("No significant genes found. Try relaxing the thresholds in the sidebar.")
        else:
            st.plotly_chart(make_top_genes_chart(de_df, n_show), use_container_width=True)

    # Gene Explorer Tab
    with tab_violin:
        st.subheader("Per-gene expression explorer")
        gene_options = de_df["gene"].tolist()
        sel_gene = st.selectbox(
            "Select a gene",
            gene_options,
            format_func=lambda g: f"{g}  (log₂FC={de_df[de_df['gene']==g]['log2FC'].values[0]:.2f})"
        )
        if sel_gene and sel_gene in df.columns:
            st.plotly_chart(make_expression_violin(de_df, df, sel_gene), use_container_width=True)

    # AI Interpretation Tab
    with tab_ai:
        st.subheader("AI biological interpretation")
        st.markdown(
            "Claude analyzes your significant DE genes and provides a pathway-level "
            "biological interpretation of the CRISPR editing effect."
        )
        if not api_key:
            st.info("Enter your Anthropic API key in the sidebar to enable this feature.")
        elif len(sig) == 0:
            st.warning("No significant genes to interpret. Relax your thresholds and re-run.")
        else:
            if st.button("Generate interpretation", type="primary"):
                with st.spinner("Analyzing gene signatures with Claude..."):
                    try:
                        interpretation = get_ai_interpretation(de_df, api_key)
                        st.session_state["ai_output"] = interpretation
                    except Exception as e:
                        st.error(f"API error: {e}")

            if "ai_output" in st.session_state:
                st.markdown(st.session_state["ai_output"])
                st.download_button(
                    "⬇ Download interpretation",
                    st.session_state["ai_output"],
                    "ai_interpretation.txt", "text/plain"
                )


if __name__ == "__main__":
    main()
