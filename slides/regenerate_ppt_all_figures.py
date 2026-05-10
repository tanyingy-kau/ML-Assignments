"""Regenerate final PPT with all figures and a concise conclusion slide."""

from pathlib import Path
import re

from pptx import Presentation
from pptx.util import Inches, Pt


REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
OUT_PPT = REPO_ROOT / "slides" / "term_project_presentation_SYNGAS--group 6 Ying.pptx"
RESULTS_PATH = REPO_ROOT / "results_summary.txt"


FIG_ORDER = [
    "fig1_data_distributions.png",
    "fig2_feature_vs_target.png",
    "fig3_mlp_training_loss.png",
    "fig4_train_vs_test_performance.png",
    "fig5_scatter_LR_train_test.png",
    "fig6_scatter_RF_train_test.png",
    "fig7_scatter_MLP_train_test.png",
    "fig8_RF_feature_importance.png",
    "fig9_error_comparison.png",
    "fig10_robustness_boxplot.png",
    "fig11_robustness_line.png",
    "fig12_scaling_ablation_rmse.png",
    "fig13_runtime_ablation.png",
]

FIG_TITLES = {
    "fig1_data_distributions.png": "Figure 1 - Data Distributions",
    "fig2_feature_vs_target.png": "Figure 2 - Feature vs Target",
    "fig3_mlp_training_loss.png": "Figure 3 - MLP Training Loss Curve",
    "fig4_train_vs_test_performance.png": "Figure 4 - Train vs Test Performance",
    "fig5_scatter_LR_train_test.png": "Figure 5 - LR Actual vs Predicted",
    "fig6_scatter_RF_train_test.png": "Figure 6 - RF Actual vs Predicted",
    "fig7_scatter_MLP_train_test.png": "Figure 7 - MLP Actual vs Predicted",
    "fig8_RF_feature_importance.png": "Figure 8 - RF Feature Importance",
    "fig9_error_comparison.png": "Figure 9 - Error Comparison",
    "fig10_robustness_boxplot.png": "Figure 10 - Robustness Boxplot",
    "fig11_robustness_line.png": "Figure 11 - Robustness Across Seeds",
    "fig12_scaling_ablation_rmse.png": "Figure 12 - Scaling Ablation (4 Combos)",
    "fig13_runtime_ablation.png": "Figure 13 - Runtime Analysis",
}


def parse_results_summary(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    out = {}

    rf_match = re.search(r"Random Forest:\s*\n\s*Train:.*\n\s*Test:\s*RMSE=([0-9.]+),\s*MAE=([0-9.]+),\s*R²=([0-9.]+)", text)
    lr_match = re.search(r"Linear Regression:\s*\n\s*Train:.*\n\s*Test:\s*RMSE=([0-9.]+),\s*MAE=([0-9.]+),\s*R²=([0-9.]+)", text)
    mlp_match = re.search(r"MLP:\s*\n\s*Train:.*\n\s*Test:\s*RMSE=([0-9.]+),\s*MAE=([0-9.]+),\s*R²=([0-9.]+)", text)
    ttest_match = re.search(r"Paired t-test:\s*t=([0-9.\-]+),\s*p=([0-9.eE\-+]+)", text)
    wil_match = re.search(r"Wilcoxon:\s*W=([0-9.\-]+),\s*p=([0-9.eE\-+]+)", text)

    if rf_match:
        out["rf_rmse"], out["rf_mae"], out["rf_r2"] = rf_match.groups()
    if lr_match:
        out["lr_rmse"], out["lr_mae"], out["lr_r2"] = lr_match.groups()
    if mlp_match:
        out["mlp_rmse"], out["mlp_mae"], out["mlp_r2"] = mlp_match.groups()
    if ttest_match:
        out["ttest_t"], out["ttest_p"] = ttest_match.groups()
    if wil_match:
        out["wil_w"], out["wil_p"] = wil_match.groups()

    return out


def add_title_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Ethanol Prediction in Syngas Fermentation"
    subtitle = slide.placeholders[1]
    subtitle.text = (
        "CIS 732/830 Final Project\n"
        "Ying Tan\n"
        "Regenerated on May 10, 2026"
    )


def add_overview_slide(prs: Presentation, summary: dict):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Overview and Key Results"

    tf = slide.shapes.placeholders[1].text_frame
    tf.clear()

    lines = [
        "Dataset: 176 samples, 7 features, target = ethanol (mM)",
        "Models: Linear Regression, Random Forest, MLP",
        f"Fixed split test: RF RMSE={summary.get('rf_rmse', '?')}, R2={summary.get('rf_r2', '?')} (best)",
        f"Fixed split test: MLP RMSE={summary.get('mlp_rmse', '?')}, R2={summary.get('mlp_r2', '?')}",
        f"Fixed split test: LR RMSE={summary.get('lr_rmse', '?')}, R2={summary.get('lr_r2', '?')}",
        f"Statistical significance: paired t-test p={summary.get('ttest_p', '?')}, Wilcoxon p={summary.get('wil_p', '?')}",
        "This deck includes ALL generated figures (Fig1-Fig13).",
    ]

    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.level = 0
        p.font.size = Pt(18 if i == 0 else 16)


def add_figure_slide(prs: Presentation, image_path: Path, title: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    title_box = slide.shapes.add_textbox(Inches(0.4), Inches(0.15), Inches(12.5), Inches(0.5))
    tf = title_box.text_frame
    tf.text = title
    tf.paragraphs[0].font.size = Pt(28)
    tf.paragraphs[0].font.bold = True

    slide.shapes.add_picture(str(image_path), Inches(0.4), Inches(0.85), width=Inches(12.5), height=Inches(6.2))



def add_conclusion_slide(prs: Presentation, summary: dict):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Conclusion"

    tf = slide.shapes.placeholders[1].text_frame
    tf.clear()

    lines = [
        "Random Forest is the best-performing model for this dataset.",
        f"Best fixed-split performance: RF RMSE={summary.get('rf_rmse', '?')}, R2={summary.get('rf_r2', '?')}.",
        "Robustness analysis across 30 random splits supports model stability.",
        "Scaling ablation confirms RF should remain unscaled in this setup.",
        "MLP requires careful optimization and shows sensitivity to settings.",
        f"Hypothesis tests are significant (t-test p={summary.get('ttest_p', '?')}, Wilcoxon p={summary.get('wil_p', '?')}).",
        "Recommendation: deploy RF as the primary soft-sensor model; extend with online updates in future work.",
    ]

    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.level = 0
        p.font.size = Pt(18 if i == 0 else 16)


def main():
    missing = [f for f in FIG_ORDER if not (FIG_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing figure files: {missing}")

    summary = parse_results_summary(RESULTS_PATH)

    prs = Presentation()
    add_title_slide(prs)
    add_overview_slide(prs, summary)

    for fig_name in FIG_ORDER:
        add_figure_slide(prs, FIG_DIR / fig_name, FIG_TITLES[fig_name])

    add_conclusion_slide(prs, summary)
    prs.save(OUT_PPT)
    print(f"Saved regenerated PPT: {OUT_PPT}")
    print(f"Slide count: {len(prs.slides)}")


if __name__ == "__main__":
    main()
