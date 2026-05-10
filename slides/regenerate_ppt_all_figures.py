"""Regenerate final PPT by reusing previous template and appending all figures with explanations."""

from pathlib import Path
import subprocess
import tempfile

from pptx import Presentation
from pptx.util import Inches, Pt


REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
OUT_PPT = REPO_ROOT / "slides" / "term_project_presentation_SYNGAS--group 6 Ying.pptx"
TEMPLATE_COMMIT = "15540ee"
TEMPLATE_REL_PATH = "slides/term_project_presentation_SYNGAS--group 6 Ying.pptx"


FIG_DATA = [
    (
        "fig1_data_distributions.png",
        "Figure 1: Data Distributions",
        [
            "Shows feature/target spread and skewness before model fitting.",
            "Confirms heterogeneous scales across variables, motivating scaling analysis.",
            "No obvious missing-data artifact in the plotted distributions.",
        ],
    ),
    (
        "fig2_feature_vs_target.png",
        "Figure 2: Feature vs Target",
        [
            "Visual check for nonlinear relations between each input and ethanol.",
            "Supports using nonlinear models (RF/MLP) in addition to LR baseline.",
            "Also helps identify potential interaction effects to investigate later.",
        ],
    ),
    (
        "fig3_mlp_training_loss.png",
        "Figure 3: MLP Training Loss Curve",
        [
            "Tracks optimization behavior across epochs for MLP.",
            "Demonstrates convergence trend and training stability characteristics.",
            "Provides evidence for optimizer sensitivity discussed in report.",
        ],
    ),
    (
        "fig4_train_vs_test_performance.png",
        "Figure 4: Train vs Test Performance",
        [
            "Compares generalization gap for LR, RF, and MLP.",
            "RF shows strongest test performance in this dataset.",
            "Used to detect overfitting/underfitting patterns across models.",
        ],
    ),
    (
        "fig5_scatter_LR_train_test.png",
        "Figure 5: LR Actual vs Predicted",
        [
            "Baseline linear fit on train/test sets.",
            "Points deviate from diagonal more than RF/MLP, indicating underfit.",
            "Serves as reference model for hypothesis testing.",
        ],
    ),
    (
        "fig6_scatter_RF_train_test.png",
        "Figure 6: RF Actual vs Predicted",
        [
            "RF predictions cluster near diagonal, especially on test data.",
            "Supports highest R2 and lowest RMSE among compared models.",
            "Main evidence for selecting RF as deployment candidate.",
        ],
    ),
    (
        "fig7_scatter_MLP_train_test.png",
        "Figure 7: MLP Actual vs Predicted",
        [
            "MLP captures nonlinear trend but is more variable than RF.",
            "Performance is competitive yet sensitive to preprocessing/optimization.",
            "Used together with ablation to discuss scaling sensitivity.",
        ],
    ),
    (
        "fig8_RF_feature_importance.png",
        "Figure 8: RF Feature Importance",
        [
            "Ranks relative contribution of each feature in RF splits.",
            "Improves interpretability for process-monitoring decisions.",
            "Guides future feature-engineering priorities.",
        ],
    ),
    (
        "fig9_error_comparison.png",
        "Figure 9: Error Comparison",
        [
            "Compares error distributions across models on the same test data.",
            "Paired view supports sample-level statistical tests (t-test/Wilcoxon).",
            "Demonstrates RF error is typically below LR error.",
        ],
    ),
    (
        "fig10_robustness_boxplot.png",
        "Figure 10: Robustness Boxplot",
        [
            "Summarizes RMSE distribution over 30 random splits.",
            "Measures stability beyond one fixed train/test partition.",
            "RF median and spread support robust generalization claim.",
        ],
    ),
    (
        "fig11_robustness_line.png",
        "Figure 11: Robustness Across Seeds",
        [
            "Seed-wise trajectories reveal variance behavior by model.",
            "Shows RF remains consistently strong across splits.",
            "Complements boxplot with temporal/seed-level visibility.",
        ],
    ),
    (
        "fig12_scaling_ablation_rmse.png",
        "Figure 12: Scaling Ablation (4 Combinations)",
        [
            "Explicitly tests all required scaler on/off combinations for RF/MLP.",
            "RF performs best unscaled in this setup; scaled RF is slightly worse.",
            "Documents preprocessing impact with direct experimental evidence.",
        ],
    ),
    (
        "fig13_runtime_ablation.png",
        "Figure 13: Runtime Analysis",
        [
            "Reports wall-clock training and inference costs.",
            "Both models are fast at inference for this dataset size.",
            "Supports practical deployment discussion in report/presentation.",
        ],
    ),
]


def export_template_from_git() -> Path:
    tmp = Path(tempfile.gettempdir()) / "ppt_template_from_git.pptx"
    cmd = [
        "git",
        "show",
        f"{TEMPLATE_COMMIT}:{TEMPLATE_REL_PATH}",
    ]
    data = subprocess.check_output(cmd, cwd=REPO_ROOT)
    tmp.write_bytes(data)
    return tmp


def add_section_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Appendix: All Figures with Explanations"
    tf = slide.shapes.placeholders[1].text_frame
    tf.clear()
    bullets = [
        "The next slides include Figure 1 through Figure 13.",
        "Each figure is accompanied by concise interpretation points.",
        "These slides are aligned with report evidence and grading feedback.",
    ]
    for i, line in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.level = 0
        p.font.size = Pt(20 if i == 0 else 18)


def add_figure_explain_slide(prs: Presentation, fig_name: str, title: str, bullets: list[str]):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    title_box = slide.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(12.7), Inches(0.55))
    tf_t = title_box.text_frame
    tf_t.text = title
    tf_t.paragraphs[0].font.size = Pt(28)
    tf_t.paragraphs[0].font.bold = True

    img = FIG_DIR / fig_name
    slide.shapes.add_picture(str(img), Inches(0.3), Inches(0.85), width=Inches(8.5), height=Inches(5.7))

    note_box = slide.shapes.add_textbox(Inches(8.95), Inches(0.85), Inches(3.95), Inches(5.8))
    tf_n = note_box.text_frame
    tf_n.word_wrap = True
    tf_n.clear()
    for i, line in enumerate(bullets):
        p = tf_n.paragraphs[0] if i == 0 else tf_n.add_paragraph()
        p.text = line
        p.level = 0
        p.font.size = Pt(15)


def main():
    missing = [n for n, _, _ in FIG_DATA if not (FIG_DIR / n).exists()]
    if missing:
        raise FileNotFoundError(f"Missing figure files: {missing}")

    template_path = export_template_from_git()
    prs = Presentation(str(template_path))

    add_section_slide(prs)
    for fig_name, title, bullets in FIG_DATA:
        add_figure_explain_slide(prs, fig_name, title, bullets)

    prs.save(OUT_PPT)
    print(f"Saved regenerated PPT: {OUT_PPT}")
    print(f"Slide count: {len(prs.slides)}")


if __name__ == "__main__":
    main()
