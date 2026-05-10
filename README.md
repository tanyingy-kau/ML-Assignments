# Syngas Fermentation Ethanol Prediction - Machine Learning Project

## Project Overview

This project applies machine learning techniques to predict ethanol production in syngas fermentation processes. The work involves comparative analysis of three models: Linear Regression, Random Forest, and Multi-Layer Perceptron (MLP).

**Track:** Track 5 (ML/Probabilistic)  
**Pillar:** Applied AI Systems (OA)  
**Course:** CIS 732/830 - Machine Learning, Kansas State University (Spring 2026)

## Key Results

| Model | Test R² | Test RMSE | 30-fold RMSE (mean±std) |
|-------|---------|-----------|------------------------|
| Linear Regression (baseline) | 0.6340 | 12.18 mM | 14.40 ± 1.81 mM |
| **Random Forest (best)** | **0.9529** | **4.37 mM** | **9.48 ± 1.83 mM** |
| MLP Neural Network | 0.9245 | 5.53 mM | 9.73 ± 2.64 mM |

- **Statistical Significance:** Paired t-test (p = 2.79×10⁻⁶), Wilcoxon (p = 3.02×10⁻⁶)

## Directory Structure

```
ML-Assignments/
├── figures/
│   ├── fig1_data_distributions.png       # Feature + target histograms
│   ├── fig2_feature_vs_target.png        # Feature vs ethanol scatter plots
│   ├── fig3_mlp_training_loss.png        # MLP training loss curve (500 epochs)
│   ├── fig4_train_vs_test_performance.png # Bar chart: Train vs Test R²/RMSE all models
│   ├── fig5_scatter_LR_train_test.png    # LR actual vs predicted (train + test)
│   ├── fig6_scatter_RF_train_test.png    # RF actual vs predicted (train + test)
│   ├── fig7_scatter_MLP_train_test.png   # MLP actual vs predicted (train + test)
│   ├── fig8_RF_feature_importance.png    # Random Forest feature importances
│   ├── fig9_error_comparison.png         # Absolute error boxplot + LR vs RF scatter
│   ├── fig10_robustness_boxplot.png      # RMSE distribution over 30 seeds
│   └── fig11_robustness_line.png         # RMSE per seed (line plot, all 3 models)
│   ├── fig12_scaling_ablation_rmse.png   # 4-combo scaling ablation (RF vs MLP RMSE)
│   └── fig13_runtime_ablation.png        # Wall-clock runtime (train + inference)
├── report/
│   ├── main.tex                          # LaTeX source (IEEEtran double-column)
│   ├── references.bib                    # BibTeX bibliography
│   └── syngas fermentation report-Ying Tan.pdf  # Final 7-page IEEE report
├── scripts/
│   ├── syngas_analysis.py               # MAIN SCRIPT: full pipeline + 13 figures
│   ├── scaling_ablation_results.csv     # 4-combo scaling ablation results table
│   └── stats_test.py                    # Legacy: metrics + statistical tests only
├── slides/
│   └── term_project_presentation_SYNGAS--group 6 Ying.pptx  # 13-slide presentation
├── sources/
│   └── code_provenance.txt              # Code provenance and AI/non-AI attribution
├── prompts/
│   └── coding_prompt_log.txt            # Source prompts/specification log
├── ML_regular.ipynb                      # Baseline notebook (PS8 interim report)
├── requirements.txt                      # Python package dependencies
├── results_summary.txt                   # Numerical results: metrics + statistical tests
└── README.md                             # This file
```

## Setup & Environment

### Requirements

- Python 3.8+
- Install all dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Data

The dataset is loaded automatically from the public [SyngasMachineLearning GitHub repository](https://github.com/garrettroell/SyngasMachineLearning) (raw CSV via HTTP). No manual download is required.

- **Size:** 176 samples
- **Features (7):** time, N₂, CO, CO₂, H₂, flow rate, composition setting
- **Target:** Ethanol concentration (mM), range 0–109

## Running the Code

### Full Analysis (recommended)
```bash
cd /path/to/ML-Assignments
python scripts/syngas_analysis.py
```

**Output (all saved automatically):**
- `figures/fig1_data_distributions.png` — Feature + target histograms
- `figures/fig2_feature_vs_target.png` — Feature vs ethanol scatter plots
- `figures/fig3_mlp_training_loss.png` — MLP loss curve (500 epochs)
- `figures/fig4_train_vs_test_performance.png` — Train/test R² and RMSE bar chart
- `figures/fig5_scatter_LR_train_test.png` — LR actual vs predicted (train + test)
- `figures/fig6_scatter_RF_train_test.png` — RF actual vs predicted (train + test)
- `figures/fig7_scatter_MLP_train_test.png` — MLP actual vs predicted (train + test)
- `figures/fig8_RF_feature_importance.png` — Feature importance bar chart
- `figures/fig9_error_comparison.png` — Absolute error boxplot
- `figures/fig10_robustness_boxplot.png` — 30-fold RMSE boxplot
- `figures/fig11_robustness_line.png` — 30-fold RMSE line plot
- `figures/fig12_scaling_ablation_rmse.png` — 4-combo scaling ablation RMSE chart
- `figures/fig13_runtime_ablation.png` — wall-clock train/inference runtime chart
- `results_summary.txt` — All numerical metrics and statistical test results
- `scripts/scaling_ablation_results.csv` — detailed ablation table for all 4 combinations

### Scaling Ablation Combinations

The script explicitly evaluates these four combinations on the same fixed split:
1. MLP scaled + RF unscaled
2. MLP scaled + RF scaled
3. MLP unscaled + RF unscaled
4. MLP unscaled + RF scaled

The results are saved in `scripts/scaling_ablation_results.csv` and visualized in `figures/fig12_scaling_ablation_rmse.png`.

### Runtime Reporting

Wall-clock runtime is measured for each model and ablation combination:
- Training time (seconds)
- Inference latency (milliseconds per sample)

Runtime plots are saved in `figures/fig13_runtime_ablation.png`.

### Compile Report (LaTeX)
```bash
cd report/
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

**Output:** `syngas fermentation report-Ying Tan.pdf` (7 pages, IEEE double-column format)

## Methodology

### Models
- **Linear Regression** — baseline, interpretable
- **Random Forest** — `n_estimators=200`, `random_state=42`; captures non-linearity
- **MLP Regressor** — `hidden_layer_sizes=(64,32)`, `max_iter=2000`; input features scaled with `StandardScaler`

### Evaluation Protocol
- **Fixed split:** 80/20 train-test (`random_state=42`), reproducible
- **30-fold robustness:** seeds 0–29, each with independent 80/20 split

### Statistical Hypothesis Testing
- **H₀:** LR and RF have equivalent prediction error on the test distribution
- **H₁:** RF has significantly lower error than LR
- **Tests:** Paired t-test + Wilcoxon signed-rank (α = 0.05)
- **Result:** Reject H₀ — both tests p ≪ 0.05 (p = 2.79×10⁻⁶)

## Compliance

- **Report Format:** IEEE/ACM double-column proceedings format ✓
- **Report Length:** 6-8 pages (7 pages) ✓
- **Report Sections:** Abstract, Introduction, Background/Related Work, Methodology, Experiments & Results (with statistical hypotheses), Conclusion ✓
- **Codebase:** requirements.txt, reproducible main script, comprehensive README ✓
- **Presentation Q&A:** Three standard reflection questions answered (Slides 11–13) ✓

## GenAI Disclosure

Per Canvas policy, a GenAI Audit Appendix is included in the final report documenting AI assistance scope, original student inputs, and reproducibility verification.

Additional repository artifacts for transparency:
- `sources/code_provenance.txt`
- `prompts/coding_prompt_log.txt`

## Authors

- **Student:** Ying Tan
- **Course:** Machine Learning (CIS 732/830), Kansas State University, Spring 2026

## License

Submitted as course work for CIS 732/830 at Kansas State University, subject to institutional academic integrity policies.

---
**Last Updated:** May 10, 2026
