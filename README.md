# Syngas Fermentation Ethanol Prediction - Machine Learning Project

## Project Overview

This project applies machine learning techniques to predict ethanol production in syngas fermentation processes. The work involves comparative analysis of three models: Linear Regression, Random Forest, and Multi-Layer Perceptron (MLP).

**Track:** Track 5 (ML/Probabilistic)  
**Pillar:** Applied AI Systems (OA)  
**Course:** CIS 732/830 - Machine Learning, Kansas State University (Spring 2026)

## Key Results

- **Best Model:** Random Forest (R² = 0.9529, RMSE = 4.37 mM)
- **Baseline:** Linear Regression (R² = 0.6408, RMSE = 12.65 mM)
- **Performance Improvement:** 65% reduction in RMSE
- **Statistical Significance:** Paired t-test (p = 2.79×10⁻⁶), Wilcoxon (p = 3.02×10⁻⁶)
- **Robustness:** Validated across 30 random seeds (RF: mean RMSE = 5.51±1.08 mM)

## Directory Structure

```
term_project_deliverables/
├── report/
│   ├── main.tex                          # Main LaTeX source document
│   ├── references.bib                    # BibTeX bibliography file
│   └── syngas fermentation report-Ying Tan.pdf  # Final 7-page IEEE-formatted report
├── slides/
│   ├── term_project_presentation_SYNGAS_FINAL.pptx  # Final 10-slide presentation deck
│   ├── generate_syngas_ppt_final.py      # Script to generate presentation with embedded charts
│   └── speaker_script_SYNGAS.txt         # Presentation speaker notes
├── scripts/
│   ├── stats_test.py                     # Model training, evaluation, and statistical testing
│   └── stat_tests_output.txt             # Logged output from statistical tests
└── README.md                             # This file
```

## Setup & Environment

### Requirements

- Python 3.8+
- Required packages (see `requirements.txt` or install via):
  ```bash
  pip install pandas numpy scikit-learn scipy matplotlib seaborn
  ```

### Data

The dataset is sourced from the [SyngasMachineLearning GitHub repository](https://github.com/thahnen/SyngasMachineLearning):
- **Size:** 176 samples
- **Features:** 7 (Temperature, Pressure, Feed Composition, etc.)
- **Target:** Ethanol production rate (mM)

**Note:** If the dataset is not in the `data/` directory, download it from the link above and place it in:
```
data/Syngas_Fermentation_ML_Data.csv
```

## Running the Code

### 1. Train Models and Generate Statistics
```bash
cd scripts/
python stats_test.py
```

**Output:**
- Model performance metrics (RMSE, MAE, R²)
- Paired t-test results (H₀: RF = LR)
- Wilcoxon signed-rank test results
- Robustness analysis across 30 seeds (0-29)
- PNG charts saved to `slides/` directory

### 2. Generate Presentation
```bash
cd slides/
python generate_syngas_ppt_final.py
```

**Output:**
- `term_project_presentation_SYNGAS_FINAL.pptx` with embedded charts

### 3. Compile Report (LaTeX)
```bash
cd report/
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

**Output:**
- `syngas fermentation report-Ying Tan.pdf` (7 pages, IEEE double-column format)

## Evaluation & Methodology

### Hypothesis Testing
- **H₀:** Random Forest performance = Linear Regression performance
- **H₁:** Random Forest performance ≠ Linear Regression performance
- **Result:** Reject H₀ (strong statistical evidence of RF superiority)

### Cross-Validation
- **Method:** 30-fold stratified cross-validation
- **Random Seeds:** 0-29 for robustness validation
- **Metric:** Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R² Score

### Key Findings
1. Random Forest significantly outperforms Linear Regression (p < 0.001)
2. MLP performs well (R² = 0.9245) but slightly underperforms RF
3. Model robustness confirmed across diverse random seeds
4. Linear Regression shows high variance (RMSE = 14.07±2.04 mM across 30 seeds)

## Files Manifest

| File | Purpose |
|------|---------|
| `main.tex` | IEEE-formatted report source (IEEEtran class, double-column) |
| `stats_test.py` | Complete ML pipeline: data loading, model training, evaluation, hypothesis testing |
| `generate_syngas_ppt_final.py` | PowerPoint generation with real data charts |
| `speaker_script_SYNGAS.txt` | Presentation delivery notes (~12.5 min, 10 slides) |
| `syngas fermentation report-Ying Tan.pdf` | Final compiled report (7 pages, 245,503 bytes) |
| `term_project_presentation_SYNGAS_FINAL.pptx` | Final presentation deck (10 slides, 120 KB) |

## Compliance

- **Report Format:** IEEE/ACM double-column proceedings format ✓
- **Report Length:** 6-8 pages (7 pages) ✓
- **Presentation Duration:** ≤15 minutes (12.5 minutes) ✓
- **Presentation Slides:** 10-15 (10 slides) ✓
- **All Canvas Guidelines:** Met ✓

## GenAI Disclosure (Appendix)

Per Canvas policy, this project includes a GenAI Audit Appendix in the main report documenting:
- **Student Contribution:** ~85-90% (problem framing, implementation, experiments, statistical analysis, conclusions) Data sources are from published paper.
- **GenAI Assistance:** ~10-15% (writing polish, formatting, editorial suggestions only)
- **Reproducibility:** All code, results, and methodology are independently verifiable

## Authors

- **Student:** Ying Tan
- **Instructor:** Dr. Hsu
- **Course:** Machine Learning (CIS 732/830), Kansas State University, Spring 2026

## License

This work is submitted as course work for CIS 732/830 at Kansas State University and is subject to institutional academic integrity policies.

## References

Complete references are provided in `references.bib` and compiled in the main report. Key sources include:
1. Scikit-learn documentation
2. SyngasMachineLearning GitHub repository
3. Statistical testing literature (paired t-test, Wilcoxon signed-rank)

---

**Last Updated:** May 9, 2026  
**Report Compilation Timestamp:** May 9, 2026 | 15:51 UTC  
**Final Status:** Ready for submission
