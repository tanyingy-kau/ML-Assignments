import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

URL = "https://raw.githubusercontent.com/garrettroell/SyngasMachineLearning/main/data/experimental_data.csv"
OUT_CSV = "/home/mike-wang/term_project_deliverables/scripts/repeated_split_summary.csv"
OUT_TXT = "/home/mike-wang/term_project_deliverables/scripts/stat_tests_output.txt"


def metric_pack(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def main():
    df = pd.read_csv(URL)
    X = df[["time", "N2", "CO", "CO2", "H2", "flow rate (mL/min)", "composition"]]
    y = df["ethanol (mM)"]

    # Fixed split for sample-level paired tests.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)
    mlp.fit(X_train_s, y_train)
    pred_mlp = mlp.predict(X_test_s)

    lr_metrics = metric_pack(y_test, pred_lr)
    rf_metrics = metric_pack(y_test, pred_rf)
    mlp_metrics = metric_pack(y_test, pred_mlp)

    err_lr = np.abs(y_test.values - pred_lr)
    err_rf = np.abs(y_test.values - pred_rf)
    t_stat, t_p = ttest_rel(err_lr, err_rf, alternative="greater")
    w_stat, w_p = wilcoxon(err_lr, err_rf, alternative="greater")

    rows = []
    for s in range(30):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=s)

        lr_s = LinearRegression().fit(X_tr, y_tr)
        p_lr = lr_s.predict(X_te)

        rf_s = RandomForestRegressor(n_estimators=200, random_state=s).fit(X_tr, y_tr)
        p_rf = rf_s.predict(X_te)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        mlp_s = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=s).fit(X_tr_s, y_tr)
        p_mlp = mlp_s.predict(X_te_s)

        for name, pred in (("LR", p_lr), ("RF", p_rf), ("MLP", p_mlp)):
            rmse, mae, r2 = metric_pack(y_te, pred)
            rows.append([s, name, rmse, mae, r2])

    rep = pd.DataFrame(rows, columns=["seed", "model", "rmse", "mae", "r2"])
    summary = rep.groupby("model")[["rmse", "mae", "r2"]].agg(["mean", "std"])
    summary.to_csv(OUT_CSV)

    lr_rmse = rep[rep["model"] == "LR"]["rmse"].values
    rf_rmse = rep[rep["model"] == "RF"]["rmse"].values
    t_stat2, t_p2 = ttest_rel(lr_rmse, rf_rmse, alternative="greater")
    w_stat2, w_p2 = wilcoxon(lr_rmse, rf_rmse, alternative="greater")

    lines = [
        "Fixed split metrics",
        f"LR  RMSE={lr_metrics[0]:.4f} MAE={lr_metrics[1]:.4f} R2={lr_metrics[2]:.4f}",
        f"RF  RMSE={rf_metrics[0]:.4f} MAE={rf_metrics[1]:.4f} R2={rf_metrics[2]:.4f}",
        f"MLP RMSE={mlp_metrics[0]:.4f} MAE={mlp_metrics[1]:.4f} R2={mlp_metrics[2]:.4f}",
        "",
        "Paired tests on sample absolute errors (H1: LR error > RF error)",
        f"paired_t_stat={t_stat:.6f}, p_value={t_p:.8f}",
        f"wilcoxon_stat={w_stat:.6f}, p_value={w_p:.8f}",
        "",
        "Seed-level RMSE tests across 30 splits (H1: LR RMSE > RF RMSE)",
        f"paired_t_stat={t_stat2:.6f}, p_value={t_p2:.8f}",
        f"wilcoxon_stat={w_stat2:.6f}, p_value={w_p2:.8f}",
        "",
        f"Summary CSV saved to: {OUT_CSV}",
    ]

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nDetailed summary saved to {OUT_TXT}")


if __name__ == "__main__":
    main()
