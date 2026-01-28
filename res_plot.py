import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



AGG_PATH = "results/agg.csv"        
OUT_DIR = "figures"           
JSON_GLOB = "results/**/*.json"

os.makedirs(OUT_DIR, exist_ok=True)



def load_agg(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

   
    df = df[df["exp_name"].notna()].copy()

  
    numeric_cols = [
        "n_qubits", "depth", "reupload", "lr", "epochs",
        "final_train_loss", "final_train_loss.1",
        "final_test_loss", "final_test_loss.1",
        "final_train_acc", "final_train_acc.1",
        "final_test_acc", "final_test_acc.1",
        "final_grad_norm", "final_grad_norm.1",
        "total_time_sec", "total_time_sec.1",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def savefig(name: str):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")


def line_with_errorbars(x, y, yerr, xlabel, ylabel, title, filename):
    plt.figure()
    plt.errorbar(x, y, yerr=yerr, marker="o", linestyle="-")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    savefig(filename)
    plt.close()


def bars_with_errorbars(labels, y, yerr, xlabel, ylabel, title, filename):
    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, y, yerr=yerr, capsize=6)
    plt.xticks(x, labels, rotation=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    savefig(filename)
    plt.close()



df = load_agg(AGG_PATH)


groups = sorted(df["group"].dropna().unique())

print("Found groups:", groups)

for g in groups:
    dfg = df[df["group"] == g].copy()

    
    g_lower = str(g).lower()

 
    if "depth" in g_lower:
        dfg = dfg.sort_values("depth")
        x = dfg["depth"].values
        y = dfg["final_test_acc"].values
        yerr = dfg["final_test_acc.1"].values

        line_with_errorbars(
            x, y, yerr,
            xlabel="Depth (layers)",
            ylabel="Final test accuracy",
            title=f"Final test accuracy vs depth ({g})",
            filename=f"{g}_final_test_acc_vs_depth.png",
        )

        
        line_with_errorbars(
            x, dfg["final_test_loss"].values, dfg["final_test_loss.1"].values,
            xlabel="Depth (layers)",
            ylabel="Final test loss",
            title=f"Final test loss vs depth ({g})",
            filename=f"{g}_final_test_loss_vs_depth.png",
        )

        line_with_errorbars(
            x, dfg["final_grad_norm"].values, dfg["final_grad_norm.1"].values,
            xlabel="Depth (layers)",
            ylabel="Final gradient norm",
            title=f"Final grad norm vs depth ({g})",
            filename=f"{g}_final_grad_norm_vs_depth.png",
        )

        line_with_errorbars(
            x, dfg["total_time_sec"].values, dfg["total_time_sec.1"].values,
            xlabel="Depth (layers)",
            ylabel="Total training time (sec)",
            title=f"Total time vs depth ({g})",
            filename=f"{g}_time_vs_depth.png",
        )

  
    elif "reup" in g_lower or "reupload" in g_lower:
        dfg = dfg.sort_values("reupload")
        x = dfg["reupload"].values
        y = dfg["final_test_acc"].values
        yerr = dfg["final_test_acc.1"].values

        line_with_errorbars(
            x, y, yerr,
            xlabel="Re-upload cycles",
            ylabel="Final test accuracy",
            title=f"Final test accuracy vs re-uploading ({g})",
            filename=f"{g}_final_test_acc_vs_reupload.png",
        )

        line_with_errorbars(
            x, dfg["final_test_loss"].values, dfg["final_test_loss.1"].values,
            xlabel="Re-upload cycles",
            ylabel="Final test loss",
            title=f"Final test loss vs re-uploading ({g})",
            filename=f"{g}_final_test_loss_vs_reupload.png",
        )

        line_with_errorbars(
            x, dfg["final_grad_norm"].values, dfg["final_grad_norm.1"].values,
            xlabel="Re-upload cycles",
            ylabel="Final gradient norm",
            title=f"Final grad norm vs re-uploading ({g})",
            filename=f"{g}_final_grad_norm_vs_reupload.png",
        )

        line_with_errorbars(
            x, dfg["total_time_sec"].values, dfg["total_time_sec.1"].values,
            xlabel="Re-upload cycles",
            ylabel="Total training time (sec)",
            title=f"Total time vs re-uploading ({g})",
            filename=f"{g}_time_vs_reupload.png",
        )

    elif "ent" in g_lower or "entang" in g_lower:
        order = ["chain", "all"]
        dfg["entanglement"] = dfg["entanglement"].astype(str)
        labels = [e for e in order if e in set(dfg["entanglement"])] + \
                 [e for e in sorted(set(dfg["entanglement"])) if e not in order]

        dfg = dfg.set_index("entanglement").loc[labels].reset_index()

        bars_with_errorbars(
            labels=labels,
            y=dfg["final_test_acc"].values,
            yerr=dfg["final_test_acc.1"].values,
            xlabel="Entanglement topology",
            ylabel="Final test accuracy",
            title=f"Final test accuracy vs entanglement ({g})",
            filename=f"{g}_final_test_acc_vs_entanglement.png",
        )

        bars_with_errorbars(
            labels=labels,
            y=dfg["final_test_loss"].values,
            yerr=dfg["final_test_loss.1"].values,
            xlabel="Entanglement topology",
            ylabel="Final test loss",
            title=f"Final test loss vs entanglement ({g})",
            filename=f"{g}_final_test_loss_vs_entanglement.png",
        )

        bars_with_errorbars(
            labels=labels,
            y=dfg["final_grad_norm"].values,
            yerr=dfg["final_grad_norm.1"].values,
            xlabel="Entanglement topology",
            ylabel="Final gradient norm",
            title=f"Final grad norm vs entanglement ({g})",
            filename=f"{g}_final_grad_norm_vs_entanglement.png",
        )

        bars_with_errorbars(
            labels=labels,
            y=dfg["total_time_sec"].values,
            yerr=dfg["total_time_sec.1"].values,
            xlabel="Entanglement topology",
            ylabel="Total training time (sec)",
            title=f"Total time vs entanglement ({g})",
            filename=f"{g}_time_vs_entanglement.png",
        )

    else:
        print(f"Skipping unknown group format: {g}")



json_files = glob.glob(JSON_GLOB, recursive=True)
if not json_files:
    print("No JSON logs found (optional step skipped).")
else:
    print(f"Found {len(json_files)} JSON logs. Plotting mean curves per experiment group...")

    rows = []
    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            cfg = obj.get("config", {})
            rows.append({
                "path": fp,
                "exp_name": obj.get("exp_name"),
                "seed": obj.get("seed"),
                "group": cfg.get("group"),
                "dataset": cfg.get("dataset"),
                "depth": cfg.get("depth"),
                "reupload": cfg.get("reupload"),
                "entanglement": cfg.get("entanglement"),
                "loss_curve": obj.get("loss_curve"),
                "test_acc_curve": obj.get("test_acc_curve"),
                "grad_norm_curve": obj.get("grad_norm_curve"),
            })
        except Exception as e:
            print("Failed reading:", fp, e)

    logs = pd.DataFrame(rows)
    logs = logs[logs["group"].notna()].copy()

    def plot_mean_curve(logs_subset, key, title, filename):
        curves = [c for c in logs_subset[key].tolist() if isinstance(c, list) and len(c) > 0]
        if not curves:
            return

        min_len = min(len(c) for c in curves)
        arr = np.array([c[:min_len] for c in curves], dtype=float)

        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = np.arange(1, min_len + 1)

        plt.figure()
        plt.plot(x, mean)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        plt.xlabel("Epoch")
        plt.ylabel(key.replace("_", " "))
        plt.title(title)
        savefig(filename)
        plt.close()
    for g in sorted(logs["group"].unique()):
        lg = logs[logs["group"] == g].copy()

        if "depth" in str(g).lower():
            # one curve per depth
            for depth in sorted(lg["depth"].dropna().unique()):
                sub = lg[lg["depth"] == depth]
                plot_mean_curve(sub, "loss_curve", f"Loss curve ({g}, depth={int(depth)})",
                                f"{g}_loss_curve_depth{int(depth)}.png")
                plot_mean_curve(sub, "test_acc_curve", f"Test accuracy curve ({g}, depth={int(depth)})",
                                f"{g}_test_acc_curve_depth{int(depth)}.png")
                plot_mean_curve(sub, "grad_norm_curve", f"Grad norm curve ({g}, depth={int(depth)})",
                                f"{g}_grad_norm_curve_depth{int(depth)}.png")

        elif "reup" in str(g).lower() or "reupload" in str(g).lower():
            for r in sorted(lg["reupload"].dropna().unique()):
                sub = lg[lg["reupload"] == r]
                plot_mean_curve(sub, "loss_curve", f"Loss curve ({g}, reupload={int(r)})",
                                f"{g}_loss_curve_reup{int(r)}.png")
                plot_mean_curve(sub, "test_acc_curve", f"Test accuracy curve ({g}, reupload={int(r)})",
                                f"{g}_test_acc_curve_reup{int(r)}.png")
                plot_mean_curve(sub, "grad_norm_curve", f"Grad norm curve ({g}, reupload={int(r)})",
                                f"{g}_grad_norm_curve_reup{int(r)}.png")

        elif "ent" in str(g).lower():
            for ent in sorted(lg["entanglement"].dropna().unique()):
                sub = lg[lg["entanglement"] == ent]
                plot_mean_curve(sub, "loss_curve", f"Loss curve ({g}, ent={ent})",
                                f"{g}_loss_curve_ent_{ent}.png")
                plot_mean_curve(sub, "test_acc_curve", f"Test accuracy curve ({g}, ent={ent})",
                                f"{g}_test_acc_curve_ent_{ent}.png")
                plot_mean_curve(sub, "grad_norm_curve", f"Grad norm curve ({g}, ent={ent})",
                                f"{g}_grad_norm_curve_ent_{ent}.png")

print("Done.")
