import os
import json
import time

import numpy as onp
import pandas as pd
import pennylane as qml
from pennylane import numpy as np



def mse_loss(y_true, y_pred):
    """Mean squared error between labels in {-1,+1} and predictions in [-1,+1}."""
    y_true = onp.asarray(y_true)
    y_pred = onp.asarray(y_pred)
    return float(onp.mean((y_true - y_pred) ** 2))


def accuracy_sign(y_true, y_pred):
    y_true = onp.asarray(y_true)
    y_pred = onp.asarray(y_pred)
    return float(onp.mean(onp.sign(y_pred) == onp.sign(y_true)))



def train_one_run(
    circuit,
    weight_shape,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    lr=0.1,
    epochs=100,
    seed=0,
    optimizer_name="gd",
    verbose=False,
    eval_every=1,   
    grad_every=1,     
    use_bias=True,    
):
   
    rng = onp.random.default_rng(seed)
    weights = np.array(0.01 * rng.standard_normal(size=weight_shape), requires_grad=True)

    if use_bias:
        bias = np.array(0.0, requires_grad=True)
    else:
        bias = None

    if optimizer_name.lower() == "gd":
        opt = qml.GradientDescentOptimizer(stepsize=lr)
    elif optimizer_name.lower() == "adam":
        opt = qml.AdamOptimizer(stepsize=lr)
    else:
        raise ValueError("optimizer_name must be 'gd' or 'adam'")

    # cost function on training set 
    def cost(w, b=None):
        if use_bias:
            preds = np.array([circuit(x, w) + b for x in X_train])
        else:
            preds = np.array([circuit(x, w) for x in X_train])
        return np.mean((y_train - preds) ** 2)

    # Logs
    loss_curve = []
    grad_norm_curve = []
    train_acc_curve = []
    test_acc_curve = []
    epoch_time_curve = []

    start_all = time.time()

    for epoch in range(epochs):
        t0 = time.time()

        #gradient norm
        grad_norm = float("nan")
        do_grad = (epoch % grad_every == 0) or (epoch == epochs - 1)

        if do_grad:
            if use_bias:
                grad_w, grad_b = qml.grad(cost, argnum=[0, 1])(weights, bias)
                flat = onp.concatenate(
                    [onp.ravel(onp.asarray(grad_w)), onp.ravel(onp.asarray(grad_b))]
                )
                grad_norm = float(onp.linalg.norm(flat))
            else:
                grad_w = qml.grad(cost)(weights, None)
                grad_norm = float(onp.linalg.norm(onp.ravel(onp.asarray(grad_w))))

        grad_norm_curve.append(grad_norm)

        # optimizer 
        if use_bias:
            weights, bias = opt.step(cost, weights, bias)
        else:
            weights = opt.step(cost, weights, None)

        # training metric
        if use_bias:
            train_preds = onp.array([circuit(x, weights) + float(bias) for x in X_train])
        else:
            train_preds = onp.array([circuit(x, weights) for x in X_train])

        loss_val = mse_loss(y_train, train_preds)
        loss_curve.append(loss_val)

        train_acc = accuracy_sign(y_train, train_preds)
        train_acc_curve.append(train_acc)

        # test metrics
        do_test = (epoch % eval_every == 0) or (epoch == epochs - 1)
        if do_test:
            if use_bias:
                test_preds = onp.array([circuit(x, weights) + float(bias) for x in X_test])
            else:
                test_preds = onp.array([circuit(x, weights) for x in X_test])
            test_acc = accuracy_sign(y_test, test_preds)
        else:
            test_acc = float("nan")

        test_acc_curve.append(test_acc)

        epoch_time_curve.append(float(time.time() - t0))

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(
                f"Epoch {epoch+1:4d}/{epochs} | loss={loss_val:.4f} | "
                f"grad_norm={grad_norm:.4e} | train_acc={train_acc:.3f} | test_acc={test_acc:.3f}"
            )

    total_time = time.time() - start_all

    # Final metrics
    if use_bias:
        final_train_preds = onp.array([circuit(x, weights) + float(bias) for x in X_train])
        final_test_preds = onp.array([circuit(x, weights) + float(bias) for x in X_test])
    else:
        final_train_preds = onp.array([circuit(x, weights) for x in X_train])
        final_test_preds = onp.array([circuit(x, weights) for x in X_test])

    # last non-nan grad norm
    final_grad_norm = next((g for g in reversed(grad_norm_curve) if not onp.isnan(g)), 0.0)

    result = {
        "seed": int(seed),
        "epochs": int(epochs),
        "lr": float(lr),
        "optimizer": optimizer_name,
        "final_train_loss": mse_loss(y_train, final_train_preds),
        "final_test_loss": mse_loss(y_test, final_test_preds),
        "final_train_acc": accuracy_sign(y_train, final_train_preds),
        "final_test_acc": accuracy_sign(y_test, final_test_preds),
        "final_grad_norm": float(final_grad_norm),
        "total_time_sec": float(total_time),
        # curves:
        "loss_curve": loss_curve,
        "grad_norm_curve": grad_norm_curve,
        "train_acc_curve": train_acc_curve,
        "test_acc_curve": test_acc_curve,
        "epoch_time_curve": epoch_time_curve,
    }
    return result



# Experiment runner
def run_experiment(
    *,
    exp_name,
    circuit,
    weight_shape,
    X_train,
    y_train,
    X_test,
    y_test,
    config,
    seeds=(0, 1, 2),
    out_dir="results",
    save_curves=True,
    verbose=False,
):
   

    os.makedirs(out_dir, exist_ok=True)
    curves_dir = os.path.join(out_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)

    rows = []
    curve_files = []

    for s in seeds:
        out = train_one_run(
            circuit,
            weight_shape,
            X_train,
            y_train,
            X_test,
            y_test,
            lr=config["lr"],
            epochs=config["epochs"],
            seed=s,
            optimizer_name=config.get("optimizer", "gd"),
            verbose=verbose,
            eval_every=config.get("eval_every", 1),
            grad_every=config.get("grad_every", 1),
            use_bias=config.get("use_bias", True),
        )

        row = {
            "exp_name": exp_name,
            **config,
            "seed": out["seed"],
            "final_train_loss": out["final_train_loss"],
            "final_test_loss": out["final_test_loss"],
            "final_train_acc": out["final_train_acc"],
            "final_test_acc": out["final_test_acc"],
            "final_grad_norm": out["final_grad_norm"],
            "total_time_sec": out["total_time_sec"],
        }
        rows.append(row)

        if save_curves:
            curve_path = os.path.join(curves_dir, f"{exp_name}_seed{int(s)}.json")
            with open(curve_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "exp_name": exp_name,
                        "seed": out["seed"],
                        "config": config,
                        "loss_curve": out["loss_curve"],
                        "grad_norm_curve": out["grad_norm_curve"],
                        "train_acc_curve": out["train_acc_curve"],
                        "test_acc_curve": out["test_acc_curve"],
                        "epoch_time_curve": out["epoch_time_curve"],
                    },
                    f,
                    indent=2,
                )
            curve_files.append(curve_path)

    df = pd.DataFrame(rows)

    # Save per-run summary
    runs_csv = os.path.join(out_dir, "runs.csv")
    if os.path.exists(runs_csv):
        df.to_csv(runs_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(runs_csv, index=False)

    # Aggregated stats
    agg_cols = [
        "final_train_loss", "final_test_loss",
        "final_train_acc", "final_test_acc",
        "final_grad_norm", "total_time_sec"
    ]

    group_cols = ["exp_name"] + [k for k in config.keys() if k != "seed"]
    agg = df.groupby(group_cols)[agg_cols].agg(["mean", "std"]).reset_index()

    agg_csv = os.path.join(out_dir, "agg.csv")
    if os.path.exists(agg_csv):
        agg.to_csv(agg_csv, mode="a", header=False, index=False)
    else:
        agg.to_csv(agg_csv, index=False)

    return df, agg, curve_files
