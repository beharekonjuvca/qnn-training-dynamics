from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from datasets import load_moons, load_circles
from circuits import make_variational_classifier, draw_circuit_example
from train import run_experiment


# Experiment settings

OUT_DIR = "results"
FIG_DIR = "figures"

SEEDS = (0, 1, 2)

# Default training hyperparams
DEFAULT_HP = {
    "lr": 0.10,
    "epochs": 120,
    "optimizer": "adam", 
}


@dataclass(frozen=True)
class DataBundle:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_data(name: str, seed: int = 42) -> DataBundle:
    
    if name == "moons":
        X_train, X_test, y_train, y_test = load_moons(n_samples=200, noise=0.10, seed=seed)
    elif name == "circles":
        X_train, X_test, y_train, y_test = load_circles(n_samples=200, noise=0.10, factor=0.50, seed=seed)
    else:
        raise ValueError("Dataset must be 'moons' or 'circles'")

    # Convert labels
    y_train = 2 * y_train - 1
    y_test = 2 * y_test - 1

    return DataBundle(name=name, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "curves"), exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


# Experiment Group A: Depth sweep
def depth_sweep(dataset: DataBundle, n_qubits: int = 2, depths: Tuple[int, ...] = (1, 2, 4, 8)):
  
    print(f"\n=== Group A: Depth sweep on {dataset.name} ===")
    for depth in depths:
        config = {
            "dataset": dataset.name,
            "group": "A_depth",
            "n_qubits": n_qubits,
            "depth": depth,
            "reupload": 1,
            "entanglement": "chain",
            "encoding_rotation": "RY",
            **DEFAULT_HP,
        }

        exp_name = f"A_{dataset.name}_q{n_qubits}_depth{depth}_reup1_chain"

        circuit, weight_shape = make_variational_classifier(
            n_qubits=n_qubits,
            depth=depth,
            reupload=1,
            encoding_rotation=config["encoding_rotation"],
            entanglement=config["entanglement"],
            measure_wire=0,
        )

        run_experiment(
            exp_name=exp_name,
            circuit=circuit,
            weight_shape=weight_shape,
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            config=config,
            seeds=SEEDS,
            out_dir=OUT_DIR,
            save_curves=True,
            verbose=False,
        )
        print(f"Saved: {exp_name}")



# Experiment Group B: Re-uploading vs no re-uploading
def reupload_sweep(dataset: DataBundle, n_qubits: int = 1, depth: int = 2, reuploads: Tuple[int, ...] = (1, 2, 4)):
   
    print(f"\n=== Group B: Re-uploading sweep on {dataset.name} ===")
    for reup in reuploads:
        config = {
            "dataset": dataset.name,
            "group": "B_reupload",
            "n_qubits": n_qubits,
            "depth": depth,
            "reupload": reup,
            "entanglement": "chain",
            "encoding_rotation": "RY",
            **DEFAULT_HP,
        }

        exp_name = f"B_{dataset.name}_q{n_qubits}_depth{depth}_reup{reup}_chain"

        circuit, weight_shape = make_variational_classifier(
            n_qubits=n_qubits,
            depth=depth,
            reupload=reup,
            encoding_rotation=config["encoding_rotation"],
            entanglement=config["entanglement"],
            measure_wire=0,
        )

        run_experiment(
            exp_name=exp_name,
            circuit=circuit,
            weight_shape=weight_shape,
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            config=config,
            seeds=SEEDS,
            out_dir=OUT_DIR,
            save_curves=True,
            verbose=False,
        )
        print(f"Saved: {exp_name}")


# Experiment Group C: Entanglement pattern
def entanglement_compare(dataset: DataBundle, n_qubits: int = 3, depth: int = 2, reupload: int = 1):
   
    print(f"\n=== Group C: Entanglement compare on {dataset.name} ===")
    for ent in ("chain", "all"):
        config = {
            "dataset": dataset.name,
            "group": "C_entanglement",
            "n_qubits": n_qubits,
            "depth": depth,
            "reupload": reupload,
            "entanglement": ent,
            "encoding_rotation": "RY",
            **DEFAULT_HP,
        }

        exp_name = f"C_{dataset.name}_q{n_qubits}_depth{depth}_reup{reupload}_{ent}"

        circuit, weight_shape = make_variational_classifier(
            n_qubits=n_qubits,
            depth=depth,
            reupload=reupload,
            encoding_rotation=config["encoding_rotation"],
            entanglement=ent,
            measure_wire=0,
        )

        run_experiment(
            exp_name=exp_name,
            circuit=circuit,
            weight_shape=weight_shape,
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            config=config,
            seeds=SEEDS,
            out_dir=OUT_DIR,
            save_curves=True,
            verbose=False,
        )
        print(f"Saved: {exp_name}")


# Optional: Save one circuit diagram for thesis
def save_circuit_diagram_text():
  
    diagram = draw_circuit_example(
        n_qubits=2,
        depth=2,
        reupload=2,
        encoding_rotation="RY",
        entanglement="chain",
    )
    path = os.path.join(FIG_DIR, "circuit_diagram_example.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(diagram)
    print(f"Saved circuit diagram text to: {path}")


def main():
    ensure_dirs()

    moons = load_data("moons", seed=42)
    circles = load_data("circles", seed=42)

    # Group A: depth sweep 
    depth_sweep(moons, n_qubits=2, depths=(1, 2, 4, 8))

    # Group B: re-uploading sweep 
    reupload_sweep(circles, n_qubits=1, depth=2, reuploads=(1, 2, 4))

    # Group C: entanglement compare
    entanglement_compare(moons, n_qubits=3, depth=2, reupload=1)

    # Save a circuit diagram for thesis figure
    save_circuit_diagram_text()

    print("\nAll experiments finished.")
    print(f"- Summary: {os.path.join(OUT_DIR, 'runs.csv')}")
    print(f"- Aggregates: {os.path.join(OUT_DIR, 'agg.csv')}")
    print(f"- Curves: {os.path.join(OUT_DIR, 'curves')}")
    print(f"- Circuit diagram: {os.path.join(FIG_DIR, 'circuit_diagram_example.txt')}")


if __name__ == "__main__":
    main()
