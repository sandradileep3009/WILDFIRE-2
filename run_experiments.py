import argparse
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config                                             
from training.train import make_tag                       

def sh(cmd):
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(ROOT), check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--horizons", type=int, nargs="+",default=list(config.HORIZONS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(config.SEEDS))
    p.add_argument("--tasks", nargs="+", default=["growth", "fullmask"])
    p.add_argument("--mask-source", choices=["gt", "predicted"], default="gt")
    p.add_argument("--ablation-history", action="store_true")
    p.add_argument("--epochs", type=int, default=config.CONVLSTM_EPOCHS)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    suffix = "_pred" if a.mask_source == "predicted" else ""
    py = sys.executable
    out_dir = config.RESULTS_DIR / ("predicted_masks" if suffix else "")

    for h in a.horizons:
        csv = config.PROCESSED_DIR / f"sequences_h{h}{suffix}.csv"
        sh([py, "preprocessing/build_sequences.py", "--horizon", h, "--mask-source", a.mask_source])
        tags = []
        variants = [True] + ([False] if a.ablation_history else [])
        for seed in a.seeds:
            for task in a.tasks:
                for use_hist in variants:
                    tag = make_tag(task, h, seed, use_hist, suffix)
                    tags.append(tag)
                    done = (config.PREDS_DIR / f"{tag}.npz").exists()
                    if done and not a.force:
                        print(f"skip {tag} (already trained)")
                        continue
                    cmd = [py, "-m", "training.train", "--task", task,
                           "--horizon", h, "--seed", seed, "--csv", csv,
                           "--epochs", a.epochs, "--tag-suffix", suffix]
                    if not use_hist:
                        cmd.append("--no-mask-history")
                    sh(cmd)

        sh([py, "-m", "evaluation.evaluate", "--horizon", h,
            "--csv", csv, "--out-dir", out_dir, "--tags", *tags])

    sh([py, "aggregate_results.py", out_dir])


if __name__ == "__main__":
    main()
