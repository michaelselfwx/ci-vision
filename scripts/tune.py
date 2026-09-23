import optuna
import json
from datetime import datetime
from pathlib import Path
from train import main

LOG_FILE = Path("tune_trials_unet_lr.txt")


def objective(trial):
    lr = trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3])

    out_dir = Path(f"../../../../train_out/unet/trial_{trial.number}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "--model", "unet",
        "--epochs", str(30),
        "--batch-size", str(8),
        "--base-channels", str(64),
        "--lr", str(lr),
        "--early-stop-patience", str(4),
        "--output-dir", out_dir.as_posix(),
        "--split-type", "rand",
        "--label-usage", "train_samp",
        "--aug-elastic",
        "--aug-expand", str(16),
    ]
    return main(argv, trial=trial)


def log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Append a record of a finished trial to LOG_FILE."""
    duration = (
        (trial.datetime_complete - trial.datetime_start).total_seconds()
        if trial.datetime_start and trial.datetime_complete
        else None
    )

    record = {
        "trial_number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": trial.params,
        "user_attrs": trial.user_attrs,
        "system_attrs": trial.system_attrs,
        "intermediate_values": trial.intermediate_values,  # per-epoch reports
        "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
        "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        "duration_seconds": duration,
        "best_so_far": {
            "trial_number": study.best_trial.number,
            "value": study.best_value,
            "params": study.best_params,
        } if study.best_trial else None,
    }

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Trial {trial.number} | state={trial.state.name} | "
                f"value={trial.value} | logged_at={datetime.now().isoformat()}\n")
        f.write("-" * 80 + "\n")
        f.write(json.dumps(record, indent=2, default=str))
        f.write("\n\n")


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="training_tune_unet_lr",
        storage="sqlite:///tune.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    study.optimize(objective, n_trials=10, callbacks=[log_trial])

    # Final summary at the end
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write("#" * 80 + "\n")
        f.write("FINAL RESULTS\n")
        f.write("#" * 80 + "\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best value (val mIoU): {study.best_value}\n")
        f.write(f"Best params: {json.dumps(study.best_params, indent=2)}\n")

    print("Best params:", study.best_params)
    print("Best val mIoU:", study.best_value)