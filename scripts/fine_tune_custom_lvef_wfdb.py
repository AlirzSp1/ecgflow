#!/usr/bin/env python3
"""Fine-tune ECGFlow on a custom WFDB low-LVEF dataset.

Metadata format:
  1. ecg_id
  2. record path without .hea/.dat extension
  3. age at ECG
  4. gender, male/female
  5. low LVEF flag, 0/1

Example:
  uv run scripts/fine_tune_custom_lvef_wfdb.py \
      --config config.json
"""
import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy import signal

from ecgflow.data import (
    MIMICIV_12CHANNEL_MEAN,
    MIMICIV_12CHANNEL_SD,
    MIMICIV_8CHANNEL_MEAN,
    MIMICIV_8CHANNEL_SD,
    PTBXL_12CHANNEL_MEAN,
    PTBXL_12CHANNEL_SD,
    PTBXL_8CHANNEL_MEAN,
    PTBXL_8CHANNEL_SD,
)


CUSTOM_DATASET_NAME = "custom_lvef_wfdb"
METADATA_COLUMNS = ["ecg_id", "record_path", "age", "gender", "low_lvef_flag"]


class CustomLvefConfig:
    metadata_file = None
    data_dir = None
    val_fraction = 0.15
    test_fraction = 0.0
    seed = 20241153
    target_sample_rate = 500.0
    target_length = 5000
    scale_source = "ptbxl"
    swap_avl_avf = False
    split_group_by = "patient_from_path"
    patient_id_regex = r"(p\d{8,})"


def _read_metadata(metadata_file):
    metadata_file = Path(metadata_file)
    df = pd.read_csv(metadata_file, sep=None, engine="python")
    if not set(METADATA_COLUMNS).issubset(df.columns):
        df = pd.read_csv(metadata_file, sep=None, engine="python", header=None)
        if df.shape[1] < len(METADATA_COLUMNS):
            raise ValueError(
                f"{metadata_file} must contain at least {len(METADATA_COLUMNS)} columns: "
                + ", ".join(METADATA_COLUMNS)
            )
        df = df.iloc[:, :len(METADATA_COLUMNS)]
        df.columns = METADATA_COLUMNS

    df = df.loc[:, METADATA_COLUMNS].copy()
    df["ecg_id"] = df["ecg_id"].astype(str)
    df["record_path"] = df["record_path"].astype(str)
    df["gender"] = df["gender"].astype(str).str.lower().str.strip()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["low_lvef_flag"] = pd.to_numeric(df["low_lvef_flag"], errors="coerce")
    valid = df["low_lvef_flag"].isin([0, 1])
    valid &= df["record_path"].str.len() > 0
    df = df[valid].reset_index(drop=True)
    df["low_lvef_flag"] = df["low_lvef_flag"].astype(np.int64)
    if df.empty:
        raise ValueError("metadata has no usable rows with binary low_lvef_flag values")
    return df


def _split_indices(labels, seed, val_fraction, test_fraction, groups=None):
    if groups is not None:
        labels = np.asarray(labels)
        groups = np.asarray(groups).astype(str)
        group_table = pd.DataFrame({"group": groups, "label": labels})
        group_labels = group_table.groupby("group", sort=False)["label"].max()
        group_split = _split_indices(
            group_labels.values,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
        split_by_group = dict(zip(group_labels.index.astype(str), group_split))
        return np.asarray([split_by_group[group] for group in groups], dtype=object)

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    split = np.empty(labels.shape[0], dtype=object)
    split[:] = "train"

    for label in sorted(np.unique(labels)):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(math.floor(n * test_fraction))
        n_val = int(math.floor(n * val_fraction))
        if val_fraction > 0 and n >= 2 and n_val == 0:
            n_val = 1
        if test_fraction > 0 and n - n_val >= 2 and n_test == 0:
            n_test = 1
        split[idx[:n_test]] = "test"
        split[idx[n_test:n_test + n_val]] = "val"

    if np.all(split != "val") and val_fraction > 0 and len(split) > 1:
        split[rng.permutation(len(split))[0]] = "val"
    return split


def _split_groups_from_metadata(df, split_group_by, patient_id_regex):
    split_group_by = (split_group_by or "none").lower()
    if split_group_by in ("none", "row", "rows"):
        return None
    if split_group_by == "ecg_id":
        return df["ecg_id"].astype(str).values
    if split_group_by == "record_path":
        return df["record_path"].astype(str).values
    if split_group_by != "patient_from_path":
        raise ValueError(
            "split_group_by must be one of: none, ecg_id, record_path, patient_from_path"
        )

    regex = re.compile(patient_id_regex)
    groups = []
    missing = 0
    for ecg_id, record_path in zip(df["ecg_id"].astype(str), df["record_path"].astype(str)):
        match = regex.search(record_path)
        if match:
            groups.append(match.group(1) if match.groups() else match.group(0))
        else:
            groups.append(ecg_id)
            missing += 1
    if missing:
        print(
            f"WARNING: patient_id_regex did not match {missing} record_path values; "
            "using ecg_id for those rows."
        )
    return np.asarray(groups, dtype=object)


def _resolve_record_path(record_path, data_dir, metadata_file):
    path = Path(record_path).expanduser()
    if path.is_absolute():
        return path
    for base in (Path(data_dir).expanduser(), Path(metadata_file).expanduser().parent):
        candidate = base / path
        if candidate.with_suffix(".hea").exists() or candidate.exists():
            return candidate
    return Path(data_dir).expanduser() / path


def _resample_waveform(wf, source_hz, target_hz):
    if source_hz is None or target_hz is None:
        return wf
    source_hz = float(source_hz)
    target_hz = float(target_hz)
    if abs(source_hz - target_hz) < 1e-6:
        return wf
    scale = 1000
    up = int(round(target_hz * scale))
    down = int(round(source_hz * scale))
    gcd = math.gcd(up, down)
    return signal.resample_poly(wf, up // gcd, down // gcd, axis=1)


def _center_crop_or_pad(wf, target_length):
    if target_length is None:
        return wf
    target_length = int(target_length)
    n = wf.shape[1]
    if n == target_length:
        return wf
    if n > target_length:
        start = (n - target_length) // 2
        return wf[:, start:start + target_length]
    pad_left = (target_length - n) // 2
    pad_right = target_length - n - pad_left
    return np.pad(wf, ((0, 0), (pad_left, pad_right)), mode="constant")


class CustomLvefWfdbData:
    def __init__(
            self,
            data_dir=None,
            use_split="train",
            trim_channels=True,
            ecg_filter=False,
            notch_freq=60.0,
            bandpass_freq=(0.01, 150),
            **kwargs):
        self.data_dir = Path(data_dir or CustomLvefConfig.data_dir).expanduser()
        self.metadata_file = Path(CustomLvefConfig.metadata_file).expanduser()
        self.use_split = use_split
        self.trim_channels = trim_channels
        self.ecg_filter = ecg_filter
        self.notch_freq = notch_freq
        self.bandpass_freq = bandpass_freq
        self.xkey_name = ["rest"]
        self.ykey = ["low_lvef_flag"]
        self.scale_y = False
        self.classes = np.array(["normal_lvef", "low_lvef"])
        self.transform = kwargs.get("transform", {})
        self.target_sample_rate = CustomLvefConfig.target_sample_rate
        self.target_length = CustomLvefConfig.target_length
        self.swap_avl_avf = CustomLvefConfig.swap_avl_avf

        if CustomLvefConfig.scale_source == "mimic":
            self.mean = MIMICIV_8CHANNEL_MEAN if trim_channels else MIMICIV_12CHANNEL_MEAN
            self.std = MIMICIV_8CHANNEL_SD if trim_channels else MIMICIV_12CHANNEL_SD
        else:
            self.mean = PTBXL_8CHANNEL_MEAN if trim_channels else PTBXL_12CHANNEL_MEAN
            self.std = PTBXL_8CHANNEL_SD if trim_channels else PTBXL_12CHANNEL_SD

        df = _read_metadata(self.metadata_file)
        split_groups = _split_groups_from_metadata(
            df,
            CustomLvefConfig.split_group_by,
            CustomLvefConfig.patient_id_regex,
        )
        df["split"] = _split_indices(
            df["low_lvef_flag"].values,
            seed=CustomLvefConfig.seed,
            val_fraction=CustomLvefConfig.val_fraction,
            test_fraction=CustomLvefConfig.test_fraction,
            groups=split_groups,
        )
        if use_split in ("all", None):
            selected = df
        else:
            selected = df[df["split"] == use_split]
        if selected.empty:
            raise ValueError(f"metadata split {use_split!r} is empty")

        self.df = selected.reset_index(drop=True)
        self.id_list = self.df["ecg_id"].values
        self.record_list = [
            _resolve_record_path(p, self.data_dir, self.metadata_file).as_posix()
            for p in self.df["record_path"].values
        ]
        self.label_data = self.df[["low_lvef_flag"]].values.astype(np.float32)
        train_age = df.loc[df["split"] == "train", "age"].dropna()
        if train_age.empty:
            self.age_mean = 0.0
            self.age_std = 1.0
        else:
            self.age_mean = float(train_age.mean())
            self.age_std = float(train_age.std(ddof=0))
            if self.age_std < 1e-6:
                self.age_std = 1.0
        age_values = self.df["age"].fillna(self.age_mean).astype(np.float32)
        age_norm = ((age_values - self.age_mean) / self.age_std).to_numpy(dtype=np.float32)
        gender_values = (
            self.df["gender"]
            .map({"female": 0.0, "male": 1.0})
            .fillna(0.5)
            .astype(np.float32)
            .to_numpy()
        )
        self.tabular_data = np.stack([age_norm, gender_values], axis=1).astype(np.float32)

    def __len__(self):
        return len(self.id_list)


class CustomLvefWfdbDataset:
    def __init__(self, data_instance, name, **kwargs):
        self.data = data_instance
        self.name = name
        self.id_list = data_instance.id_list
        self.record_list = data_instance.record_list
        self.label_data = data_instance.label_data
        self.tabular_data = data_instance.tabular_data
        self.classes = data_instance.classes
        self.xkey_name = data_instance.xkey_name
        self.ykey = data_instance.ykey
        self.trim_channels = data_instance.trim_channels
        self.scale_y = data_instance.scale_y
        self.transform = data_instance.transform

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        import torch

        data = self.data
        rec = wfdb.rdrecord(self.record_list[idx])
        wf = rec.p_signal.T
        if data.swap_avl_avf and wf.shape[0] >= 6:
            wf = wf[(0, 1, 2, 3, 5, 4) + tuple(range(6, wf.shape[0])), :]
        if self.trim_channels:
            if wf.shape[0] < 8:
                raise ValueError(
                    f"record {self.record_list[idx]} has {wf.shape[0]} channels; "
                    "8-channel ECGFlow input requires at least 8"
                )
            if wf.shape[0] >= 12:
                wf = wf[(0, 1) + tuple(range(6, 12)), :]
            else:
                wf = wf[:8, :]

        wf = _resample_waveform(wf, getattr(rec, "fs", None), data.target_sample_rate)
        wf = _center_crop_or_pad(wf, data.target_length)
        wf = torch.tensor(wf)
        if self.transform:
            for key in self.xkey_name:
                wf = self.transform[key](wf)
        if wf.shape[0] == 1:
            wf = torch.squeeze(wf, axis=0)
        return dict(
            X=wf.float(),
            tabular=torch.from_numpy(self.tabular_data[idx]).float(),
            label=torch.from_numpy(self.label_data[idx]).float(),
        )


def _custom_dataset_factory(name, **kwargs):
    if name != CUSTOM_DATASET_NAME:
        raise ValueError(f"Unknown custom dataset {name}")
    data = CustomLvefWfdbData(**kwargs)
    return CustomLvefWfdbDataset(data, name, **kwargs)


def _patch_ecgflow_dataset_registration():
    import ecgflow.data as ecgflow_data
    import ecgflow.datasets as ecgflow_datasets
    import timm.data.dataset_factory as timm_dataset_factory

    ecgflow_data.data_factory[CUSTOM_DATASET_NAME] = CustomLvefWfdbData
    ecgflow_datasets.data_factory[CUSTOM_DATASET_NAME] = CustomLvefWfdbData
    ecgflow_datasets.dataset_factory = _custom_dataset_factory
    if CUSTOM_DATASET_NAME not in timm_dataset_factory._ECGFLOW_DS:
        timm_dataset_factory._ECGFLOW_DS.append(CUSTOM_DATASET_NAME)
    timm_dataset_factory.ecgflow_dataset_factory = _custom_dataset_factory


def _load_train_module():
    train_path = Path(__file__).resolve().parent / "train.py"
    spec = importlib.util.spec_from_file_location("ecgflow_scripts_train", train_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json_config(config_file):
    if not config_file:
        default_config = Path("config.json")
        if not default_config.exists():
            return {}, None
        config_file = default_config

    config_path = Path(config_file).expanduser()
    with open(config_path, "rt", encoding="utf-8") as fp:
        config = json.load(fp)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a JSON object")

    normalized = {}
    for key, value in config.items():
        normalized[key.replace("-", "_")] = value
    return normalized, config_path


def _format_run_name(run_name):
    if not run_name:
        return run_name
    return run_name.format(datetime=datetime.now().strftime("%Y-%m-%d-%H:%M"))


def _json_safe(value):
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_distributed_env():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _num_visible_cuda_devices(cuda_visible_devices):
    if not cuda_visible_devices:
        return 1
    devices = [item.strip() for item in str(cuda_visible_devices).split(",")]
    devices = [item for item in devices if item]
    return max(1, len(devices))


def _apply_distributed_env(args, env):
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    if args.nccl_safe_mode:
        env.setdefault("NCCL_IB_DISABLE", "1")
        env.setdefault("NCCL_P2P_DISABLE", "1")
        env.setdefault("NCCL_SHM_DISABLE", "1")

    for key, value in args.distributed_env.items():
        env[str(key)] = str(value)


def _launch_distributed_if_needed(args):
    if _is_distributed_env():
        return

    nproc = _num_visible_cuda_devices(args.cuda_visible_devices)
    if nproc <= 1 or not args.distributed:
        return

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    _apply_distributed_env(args, env)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    if args.wandb_run_name:
        cmd.extend(["--wandb-run-name", args.wandb_run_name])
    print(
        f"Launching distributed training with {nproc} processes "
        f"on CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')} "
        f"(backend={args.distributed_backend}, nccl_safe_mode={args.nccl_safe_mode})"
    )
    raise SystemExit(subprocess.run(cmd, env=env).returncode)


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("-c", "--config", default=None, help="JSON config file.")
    config_args, remaining = config_parser.parse_known_args()
    config, config_path = _load_json_config(config_args.config)

    parser = argparse.ArgumentParser(
        description="Fine-tune the SSL-pretrained ECGFlow 1dViT on custom WFDB low-LVEF labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", default=config_args.config or (config_path.as_posix() if config_path else None),
                        help="JSON config file.")
    parser.add_argument("--metadata-file", default=config.get("metadata_file"), help="CSV/TSV metadata file.")
    parser.add_argument("--data-dir", default=config.get("data_dir"), help="Root for relative WFDB record paths.")
    parser.add_argument("--pretrained-path", default=config.get("pretrained_path"), help="SSL checkpoint path.")
    parser.add_argument("--output", default=config.get("output", "~/ecgflow/experiments/custom_lvef"))
    parser.add_argument("--experiment", default=config.get("experiment", "mvtst-p50-d12-h8.custom-lvef"))
    parser.add_argument("--model", default=config.get("model", "mvtst_base_patch50"))
    parser.add_argument("--use-demographics", action=argparse.BooleanOptionalAction,
                        default=config.get("use_demographics", True),
                        help="Fuse normalized age and gender with ECG features.")
    parser.add_argument("--tabular-hidden-dim", type=int, default=config.get("tabular_hidden_dim", 32),
                        help="Hidden dimension for the age/gender fusion branch.")
    parser.add_argument("--tabular-dropout", type=float, default=config.get("tabular_dropout", 0.1),
                        help="Dropout for the age/gender fusion branch.")
    parser.add_argument("--seed", type=int, default=config.get("seed", 20241153))
    parser.add_argument("--batch-size", type=int, default=config.get("batch_size", 128))
    parser.add_argument("--epochs", type=int, default=config.get("epochs", 100))
    parser.add_argument("--ft-top", type=int, default=config.get("ft_top", 6),
                        help="Number of top transformer blocks to fine-tune.")
    parser.add_argument("--early-stop", type=int, default=config.get("early_stop", 4),
                        help="Stop after N epochs without improvement in the selection metric.")
    parser.add_argument("--workers", type=int, default=config.get("workers", 8))
    parser.add_argument("--checkpoint-hist", type=int, default=config.get("checkpoint_hist", 3),
                        help="Number of best epoch checkpoints to keep.")
    parser.add_argument("--eval-metric", default=config.get("eval_metric", "Sensitivity"),
                        help="Validation metric used for best checkpoint and early stopping.")
    parser.add_argument("--metric-threshold", type=float, default=config.get("metric_threshold", 0.3),
                        help="Probability threshold for Accuracy/Sensitivity/Specificity.")
    parser.add_argument("--target-sensitivity", type=float, default=config.get("target_sensitivity", 0.8),
                        help="Sensitivity target for ROC-derived screening checkpoint metrics.")
    parser.add_argument("--lr", type=float, default=config.get("lr", 5e-5))
    parser.add_argument("--opt", default=config.get("opt", "adamw"),
                        help="Optimizer passed to train.py.")
    parser.add_argument("--sched", default=config.get("sched", "cosine"),
                        help="LR scheduler passed to train.py.")
    parser.add_argument("--sched-on-updates", action=argparse.BooleanOptionalAction,
                        default=config.get("sched_on_updates", False),
                        help="Step the scheduler on every optimizer update instead of per epoch.")
    parser.add_argument("--warmup-epochs", type=int, default=config.get("warmup_epochs", 2),
                        help="Number of LR warmup epochs.")
    parser.add_argument("--patience-epochs", type=int, default=config.get("patience_epochs", 2),
                        help="Patience for plateau scheduler.")
    parser.add_argument("--cooldown-epochs", type=int, default=config.get("cooldown_epochs", 0),
                        help="Cooldown epochs after LR reduction.")
    parser.add_argument("--decay-rate", type=float, default=config.get("decay_rate", 0.5),
                        help="LR decay factor for supported schedulers.")
    parser.add_argument("--weight-decay", type=float, default=config.get("weight_decay", 0.01),
                        help="Optimizer weight decay.")
    parser.add_argument("--bce-pos-weight", type=float, default=config.get("bce_pos_weight", 1.5),
                        help="Positive-class weight for BCE loss.")
    parser.add_argument("--val-fraction", type=float, default=config.get("val_fraction", 0.15))
    parser.add_argument("--test-fraction", type=float, default=config.get("test_fraction", 0.0))
    parser.add_argument(
        "--split-group-by",
        choices=("none", "ecg_id", "record_path", "patient_from_path"),
        default=config.get("split_group_by", "patient_from_path"),
        help=(
            "Keep all rows from the same group in one split. "
            "For MIMIC paths, patient_from_path extracts IDs such as p10000764."
        ),
    )
    parser.add_argument(
        "--patient-id-regex",
        default=config.get("patient_id_regex", r"(p\d{8,})"),
        help="Regex used by split_group_by=patient_from_path.",
    )
    parser.add_argument("--target-sample-rate", type=float, default=config.get("target_sample_rate", 500.0))
    parser.add_argument("--target-length", type=int, default=config.get("target_length", 5000))
    parser.add_argument("--notch-freq", type=float, default=config.get("notch_freq", 60.0))
    parser.add_argument("--bandpass-freq", type=float, nargs=2, default=config.get("bandpass_freq", (0.01, 150)))
    parser.add_argument("--filter-wf", action=argparse.BooleanOptionalAction,
                        default=config.get("filter_wf", False),
                        help="Apply ECGFlow notch/bandpass filter.")
    parser.add_argument("--scale-source", choices=("ptbxl", "mimic"), default=config.get("scale_source", "ptbxl"))
    parser.add_argument("--input-channels", type=int, choices=(8, 12), default=config.get("input_channels", 8))
    parser.add_argument("--swap-avl-avf", action=argparse.BooleanOptionalAction,
                        default=config.get("swap_avl_avf", False),
                        help="Swap aVL/aVF before channel trimming.")
    parser.add_argument("--log-wandb", action=argparse.BooleanOptionalAction,
                        default=config.get("log_wandb", False),
                        help="Log train and validation metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", default=config.get("wandb_project"),
                        help="Weights & Biases project name.")
    parser.add_argument("--wandb-run-name", default=config.get("wandb_run_name"),
                        help="Weights & Biases run name.")
    parser.add_argument("--wandb-entity", default=config.get("wandb_entity"),
                        help="Weights & Biases entity/team.")
    parser.add_argument("--wandb-tags", nargs="*", default=config.get("wandb_tags", []),
                        help="Weights & Biases tags.")
    parser.add_argument("--wandb-skip-system-info", action=argparse.BooleanOptionalAction,
                        default=config.get("wandb_skip_system_info", True),
                        help="Disable W&B system stats, machine info, metadata, git, and code logging.")
    parser.add_argument(
        "--cuda-visible-devices",
        default=config.get("cuda_visible_devices"),
        help=(
            "Set CUDA_VISIBLE_DEVICES for this training process before PyTorch "
            "is imported, e.g. '0', '1', or '0,2'."
        ),
    )
    parser.add_argument("--distributed", action=argparse.BooleanOptionalAction,
                        default=config.get("distributed", True),
                        help="Launch multi-process DDP when multiple CUDA devices are visible.")
    parser.add_argument("--distributed-backend", default=config.get("distributed_backend", "nccl"),
                        help="Distributed process-group backend passed to the trainer.")
    parser.add_argument("--nccl-safe-mode", action=argparse.BooleanOptionalAction,
                        default=config.get("nccl_safe_mode", True),
                        help="Set conservative NCCL env vars for container multi-GPU stability.")
    parser.add_argument(
        "--extra-train-args",
        nargs="*",
        default=config.get("extra_train_args", []),
        help="Additional arguments passed through to scripts/train.py.",
    )

    args, passthrough = parser.parse_known_args(remaining)
    missing = []
    if not args.metadata_file:
        missing.append("metadata_file")
    if not args.pretrained_path:
        missing.append("pretrained_path")
    if missing:
        parser.error(
            "missing required argument(s): "
            + ", ".join(missing)
            + ". Provide them in config.json or on the command line."
        )
    if not isinstance(args.extra_train_args, list):
        parser.error("extra_train_args must be a list of command-line tokens")
    if args.wandb_tags is None:
        args.wandb_tags = []
    if not isinstance(args.wandb_tags, list):
        parser.error("wandb_tags must be a list of strings")
    args.distributed_env = config.get("distributed_env", {})
    if not isinstance(args.distributed_env, dict):
        parser.error("distributed_env must be a JSON object")
    args.wandb_run_name = _format_run_name(args.wandb_run_name)
    if args.wandb_run_name:
        args.experiment = args.wandb_run_name
    elif not args.experiment:
        args.experiment = "custom_lvef"
    passthrough = [str(item) for item in args.extra_train_args] + passthrough
    return args, passthrough


def main():
    args, passthrough = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    _launch_distributed_if_needed(args)

    metadata_file = Path(args.metadata_file).expanduser()
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else metadata_file.parent

    CustomLvefConfig.metadata_file = metadata_file
    CustomLvefConfig.data_dir = data_dir
    CustomLvefConfig.val_fraction = args.val_fraction
    CustomLvefConfig.test_fraction = args.test_fraction
    CustomLvefConfig.seed = args.seed
    CustomLvefConfig.target_sample_rate = args.target_sample_rate
    CustomLvefConfig.target_length = args.target_length
    CustomLvefConfig.scale_source = args.scale_source
    CustomLvefConfig.swap_avl_avf = args.swap_avl_avf
    CustomLvefConfig.split_group_by = args.split_group_by
    CustomLvefConfig.patient_id_regex = args.patient_id_regex

    _patch_ecgflow_dataset_registration()
    train_module = _load_train_module()

    train_argv = [
        "train.py",
        "--seed", str(args.seed),
        "--data-dir", data_dir.as_posix(),
        "--train-split", "train",
        "--val-split", "validation",
        "--early-stop", str(args.early_stop),
        "--ft-top", str(args.ft_top),
        "--pretrained",
        "--pretrained-path", str(Path(args.pretrained_path).expanduser()),
        "--dataset", f"ecgflow/{CUSTOM_DATASET_NAME}",
        "--no-aug",
        "--no-prefetcher",
        "--model", args.model,
        "--model-kwargs", f"img_size={args.target_length}",
        "--input-size", "1", str(args.input_channels), str(args.target_length),
        "--num-classes", "2",
        "--warmup-epochs", str(args.warmup_epochs),
        "--opt", args.opt,
        "--epochs", str(args.epochs),
        "--bce-loss",
        "--smoothing", "0.001",
        "--sched", args.sched,
        "--lr", str(args.lr),
        "--lr-k-decay", "1",
        "--patience-epochs", str(args.patience_epochs),
        "--cooldown-epochs", str(args.cooldown_epochs),
        "--decay-rate", str(args.decay_rate),
        "--weight-decay", str(args.weight_decay),
        "--batch-size", str(args.batch_size),
        "--eval-metric", args.eval_metric,
        "--binary-metric-threshold", str(args.metric_threshold),
        "--target-sensitivity", str(args.target_sensitivity),
        "--workers", str(args.workers),
        "--checkpoint-hist", str(args.checkpoint_hist),
        "--dist-backend", args.distributed_backend,
        "--notch-freq", str(args.notch_freq),
        "--bandpass-freq", str(args.bandpass_freq[0]), str(args.bandpass_freq[1]),
        "--output", str(Path(args.output).expanduser()),
        "--experiment", args.experiment,
    ]
    if args.sched_on_updates:
        train_argv.append("--sched-on-updates")
    if args.use_demographics:
        train_argv.extend([
            "--use-demographics",
            "--tabular-hidden-dim", str(args.tabular_hidden_dim),
            "--tabular-dropout", str(args.tabular_dropout),
        ])
    if args.bce_pos_weight is not None:
        train_argv.extend(["--bce-pos-weight", str(args.bce_pos_weight)])
    if args.filter_wf:
        train_argv.append("--filter-wf")
    if args.log_wandb:
        train_argv.append("--log-wandb")
    if args.wandb_project:
        train_argv.extend(["--wandb-project", args.wandb_project])
    if args.wandb_run_name:
        train_argv.extend(["--wandb-run-name", args.wandb_run_name])
    if args.wandb_entity:
        train_argv.extend(["--wandb-entity", args.wandb_entity])
    if args.wandb_tags:
        train_argv.append("--wandb-tags")
        train_argv.extend([str(tag) for tag in args.wandb_tags])
    if args.wandb_skip_system_info:
        train_argv.append("--wandb-skip-system-info")
    if args.log_wandb:
        wandb_config = {
            "custom_config": _json_safe(vars(args)),
            "custom_passthrough_args": _json_safe(passthrough),
        }
        train_argv.extend(["--wandb-config-json", json.dumps(wandb_config)])
    train_argv.extend(passthrough)

    old_argv = sys.argv
    try:
        sys.argv = train_argv
        train_module.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
