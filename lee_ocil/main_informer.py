import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from exp.exp_config import InformerConfig
from exp.exp_informer import Exp_Informer
from models.activations import activation_family


DATASET_DEFAULTS = {
    "ETTh1": {
        "root_path": "./ETT-small",
        "data_path": "ETTh1.csv",
        "target": "OT",
        "freq": "h",
    },
    "ETTh2": {
        "root_path": "./ETT-small",
        "data_path": "ETTh2.csv",
        "target": "OT",
        "freq": "h",
    },
    "Solar": {
        "root_path": "../Solar",
        "data_path": "PV_Solar_Station_1.csv",
        "target": "Power",
        "freq": "t",
    },
    "Solar1": {
        "root_path": "../Solar",
        "data_path": "Site_1_50MW.csv",
        "target": "Power",
        "freq": "t",
    },
    "Solar5": {
        "root_path": "../Solar",
        "data_path": "Site_5_110MW.csv",
        "target": "Power",
        "freq": "t",
    },
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value}")


def parse_int_list(value):
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_parser():
    parser = argparse.ArgumentParser(description="Informer activation experiment runner")

    parser.add_argument("--model", default=None, choices=["informer", "informerstack"])
    parser.add_argument("--data", default=None, choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Solar", "Solar1", "Solar5", "custom", "WTH", "ECL"])
    parser.add_argument("--root_path", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--features", default=None, choices=["M", "S", "MS"])
    parser.add_argument("--target", default=None)
    parser.add_argument("--freq", default=None)
    parser.add_argument("--detail_freq", default=None)
    parser.add_argument("--cols", default=None)

    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--label_len", type=int, default=None)
    parser.add_argument("--pred_len", type=int, default=None)
    parser.add_argument("--enc_in", type=int, default=None)
    parser.add_argument("--dec_in", type=int, default=None)
    parser.add_argument("--c_out", type=int, default=None)

    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--e_layers", type=int, default=None)
    parser.add_argument("--d_layers", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--factor", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--attn", default=None, choices=["prob", "full"])
    parser.add_argument("--embed", default=None)
    parser.add_argument("--distil", type=str2bool, default=None)
    parser.add_argument("--mix", type=str2bool, default=None)
    parser.add_argument("--output_attention", type=str2bool, default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--train_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--itr", type=int, default=None)
    parser.add_argument("--lradj", default=None)
    parser.add_argument("--use_amp", type=str2bool, default=None)

    parser.add_argument("--activation", default=None,
                        choices=[
                            "gelu",
                            "relu",
                            "tanh",
                            "softsign",
                            "scaled_tanh",
                            "gelu_sin",
                            "relu_sin",
                            "tanh_sin",
                            "gelu_cos",
                            "relu_cos",
                            "tanh_cos",
                            "tanh_rand",
                            "lee",
                            "dynamic_gelu_sin",
                        ])
    parser.add_argument("--perturb_amplitude", "--amplitude", type=float, default=None)
    parser.add_argument("--perturb_frequency", "--frequency", type=float, default=None)
    parser.add_argument("--perturb_phase", "--phase", type=float, default=None)
    parser.add_argument("--lee_type", type=int, default=None)
    parser.add_argument("--encoder_lee_types", default=None)
    parser.add_argument("--decoder_lee_types", default=None)
    parser.add_argument("--lee_grid_search", type=str2bool, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_gpu", type=str2bool, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--use_multi_gpu", type=str2bool, default=None)

    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--logs_dir", default=None)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--server_id", default=None)
    parser.add_argument("--server_ip", default=None)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--des", default=None)

    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--smoke_flag", default="train", choices=["train", "val", "test"])
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--no_save_pred", action="store_true")
    parser.add_argument("--no_save_true", action="store_true")
    return parser


def apply_overrides(config, args):
    for key, value in vars(args).items():
        if value is None or key in {"smoke_flag", "no_save_pred", "no_save_true"}:
            continue
        if hasattr(config, key):
            setattr(config, key, value)

    defaults = DATASET_DEFAULTS.get(config.data, {})
    if args.root_path is None and "root_path" in defaults:
        config.root_path = defaults["root_path"]
    if args.data_path is None and "data_path" in defaults:
        config.data_path = defaults["data_path"]
    if args.target is None and "target" in defaults:
        config.target = defaults["target"]
    if args.freq is None and "freq" in defaults:
        config.freq = defaults["freq"]
    if args.detail_freq is None:
        config.detail_freq = config.freq

    config.activation = config.activation.lower()
    config.activation_family = activation_family(config.activation)
    config.use_lee = config.activation == "lee"
    config.lee_oscillator = config.use_lee
    config.save_pred = not args.no_save_pred
    config.save_true = not args.no_save_true

    if args.encoder_lee_types is not None:
        config.encoder_lee_types = parse_int_list(args.encoder_lee_types)
    elif config.activation == "lee":
        config.encoder_lee_types = [config.lee_type] * config.e_layers
    else:
        config.encoder_lee_types = [config.lee_type] * config.e_layers

    if args.decoder_lee_types is not None:
        config.decoder_lee_types = parse_int_list(args.decoder_lee_types)
    elif config.activation == "lee":
        config.decoder_lee_types = [config.lee_type] * config.d_layers
    else:
        config.decoder_lee_types = [config.lee_type] * config.d_layers

    if config.run_id is None:
        config.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    config.results_dir = os.path.normpath(config.results_dir)
    config.logs_dir = os.path.normpath(config.logs_dir)
    config.summary_path = os.path.normpath(config.summary_path)
    config.checkpoints = os.path.normpath(config.checkpoints)

    infer_dimensions(config, args)
    config.use_gpu = bool(torch.cuda.is_available() and config.use_gpu)
    return config


def infer_dimensions(config, args):
    csv_path = os.path.join(config.root_path, config.data_path)
    if os.path.exists(csv_path):
        columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
        if config.features in {"M", "MS"}:
            feature_count = len([column for column in columns if column != "date"])
            output_count = 1 if config.features == "MS" else feature_count
        else:
            feature_count = 1
            output_count = 1
        if args.enc_in is None:
            config.enc_in = feature_count
        if args.dec_in is None:
            config.dec_in = feature_count
        if args.c_out is None:
            config.c_out = output_count


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_setting(config):
    parts = [
        "informer",
        config.data,
        config.activation,
        f"a{config.perturb_amplitude:g}",
    ]
    if config.activation == "lee":
        parts.append(f"lee{config.lee_type}")
    parts.extend([
        f"ft{config.features}",
        f"sl{config.seq_len}",
        f"ll{config.label_len}",
        f"pl{config.pred_len}",
        f"dm{config.d_model}",
        f"nh{config.n_heads}",
        f"el{config.e_layers}",
        f"dl{config.d_layers}",
        f"df{config.d_ff}",
        f"fc{config.factor}",
        f"seed{config.seed}",
        config.server_id,
    ])
    if config.des:
        parts.append(config.des)
    return "_".join(str(part).replace("/", "-") for part in parts)


def dump_runtime_config(config):
    payload = {
        key: value
        for key, value in vars(config).items()
        if isinstance(value, (str, int, float, bool, list, tuple)) or value is None
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = apply_overrides(InformerConfig(), args)
    set_seed(config.seed)
    dump_runtime_config(config)

    exp = Exp_Informer(config)
    setting = build_setting(config)

    if args.smoke_test:
        exp.smoke_test_data(flag=args.smoke_flag)
        return

    if config.lee_grid_search:
        print(">>>>>>>开始Lee振荡器网格搜索>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        best_config = exp.grid_search()
        config.encoder_lee_types = list(best_config["encoder_types"])
        config.decoder_lee_types = list(best_config["decoder_types"])
        exp = Exp_Informer(config)
        setting = best_config["setting"]

    if not args.test_only:
        print(f">>>>>>>开始训练 : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        exp.train(setting)

    if not args.train_only:
        print(f">>>>>>>测试 : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting)


if __name__ == "__main__":
    main()
