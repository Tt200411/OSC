import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Pred
from exp.exp_config import InformerConfig
from exp.exp_informer import Exp_Informer
from models.activations import activation_family


ETT_DEFAULTS = {
    "ETTh1": ("./ETT-small", "ETTh1.csv", "OT", "h"),
    "ETTh2": ("./ETT-small", "ETTh2.csv", "OT", "h"),
    "ETTm1": ("./ETT-small", "ETTm1.csv", "OT", "t"),
    "ETTm2": ("./ETT-small", "ETTm2.csv", "OT", "t"),
}

SOLAR_DEFAULTS = {
    "Solar": ("../Solar", "PV_Solar_Station_1.csv", "Power", "t"),
    "Solar1": ("../Solar", "Site_1_50MW.csv", "Power", "t"),
    "Solar2": ("../Solar", "Site_2_130MW.csv", "Power", "t"),
    "Solar3": ("../Solar", "Site_3_30MW.csv", "Power", "t"),
    "Solar4": ("../Solar", "Site_4_130MW.csv", "Power", "t"),
    "Solar5": ("../Solar", "Site_5_110MW.csv", "Power", "t"),
    "Solar6": ("../Solar", "Site_6_35MW.csv", "Power", "t"),
    "Solar7": ("../Solar", "Site_7_30MW.csv", "Power", "t"),
    "Solar8": ("../Solar", "Site_8_30MW.csv", "Power", "t"),
}

DATASET_DEFAULTS = {
    **{
        name: {"root_path": root, "data_path": data_path, "target": target, "freq": freq}
        for name, (root, data_path, target, freq) in ETT_DEFAULTS.items()
    },
    **{
        name: {"root_path": root, "data_path": data_path, "target": target, "freq": freq}
        for name, (root, data_path, target, freq) in SOLAR_DEFAULTS.items()
    },
}

ACTIVATION_CHOICES = [
    "gelu",
    "relu",
    "swish",
    "silu",
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
]


class Phase4ExpInformer(Exp_Informer):
    def _get_data(self, flag):
        args = self.config
        if args.data in {"ETTh1", "ETTh2"}:
            Data = Dataset_ETT_hour
        elif args.data in {"ETTm1", "ETTm2"}:
            Data = Dataset_ETT_minute
        elif args.data.startswith("Solar") or args.data in {"WTH", "ECL", "custom"}:
            Data = Dataset_Custom
        else:
            raise ValueError(f"Unsupported Phase-4 dataset: {args.data}")

        timeenc = 0 if args.embed != "timeF" else 1
        if flag == "test":
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == "pred":
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
        return data_set, data_loader


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
    parser = argparse.ArgumentParser(description="Phase-4 Informer activation experiment runner")
    parser.add_argument("--model", default=None, choices=["informer", "informerstack"])
    parser.add_argument(
        "--data",
        default=None,
        choices=[
            "ETTh1",
            "ETTh2",
            "ETTm1",
            "ETTm2",
            "Solar",
            "Solar1",
            "Solar2",
            "Solar3",
            "Solar4",
            "Solar5",
            "Solar6",
            "Solar7",
            "Solar8",
            "custom",
            "WTH",
            "ECL",
        ],
    )
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
    parser.add_argument("--activation", default=None, choices=ACTIVATION_CHOICES)
    parser.add_argument("--encoder_activation", default=None, choices=ACTIVATION_CHOICES)
    parser.add_argument("--decoder_activation", default=None, choices=ACTIVATION_CHOICES)
    parser.add_argument("--output_activation", default=None, choices=["linear", "tanh"])
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
    config.encoder_activation = (config.encoder_activation or config.activation).lower()
    config.decoder_activation = (config.decoder_activation or config.activation).lower()
    config.output_activation = (config.output_activation or "linear").lower()
    config.activation_signature = (
        f"act={config.activation}|enc={config.encoder_activation}|"
        f"dec={config.decoder_activation}|out={config.output_activation}|"
        f"a={config.perturb_amplitude:g}|lee={config.lee_type}"
    )
    config.activation_family = activation_family(config.activation)
    config.use_lee = config.activation == "lee"
    config.lee_oscillator = config.use_lee
    config.save_pred = not args.no_save_pred
    config.save_true = not args.no_save_true

    if args.encoder_lee_types is not None:
        config.encoder_lee_types = parse_int_list(args.encoder_lee_types)
    else:
        config.encoder_lee_types = [config.lee_type] * config.e_layers
    if args.decoder_lee_types is not None:
        config.decoder_lee_types = parse_int_list(args.decoder_lee_types)
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_setting(config):
    parts = ["informer", config.data, config.activation, f"a{config.perturb_amplitude:g}"]
    if config.activation == "lee":
        parts.append(f"lee{config.lee_type}")
    if (
        config.encoder_activation != config.activation
        or config.decoder_activation != config.activation
        or config.output_activation != "linear"
    ):
        parts.append(
            "enc{}_dec{}_out{}".format(
                config.encoder_activation,
                config.decoder_activation,
                config.output_activation,
            )
        )
    parts.extend(
        [
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
        ]
    )
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

    exp = Phase4ExpInformer(config)
    setting = build_setting(config)

    if args.smoke_test:
        exp.smoke_test_data(flag=args.smoke_flag)
        return

    if config.lee_grid_search:
        print(">>>>>>>开始Lee振荡器网格搜索>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        best_config = exp.grid_search()
        config.encoder_lee_types = list(best_config["encoder_types"])
        config.decoder_lee_types = list(best_config["decoder_types"])
        exp = Phase4ExpInformer(config)
        setting = best_config["setting"]

    if not args.test_only:
        print(f">>>>>>>开始训练 : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        exp.train(setting)
    if not args.train_only:
        print(f">>>>>>>测试 : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting)


if __name__ == "__main__":
    main()
