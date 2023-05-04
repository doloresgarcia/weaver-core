#!/usr/bin/env python

import os
import ast
import sys
import shutil
import glob
import argparse
import functools
import numpy as np
import math
import torch
from torch.utils.data import DataLoader
from weaver.utils.logger import _logger, _configLogger
from weaver.utils.dataset import SimpleIterDataset
from weaver.utils.import_tools import import_module
from weaver.utils.parser_args import parser
from weaver.utils.trainer_utils import (
    iotest,
    save_root,
    save_parquet,
    model_setup,
    to_filelist,
    train_load,
    test_load,
    onnx,
    flops,
    profile,
    optim,
)


def _main(args):
    _logger.info("args:\n - %s", "\n - ".join(str(it) for it in args.__dict__.items()))

    # export to ONNX
    if args.export_onnx:
        onnx(args)
        return

    if args.file_fraction < 1:
        _logger.warning(
            "Use of `file-fraction` is not recommended in general -- prefer using `data-fraction` instead."
        )

    # classification/regression mode
    if args.regression_mode:
        _logger.info("Running in regression mode")
        from weaver.utils.nn.tools import train_regression as train
        from weaver.utils.nn.tools import evaluate_regression as evaluate
    elif args.graphs:
        from weaver.utils.nn.tools_san import train_classification as train
        from weaver.utils.nn.tools_san import evaluate_classification as evaluate
    else:
        _logger.info("Running in classification mode")
        from weaver.utils.nn.tools import train_classification as train
        from weaver.utils.nn.tools import evaluate_classification as evaluate

    # training/testing mode
    training_mode = not args.predict

    # device
    # if args.gpus:
    # distributed training
    #    if args.backend is not None:
    #       local_rank = args.local_rank
    #       torch.cuda.set_device(local_rank)
    #       gpus = [local_rank]
    #       dev = torch.device(local_rank)
    #       torch.distributed.init_process_group(backend=args.backend)
    #        _logger.info(f"Using distributed PyTorch with {args.backend} backend")
    #   else:
    #       gpus = [int(i) for i in args.gpus.split(",")]
    #       dev = torch.device(gpus[0])
    # else:
    gpus = None
    dev = torch.device("cpu")

    if training_mode:
        (
            train_loader,
            val_loader,
            data_config,
            train_input_names,
            train_label_names,
        ) = train_load(args)
    else:
        test_loaders, data_config = test_load(args)

    model, model_info, loss_func = model_setup(args, data_config)
    orig_model = model
    
    if args.profile:
        profile(args, model, model_info, device=dev)
        return
    
    if args.data_test:

        if not args.model_prefix.endswith(".onnx"):
            if args.predict_gpus:
                gpus = [int(i) for i in args.predict_gpus.split(",")]
                dev = torch.device(gpus[0])
            else:
                gpus = None
                dev = torch.device("cpu")
            model = orig_model.to(dev)
            model_path = (
                args.model_prefix
                if args.model_prefix.endswith(".pt")
                else args.model_prefix + "_best_epoch_state.pt"
            )
            
            model.load_state_dict(torch.load(model_path, map_location=dev))
            model = model.to(dev)

        for name, get_test_loader in test_loaders.items():
            test_loader = get_test_loader()
            # run prediction
            if args.model_prefix.endswith(".onnx"):
                _logger.info("Loading model %s for eval" % args.model_prefix)
                from weaver.utils.nn.tools import evaluate_onnx

                test_metric, scores, labels, observers = evaluate_onnx(
                    args.model_prefix, test_loader
                )
            else:
                test_metric, scores, labels, observers = evaluate(
                    model,
                    test_loader,
                    dev,
                    epoch=None,
                    for_training=False,
                    tb_helper=tb,
                )
            _logger.info("Test metric %.5f" % test_metric, color="bold")
            del test_loader

            if args.predict_output:
                if "/" not in args.predict_output:
                    predict_output = os.path.join(
                        os.path.dirname(args.model_prefix),
                        "predict_output",
                        args.predict_output,
                    )
                else:
                    predict_output = args.predict_output
                os.makedirs(os.path.dirname(predict_output), exist_ok=True)
                if name == "":
                    output_path = predict_output
                else:
                    base, ext = os.path.splitext(predict_output)
                    output_path = base + "_" + name + ext
                if output_path.endswith(".root"):
                    save_root(args, output_path, data_config, scores, labels, observers)
                else:
                    save_parquet(args, output_path, scores, labels, observers)
                _logger.info("Written output to %s" % output_path, color="bold")


def main():
    args = parser.parse_args()

    if args.samples_per_epoch is not None:
        if args.steps_per_epoch is None:
            args.steps_per_epoch = args.samples_per_epoch // args.batch_size
        else:
            raise RuntimeError(
                "Please use either `--steps-per-epoch` or `--samples-per-epoch`, but not both!"
            )

    if args.samples_per_epoch_val is not None:
        if args.steps_per_epoch_val is None:
            args.steps_per_epoch_val = args.samples_per_epoch_val // args.batch_size
        else:
            raise RuntimeError(
                "Please use either `--steps-per-epoch-val` or `--samples-per-epoch-val`, but not both!"
            )

    if args.steps_per_epoch_val is None and args.steps_per_epoch is not None:
        args.steps_per_epoch_val = round(
            args.steps_per_epoch * (1 - args.train_val_split) / args.train_val_split
        )
    if args.steps_per_epoch_val is not None and args.steps_per_epoch_val < 0:
        args.steps_per_epoch_val = None

    if "{auto}" in args.model_prefix or "{auto}" in args.log:
        import hashlib
        import time

        model_name = (
            time.strftime("%Y%m%d-%H%M%S")
            + "_"
            + os.path.basename(args.network_config).replace(".py", "")
        )
        if len(args.network_option):
            model_name = (
                model_name
                + "_"
                + hashlib.md5(str(args.network_option).encode("utf-8")).hexdigest()
            )
        model_name += "_{optim}_lr{lr}_batch{batch}".format(
            lr=args.start_lr, optim=args.optimizer, batch=args.batch_size
        )
        args._auto_model_name = model_name
        args.model_prefix = args.model_prefix.replace("{auto}", model_name)
        args.log = args.log.replace("{auto}", model_name)
        print("Using auto-generated model prefix %s" % args.model_prefix)

    if args.predict_gpus is None:
        args.predict_gpus = args.gpus

    args.local_rank = (
        None if args.backend is None else int(os.environ.get("LOCAL_RANK", "0"))
    )

    stdout = sys.stdout
    if args.local_rank is not None:
        args.log += ".%03d" % args.local_rank
        if args.local_rank != 0:
            stdout = None
    _configLogger("weaver", stdout=stdout, filename=args.log)

    if args.cross_validation:
        model_dir, model_fn = os.path.split(args.model_prefix)
        var_name, kfold = args.cross_validation.split("%")
        kfold = int(kfold)
        for i in range(kfold):
            _logger.info(f"\n=== Running cross validation, fold {i} of {kfold} ===")
            args.model_prefix = os.path.join(f"{model_dir}_fold{i}", model_fn)
            args.extra_selection = f"{var_name}%{kfold}!={i}"
            args.extra_test_selection = f"{var_name}%{kfold}=={i}"
            _main(args)
    else:
        _main(args)


if __name__ == "__main__":
    main()
