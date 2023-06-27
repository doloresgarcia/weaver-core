#!/usr/bin/env python

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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

    print("TESTING THIS", args.backend)
    # device
    if args.gpus:
        # distributed training
        if args.backend is not None:
            local_rank = args.local_rank
            torch.cuda.set_device(local_rank)
            gpus = [local_rank]
            dev = torch.device(local_rank)
            torch.distributed.init_process_group(backend=args.backend)
            _logger.info(f"Using distributed PyTorch with {args.backend} backend")
        else:
            gpus = [int(i) for i in args.gpus.split(",")]
            dev = torch.device(gpus[0])
            local_rank = 0
    else:
        gpus = None
        dev = torch.device("cpu")
        local_rank = 0

    # load data
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

    if args.io_test:
        data_loader = (
            train_loader if training_mode else list(test_loaders.values())[0]()
        )
        iotest(args, data_loader)
        return

    model, model_info, loss_func = model_setup(args, data_config)

    # TODO: load checkpoint
    # if args.backend is not None:
    #     load_checkpoint()

    if args.print:
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print("TOTAL PARAMETERS", pytorch_total_params)
        return

    if args.profile:
        profile(args, model, model_info, device=dev)
        return

    if args.tensorboard:
        from weaver.utils.nn.tools import TensorboardHelper

        tb = TensorboardHelper(
            tb_comment=args.tensorboard, tb_custom_fn=args.tensorboard_custom_fn
        )
    else:
        tb = None

    # note: we should always save/load the state_dict of the original model, not the one wrapped by nn.DataParallel
    # so we do not convert it to nn.DataParallel now
    orig_model = model

    if training_mode:
        if args.log_wandb and local_rank == 0:
            import wandb
            from weaver.utils.logger_wandb import log_wandb_init

            wandb.init(project=args.wandb_projectname, entity=args.wandb_entity)
            wandb.run.name = args.wandb_displayname
            log_wandb_init(args)

        model = orig_model.to(dev)
        print("MODEL DEVICE", next(model.parameters()).is_cuda)

        # DistributedDataParallel
        if args.backend is not None:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=gpus, output_device=local_rank
            )

        # optimizer & learning rate
        opt, scheduler = optim(args, model, dev)

        # DataParallel
        if args.backend is None:
            if gpus is not None and len(gpus) > 1:
                # model becomes `torch.nn.DataParallel` w/ model.module being the original `torch.nn.Module`
                model = torch.nn.DataParallel(model, device_ids=gpus)
            # model = model.to(dev)

        # lr finder: keep it after all other setups
        if args.lr_finder is not None:
            start_lr, end_lr, num_iter = args.lr_finder.replace(" ", "").split(",")
            from weaver.utils.lr_finder import LRFinder

            lr_finder = LRFinder(
                model,
                opt,
                loss_func,
                device=dev,
                input_names=train_input_names,
                label_names=train_label_names,
            )
            lr_finder.range_test(
                train_loader,
                start_lr=float(start_lr),
                end_lr=float(end_lr),
                num_iter=int(num_iter),
            )
            lr_finder.plot(
                output="lr_finder.png"
            )  # to inspect the loss-learning rate graph
            return

        # training loop
        best_valid_metric = np.inf if args.regression_mode else 0
        grad_scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
        for epoch in range(args.num_epochs):
            if args.load_epoch is not None:
                if epoch <= args.load_epoch:
                    continue
            _logger.info("-" * 50)
            _logger.info("Epoch #%d training" % epoch)
            # print('number of steps per epoch', args.steps_per_epoch)
            train(
                model,
                loss_func,
                opt,
                scheduler,
                train_loader,
                dev,
                epoch,
                steps_per_epoch=args.steps_per_epoch,
                grad_scaler=grad_scaler,
                tb_helper=tb,
                logwandb=args.log_wandb,
                local_rank=local_rank,
                args=args,
            )
            if args.model_prefix and (args.backend is None or local_rank == 0):
                dirname = os.path.dirname(args.model_prefix)
                if dirname and not os.path.exists(dirname):
                    os.makedirs(dirname)
                state_dict = (
                    model.module.state_dict()
                    if isinstance(
                        model,
                        (
                            torch.nn.DataParallel,
                            torch.nn.parallel.DistributedDataParallel,
                        ),
                    )
                    else model.state_dict()
                )
                torch.save(state_dict, args.model_prefix + "_epoch-%d_state.pt" % epoch)
                torch.save(
                    opt.state_dict(),
                    args.model_prefix + "_epoch-%d_optimizer.pt" % epoch,
                )
            # if args.backend is not None and local_rank == 0:
            # TODO: save checkpoint
            #     save_checkpoint()

            _logger.info("Epoch #%d validating" % epoch)
            valid_metric = evaluate(
                model,
                val_loader,
                dev,
                epoch,
                loss_func=loss_func,
                steps_per_epoch=args.steps_per_epoch_val,
                tb_helper=tb,
                logwandb=args.log_wandb,
                local_rank=local_rank,
            )
            is_best_epoch = (
                (valid_metric < best_valid_metric)
                if args.regression_mode
                else (valid_metric > best_valid_metric)
            )
            if is_best_epoch:
                best_valid_metric = valid_metric
                if args.model_prefix and (args.backend is None or local_rank == 0):
                    shutil.copy2(
                        args.model_prefix + "_epoch-%d_state.pt" % epoch,
                        args.model_prefix + "_best_epoch_state.pt",
                    )
                    # torch.save(model, args.model_prefix + '_best_epoch_full.pt')
            _logger.info(
                "Epoch #%d: Current validation metric: %.5f (best: %.5f)"
                % (epoch, valid_metric, best_valid_metric),
                color="bold",
            )

    if args.data_test:
        if args.backend is not None and local_rank != 0:
            return
        if training_mode:
            del train_loader, val_loader
            test_loaders, data_config = test_load(args)

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
            _logger.info("Loading model %s for eval" % model_path)
            model.load_state_dict(torch.load(model_path, map_location=dev))
            if gpus is not None and len(gpus) > 1:
                model = torch.nn.DataParallel(model, device_ids=gpus)
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
    if args.backend is not None:
        port = find_free_port()
        args.port = port
        world_size = torch.cuda.device_count()
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


def find_free_port():
    """https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number"""
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


if __name__ == "__main__":
    main()
