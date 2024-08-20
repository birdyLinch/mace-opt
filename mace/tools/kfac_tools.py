"""Utilities for getting optimizers for computer vision examples."""

from __future__ import annotations

import argparse
from typing import Callable

import torch
import torch.distributed as dist
import torch.optim as optim

import kfac


def get_kfac(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> tuple[
    kfac.preconditioner.KFACPreconditioner | None,
    kfac.scheduler.LambdaParamScheduler | None,
]:
    """Get optimizer, preconditioner, and scheduler."""
    use_kfac = True if args.kfac_inv_update_steps > 0 else False

    grad_worker_fraction: kfac.enums.DistributedStrategy | float
    if args.kfac_strategy == 'comm-opt':
        grad_worker_fraction = kfac.enums.DistributedStrategy.COMM_OPT
    elif args.kfac_strategy == 'mem-opt':
        grad_worker_fraction = kfac.enums.DistributedStrategy.MEM_OPT
    elif args.kfac_strategy == 'hybrid-opt':
        grad_worker_fraction = args.kfac_grad_worker_fraction
    else:
        raise ValueError(
            f'Unknown KFAC Comm Method: {args.kfac_strategy}',
        )

    if use_kfac:
        #import torch.distributed as dist
        #print(f"rank {dist.get_rank()}: start kfac.preconditioner.KFACPreconditioner -- kfac_tools.py")
        #import ipdb; ipdb.set_trace()
        preconditioner = kfac.preconditioner.KFACPreconditioner(
            model,
            factor_update_steps=args.kfac_factor_update_steps,
            inv_update_steps=args.kfac_inv_update_steps,
            damping=args.kfac_damping,
            factor_decay=args.kfac_factor_decay,
            kl_clip=args.kfac_kl_clip,
            lr=lambda x: optimizer.param_groups[0]['lr'],
            accumulation_steps=args.batches_per_allreduce,
            allreduce_bucket_cap_mb=25,
            colocate_factors=args.kfac_colocate_factors,
            compute_method=kfac.enums.ComputeMethod.INVERSE
            if args.kfac_inv_method
            else kfac.enums.ComputeMethod.EIGEN,
            grad_worker_fraction=grad_worker_fraction,
            grad_scaler=args.grad_scaler if 'grad_scaler' in args else None,
            skip_layers=args.kfac_skip_layers,
            update_factors_in_hook=False,
        )
        #print(preconditioner)
        #print(f"rank {dist.get_rank()}: finish kfac.preconditioner.KFACPreconditioner -- kfac_tools.py")

        def get_lambda(
            alpha: int,
            epochs: list[int] | None,
        ) -> Callable[[int], float]:
            """Create lambda function for param scheduler."""
            if epochs is None:
                _epochs = []
            else:
                _epochs = epochs

            def scale(epoch: int) -> float:
                """Compute current scale factor using epoch."""
                factor = 1.0
                for e in _epochs:
                    if epoch >= e:
                        factor *= alpha
                return factor

            return scale

        kfac_param_scheduler = kfac.scheduler.LambdaParamScheduler(
            preconditioner,
            damping_lambda=get_lambda(
                args.kfac_damping_alpha,
                args.kfac_damping_decay,
            ),
            factor_update_steps_lambda=get_lambda(
                args.kfac_update_steps_alpha,
                args.kfac_update_steps_decay,
            ),
            inv_update_steps_lambda=get_lambda(
                args.kfac_update_steps_alpha,
                args.kfac_update_steps_decay,
            ),
        )
    else:
        preconditioner = None
        kfac_param_scheduler = None

    return preconditioner, kfac_param_scheduler
