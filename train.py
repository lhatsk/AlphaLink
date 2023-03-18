import argparse
import logging
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
#os.environ["MASTER_ADDR"]="10.119.81.14"
#os.environ["MASTER_PORT"]="42069"
#os.environ["NODE_RANK"]="0"
from functools import partial

import random
import time

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins.training_type import DeepSpeedPlugin
from pytorch_lightning.plugins.environments import SLURMEnvironment
import torch

from openfold.config_refinement import model_config
from openfold.data.data_modules import (
    OpenFoldDataModule,
    DummyDataLoader,
    OpenFoldSingleDataset,
    OpenFoldBatchCollator,
)
from openfold.model.model import AlphaFold
from openfold.model.torchscript import script_preset_
from openfold.utils.callbacks import (
    EarlyStoppingVerbose,
)
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.argparse import remove_arguments
from openfold.utils.loss import AlphaFoldLoss
from openfold.utils.seed import seed_everything
from openfold.utils.tensor_utils import tensor_tree_map
from scripts.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint
)

from openfold.utils.import_weights import (
    import_jax_weights_,
)

from openfold.utils.logger import PerformanceLoggingCallback
import torch.optim as optim


class OpenFoldWrapper(pl.LightningModule):
    def __init__(self, config, model):
        super(OpenFoldWrapper, self).__init__()
        self.config = config
        self.model = model #AlphaFold(config)
        self.loss = AlphaFoldLoss(config.loss)
        self.ema = ExponentialMovingAverage(
            model=self.model, decay=config.ema.decay
        )
        self.cached_weights = None

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        if(self.ema.device != batch["aatype"].device):
            self.ema.to(batch["aatype"].device)

        # Run the model
        outputs = self(batch)
        
        # Remove the recycling dimension
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # Compute loss
        loss = self.loss(outputs, batch)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # At the start of validation, load the EMA weights
        if(self.cached_weights is None):
            self.cached_weights = self.model.state_dict()
            self.model.load_state_dict(self.ema.state_dict()["params"])
        
        # Calculate validation loss
        outputs = self(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)
        loss = self.loss(outputs, batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, _):
        # Restore the model weights to normal
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def configure_optimizers(self, 
        learning_rate: float = 1e-3,
        eps: float = 1e-8
    ) -> torch.optim.Adam:
        # Ignored as long as a DeepSpeed optimizer is configured
        return torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            eps=eps
        )

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()


def _gen_batch_collator(batch_seed, config, stage):
    """ We want each process to use the same batch collation seed """
    generator = torch.Generator()
    if(batch_seed is not None):
        generator = generator.manual_seed(batch_seed)
    collate_fn = OpenFoldBatchCollator(
        config, generator, stage
    )
    return collate_fn

def main(args):
    if(args.seed is not None):
        seed_everything(args.seed) 

    config = model_config(
        "model_5", 
        train=True, 
        low_prec=(args.precision == 16)
    ) 
    

    model = AlphaFold(config)   
    
    import_jax_weights_(model, "openfold/params/params_model_5.npz", version="model_5")

    train_dataset_gen = partial(OpenFoldSingleDataset,
        template_mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        config=config.data,
        kalign_binary_path=args.kalign_binary_path,
        template_release_dates_cache_path=
            args.template_release_dates_cache_path,
        obsolete_pdbs_file_path=None,
    )

    train_dataset = train_dataset_gen(
            data_dir=args.train_data_dir,
            alignment_dir=args.train_alignment_dir,
            mapping_path=args.train_mapping_path,
            feature_dir='/localscratch/openfold_features',
            crosslink_dir='/localscratch/openfold_xl',
            max_template_hits=config.data.train.max_template_hits,
            shuffle_top_k_prefiltered=
                config.data.train.shuffle_top_k_prefiltered,
            treat_pdb_as_distillation=False,
            mode="train",
            _output_raw=True,
        )

    val_dataset_gen = partial(OpenFoldSingleDataset,
        template_mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        config=config.data,
        kalign_binary_path=args.kalign_binary_path,
        template_release_dates_cache_path=
            args.template_release_dates_cache_path,
        obsolete_pdbs_file_path=None,
    )

    val_dataset = val_dataset_gen(
                data_dir=args.val_data_dir,
                alignment_dir=args.val_alignment_dir,
                mapping_path=None,
                feature_dir='/localscratch/openfold_validation_features',
                crosslink_dir='/localscratch/openfold_validation_xl',
                max_template_hits=config.data.eval.max_template_hits,
                mode="val",
                _output_raw=True,
            )


    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=8, shuffle=True, collate_fn=_gen_batch_collator(args.seed, config.data, "train"))

    validation_loader = torch.utils.data.DataLoader(val_dataset, num_workers=8, collate_fn=_gen_batch_collator(args.seed, config.data, "eval"))




    script_preset_(model)

    model = model.cuda()

    loss = AlphaFoldLoss(config.loss)
    ema = ExponentialMovingAverage(
            model=model, decay=config.ema.decay)

    optimizer = optim.Adam(model.parameters(), lr = 5e-4)#, weight_decay=1e-3)
    # optimizer = optim.AdamW(model.parameters(), lr = 5e-4, weight_decay=1e-3)

    n_training = len(train_dataset)

    n_validation = len(val_dataset)

    best = float("inf")

    print(n_training, n_validation)

    scaler = torch.cuda.amp.GradScaler()

    import time

    for e in range(5000):
        model.train()
        for X in train_loader:
            optimizer.zero_grad() # zero the gradient buffers

            start = time.time()


            # with torch.cuda.amp.autocast():
            X = { k: v.cuda() for k,v in X.items() }
            outputs = model(X)

            batch = tensor_tree_map(lambda t: t[..., -1], X)

            # Compute loss
            loss_ = loss(outputs, batch)
            loss_.backward()
            optimizer.step()
            # scaler.scale(loss_).backward()
            # scaler.step(optimizer)
            # scaler.update()
            end = time.time()
            print(loss_.item(), end - start)

        ema.update(model)


def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data_dir", type=str,
        help="Directory containing training mmCIF files"
    )
    parser.add_argument(
        "train_alignment_dir", type=str,
        help="Directory containing precomputed training alignments"
    )
    parser.add_argument(
        "template_mmcif_dir", type=str,
        help="Directory containing mmCIF files to search for templates"
    )
    parser.add_argument(
        "output_dir", type=str,
        help='''Directory in which to output checkpoints, logs, etc. Ignored
                if not on rank 0'''
    )
    parser.add_argument(
        "max_template_date", type=str,
        help='''Cutoff for all templates. In training mode, templates are also 
                filtered by the release date of the target'''
    )
    parser.add_argument(
        "--distillation_data_dir", type=str, default=None,
        help="Directory containing training PDB files"
    )
    parser.add_argument(
        "--distillation_alignment_dir", type=str, default=None,
        help="Directory containing precomputed distillation alignments"
    )
    parser.add_argument(
        "--val_data_dir", type=str, default=None,
        help="Directory containing validation mmCIF files"
    )
    parser.add_argument(
        "--val_alignment_dir", type=str, default=None,
        help="Directory containing precomputed validation alignments"
    )
    parser.add_argument(
        "--kalign_binary_path", type=str, default='/usr/bin/kalign',
        help="Path to the kalign binary"
    )
    parser.add_argument(
        "--train_mapping_path", type=str, default=None,
        help='''Optional path to a .json file containing a mapping from
                consecutive numerical indices to sample names. Used to filter
                the training set'''
    )
    parser.add_argument(
        "--distillation_mapping_path", type=str, default=None,
        help="""See --train_mapping_path"""
    )
    parser.add_argument(
        "--template_release_dates_cache_path", type=str, default=None,
        help="""Output of scripts/generate_mmcif_cache.py run on template mmCIF
                files."""
    )
    parser.add_argument(
        "--use_small_bfd", type=bool_type, default=False,
        help="Whether to use a reduced version of the BFD database"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--deepspeed_config_path", type=str, default=None,
        help="Path to DeepSpeed config. If not provided, DeepSpeed is disabled"
    )
    parser.add_argument(
        "--checkpoint_best_val", type=bool_type, default=True,
        help="""Whether to save the model parameters that perform best during
                validation"""
    )
    parser.add_argument(
        "--early_stopping", type=bool_type, default=False,
        help="Whether to stop training when validation loss fails to decrease"
    )
    parser.add_argument(
        "--min_delta", type=float, default=0,
        help="""The smallest decrease in validation loss that counts as an 
                improvement for the purposes of early stopping"""
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--resume_from_ckpt", type=str, default=None,
        help="Path to a model checkpoint from which to restore training state"
    )
    parser.add_argument(
        "--resume_model_weights_only", type=bool_type, default=False,
        help="Whether to load just model weights as opposed to training state"
    )
    parser.add_argument(
        "--log_performance", action='store_true',
        help="Measure performance"
    )
    parser = pl.Trainer.add_argparse_args(parser)
   
    # Disable the initial validation pass
    parser.set_defaults(
        num_sanity_val_steps=0,
    )

    # Remove some buggy/redundant arguments introduced by the Trainer
    remove_arguments(parser, ["--accelerator", "--resume_from_checkpoint"]) 

    args = parser.parse_args()

    if(args.seed is None and 
        ((args.gpus is not None and args.gpus > 1) or 
         (args.num_nodes is not None and args.num_nodes > 1))):
        raise ValueError("For distributed training, --seed must be specified")

    main(args)
