# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
from copy import deepcopy
from datetime import date
import logging
import math
import numpy as np
import os

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

import pickle
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict
)
import random
import sys
import time
import torch
import re

torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if(
    torch_major_version > 1 or 
    (torch_major_version == 1 and torch_minor_version >= 12)
):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

torch.set_grad_enabled(False)

from openfold.config_crosslinks import model_config, NUM_RES
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.model.model import AlphaFold
from openfold.model.torchscript import script_preset_
from openfold.np import residue_constants, protein
import openfold.np.relax.relax as relax
from openfold.utils.import_weights import (
    import_jax_weights_,
)
from openfold.utils.tensor_utils import (
    tensor_tree_map,
)

from scripts.utils import add_data_args

from openfold.data.msa_subsampling import subsample_msa_sequentially, get_eff

TRACING_INTERVAL = 50


def precompute_alignments(tag, seq, alignment_dir, args):
    tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
    with open(tmp_fasta_path, "w") as fp:
        fp.write(f">{tag}\n{seq}")

    local_alignment_dir = os.path.join(alignment_dir, tag)

    if(args.use_precomputed_alignments is None and not os.path.isdir(local_alignment_dir)):
        logger.info(f"Generating alignments for {tag}...")
            
        os.makedirs(local_alignment_dir)

        alignment_runner = data_pipeline.AlignmentRunner(
            jackhmmer_binary_path=args.jackhmmer_binary_path,
            hhblits_binary_path=args.hhblits_binary_path,
            hhsearch_binary_path=args.hhsearch_binary_path,
            uniref90_database_path=args.uniref90_database_path,
            mgnify_database_path=args.mgnify_database_path,
            bfd_database_path=args.bfd_database_path,
            uniclust30_database_path=args.uniclust30_database_path,
            pdb70_database_path=args.pdb70_database_path,
            no_cpus=args.cpus,
        )
        alignment_runner.run(
            tmp_fasta_path, local_alignment_dir
        )
    else:
        logger.info(
            f"Using precomputed alignments for {tag} at {alignment_dir}..."
        )

    # Remove temporary FASTA file
    os.remove(tmp_fasta_path)


def round_up_seqlen(seqlen):
    return int(math.ceil(seqlen / TRACING_INTERVAL)) * TRACING_INTERVAL


def run_model(model, batch, tag, args):
    with torch.no_grad(): 
        # Disable templates if there aren't any in the batch
        model.config.template.enabled = model.config.template.enabled and any([
            "template_" in k for k in batch
        ])

        logger.info(f"Running inference for {tag}...")
        t = time.perf_counter()
        out = model(batch)
        inference_time = time.perf_counter() - t
        logger.info(f"Inference time: {inference_time}")
   
    return out


def prep_output(out, batch, feature_dict, feature_processor, args):
    plddt = out["plddt"]
    mean_plddt = np.mean(plddt)
    
    plddt_b_factors = np.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )

    if(args.subtract_plddt):
        plddt_b_factors = 100 - plddt_b_factors

    # Prep protein metadata
    template_domain_names = []
    template_chain_index = None
    if(feature_processor.config.common.use_templates and "template_domain_names" in feature_dict):
        template_domain_names = [
            t.decode("utf-8") for t in feature_dict["template_domain_names"]
        ]

        # This works because templates are not shuffled during inference
        template_domain_names = template_domain_names[
            :feature_processor.config.predict.max_templates
        ]

        if("template_chain_index" in feature_dict):
            template_chain_index = feature_dict["template_chain_index"]
            template_chain_index = template_chain_index[
                :feature_processor.config.predict.max_templates
            ]

    no_recycling = feature_processor.config.common.max_recycling_iters
    remark = ', '.join([
        f"no_recycling={no_recycling}",
        f"max_templates={feature_processor.config.predict.max_templates}",
        f"config_preset=model_5_ptm",
    ])

    # For multi-chain FASTAs
    ri = feature_dict["residue_index"]
    chain_index = (ri - np.arange(ri.shape[0])) / args.multimer_ri_gap
    chain_index = chain_index.astype(np.int64)
    cur_chain = 0
    prev_chain_max = 0
    for i, c in enumerate(chain_index):
        if(c != cur_chain):
            cur_chain = c
            prev_chain_max = i + cur_chain * args.multimer_ri_gap

        batch["residue_index"][i] -= prev_chain_max

    unrelaxed_protein = protein.from_prediction(
        features=batch,
        result=out,
        b_factors=plddt_b_factors,
        chain_index=chain_index,
        remark=remark,
        parents=template_domain_names,
        parents_chain_index=template_chain_index,
    )

    return unrelaxed_protein


def parse_fasta(data):
    data = re.sub('>$', '', data, flags=re.M)
    lines = [
        l.replace('\n', '')
        for prot in data.split('>') for l in prot.strip().split('\n', 1)
    ][1:]
    tags, seqs = lines[::2], lines[1::2]

    tags = [t.split()[0] for t in tags]

    return tags, seqs


def generate_feature_dict(
    tags,
    seqs,
    alignment_dir,
    data_processor,
    args,
):
    tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
    if len(seqs) == 1:
        tag = tags[0]
        seq = seqs[0]
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(alignment_dir, tag)
        feature_dict = data_processor.process_fasta(
            fasta_path=tmp_fasta_path, alignment_dir=local_alignment_dir
        )
    else:
        with open(tmp_fasta_path, "w") as fp:
            fp.write(
                '\n'.join([f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)])
            )
        feature_dict = data_processor.process_multiseq_fasta(
            fasta_path=tmp_fasta_path, super_alignment_dir=alignment_dir,
        )

    # Remove temporary FASTA file
    os.remove(tmp_fasta_path)

    return feature_dict


def get_model_basename(model_path):
    return os.path.splitext(
                os.path.basename(
                    os.path.normpath(model_path)
                )
            )[0]


def make_output_directory(output_dir, model_name, multiple_model_mode):
    if multiple_model_mode:
        prediction_dir = os.path.join(output_dir, "predictions", model_name)
    else:
        prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    return prediction_dir


def count_models_to_evaluate(openfold_checkpoint_path, jax_param_path):
    model_count = 0
    if openfold_checkpoint_path:
        model_count += len(openfold_checkpoint_path.split(","))
    if jax_param_path:
        model_count += len(jax_param_path.split(","))
    return model_count


def load_models_from_command_line(args, config):
    # Create the output directory

    model = AlphaFold(config)
    model = model.eval()
    checkpoint_basename = get_model_basename(args.checkpoint_path)

    sd = torch.load(args.checkpoint_path)['ema']['params']
    # sd = {("model." + k):v for k,v in sd.items()}

    model.load_state_dict(sd)
    
    model = model.to(args.model_device)
    logger.info(
        f"Loaded OpenFold parameters at {args.checkpoint_path}..."
    )
    output_directory = make_output_directory(args.output_dir, checkpoint_basename, False)

    return model, output_directory


def list_files_with_extensions(dir, extensions):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]



def load_crosslinks(crosslink_csv, fdr, seq):
    links = np.loadtxt(crosslink_csv,delimiter=',')

    n = len(seq)

    crosslinks = np.zeros((n,n,1))
    grouping = np.zeros((n,n,1))

    groups = np.arange(len(links))+1 # 0th group is no crosslink
    
    if links.shape[1] == 3:
        for i_, (i,j,fdr) in enumerate(links):
            i = int(i)
            j = int(j)
            crosslinks[i,j,0] = crosslinks[j,i,0] = fdr
            grouping[i,j,0] = grouping[j,i,0] = groups[i_]

    else:
        for i_, (i,j) in enumerate(links):
            i = int(i)
            j = int(j)
            crosslinks[i,j,0] = crosslinks[j,i,0] = fdr
            grouping[i,j,0] = grouping[j,i,0] = groups[i_]
    
    logger.info(
        f"Loaded {np.sum(crosslinks > 0) // 2} crosslinks..."
    )
    

    return crosslinks, grouping


def main(args):
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    config = model_config('model_5_ptm')
    
    
    # template_featurizer = templates.TemplateHitFeaturizer(
    #     mmcif_dir=args.template_mmcif_dir,
    #     max_template_date=args.max_template_date,
    #     max_hits=config.data.predict.max_templates,
    #     kalign_binary_path=args.kalign_binary_path,
    #     release_dates_path=args.release_dates_path,
    #     obsolete_pdbs_path=args.obsolete_pdbs_path
    # )

    data_processor = data_pipeline.DataPipeline(
        template_featurizer=None,
    )

    output_dir_base = args.output_dir
    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2**32)
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)
    
    feature_processor = feature_pipeline.FeaturePipeline(config.data)
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)
    if args.use_precomputed_alignments is None:
        alignment_dir = os.path.join(output_dir_base, "alignments")
    else:
        alignment_dir = args.use_precomputed_alignments


    with open(args.fasta, "r") as fp:
        data = fp.read()

    tag, seq = parse_fasta(data)

    tag = tag[0]
    seq = seq[0]

    # feature_dicts = {}

    model, output_directory = load_models_from_command_line(args, config)

    cur_tracing_interval = 0

    output_name = f'{tag}_model_5_ptm_crosslinks'
    if args.output_postfix is not None:
        output_name = f'{output_name}_{args.output_postfix}'

    # Does nothing if the alignments have already been computed
    precompute_alignments(tag, seq, alignment_dir, args)

    # feature_dict = feature_dicts.get(tag, None)
    # if(feature_dict is None):
    feature_dict = generate_feature_dict(
        [tag],
        [seq],
        alignment_dir,
        data_processor,
        args,
    )


    if args.crosslinks.endswith('.pt'):
        crosslinks = torch.load(args.crosslinks)
        feature_dict['xl'] = crosslinks['xl']
        feature_dict['xl_grouping'] = crosslinks['grouping']
    elif args.crosslinks.endswith('.csv'):
        crosslinks, grouping = load_crosslinks(args.crosslinks, args.fdr, seq)
        feature_dict['xl'] = crosslinks
        feature_dict['xl_grouping'] = grouping
    else:
        print("Crosslinks need to be either given as a CSV or already as a tensor")
        sys.exit(0)


    # subsample MSAs to specified Neff
    msa = feature_dict['msa']

    logger.info(
        f"Subsampling MSA to Neff={args.neff}..."
    )
    # msa = subsample_msa_sequentially(msa, neff=args.neff)
    feature_dict['msa'] = msa

    processed_feature_dict = feature_processor.process_features(
        feature_dict, mode='predict',
    )

    processed_feature_dict = {
        k:torch.as_tensor(v, device=args.model_device) 
        for k,v in processed_feature_dict.items()
    }

    out = run_model(model, processed_feature_dict, tag, args)

    # Toss out the recycling dimensions --- we don't need them anymore
    processed_feature_dict = tensor_tree_map(
        lambda x: np.array(x[..., -1].cpu()), 
        processed_feature_dict
    )
    out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

    plddt = out["plddt"]

    plddt_b_factors = np.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )

    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=out,
        b_factors=plddt_b_factors
    )

    unrelaxed_output_path = os.path.join(
        output_directory, f'{output_name}_unrelaxed.pdb'
    )


    with open(unrelaxed_output_path, 'w') as fp:
        fp.write(protein.to_pdb(unrelaxed_protein))

    logger.info(f"Output written to {unrelaxed_output_path}...")

    if not args.skip_relaxation:        
        amber_relaxer = relax.AmberRelaxation(
            **config.relax
        )
        
        relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)

        # Save the relaxed PDB.
        relaxed_output_path = os.path.join(
            output_directory, f'{output_name}_relaxed.pdb'
        )
        with open(relaxed_output_path, 'w') as fp:
            fp.write(relaxed_pdb_str)
        
        logger.info(f"Relaxed output written to {relaxed_output_path}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta", type=str,
        help="Path to FASTA file, one sequence per file"
    )
    parser.add_argument(
        "crosslinks", type=str,
    )
    parser.add_argument(
        "--fdr", type=float, default=0.05,
        help="""Number of CPUs with which to run alignment tools"""
    )

    parser.add_argument(
        "--use_precomputed_alignments", type=str, default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored."""
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--model_device", type=str, default="cuda:0",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")"""
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default='resources/AlphaLink_params/finetuning_model_5_ptm_CACA_10A.pt',
        help="""Path to OpenFold checkpoint (.pt file)"""
    )
    parser.add_argument(
        "--save_outputs", action="store_true", default=False,
        help="Whether to save all model outputs, including embeddings, etc."
    )
    parser.add_argument(
        "--cpus", type=int, default=4,
        help="""Number of CPUs with which to run alignment tools"""
    )
    parser.add_argument(
        "--preset", type=str, default='full_dbs',
        choices=('reduced_dbs', 'full_dbs')
    )
    parser.add_argument(
        "--output_postfix", type=str, default=None,
        help="""Postfix for output prediction filenames"""
    )
    parser.add_argument(
        "--data_random_seed", type=int, default=None
    )
    parser.add_argument(
        "--skip_relaxation", action="store_true", default=False,
    )
    parser.add_argument(
        "--neff", type=float, default=10,
        help="""MSAs are subsampled to specified Neff"""
    )
    parser.add_argument(
        "--subtract_plddt", action="store_true", default=False,
        help=""""Whether to output (100 - pLDDT) in the B-factor column instead
                 of the pLDDT itself"""
    )
    add_data_args(parser)
    args = parser.parse_args()

    if(args.model_device == "cpu" and torch.cuda.is_available()):
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )

    main(args)
