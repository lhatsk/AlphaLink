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
import logging
import math
import numpy as np
import os

from openfold.utils.script_utils import load_models_from_command_line, parse_fasta, run_model, prep_output, \
    update_timings, relax_protein

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

import pickle

import random
import time
import torch

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

from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.np import residue_constants, protein
import openfold.np.relax.relax as relax

from openfold.utils.tensor_utils import (
    tensor_tree_map,
)
from openfold.utils.trace_utils import (
    pad_feature_dict_seq,
    trace_model_,
)
from scripts.utils import add_data_args


TRACING_INTERVAL = 50


import math
from scipy.spatial.distance import pdist, squareform

# adapted from: https://github.com/sokrypton/GREMLIN_CPP

def get_eff(msa, eff_cutoff=0.8): # eff_cutoff=0.62 for metapsicov
    if msa.ndim == 3: msa = msa.argmax(-1)
    # pairwise identity
    msa_sm = 1.0 - squareform(pdist(msa,"hamming"))
    # weight for each sequence
    msa_w = (msa_sm >= eff_cutoff).astype(float)
    msa_w = 1/np.sum(msa_w,-1)

    return msa_w


# if cap_msa is enabled, we bypass the ExtraMSAStack, helps with determinism for |MSA| < 128
def subsample_msa_sequentially(msa, neff=10, eff_cutoff=0.8, cap_msa=True):
    if msa.shape[0] == 1:
        return np.array([0])

    indices = [0]

    idx = np.arange(msa.shape[0] - 1) + 1
    np.random.shuffle(idx)

    new = [msa[0]]

    for i in idx:
        new.append(msa[i])
        indices.append(i)
        neff_ = get_eff(np.array(new), eff_cutoff=eff_cutoff).sum()

        if cap_msa:
            if neff_ > neff or len(new) > 126:
                new.pop()
                indices.pop()
                break
        else:
            if neff_ > neff:
                new.pop()
                indices.pop()
                break

    return np.array(indices)


def precompute_alignments(tags, seqs, alignment_dir, args):
    for tag, seq in zip(tags, seqs):
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

def list_files_with_extensions(dir, extensions):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]

def load_distogram(distogram_csv, seq):
    links = np.loadtxt(crosslink_csv)#,delimiter=',')

    n = len(seq)

    distogram = np.zeros((n,n,128))

    for row in links:
        i = int(row[0]) - 1
        j = int(row[1]) - 1
        distogram[i,j] = distogram[j,i] = row[2:]
    
    logger.info(
        f"Loaded {np.sum(np.max(distogram,axis=-1) > 0) // 2} restraints..."
    )

    return distogram

def main(args):
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    config = model_config(args.config_preset, long_sequence_inference=args.long_sequence_inference)
    
    if(args.trace_model):
        if(not config.data.predict.fixed_size):
            raise ValueError(
                "Tracing requires that fixed_size mode be enabled in the config"
            )
    
    # template_featurizer = templates.TemplateHitFeaturizer(
    #     mmcif_dir=args.template_mmcif_dir,
    #     max_template_date=args.max_template_date,
    #     max_hits=config.data.predict.max_templates,
    #     kalign_binary_path=args.kalign_binary_path,
    #     release_dates_path=args.release_dates_path,
    #     obsolete_pdbs_path=args.obsolete_pdbs_path
    # )

    data_processor = data_pipeline.DataPipeline(
        template_featurizer=None, #template_featurizer,
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

    tag_list = []
    seq_list = []
    for fasta_file in list_files_with_extensions(args.fasta_dir, (".fasta", ".fa")):
        # Gather input sequences
        with open(os.path.join(args.fasta_dir, fasta_file), "r") as fp:
            data = fp.read()
    
        tags, seqs = parse_fasta(data)
        # assert len(tags) == len(set(tags)), "All FASTA tags must be unique"
        tag = '-'.join(tags)

        tag_list.append((tag, tags))
        seq_list.append(seqs)

    seq_sort_fn = lambda target: sum([len(s) for s in target[1]])
    sorted_targets = sorted(zip(tag_list, seq_list), key=seq_sort_fn)
    feature_dicts = {}
    model_generator = load_models_from_command_line(
        config,
        args.model_device,
        args.checkpoint_path,
        None,
        args.output_dir)
    for model, output_directory in model_generator:
        cur_tracing_interval = 0
        for (tag, tags), seqs in sorted_targets:
            output_name = f'{tag}_{args.config_preset}'
            if args.output_postfix is not None:
                output_name = f'{output_name}_{args.output_postfix}'
    
            if args.features:
                feature_dict = pickle.load(open(args.features,'rb'))
            else:
                # Does nothing if the alignments have already been computed
                precompute_alignments(tags, seqs, alignment_dir, args)
            
                feature_dict = feature_dicts.get(tag, None)
                if(feature_dict is None):
                    feature_dict = generate_feature_dict(
                        tags,
                        seqs,
                        alignment_dir,
                        data_processor,
                        args,
                    )

                    if(args.trace_model):
                        n = feature_dict["aatype"].shape[-2]
                        rounded_seqlen = round_up_seqlen(n)
                        feature_dict = pad_feature_dict_seq(
                            feature_dict, rounded_seqlen,
                        )

                    feature_dicts[tag] = feature_dict

            if args.crosslinks.endswith('.pt'):
                crosslinks = torch.load(args.crosslinks)
                feature_dict['xl'] = crosslinks['xl']
            elif args.crosslinks.endswith('.csv'):
                crosslinks = load_distogram(args.crosslinks, seq)
                feature_dict['xl'] = crosslinks
            else:
                print("Crosslinks need to be either given as a CSV or already as a tensor")
                sys.exit(0)

            # subsample MSAs to specified Neff
            msa = feature_dict['msa']

            if args.neff:
                logger.info(
                    f"Subsampling MSA to Neff={args.neff}..."
                )
                indices = subsample_msa_sequentially(msa, neff=args.neff)
                feature_dict['msa'] = msa[indices]
                feature_dict['deletion_matrix'] = feature_dict['deletion_matrix'][indices]

            processed_feature_dict = feature_processor.process_features(
                feature_dict, mode='predict',
            )

            processed_feature_dict = {
                k:torch.as_tensor(v, device=args.model_device) 
                for k,v in processed_feature_dict.items()
            }

            if(args.trace_model):
                if(rounded_seqlen > cur_tracing_interval):
                    logger.info(
                        f"Tracing model at {rounded_seqlen} residues..."
                    )
                    t = time.perf_counter()
                    trace_model_(model, processed_feature_dict)
                    tracing_time = time.perf_counter() - t
                    logger.info(
                        f"Tracing time: {tracing_time}"
                    )
                    cur_tracing_interval = rounded_seqlen

            out = run_model(model, processed_feature_dict, tag, args.output_dir)

            # Toss out the recycling dimensions --- we don't need them anymore
            processed_feature_dict = tensor_tree_map(
                lambda x: np.array(x[..., -1].cpu()), 
                processed_feature_dict
            )
            out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

            unrelaxed_protein = prep_output(
                out, 
                processed_feature_dict, 
                feature_dict, 
                feature_processor, 
                args.config_preset,
                args.multimer_ri_gap,
                args.subtract_plddt
            )

            unrelaxed_output_path = os.path.join(
                output_directory, f'{output_name}_unrelaxed.pdb'
            )

            with open(unrelaxed_output_path, 'w') as fp:
                fp.write(protein.to_pdb(unrelaxed_protein))

            logger.info(f"Output written to {unrelaxed_output_path}...")
            
            if not args.skip_relaxation:
                # Relax the prediction.
                logger.info(f"Running relaxation on {unrelaxed_output_path}...")
                relax_protein(config, args.model_device, unrelaxed_protein, output_directory, output_name)

            if args.save_outputs:
                output_dict_path = os.path.join(
                    output_directory, f'{output_name}_output_dict.pkl'
                )
                with open(output_dict_path, "wb") as fp:
                    pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)

                logger.info(f"Model output written to {output_dict_path}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta_dir", type=str,
        help="Path to directory containing FASTA files, one sequence per file"
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
        "--config_preset", type=str, default="model_5_ptm",
        help="""Name of a model config preset defined in openfold/config.py"""
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
        "--features", type=str,
        help="Feature pickle"
    )
    parser.add_argument(
        "--distograms", action="store_true", default=False,
        help="Switch to distogram mode"
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
        "--data_random_seed", type=str, default=None
    )
    parser.add_argument(
        "--skip_relaxation", action="store_true", default=False,
    )
    parser.add_argument(
        "--neff", type=float,
        help="""MSAs are subsampled to specified Neff"""
    )
    parser.add_argument(
        "--multimer_ri_gap", type=int, default=200,
        help="""Residue index offset between multiple sequences, if provided"""
    )
    parser.add_argument(
        "--trace_model", action="store_true", default=False,
        help="""Whether to convert parts of each model to TorchScript.
                Significantly improves runtime at the cost of lengthy
                'compilation.' Useful for large batch jobs."""
    )
    parser.add_argument(
        "--subtract_plddt", action="store_true", default=False,
        help=""""Whether to output (100 - pLDDT) in the B-factor column instead
                 of the pLDDT itself"""
    )
    parser.add_argument(
        "--long_sequence_inference", action="store_true", default=False,
        help="""enable options to reduce memory usage at the cost of speed, helps longer sequences fit into GPU memory, see the README for details"""
    )
    add_data_args(parser)
    args = parser.parse_args()

    if(args.model_device == "cpu" and torch.cuda.is_available()):
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )

    main(args)
