Code for Paper "Protein structure prediction with in-cell photo-crosslinking mass spectrometry and deep learning"

# AlphaLink

AlphaLink extends [OpenFold](https://github.com/aqlaboratory/openfold) with crosslinking data.


## System Requirements

### Hardware requirements
AlphaLink requires only standard computer hardware with sufficient RAM for bigger proteins. Computation can be accelerated with a NVIDIA GPU with ideally >32GB VRAM.

### Software requirements

This package is supported on Linux.

## Installation

Please refer to the [OpenFold GitHub](https://github.com/aqlaboratory/openfold).

## Crosslinking data

Crosslinking data can be included either as a PyTorch dictionary with NumPy arrays: 'xl' and 'grouping' with shape LxLx1 where L is the length of the protein or as a csv with the following format:

2 or 3 columns: i j fdr (optional)

```
128 163 0.05
147 77 0.05
147 41 0.05
```

The indices i,j are 0-based. fdr is between 0 and 1.

Distograms have shape LxLx128 with the following binning: torch.arange(2.3125,42,0.3125) and no group embedding. Last bin is a catch-all bin. The probabilities should sum up to 1. To use distograms, you have to set the distograms flag to True in config_crosslinks.py xl_embedder.

## Usage

AlphaLink expects a FASTA file containing a single sequence, the crosslinks, and databases for template/ MSA search , [see also OpenFold Inference](https://github.com/aqlaboratory/openfold#inference).

```
python3 predict_with_crosslinks.py --checkpoint_path resources/AlphaLink_params/finetuning_model_5_ptm_CACA_10A.pt 7K3N_A.fasta photoL.csv uniref90.fasta mgy_clusters.fa pdb70/pdb70 pdb_mmcif/mmcif_files uniclust30_2018_08/uniclust30_2018_08
```

MSA generation can be skipped if there are precomputed alignments:

```
python3 predict_with_crosslinks.py --use_precomputed_alignments msa/ --checkpoint_path resources/AlphaLink_params/finetuning_model_5_ptm_CACA_10A.pt  7K3N_A.fasta photoL.csv uniref90.fasta mgy_clusters.fa pdb70/pdb70 pdb_mmcif/mmcif_files uniclust30_2018_08/uniclust30_2018_08
```

## Network weights

Can be downloaded here: 

https://www.dropbox.com/s/8npy4d6q86eqpfn/finetuning_model_5_ptm_CACA_10A.pt.gz?dl=0
https://www.dropbox.com/s/5jmb8pxmt5rr751/finetuning_model_5_ptm_distogram.pt.gz?dl=0

They need to be unpacked (gunzip).

## PDB models are available at: https://www.dropbox.com/sh/yrto5tzo7u1atqg/AABy2SdP-WFOanp7eOKr3eeoa?dl=0

## Reproduction instructions

We eliminated all non-determinism (MSA masking), since with low neff targets, different MSA masking can have a big effect.

## Copyright notice

While AlphaFold's and, by extension, OpenFold's source code is licensed under
the permissive Apache Licence, Version 2.0, DeepMind's pretrained parameters 
fall under the CC BY 4.0 license, a copy of which is downloaded to 
`openfold/resources/params` by the installation script. Note that the latter
replaces the original, more restrictive CC BY-NC 4.0 license as of January 2022.


## Citing this work

Any work that cites AlphaLink should also cite AlphaFold and OpenFold.
