# AlphaLink

AlphaLink extends [OpenFold](https://github.com/aqlaboratory/openfold) with crosslinking MS data or other experimental distance restraint or contact information by explicitly incorporating them in the OpenFold architecture.

## System Requirements

### Hardware requirements
AlphaLink requires only standard computer hardware with sufficient RAM for bigger proteins. Computation can be accelerated with a NVIDIA GPU with ideally >32GB VRAM.

### Software requirements

This package is supported on Linux.

## Installation

Please refer to the [OpenFold GitHub](https://github.com/aqlaboratory/openfold) for installation instructions. 

## Crosslinking data

Crosslinking MS data can be included either as a PyTorch dictionary with NumPy arrays: 'xl' and 'grouping' with shape LxLx1 where L is the length of the protein or as a space-separated file with the following format:

residueFrom residueTo FDR
```
128 163 0.05
147 77 0.05
147 41 0.05
```

residueFrom and residueTo are the residues crosslinked to each other (sequence numbering starts at 1). FDR is between 0 and 1.

The software may then be run with models based on upper bound distance thresholds or using generalized distograms. Distograms have shape LxLx128 with the following binning: torch.arange(2.3125,42,0.3125) and no group embedding. Last bin is a catch-all bin. The probabilities should sum up to 1. To use distograms, you have to set the distograms flag to True in the xl_embedder in config_crosslinks.py.

## Usage

AlphaLink expects a FASTA file containing a single sequence, the crosslinking MS residue pairs, and databases for template/ MSA search , [see also OpenFold Inference](https://github.com/aqlaboratory/openfold#inference).

```
python predict_with_crosslinks.py --checkpoint_path resources/AlphaLink_params/finetuning_model_5_ptm_CACA_10A.pt 7K3N_A.fasta photoL.csv uniref90.fasta mgy_clusters.fa pdb70/pdb70 pdb_mmcif/mmcif_files uniclust30_2018_08/uniclust30_2018_08
```

MSA generation can be skipped if there are precomputed alignments:

```
python predict_with_crosslinks.py --use_precomputed_alignments msa/ --checkpoint_path resources/AlphaLink_params/finetuning_model_5_ptm_CACA_10A.pt  7K3N_A.fasta photoL.csv uniref90.fasta mgy_clusters.fa pdb70/pdb70 pdb_mmcif/mmcif_files uniclust30_2018_08/uniclust30_2018_08
```

## Network weights

Can be downloaded here: 

https://www.dropbox.com/s/8npy4d6q86eqpfn/finetuning_model_5_ptm_CACA_10A.pt.gz?dl=0 
https://www.dropbox.com/s/5jmb8pxmt5rr751/finetuning_model_5_ptm_distogram.pt.gz?dl=0

They need to be unpacked (gunzip).

## AlphaLink IHM model deposition [alphalink-ihm-template](https://github.com/grandrea/alphalink-ihm-template)

Models generated with AlphaLink using experimental restraints can be published as integrative/hybrid models in PDB-Dev [PDB-Dev](https://pdb-dev.wwpdb.org/) using this script. Requires [python-ihm](https://github.com/ihmwg/python-ihm).

Takes a .csv file with the crosslinking MS restraints, uniprot accession code and system name to generate a pdb-dev compliant file for deposition. Takes an mmcif file as an input.

First, generate an mmcif file from the .pdb output of AlphaLink using [Maxit](https://sw-tools.rcsb.org/apps/MAXIT/index.html).

Then, edit the make_ihm script to include authors, publication, system name, entity source, deposition database and details as you need.

Then you can run with

```
python make_ihm.py
```

## Reproducibility instructions

We eliminated all non-determinism (MSA masking), since with low Neff targets, different MSA masking can have a big effect.

The models generated for the [AlphaLink paper]() are deposited in [ModelArchive]() and [PDB-Dev](). The restraints used in the modeling are available as supplementary tables to the AlphaLink paper.

## Copyright notice

While AlphaFold's and, by extension, OpenFold's source code is licensed under
the permissive Apache Licence, Version 2.0, DeepMind's pretrained parameters 
fall under the CC BY 4.0 license, a copy of which is downloaded to 
`openfold/resources/params` by the installation script. Note that the latter
replaces the original, more restrictive CC BY-NC 4.0 license as of January 2022.

## Citing this work

Cite the AlphaLink paper:
"Protein structure prediction with in-cell photo-crosslinking mass spectrometry and deep learning", Nat. Biotech. XXX doi:YYY .

Any work that cites AlphaLink should also cite AlphaFold and OpenFold.
