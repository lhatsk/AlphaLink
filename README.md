![header ](imgs/T1064_pred.png)
_Figure: AlphaLink prediction (teal) of T1064 with simulated crosslinks (blue)_

# AlphaLink

[AlphaLink](https://www.nature.com/articles/s41587-023-01704-z) predicts protein structures using deep learning given a sequence and a set of experimental contacts. It extends [OpenFold](https://github.com/aqlaboratory/openfold) with crosslinking MS data or other experimental distance restraint by explicitly incorporating them in the OpenFold architecture. The experimental distance restraints may be represented in one of two forms:

1. As contacts/upper bound distance restraints
2. As distance distributions (distograms) (flag --distograms)

For (1), we trained our network with 10 Angstrom Ca-Ca and show robust rejection of experimental noise and false restraints. The distogram representation (2) allows the user to input longer restraints, for example corresponding to crosslinkers with spacers like BS3 or DSS or to NMR PRE distance restraints.


## Installation

Please refer to the [OpenFold GitHub](https://github.com/aqlaboratory/openfold#installation-linux) for installation instructions of the required packages. AlphaLink requires the same packages, since it builds on top of OpenFold.  

## Crosslinking data

Crosslinking MS data can be included either as a PyTorch dictionary with NumPy arrays: 'xl' and 'grouping' with shape LxLx1 where L is the length of the protein or as a space-separated file with the following format:

residueFrom residueTo FDR
```
128 163 0.05
147 77 0.05
147 41 0.05
```

residueFrom and residueTo are the residues crosslinked to each other (sequence numbering starts at 1). FDR is between 0 and 1. CSV format is not supported for distograms.

The software may then be run with models based on upper bound distance thresholds or using generalized distograms. Distograms have shape LxLx128 with the following binning: numpy.arange(2.3125,42,0.3125) + a catch-all bin in the end for distances >= 42A and no group embedding. Last bin is a catch-all bin. The probabilities should sum up to 1. To use distograms, you have to run predict_with_crosslinks.py with the --distograms flag.

Distograms can also be given as a space-separated file with the following format:

residueFrom residueTo 1..128
```
128 163 0.05 0.05 0.05 0.05 ...
147 77 0.01 0.015 0.05 0.05 ...
147 41 0.04 0.1 0.05 0.052 ...
```
residueFrom and residueTo are the residues crosslinked to each other (sequence numbering starts at 1). Columns 2-130 contain the probability for each bin in numpy.arange(2.3125,42,0.3125)- i.e. the probability of each bin in a distogram going from 2.3125 to 42 Angstrom. The 128th bin is a catch-all bin for distances >= 42. Each restraint can have a different distribution, any uncertainty has to be encoded in the distribution. There is no additional FDR parameter.

Distance distributions for AlphaLink can be automatically generated from restraint lists with the script preprocessing_distributions.py.
```
     python preprocessing_distributions.py --infile restraints.csv
```

Where restraints.csv is a comma-separated file containing residueFrom,residueTo,meanDistance,standard deviation, distribution type (normal/log-normal). For example:

    12,135,15.0,5.0,normal
    
For a restraint between residue 12 and 135 imposed as a normal distribution with a mean distance of 15 Angstrom and a standard deviation of 10 Angstrom.

preprocessing_distributions.py will generate a restraint list with distance distributions binned in 128-bin distograms that can be given to AlphaLink when run with the --distograms flag

```
     python predict_with_crosslinks.py 7K3N_A.fasta restraints.csv --distograms --checkpoint_path resources/AlphaLink_params/finetuning_model_5_ptm_distogram.pt --uniref90_database_path uniref90.fasta --mgnify_database_path mgy_clusters.fa --pdb70_database_path pdb70/pdb70 --uniclust30_database_path uniclust30_2018_08/uniclust30_2018_08
```
## MSA subsampling

MSAs can be subsampled to a given Neff with --neff. 

## Usage

AlphaLink expects a FASTA file containing a single sequence, the crosslinking MS residue pairs, and databases for template/ MSA search, [see also OpenFold Inference](https://github.com/aqlaboratory/openfold#inference).

```
python predict_with_crosslinks.py 7K3N_A.fasta photoL.csv --checkpoint_path resources/AlphaLink_params/finetuning_model_5_ptm_CACA_10A.pt --uniref90_database_path uniref90.fasta --mgnify_database_path mgy_clusters.fa --pdb70_database_path pdb70/pdb70 --uniclust30_database_path uniclust30_2018_08/uniclust30_2018_08
```

MSA generation can be skipped if there are precomputed alignments:

```
python predict_with_crosslinks.py 7K3N_A.fasta photoL.csv --use_precomputed_alignments msa/ --checkpoint_path resources/AlphaLink_params/finetuning_model_5_ptm_CACA_10A.pt  --uniref90_database_path uniref90.fasta --mgnify_database_path mgy_clusters.fa --pdb70_database_path pdb70/pdb70 --uniclust30_database_path uniclust30_2018_08/uniclust30_2018_08
```

or with precomputed features (pickle) with --features

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

The models generated for the [AlphaLink paper](https://www.nature.com/articles/s41587-023-01704-z) are deposited in [ModelArchive](https://modelarchive.org/doi/10.5452/ma-rap-alink) and [PDB-Dev](https://pdb-dev.wwpdb.org/entry.html?PDBDEV_00000165). The restraints used in the modeling are available as supplementary tables to the AlphaLink paper.

## Copyright notice

While AlphaFold's and, by extension, OpenFold's source code is licensed under
the permissive Apache Licence, Version 2.0, DeepMind's pretrained parameters 
fall under the CC BY 4.0 license, a copy of which is downloaded to 
`openfold/resources/params` by the installation script. Note that the latter
replaces the original, more restrictive CC BY-NC 4.0 license as of January 2022.

## Citing this work

Cite the AlphaLink paper:
"Protein structure prediction with in-cell photo-crosslinking mass spectrometry and deep learning", Nat. Biotech. XXX doi:10.1038/s41587-023-01704-z.

Any work that cites AlphaLink should also cite AlphaFold and OpenFold.
