Code for Paper AlphaLink: bringing protein structure prediction in situ with in-cell photo-crosslinking mass spectrometry

# AlphaLink

AlphaLink extends [OpenFold](https://github.com/aqlaboratory/openfold) with crosslinking data.

## Installation

Please refer to the [OpenFold GitHub](https://github.com/aqlaboratory/openfold).

## Crosslinking data

AlphaLink expects a csv with 2 or 3 columns with the following format:

i j fdr (optional)

The indices i,j are 0-based.

## Usage

python3 predict_with_crosslinks.py 7K3N_A.fasta photoK.csv

The script looks for pre-computed MSAs in the alignments/ folder. 

## Copyright notice

While AlphaFold's and, by extension, OpenFold's source code is licensed under
the permissive Apache Licence, Version 2.0, DeepMind's pretrained parameters 
fall under the CC BY 4.0 license, a copy of which is downloaded to 
`openfold/resources/params` by the installation script. Note that the latter
replaces the original, more restrictive CC BY-NC 4.0 license as of January 2022.


## Citing this work

<!-- For now, cite OpenFold as follows: -->

<!-- ```bibtex
@software{Ahdritz_OpenFold_2021,
  author = {Ahdritz, Gustaf and Bouatta, Nazim and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and AlQuraishi, Mohammed},
  doi = {10.5281/zenodo.5709539},
  month = {11},
  title = {{OpenFold}},
  url = {https://github.com/aqlaboratory/openfold},
  year = {2021}
}
``` -->

Any work that cites AlphaLink should also cite AlphaFold and OpenFold.
