---
license: cc-by-4.0
---

Mirror of OpenFold parameters as provided in https://github.com/aqlaboratory/openfold. Stopgap solution as the original download link was down. All rights to the authors.

OpenFold model parameters, v. 06_22. 

# Training details:

Trained using OpenFold on 44 A100s using the training schedule from Table 4 in
the AlphaFold supplement. AlphaFold was used as the pre-distillation model. 
Training data is hosted publicly in the "OpenFold Training Data" RODA repository.

# Parameter files:

Parameter files fall into the following categories:

    initial_training.pt: 
        OpenFold at the end of the initial training phase.
    finetuning_x.pt:
        Checkpoints in chronological order corresponding to peaks in the 
        validation LDDT-Ca during the finetuning phase. Roughly evenly spaced 
        across the 45 finetuning epochs.
    finetuning_ptm_x.pt:
        Checkpoints in chronological order corresponding to peaks inthe pTM
        training phase, a short additional training phase that takes place 
        after finetuning. Models in this category include the pTM module and 
        comprise the most recent of the checkpoints in this directory.
    
Average validation LDDT-Ca scores for each of the checkpoints are listed below. 
The validation set contains approximately 180 chains drawn from CAMEO over a 
three-month period at the end of 2021.

    initial_training: 0.9088
    finetuning_ptm_1: 0.9075
    finetuning_ptm_2: 0.9097
    finetuning_1: 0.9089
    finetuning_2: 0.9061
    finetuning_3: 0.9075
    finetuning_4: 0.9059
    finetuning_5: 0.9054