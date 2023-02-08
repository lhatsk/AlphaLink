# https://github.com/grandrea/alphalink-ihm-template/
# Author Andrea Graziadei
import ihm
import ihm.location
import ihm.dataset
import ihm.restraint
import ihm.protocol
import ihm.model
import ihm.cross_linkers
import ihm.reference
import ihm.dumper
import ihm.representation
import pandas as pd
import ihm.reader
import ihm.citations
import sys

if len(sys.argv)!=4:
    print("usage: python make_ihm.py model.cif ")

model_file = "model_1.cif" #model file converted to cif by MAXIT
uniprot_accession ="P0A910"
protein_name ="OmpA"
restraint_file = "restraint.csv" # a file with 3 space-separated columns for residue from, residue to and restraint confidence (1-FDR)

with open(model_file) as fh:
    system = ihm.reader.read(fh)
system=system[0]

#change to PRIDE or whatever repository you have your cx-ms results in
crosslink_dataset = ihm.dataset.CXMSDataset(ihm.location.DatabaseLocation(db_name="jPOSTrepo",
                                            db_code="JPST001851"))


model_sequence = ihm.reference.UniProtSequence.from_accession(uniprot_accession)

#this would need to be a loop for systems with more than one entity
entityA = system.entities[0]
entityA.references = [model_sequence]

#change source as needed using ihm terms
entityA.source = ihm.source.Natural(ncbi_taxonomy_id=83333,
                                    common_name="Escherichia coli",
                                    scientific_name="Escherichia coli K12",
                                    strain="K12")

entityA.description = (protein_name)
asymA = system.asym_units[0]
asymA.description = str(protein_name)
asymA.details = str(protein_name)
assembly = ihm.Assembly((asymA,), name='Modeled assembly', description="modeled assembly")

title_string = str("model of " + protein_name)

system.title = title_string

#change authors as needed
system.authors = ["author 1",
                  "author 2"]

citation = ihm.Citation(
    pmid=None,
    title='mytitle',
    journal='myjournal', volume=None, page_range=(1, 2), year=2023,
    authors=["author 1",
             "author 2"],
    doi=None)
system.citations.append(alphalink)

alphalink_software = ihm.Software(name="AlphaLink",
                                  classification="model building",
                                  description="protein structure prediction by deep learning assisted by experimental distance restraints",
                                  location="https://github.com/lhatsk/AlphaLink",
                                  version="1.0",
                                  citation=alphalink)

system.software.append(alphalink_software)

rep = ihm.representation.Representation([ihm.representation.AtomicSegment(asymA, rigid=False)])

# Since our input models are plain PDBx, not IHM, we need to add additional
# required information on the model representation and assembly. This may be
# better handled by passing the file through python-ihm's
# util/make-mmcif.py first.
for state_group in system.state_groups:
        for state in state_group:
            for model_group in state:
                for model in model_group:
                    if not model.assembly:
                        model.assembly = assembly
                    if not model.representation:
                        model.representation = rep

#define crosslinker. Here, photo-leucine.
photo_leucine = ihm.ChemDescriptor(auth_name="L-Photo-Leucine",
                                   chem_comp_id=None,
                                   smiles="CC1(C[C@H](N)C(O)=O)N=N1",
                                   inchi="1S/C5H9N3O2/c1-5(7-8-5)2-3(6)4(9)10/h3H,2,6H2,1H3,(H,9,10)/t3-/m0/s1",
                                   inchi_key="MJRDGTVDJKACQZ-VKHMYHEASA-N",
                                   common_name="L-Photo-Leucine")


#if not using photo-leucine, use ihm.crosslinkers definitions
crosslink_restraint = ihm.restraint.CrossLinkRestraint(dataset=crosslink_dataset,
                                                       linker=photo_leucine)

crosslink_df = pd.read_csv(restraint_file, sep=" ", names=["from", "to", "confidence"], header=None)
crosslinks = []
# Usually cross-links use an upper bound restraint on the distance
distance = ihm.restraint.UpperBoundDistanceRestraint(10)
for index, line in crosslink_df.iterrows():
    res1ind = int(line["from"])
    res2ind = int(line["to"])
    # This assumes that residue indices in the CSV file map 1:1 to mmCIF
    # seq_ids. Verify by checking the residue names in the ihm_cross_link_list
    # in the output mmCIF. You may need to add an offset or otherwise map
    # the residue indices, because it looks off to me.
    residue_pair = ihm.restraint.ExperimentalCrossLink(
        residue1=entityA.residue(res1ind), residue2=entityA.residue(res2ind))
    # This takes a list of all ambiguous cross-links. Here we're saying there
    # is no ambiguity.
    crosslink_restraint.experimental_cross_links.append([residue_pair])
    residue_pair_restraint = ihm.restraint.ResidueCrossLink(experimental_cross_link=residue_pair,
                                                            asym1=asymA,
                                                            asym2=asymA,
                                                            psi=(1 - line["confidence"]),
                                                            distance=distance)
    crosslink_restraint.cross_links.append(residue_pair_restraint)

system.restraints.append(crosslink_restraint)

all_datasets = ihm.dataset.DatasetGroup((crosslink_dataset,))

protocol = ihm.protocol.Protocol(name='AlphaLink')
protocol.steps.append(ihm.protocol.Step(
    assembly=system.complete_assembly, dataset_group=all_datasets,
    method='AlphaLink', name='AlphaLink',
    num_models_begin=0, num_models_end=5, multi_scale=False, ensemble=False))

for state_group in system.state_groups:
        for state in state_group:
            for model_group in state:
                for model in model_group:
                    model.protocol = protocol
                    if not model.representation:
                        model.representation = rep

with open("model.cif", "w", encoding="utf-8") as fh:
    ihm.dumper.write(fh, [system])

