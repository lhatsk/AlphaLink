import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(prog='AlphaLink preprocessing',
                    description='takes a restraint list and returns 128-bin distogram per restraint',
                    epilog='usage: python preprocessing_distributions.py --infile restraints.csv')

parser.add_argument("--infile", metavar="restraints.csv",
                    required=True,
                    type=str,
                    help= str("the input is a comma-separated file formatted " +
          "as follows:\n"+
          "residueFrom,residueTo,mean_dist,stdev,distribution_type\n"+
          "residue numbering starts at 1.\n"+
          "Distribution types to choose from are 'normal' or 'log-normal'\n"+
          "For custom distributions see the numpy random distributions list "+
          "to generate 128-bin distributions.\n"+
          "For upper-bound restraints, use normal AlphaLink restraint input.\n\n"+
          "example line in input file:\n"+
          "123,415,10.0,5.0,normal\n"+
          "to impose a restraint between residue 123 and residue 415 with a gaussian "+
          "probability distribution centered around 10.0 Angstrom and a standard "+
          "deviation of 5 Angstrom\n"))

parser.add_argument("--outfile", metavar="restraint_distributions.csv",
                    required=False,
                    type=str,
                    default="restraint_distributions.csv",
                    help="output file name")

args = parser.parse_args()

matplotlib.use('Agg')

np.random.seed(123)

restraints = np.genfromtxt(args.infile,
                           names=["From", "To", "mu", "sigma", "type"],
                           delimiter=",",
                           dtype=None,
                           encoding=None)

if len(restraints.shape) == 1:
    restraints = np.array([restraints])

distogram = []
for line in restraints:
    #convert to 0-based residue index
    res_from_0 = line["From"] #- 1
    res_to_0 = line["To"] #- 1
    if line["type"] == "normal":
        sample = np.random.normal(line["mu"], line["sigma"], size=10000)
    elif line["type"] == "log-normal":
        sample = np.random.lognormal(line["mu"], line["sigma"], size=10000)
    else:
        print("cannot parse restraint type in line\n")
        print(line)
        sys.exit()

    n, bins, p = plt.hist(sample, bins=np.arange(2.3125, 42.625, 0.3125),
                          density=True)
    n /= np.sum(n)
    n = n.tolist()
    distogram.append([res_from_0, res_to_0]+list(n))

distogram = np.array(distogram)

np.savetxt(args.outfile,
           distogram,
           delimiter=" ")
