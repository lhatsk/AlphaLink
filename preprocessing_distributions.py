import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('Agg')

np.random.seed(123)

if len(sys.argv) != 2:
    print("usage:python preprocessing_distributions.py restraints.csv\n\n",
          "where restraints.csv is a comma-separated file formatted "
          "as follows:\n",
          "residueFrom,residueTo,mean_dist,stdev,distribution_type\n",
          "residue numbering starts at 1.\n",
          "Distribution types to choose from are 'normal' or 'log-normal'\n",
          "For custom distributions see the numpy random distributions list"
          "to generate 128-bin distributions.\n"
          "For upper-bound restraints, use normal AlphaLink restraint input.\n\n",
          "example line in input file:\n",
          "123,415,10.0,5.0,normal\n"
          "to impose a restraint between residue 123 and residue 415 with a gaussian "
          "probability distribution centered around 10.0 Angstrom and a standard "
          "deviation of 5 Angstrom\n")
    sys.exit()

else:
    restraints = np.genfromtxt(str(sys.argv[1]),
                               names=["From", "To", "mu", "sigma", "type"],
                               delimiter=",",
                               dtype=None,
                               encoding=None)




restraint_array = []
for line in restraints:
    #convert to 0-based residue index
    res_from_0 = line["From"] - 1
    res_to_0 = line["To"] - 1
    if line["type"] == "normal":

        sample = np.random.normal(line["mu"], line["sigma"], size=10000)
        n, bins, p = plt.hist(sample, bins=np.arange(2.3125, 42, 0.3125),
                              density=True)
        n = n.tolist()
        restraint_line = [res_from_0, res_to_0]

        for element in n:
            restraint_line.append(element)
        restraint_array.append(restraint_line)

    elif line["type"] == "log-normal":
        sample = np.random.lognormal(line["mu"], line["sigma"], size=10000)
        n, bins, p = plt.hist(sample, bins=np.arange(2.3125, 42, 0.3125),
                              density=True)
        n = n.tolist()
        restraint_line = [res_from_0, res_to_0]

        for element in n:
            restraint_line.append(element)
        restraint_array.append(restraint_line)
    else:
        print("cannot parse restraint type in line\n")
        print(line)
        sys.exit()

restraint_array = np.array(restraint_array)
print(restraint_array)

np.savetxt("restraint_distributions.csv",
           restraint_array,
           delimiter=",")