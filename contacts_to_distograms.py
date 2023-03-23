import numpy as np
from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser(description='Generate distograms for binary contacts')
    parser.add_argument('--csv',
                        help='CSV with contacts: i j FDR',
                        required=True)
    parser.add_argument('--cutoff',
                        help='cutoff in A',
                        required=False,
                        default=10)
    parser.add_argument('--output',
                        help='Output CSV with distogram restraints',
                        required=True)                   
    args = parser.parse_args()
    return args

distogram_bins = np.arange(2.3125,42,0.3125)

def get_uniform(cutoff, fdr):
    d = np.ones(128)
    maximum_bin = np.argmax(distogram_bins > cutoff)
    d[:maximum_bin] /= np.sum(d[:maximum_bin])
    d[:maximum_bin] *= 1 - fdr
    d[maximum_bin:] /= np.sum(d[maximum_bin:])
    d[maximum_bin:] *= fdr

    return d

def main():
    args = parse_arguments()

    contacts = np.loadtxt(args.csv)

    with open(args.output, 'w') as f:
        for i,j,fdr in contacts:
            line = [i,j]
            distogram = get_uniform(args.cutoff, fdr)
            line += list(distogram)
            f.write(' '.join([str(e) for e in line]) + '\n')

if __name__ == "__main__":
    main()