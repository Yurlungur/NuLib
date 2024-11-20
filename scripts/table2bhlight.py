#!/usr/bin/env python

import h5py
import numpy as np
from argparse import ArgumentParser

HPL = 6.6260693e-27
EV  = 1.60217653e-12 
MEV = 1.0e6*EV

parser = ArgumentParser(
    description="Rename fields and rearrange indices to convert NuLib table to nubhlight format")
parser.add_argument("infile",
                    type=str,
                    help='Name of input file')
parser.add_argument("-o","--outfile",
                    type=str,
                    default = None,
                    help="Optional output filename. Filename otherwise computed from nulib file")

if __name__ == "__main__":
    args = parser.parse_args()

    if args.outfile is not None:
        outfile = args.outfile
    else:
        outfile = args.infile.rstrip('.h5') + '_bhlight.h5'

    print("Reading {}...".format(args.infile))
    with h5py.File(args.infile, 'r') as fin:
        rho = fin['rho_points'][:]
        T = fin['temp_points'][:]
        Ye = fin['ye_points'][:]
        emis = fin['emissivities'][:]
        opac = fin['absorption_opacity'][:]
        e = fin['neutrino_energies'][:]
    lrho = np.log10(rho)
    lT = np.log10(T)
    nu = e*MEV/HPL
    lnu = np.log(nu)

    emis_new = HPL*np.moveaxis(emis,[0,1],[4,3])/MEV
    opac_new = np.moveaxis(opac,[0,1],[4,3])

    print("Writing {}...".format(outfile))
    with h5py.File(outfile, 'w') as fout:
        fout.create_dataset('Ye', data=Ye)
        fout.create_dataset('emis', data=emis_new)
        fout.create_dataset('lT', data=lT)
        fout.create_dataset('lnu', data=lnu)
        fout.create_dataset('lrho', data=lrho)
        fout.create_dataset('opac', data=opac_new)
    print("Done")
