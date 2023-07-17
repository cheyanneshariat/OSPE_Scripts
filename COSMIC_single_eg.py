""" 
Example of Using COSMIC, for full details, refer to the documentation page:
https://cosmic-popsynth.github.io/docs/stable/index.html
"""

#install cosmic
!pip3 install cosmic-popsynth #(can do this on unix command-line as well, I used Jupyter Notebook)


#import cosmic stuff
from cosmic import _evolvebin
from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.evolve import Evolve

#import pandas
import pandas as pd


#initialize a binary with specified parameters
single_binary = InitialBinaryTable.InitialBinaries(m1=1, m2=1, porb=.5, ecc=0.5, tphysf=13700.0, kstar1=1, kstar2=1, metallicity=0.002,sep=2.,
                                                   rad_1 = 1, rad_2=1,renv_1=0.3,renv_2=0.3)
#just need this...
BSEDict = {'xi': 1.0, 'bhflag': 1, 'neta': 0.5, 'windflag': 3, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.001, 'pts3': 0.02, 'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 'tflag': 1, 'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0, 'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 'bhsigmafrac' : 1.0, 'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.25, 'ecsn_mlow' : 1.6, 'aic' : 1, 'ussn' : 0, 'sigmadiv' :-20.0, 'qcflag' : 1, 'eddlimflag' : 0, 'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 1, 'ST_tide' : 1, 'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 0, 'zsun' : 0.014, 'bhms_coll_flag' : 0, 'don_lim' : -1, 'acc_lim' : -1}

#evolve the binary
bpp, bcm, initC, kick_info = Evolve.evolve(initialbinarytable=single_binary, BSEDict=BSEDict, 
                                           timestep_conditions =[['RRLO_1>=1', 'dtp=0.0'], ['RRLO_2>=1', 'dtp=0.0'],['binstate==1', 'dtp=0.0'], ['binstate==0', 'dtp=0.0']   ]
                                           )

#initial conditions (make sure everthing looks good here)
display(initC)


#default printouts
display(bpp)

#printout at specified times from timestep_conditions
display(bcm)


