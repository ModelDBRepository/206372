This is the readme for the model associated with the paper:

Chavlis S, Petrantonakis PC, Poirazi P (2017) Dendrites of dentate
gyrus granule cells contribute to pattern separation by controlling
sparsity. Hippocampus

This python model was contributed by S Chavlis.

Neurons folder

The codes there are for various validation tests in order to create figure2 and several supplement figures

3,6 and 12 dendrites folders

The main code for every model with 12, 6 and 3 dendrites on Granule Cells.

The code represents one Trial for a given input pattern as well as a specific connectivity in the ConnectivityMatrices_#dendrites folder

Need 50 Trials of each code with 50 different input patterns. 

Each main code should run for different overlaps, specifric comments inside code.


For many input patterns use the following code inputs.py

#####################################################
from brian import  reinit,clear
import numpy as np
import random as pyrandom
import sys

def input_patterns(trial_i):
    reinit(states = True)
    clear(erase = True, all = True)

    Trial = trial_i[0]
    # Initial pattern
    scale_fac = 2

    N_input   = 100 * scale_fac
    d_input   = 0.10 # active input density

    # Active pattern of neurons
    active   = sorted(pyrandom.sample(xrange(N_input), int(d_input*N_input)))
    np.save('active_pattern_'+str(Trial)+'.npy', active)

    return

jobidx = int(sys.argv[1])
results = input_patterns([jobidx]) # launches multiple processes
#####################################################


python inputs.py <number form 1 to 50>


