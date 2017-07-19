#==============================================================================
# Network of Dentate gyrus based on Myers and Scharfman, Hippocampus 2008
#==============================================================================

# External input --> 400 cells, PoissonGroup
# Granule cells --> 2000 cells into 100 clusters
# Basket cell (GABAergic) --> 1 per cluster (100 cells)
# Hilar mossy cells --> 80 cells
# HIPP --> 40 cells
#
#==============================================================================
# ****************************************************************************
#==============================================================================
# CONNECTIONS
# 1. input ---> granule cells : each to 20% of granule cells randomly (excitation)
# 2. input ---> HIPP : each to 20% of HIPP randomly (excitation)
# 3. mossy ---> granule cells: each to 20% randomly (excitation)
# 4. HIPP ---> granule cells: each to 20% randomly (feed forward inhibition)
# 5. granule cells ---> mossy cells: eack to 20% randomly (excitation)
# 6. basket cells <---> granule cell: one-to-all (feedback inhibition)
####################################################################################

import os
from brian import *
from brian.library.ionic_currents import *
from brian.library.IF import *
import numpy as np
import time
import math
import random as pyrandom


Trial = 1

# General Parameters
scale_fac = 4
N_input   = 100 * scale_fac
N_granule = 500 * scale_fac
N_basket  =  25 * scale_fac
N_mossy   =  20 * scale_fac
N_hipp    =  10 * scale_fac
d_input   = 0.10 # active input density

reinit(states = True)
clear(erase   = True, all = True)

# Active pattern of neurons
active   = sorted(pyrandom.sample(xrange(N_input), int(d_input*N_input)))
np.save('active_pattern_Trial'+str(Trial)+'.npy', active)
inactive = [x for x in xrange(N_input) if x not in active]

# For second pattern with 90% overlap!
#overlap = '90'
#active_pattern = np.load('active_pattern_Trial'+str(Trial)+'.npy')
#n = int( round((1 - int(overlap)*0.01)*d_input*N_input) )
#
#removal = pyrandom.sample(active_pattern, n)
#extra   = pyrandom.sample([x for x in xrange(N_input) if x not in active_pattern], n)
#
#active   = [x for x in active_pattern if x not in removal]
#active  += extra
#active   = list(sort(active))
#inactive = [x for x in xrange(N_input) if x not in active]

print "\nBuilding the Network... "
# CONNECTIVITY PARAMETERS

# General parameters
E_nmda     =   0     * mV       # NMDA reversal potential
E_ampa     =   0     * mV       # AMPA reversal potential
E_gaba     = -86     * mV       # GABA reversal potential
gamma      =   0.072 * mV**-1   # Mg Concentration factor
alpha_nmda =   0.5   * ms**-1   # NMDA scale factor
alpha_ampa =   1     * ms**-1   # AMPA scale factor
alpha_gaba =   1     * ms**-1   # GABA scale factor

# EC CELLS ---> GRANULE CELLS
g_ampa_eg = 0.8066 * nS
g_nmda_eg = 1.0800 * g_ampa_eg

# EC CELLS ---> HIPP CELLS
g_ampa_eh = 0.24 * nS
g_nmda_eh = 1.15 * g_ampa_eh

# SOURCE: GRANULE CELLS
# GRANULE CELLS ---> BASKET CELLS
g_ampa_gb = 0.21 * nS
g_nmda_gb = 1.50 * g_ampa_gb

# GRANULE CELLS ---> MOSSY CELLS
g_ampa_gm = 0.50 * nS
g_nmda_gm = 1.05 * g_ampa_gm

# SOURCE: MOSSY CELLS
# MOSSY CELLS ---> GRANULE CELLS
g_ampa_mg = 0.1066 * nS
g_nmda_mg = 1.0800 * g_ampa_mg

# MOSSY CELLS ---> BASKET CELLS
g_ampa_mb = 0.35 * nS
g_nmda_mb = 1.10 * g_ampa_mb

# SOURCE: BASKET CELLS
# BASKET CELLS ---> GRANULE CELLS
g_gaba_bg = 14.0 * nS

# SOURCE: HIPP CELLS
# HIPP CELLS ---> GRANULE CELLS
g_gaba_hg = 0.12 * nS
#=======================================================================================================================


#=======================================================================================================================
# INPUT CELLS (ENTORHINAL CORTEX)
from poisson_input import *
rate = 40*Hz
simtime = 500*ms
t1 = 300 * ms
t2 =  10 * ms
spiketimes = poisson_input(active, N_input, rate, simtime, t1, t2)
Input_ec = SpikeGeneratorGroup(N_input, spiketimes)
#=======================================================================================================================

# GRANULE CELLS
# Parameters
gl      =   0.00003  * siemens/(cm**2)  # leakage conductance
gl_dend =   0.00001 * siemens/(cm**2)  # leakage conductance
El_soma = -87.0      * mV               # reversal-resting potential
El_dend = -82.0      * mV               # reversal-resting potential
Cm      =   1.0      * uF/(cm**2)       # membrane capacitance
Cm_dend =   2.5      * uF/(cm**2)       # membrane capacitance
v_th    = -56.0      * mV               # threshold potential
v_reset = -74.0      * mV               # reset potential

# Morphology
# Soma
length_soma = 18.0 * um
diam_soma   = 12.0 * um
area_soma   = math.pi * diam_soma * length_soma
# Dendrites
Nseg    = 15 # Number of dendritic compartments
Nbranch = 3 # Number of main branches
Ntips   = 6 # Number of distal dendritic compartments
distal_l   = 83.0 * um
medial_l   = 83.0 * um
proximal_l = 83.0 * um
length_dend  = [distal_l, distal_l, medial_l, medial_l, proximal_l]
length_dend *= Nbranch
distal_d   = 0.80 * um
medial_d   = 0.90 * um
proximal_d = 1.00 * um
diam_dend  = [distal_d, distal_d, medial_d, medial_d, proximal_d]
diam_dend *= Nbranch
area_dend  = [math.pi*x*y for x,y in zip(length_dend, diam_dend)]

# AMPA/NMDA/GABA Kinetics
t_nmda_decay_g = 50.0  * ms  # NMDA decay time constant
t_nmda_rise_g  =  0.33 * ms  # NMDA rise time constant
t_ampa_decay_g =  2.5  * ms  # AMPA decay time constant
t_ampa_rise_g  =  0.1  * ms  # AMPA rise time constant
t_gaba_decay_g =  6.8  * ms  # GABA decay time constant
t_gaba_rise_g  =  0.9  * ms  # GABA rise time constant

# AMPA/NMDA/GABA Model Parameters
gamma_g      = 0.04 * mV**-1 # the steepness of Mg sensitivity of Mg unblock
Mg           = 2.0  # [mM]--mili Molar - the extracellular Magnesium concentration
eta          = 0.2 # [mM**-1] -1- mili Molar **(-1) - Magnesium sensitivity of unblock
alpha_nmda_g = 2.0  * ms**-1
alpha_ampa   = 1.0  * ms**-1
alpha_gaba   = 1.0  * ms**-1

# Axial resistances
Ri = 210.0 * ohm * cm
ra0 = Ri * 4 / (pi * distal_d ** 2)
ra1 = Ri * 4 / (pi * medial_d ** 2)
ra2 = Ri * 4 / (pi * proximal_d ** 2)
Ra_0 = ra0 * distal_l
Ra_1 = ra1 * medial_l
Ra_2 = ra2 * proximal_l

# NOISE
g_ampa_gn = 0.008 * nS
g_nmda_gn = 0.008 * nS
# AHP patrameters
tau_ahp = 45*ms
g_ahp   = 2*nS
# Synaptic current equations @ SOMA
eq_soma = Equations('''
I_synS = I_gaba_bg - I_inj + I_Sahp              : amp
I_Sahp                                           : amp
dI_Sahp/dt = (g_ahp*(vm-El_soma)-I_Sahp)/tau_ahp : amp
I_gaba_bg = g_gaba_bg*(vm - E_gaba)*s_gaba_bg    : amp
s_gaba_bg                                        : 1
I_inj                                            : amp
''')


# Synaptic current equations
eq_dend = Equations('''
I_synD    = I_nmda_eg + I_ampa_eg + I_nmda_mg + I_ampa_mg + I_gaba_hg + I_nmda_gn + I_ampa_gn : amp
I_nmda_eg = g_nmda_eg*(vm - E_nmda)*s_nmda_eg*(1.0/(1 + eta*Mg*exp(-gamma_g*vm)))             : amp
I_ampa_eg = g_ampa_eg*(vm - E_ampa)*s_ampa_eg                                                 : amp
s_nmda_eg                                                                                     : 1
s_ampa_eg                                                                                     : 1
I_nmda_mg = g_nmda_mg*(vm - E_nmda)*s_nmda_mg*(1.0/(1 + eta*Mg*exp(-gamma_g*vm)))             : amp
I_ampa_mg = g_ampa_mg*(vm - E_ampa)*s_ampa_mg                                                 : amp
s_nmda_mg                                                                                     : 1
s_ampa_mg                                                                                     : 1
I_nmda_gn = g_nmda_gn*(vm - E_nmda)*s_nmda_gn*(1.0/(1 + eta*Mg*exp(-gamma_g*vm)))             : amp
I_ampa_gn = g_ampa_gn*(vm - E_ampa)*s_ampa_gn                                                 : amp
s_nmda_gn                                                                                     : 1
s_ampa_gn                                                                                     : 1
I_gaba_hg = g_gaba_hg*(vm - E_gaba)*s_gaba_hg                                                 : amp
s_gaba_hg                                                                                     : 1
''')

# Soma equation
eqs_soma  = MembraneEquation(Cm * area_soma)
eqs_soma += leak_current(gl * area_soma, El_soma)
eqs_soma += IonicCurrent('I = I_synS : amp')
eqs_soma += eq_soma

# Dendrite equations
eqs_dendrite = {}
#    area_dend *= 3
for seg in xrange(Nseg):
    eqs_dendrite[seg]  = MembraneEquation(Cm_dend * area_dend[seg])
    eqs_dendrite[seg] += leak_current(gl_dend * area_dend[seg], El_dend, current_name = 'Il')
    eqs_dendrite[seg] += IonicCurrent('I = I_synD: amp') + eq_dend

granule_eqs = Compartments({'soma' : eqs_soma,
                           'dend001': eqs_dendrite[0],
                           'dend002': eqs_dendrite[1],
                           'dend011': eqs_dendrite[2],
                           'dend012': eqs_dendrite[3],
                           'dend02': eqs_dendrite[4],
                           'dend101': eqs_dendrite[5],
                           'dend102': eqs_dendrite[6],
                           'dend111': eqs_dendrite[7],
                           'dend112': eqs_dendrite[8],
                           'dend12': eqs_dendrite[9],
                           'dend201': eqs_dendrite[10],
                           'dend202': eqs_dendrite[11],
                           'dend211': eqs_dendrite[12],
                           'dend212': eqs_dendrite[13],
                           'dend22': eqs_dendrite[14]})

granule_eqs.connect('dend001', 'dend011', Ra_0)
granule_eqs.connect('dend002', 'dend012', Ra_0)
granule_eqs.connect('dend011', 'dend02', Ra_1)
granule_eqs.connect('dend012', 'dend02', Ra_1)
granule_eqs.connect('dend02', 'soma', Ra_2)

granule_eqs.connect('dend101', 'dend111', Ra_0)
granule_eqs.connect('dend102', 'dend112', Ra_0)
granule_eqs.connect('dend111', 'dend12', Ra_1)
granule_eqs.connect('dend112', 'dend12', Ra_1)
granule_eqs.connect('dend12', 'soma', Ra_2)

granule_eqs.connect('dend201', 'dend211', Ra_0)
granule_eqs.connect('dend202', 'dend212', Ra_0)
granule_eqs.connect('dend211', 'dend22', Ra_1)
granule_eqs.connect('dend212', 'dend22', Ra_1)
granule_eqs.connect('dend22', 'soma', Ra_2)

granule = NeuronGroup(N_granule, model = granule_eqs, threshold = 'vm_soma > v_th',
                     reset = 'vm_soma = v_reset; I_Sahp_soma += 0.0450*nA',
                     refractory = 20 * ms, compile = True, freeze = True)

# Initialization of membrane potential
granule.vm_soma = El_soma
# 1st branch
granule.vm_dend001 = El_dend
granule.vm_dend002 = El_dend
granule.vm_dend011 = El_dend
granule.vm_dend012 = El_dend
granule.vm_dend02  = El_dend

# 2nd branch
granule.vm_dend101 = El_dend
granule.vm_dend102 = El_dend
granule.vm_dend111 = El_dend
granule.vm_dend112 = El_dend
granule.vm_dend12  = El_dend

# 3rd branch
granule.vm_dend201 = El_dend
granule.vm_dend202 = El_dend
granule.vm_dend211 = El_dend
granule.vm_dend212 = El_dend
granule.vm_dend22  = El_dend


#Clustering of granule cells
counter = 20
N_cl = len(granule)/counter
granule_cl = {}
for gran in xrange(N_cl):
    granule_cl[gran] = granule.subgroup(counter)
#=======================================================================================================================

#=======================================================================================================================
# BASKET CELLS

# Parameters
gl_b         =  18.054  * nS # leakage conductance
El_b         = -52      * mV # reversal-resting potential
Cm_b         =   0.1793 * nF # membrane capacitance
v_th_b       = -39      * mV # threshold potential
v_reset_b    = -45      * mV # reset potential
DeltaT_b     =   2      * mV # slope factor

# Synaptic Parameters
gamma      =   0.072 * mV**-1   # Mg Concentration factor
alpha_nmda =   0.5   * ms**-1   # NMDA scale factor
alpha_ampa =   1     * ms**-1   # AMPA scale factor
alpha_gaba =   1     * ms**-1   # GABA scale factor

#AMPA/NMDA Kinetics
t_nmda_decay_b = 130.0  * ms # NMDA decay time constant
t_nmda_rise_b  =  10.0  * ms # NMDA rise time constant
t_ampa_decay_b =   4.2  * ms # AMPA decay time constant
t_ampa_rise_b  =   1.2  * ms # AMPA rise time constant

# NOISE
g_nmda_bn       =   2.5 * nS # NMDA maximum conductance
g_ampa_bn       =   3.5 * nS # AMPA maximum conductance
t_nmda_decay_bn = 130   * ms # NMDA decay time constant
t_nmda_rise_bn  =  10   * ms # NMDA rise time constant
t_ampa_decay_bn =   4.2 * ms # AMPA decay time constant
t_ampa_rise_bn  =   1.2 * ms # AMPA rise time constant


# Synaptic current equations
eq_soma_b = Equations('''
I_syn_b = I_nmda_gb + I_ampa_gb + I_nmda_mb + I_ampa_mb + I_nmda_bn + I_ampa_bn      : amp
I_nmda_gb = g_nmda_gb*(vm - E_nmda)*s_nmda_gb*(1.0/(1 + exp(-gamma*vm)*(1.0/3.57)))  : amp
I_ampa_gb = g_ampa_gb*(vm - E_ampa)*s_ampa_gb                                        : amp
s_nmda_gb                                                                            : 1
s_ampa_gb                                                                            : 1
I_nmda_mb = g_nmda_mb*(vm - E_nmda)*s_nmda_mb*(1.0/(1 + exp(-gamma*vm)*(1.0/3.57)))  : amp
I_ampa_mb = g_ampa_mb*(vm - E_ampa)*s_ampa_mb                                        : amp
s_nmda_mb                                                                            : 1
s_ampa_mb                                                                            : 1
I_nmda_bn  = g_nmda_bn*(vm - E_nmda)*s_nmda_bn*(1.0/(1 + exp(-gamma*vm)*(1.0/3.57))) : amp
I_ampa_bn  = g_ampa_bn*(vm - E_ampa)*s_ampa_bn                                       : amp
s_nmda_bn                                                                            : 1
s_ampa_bn                                                                            : 1
''')

# Brette-Gerstner
basket_eqs = Brette_Gerstner(Cm_b, gl_b, El_b, v_th_b, DeltaT_b, tauw = 100 * ms, a = .1 * nS)
basket_eqs += IonicCurrent('I = I_syn_b : amp')
basket_eqs += eq_soma_b

basket = NeuronGroup(N_basket, model = basket_eqs, threshold = 'vm > v_th_b',
                     reset = AdaptiveReset(Vr=v_reset_b, b = 0.0205*nA),
                     refractory = 2 * ms, compile = True)

# Initialization of membrane potential
basket.vm = El_b

basket_cl = {}
for bb in xrange(N_cl):
    basket_cl[bb] = basket.subgroup(1)
#=======================================================================================================================


#=======================================================================================================================
# MOSSY CELLS

# Parameters
gl_m           =   4.53   * nS      # leakage conductance
El_m           = -64      * mV      # reversal-resting potential
Cm_m           =   0.2521 * nfarad  # membrane capacitance
v_th_m         = -42      * mV      # threshold potential
v_reset_m      = -49      * mV      # reset potential
DeltaT_m       =   2      * mV      # slope factor

# Synaptic Parameters
gamma      =   0.072 * mV**-1   # Mg Concentration factor
alpha_nmda =   0.5   * ms**-1   # NMDA scale factor
alpha_ampa =   1     * ms**-1   # AMPA scale factor
alpha_gaba =   1     * ms**-1   # GABA scale factor

#AMPA/NMDA Kinetics
t_nmda_decay_m = 100     * ms  # NMDA decay time constant
t_nmda_rise_m  =   4     * ms  # NMDA rise time constant
t_ampa_decay_m =   6.2   * ms  # AMPA decay time constant
t_ampa_rise_m  =   0.5   * ms  # AMPA rise time constant

# Noise model Parameters
g_nmda_mn       =   4.465 * nS  # NMDA maximum conductance
g_ampa_mn       =   4.7   * nS  # AMPA maximum conductance
t_nmda_decay_mn = 100     * ms  # NMDA decay time constant
t_nmda_rise_mn  =   4     * ms  # NMDA rise time constant
t_ampa_decay_mn =   6.2   * ms  # AMPA decay time constant
t_ampa_rise_mn  =   0.5   * ms  # AMPA rise time constant

# Synaptic current equations
eq_soma_m = Equations('''
I_syn_m   = I_ampa_gm + I_nmda_gm + I_ampa_mn + I_nmda_mn                           : amp
I_nmda_gm = g_nmda_gm*(vm - E_nmda)*s_nmda_gm*(1.0/(1 + exp(-gamma*vm)*(1.0/3.57))) : amp
I_ampa_gm = g_ampa_gm*(vm - E_ampa)*s_ampa_gm                                       : amp
s_nmda_gm                                                                           : 1
s_ampa_gm                                                                           : 1
I_nmda_mn = g_nmda_mn*(vm - E_nmda)*s_nmda_mn*(1.0/(1 + exp(-gamma*vm)*(1.0/3.57))) : amp
I_ampa_mn = g_ampa_mn*(vm - E_ampa)*s_ampa_mn                                       : amp
s_nmda_mn                                                                           : 1
s_ampa_mn                                                                           : 1
''')

# Brette-Gerstner
mossy_eqs  = Brette_Gerstner(Cm_m, gl_m, El_m, v_th_m, DeltaT_m, tauw = 180 * ms, a = 1 * nS)
mossy_eqs += IonicCurrent('I = I_syn_m : amp')
mossy_eqs += eq_soma_m

mossy = NeuronGroup(N_mossy, model = mossy_eqs, threshold = 'vm > v_th_m',
                     reset = AdaptiveReset(Vr=v_reset_m, b = 0.0829*nA),
                     refractory = 2 * ms, compile = True)

# Initialization of membrane potential
mossy.vm = El_m
#=======================================================================================================================

#=======================================================================================================================
# HIPP CELLS
# Parameters
gl_h      =   1.930  * nS # leakage conductance
El_h      = -59      * mV # reversal-resting potential
Cm_h      =  0.0584  * nF # membrane capacitance
v_th_h    = -50      * mV # threshold potential
v_reset_h = -56      * mV # reset potential
DeltaT_h  =   2      * mV # slope factor

# Synaptic Parameters
gamma      =   0.072 * mV**-1   # Mg Concentration factor
alpha_nmda =   0.5   * ms**-1   # NMDA scale factor
alpha_ampa =   1     * ms**-1   # AMPA scale factor
alpha_gaba =   1     * ms**-1   # GABA scale factor

#AMPA/NMDA Kinetics
t_nmda_decay_h = 110    * ms  # NMDA decay time constant
t_nmda_rise_h  =   4.8  * ms  # NMDA rise time constant
t_ampa_decay_h =  11.0  * ms  # AMPA decay time constant
t_ampa_rise_h  =   2.0  * ms  # AMPA rise time constant
# NOISE
g_nmda_hn       =   0.2 * nS  # NMDA maximum conductance
g_ampa_hn       =   0.2 * nS  # AMPA maximum conductance
t_nmda_decay_hn = 100   * ms  # NMDA decay time constant
t_nmda_rise_hn  =  5.0  * ms  # NMDA rise time constant
t_ampa_decay_hn = 11.0  * ms  # AMPA decay time constant
t_ampa_rise_hn  =  2.0  * ms  # AMPA rise time constant

# Synaptic current equations
eq_soma_h = Equations('''
I_syn_h   = I_nmda_eh + I_ampa_eh + I_nmda_hn + I_ampa_hn                           : amp
I_nmda_eh = g_nmda_eh*(vm - E_nmda)*s_nmda_eh*1./(1 + exp(-gamma*vm)/3.57)          : amp
I_ampa_eh = g_ampa_eh*(vm - E_ampa)*s_ampa_eh                                       : amp
s_nmda_eh                                                                           : 1
s_ampa_eh                                                                           : 1
I_nmda_hn = g_nmda_hn*(vm - E_nmda)*s_nmda_hn*(1.0/(1 + exp(-gamma*vm)*(1.0/3.57))) : amp
I_ampa_hn = g_ampa_hn*(vm - E_ampa)*s_ampa_hn                                       : amp
s_nmda_hn                                                                           : 1
s_ampa_hn                                                                           : 1
''')

# Brette-Gerstner
hipp_eqs  = Brette_Gerstner(Cm_h, gl_h, El_h, v_th_h, DeltaT_h, tauw = 93 * ms, a = .82 * nS)
hipp_eqs += IonicCurrent('I = I_syn_h : amp')
hipp_eqs += eq_soma_h


hipp = NeuronGroup(N_hipp, model = hipp_eqs, threshold = 'vm > v_th_h',
                     reset = AdaptiveReset(Vr=v_reset_h, b = 0.015*nA),
                     refractory = 3 * ms, compile = True)

# Initialization of membrane potential
hipp.vm = El_h
#=======================================================================================================================

#=======================================================================================================================
# ***************************************  C  O  N  N  E  C  T  I  O  N  S  ********************************************
#=======================================================================================================================
os.chdir('ConnectivityMatrices_6dendrites/')

#  EC CELLS ----> GRANULE CELLS
a = 1.0
# Synapses at 1st dendrite
nmda_eqs_eg1 = '''
dj_eg1/dt = -j_eg1 / t_nmda_decay_g + alpha_nmda_g * x_eg1 * (1 - j_eg1) : 1
dx_eg1/dt = -x_eg1 / t_nmda_rise_g                                       : 1
wNMDA_eg1                                                                : 1
'''
synNMDA_eg1 = Synapses(Input_ec, granule, model = nmda_eqs_eg1, pre = 'x_eg1 += wNMDA_eg1', implicit=True, freeze=True)
granule.s_nmda_eg_dend001 = synNMDA_eg1.j_eg1
synNMDA_eg1.load_connectivity('syn_eg1.txt')
synNMDA_eg1.wNMDA_eg1[:, :] = 1.0*a
synNMDA_eg1.delay[:, :]     = 3 * ms

ampa_eqs_eg1 = '''
dy_eg1/dt = -y_eg1 / t_ampa_decay_g + alpha_ampa * h_eg1 * (1 - y_eg1) : 1
dh_eg1/dt = -h_eg1 / t_ampa_rise_g                                     : 1
wAMPA_eg1                                                              : 1
'''
synAMPA_eg1 = Synapses(Input_ec, granule, model = ampa_eqs_eg1, pre = 'h_eg1 += wAMPA_eg1', implicit=True, freeze=True)
granule.s_ampa_eg_dend001 = synAMPA_eg1.y_eg1
synAMPA_eg1.load_connectivity('syn_eg1.txt')
synAMPA_eg1.wAMPA_eg1[:, :] = 1.0*a
synAMPA_eg1.delay[:, :]     = 3 * ms

# Synapses at 2nd dendrite
nmda_eqs_eg2 = '''
dj_eg2/dt = -j_eg2 / t_nmda_decay_g + alpha_nmda_g * x_eg2 * (1 - j_eg2) : 1
dx_eg2/dt = -x_eg2 / t_nmda_rise_g                                       : 1
wNMDA_eg2                                                                : 1
'''
synNMDA_eg2 = Synapses(Input_ec, granule, model = nmda_eqs_eg2, pre = 'x_eg2 += wNMDA_eg2', implicit=True, freeze=True)
granule.s_nmda_eg_dend002 = synNMDA_eg2.j_eg2
synNMDA_eg2.load_connectivity('syn_eg2.txt')
synNMDA_eg2.wNMDA_eg2[:, :] = 1.0*a
synNMDA_eg2.delay[:, :]     = 3 * ms

ampa_eqs_eg2 = '''
dy_eg2/dt = -y_eg2 / t_ampa_decay_g + alpha_ampa * h_eg2 * (1 - y_eg2) : 1
dh_eg2/dt = -h_eg2 / t_ampa_rise_g                                     : 1
wAMPA_eg2                                                              : 1
'''
synAMPA_eg2 = Synapses(Input_ec, granule, model = ampa_eqs_eg2, pre = 'h_eg2 += wAMPA_eg2', implicit=True, freeze=True)
granule.s_ampa_eg_dend002 = synAMPA_eg2.y_eg2
synAMPA_eg2.load_connectivity('syn_eg2.txt')
synAMPA_eg2.wAMPA_eg2[:, :] = 1.0*a
synAMPA_eg2.delay[:, :]     = 3 * ms


# Synapses at 3rd dendrite
nmda_eqs_eg3 = '''
dj_eg3/dt = -j_eg3 / t_nmda_decay_g + alpha_nmda_g * x_eg3 * (1 - j_eg3) : 1
dx_eg3/dt = -x_eg3 / t_nmda_rise_g                                       : 1
wNMDA_eg3                                                                : 1
'''
synNMDA_eg3 = Synapses(Input_ec, granule, model = nmda_eqs_eg3, pre = 'x_eg3 += wNMDA_eg3', implicit=True, freeze=True)
granule.s_nmda_eg_dend101 = synNMDA_eg3.j_eg3
synNMDA_eg3.load_connectivity('syn_eg3.txt')
synNMDA_eg3.wNMDA_eg3[:, :] = 1.0*a
synNMDA_eg3.delay[:, :]     = 3 * ms

ampa_eqs_eg3 = '''
dy_eg3/dt = -y_eg3 / t_ampa_decay_g + alpha_ampa * h_eg3 * (1 - y_eg3) : 1
dh_eg3/dt = -h_eg3 / t_ampa_rise_g                                     : 1
wAMPA_eg3                                                              : 1
'''
synAMPA_eg3 = Synapses(Input_ec, granule, model = ampa_eqs_eg3, pre = 'h_eg3 += wAMPA_eg3', implicit=True, freeze=True)
granule.s_ampa_eg_dend101 = synAMPA_eg3.y_eg3
synAMPA_eg3.load_connectivity('syn_eg3.txt')
synAMPA_eg3.wAMPA_eg3[:, :] = 1.0*a
synAMPA_eg3.delay[:, :]     = 3 * ms


# Synapses at 4th dendrite
nmda_eqs_eg4 = '''
dj_eg4/dt = -j_eg4 / t_nmda_decay_g + alpha_nmda_g * x_eg4 * (1 - j_eg4) : 1
dx_eg4/dt = -x_eg4 / t_nmda_rise_g                                       : 1
wNMDA_eg4                                                                : 1
'''
synNMDA_eg4 = Synapses(Input_ec, granule, model = nmda_eqs_eg4, pre = 'x_eg4 += wNMDA_eg4', implicit=True, freeze=True)
granule.s_nmda_eg_dend102 = synNMDA_eg4.j_eg4
synNMDA_eg4.load_connectivity('syn_eg4.txt')
synNMDA_eg4.wNMDA_eg4[:, :] = 1.0*a
synNMDA_eg4.delay[:, :]     = 3 * ms

ampa_eqs_eg4 = '''
dy_eg4/dt = -y_eg4 / t_ampa_decay_g + alpha_ampa * h_eg4 * (1 - y_eg4) : 1
dh_eg4/dt = -h_eg4 / t_ampa_rise_g                                     : 1
wAMPA_eg4                                                              : 1
'''
synAMPA_eg4 = Synapses(Input_ec, granule, model = ampa_eqs_eg4, pre = 'h_eg4 += wAMPA_eg4', implicit=True, freeze=True)
granule.s_ampa_eg_dend102 = synAMPA_eg4.y_eg4
synAMPA_eg4.load_connectivity('syn_eg4.txt')
synAMPA_eg4.wAMPA_eg4[:, :] = 1.0*a
synAMPA_eg4.delay[:, :]     = 3 * ms

# Synapses at 5th dendrite
nmda_eqs_eg5 = '''
dj_eg5/dt = -j_eg5 / t_nmda_decay_g + alpha_nmda_g * x_eg5 * (1 - j_eg5) : 1
dx_eg5/dt = -x_eg5 / t_nmda_rise_g                                       : 1
wNMDA_eg5                                                                : 1
'''
synNMDA_eg5 = Synapses(Input_ec, granule, model = nmda_eqs_eg5, pre = 'x_eg5 += wNMDA_eg5', implicit=True, freeze=True)
granule.s_nmda_eg_dend201 = synNMDA_eg5.j_eg5
synNMDA_eg5.load_connectivity('syn_eg5.txt')
synNMDA_eg5.wNMDA_eg5[:, :] = 1.0*a
synNMDA_eg5.delay[:, :]     = 3 * ms

ampa_eqs_eg5 = '''
dy_eg5/dt = -y_eg5 / t_ampa_decay_g + alpha_ampa * h_eg5 * (1 - y_eg5) : 1
dh_eg5/dt = -h_eg5 / t_ampa_rise_g                                     : 1
wAMPA_eg5                                                              : 1
'''
synAMPA_eg5 = Synapses(Input_ec, granule, model = ampa_eqs_eg5, pre = 'h_eg5 += wAMPA_eg5', implicit=True, freeze=True)
granule.s_ampa_eg_dend201 = synAMPA_eg5.y_eg5
synAMPA_eg5.load_connectivity('syn_eg5.txt')
synAMPA_eg5.wAMPA_eg5[:, :] = 1.0*a
synAMPA_eg5.delay[:, :]     = 3 * ms


# Synapses at 6th dendrite
nmda_eqs_eg6 = '''
dj_eg6/dt = -j_eg6 / t_nmda_decay_g + alpha_nmda_g * x_eg6 * (1 - j_eg6) : 1
dx_eg6/dt = -x_eg6 / t_nmda_rise_g                                       : 1
wNMDA_eg6                                                                : 1
'''
synNMDA_eg6 = Synapses(Input_ec, granule, model = nmda_eqs_eg6, pre = 'x_eg6 += wNMDA_eg6', implicit=True, freeze=True)
granule.s_nmda_eg_dend202 = synNMDA_eg6.j_eg6
synNMDA_eg6.load_connectivity('syn_eg6.txt')
synNMDA_eg6.wNMDA_eg6[:, :] = 1.0*a
synNMDA_eg6.delay[:, :]     = 3 * ms

ampa_eqs_eg6 = '''
dy_eg6/dt = -y_eg6 / t_ampa_decay_g + alpha_ampa * h_eg6 * (1 - y_eg6) : 1
dh_eg6/dt = -h_eg6 / t_ampa_rise_g                                     : 1
wAMPA_eg6                                                              : 1
'''
synAMPA_eg6 = Synapses(Input_ec, granule, model = ampa_eqs_eg6, pre = 'h_eg6 += wAMPA_eg6', implicit=True, freeze=True)
granule.s_ampa_eg_dend202 = synAMPA_eg6.y_eg6
synAMPA_eg6.load_connectivity('syn_eg6.txt')
synAMPA_eg6.wAMPA_eg6[:, :] = 1.0*a
synAMPA_eg6.delay[:, :]     = 3 * ms


# EC CELLS ---> HIPP CELLS
# The NMDA/AMPA synapses @ hipp cell
nmda_eqs_eh = '''
dj_eh/dt = -j_eh / t_nmda_decay_h + alpha_nmda * x_eh * (1 - j_eh) : 1
dx_eh/dt = -x_eh / t_nmda_rise_h                                   : 1
wNMDA_eh                                                           : 1
'''
synNMDA_eh = Synapses(Input_ec, hipp, model = nmda_eqs_eh, pre = 'x_eh += wNMDA_eh', implicit=True, freeze=True)
hipp.s_nmda_eh = synNMDA_eh.j_eh
synNMDA_eh.load_connectivity('syn_eh.txt')
synNMDA_eh.wNMDA_eh[:, :] = 1.0
synNMDA_eh.delay[:, :]    = 3.0 * ms

ampa_eqs_eh = '''
dy_eh/dt = -y_eh / t_ampa_decay_h + h_eh*alpha_ampa*(1 - y_eh) : 1
dh_eh/dt = -h_eh / t_ampa_rise_h                               : 1
wAMPA_eh                                                       : 1
'''
synAMPA_eh = Synapses(Input_ec, hipp, model = ampa_eqs_eh, pre = 'h_eh += wAMPA_eh', implicit=True, freeze=True)
hipp.s_ampa_eh = synAMPA_eh.y_eh
synAMPA_eh.load_connectivity('syn_eh.txt')
synAMPA_eh.wAMPA_eh[:, :] = 1.0
synAMPA_eh.delay[:, :]    = 3.0 * ms

# GRANULE CELLS ---> MOSSY CELLS
# The NMDA/AMPA synapses @ mossy cell
nmda_eqs_gm = '''
dj_gm/dt = -j_gm / t_nmda_decay_m + alpha_nmda * x_gm * (1 - j_gm) : 1
dx_gm/dt = -x_gm / t_nmda_rise_m                                   : 1
wNMDA_gm                                                           : 1
'''
synNMDA_gm = Synapses(granule, mossy, model = nmda_eqs_gm, pre = 'x_gm += wNMDA_gm', implicit=True, freeze=True)
mossy.s_nmda_gm = synNMDA_gm.j_gm
synNMDA_gm.load_connectivity('syn_gm.txt')
synNMDA_gm.wNMDA_gm[:, :] = 1.0
synNMDA_gm.delay[:, :]    = 1.5 * ms

ampa_eqs_gm = '''
dy_gm/dt = -y_gm / t_ampa_decay_m + h_gm*alpha_ampa*(1 - y_gm) : 1
dh_gm/dt = -h_gm / t_ampa_rise_m                               : 1
wAMPA_gm                                                       : 1
'''
synAMPA_gm = Synapses(granule, mossy, model = ampa_eqs_gm, pre = 'h_gm += wAMPA_gm', implicit=True, freeze=True)
mossy.s_ampa_gm = synAMPA_gm.y_gm
synAMPA_gm.load_connectivity('syn_gm.txt')
synAMPA_gm.wAMPA_gm[:, :] = 1.0
synAMPA_gm.delay[:, :]    = 1.5 * ms

# GRANULE CELLS ---> BASKET CELLS
# The NMDA/AMPA synapses @ basket cell
synNMDA_gb = {}
synAMPA_gb = {}
for gtob in xrange(N_cl):
    nmda_eqs_gb = '''
    dj_gb/dt = -j_gb / t_nmda_decay_b + alpha_nmda * x_gb * (1 - j_gb) : 1
    dx_gb/dt = -x_gb / t_nmda_rise_b                                   : 1
    wNMDA_gb                                                           : 1
    '''
    synNMDA_gb[gtob] = Synapses(granule_cl[gtob], basket_cl[gtob], model = nmda_eqs_gb, pre = 'x_gb += wNMDA_gb', implicit=True, freeze=True)
    basket_cl[gtob].s_nmda_gb = synNMDA_gb[gtob].j_gb
    synNMDA_gb[gtob].connect_random(granule_cl[gtob], basket_cl[gtob], sparseness = 1.0)
    synNMDA_gb[gtob].wNMDA_gb[:, :] = 1.0
    synNMDA_gb[gtob].delay[:, :]    = 0.8 * ms

    ampa_eqs_gb = '''
    dy_gb/dt = -y_gb / t_ampa_decay_b + h_gb*alpha_ampa*(1 - y_gb) : 1
    dh_gb/dt = -h_gb / t_ampa_rise_b                               : 1
    wAMPA_gb                                                       : 1
    '''
    synAMPA_gb[gtob] = Synapses(granule_cl[gtob], basket_cl[gtob], model = ampa_eqs_gb, pre = 'h_gb += wAMPA_gb', implicit=True, freeze=True)
    basket_cl[gtob].s_ampa_gb = synAMPA_gb[gtob].y_gb
    synAMPA_gb[gtob].connect_random(granule_cl[gtob], basket_cl[gtob], sparseness = 1.0)
    synAMPA_gb[gtob].wAMPA_gb[:, :] = 1.0
    synAMPA_gb[gtob].delay[:, :]    = 0.8 * ms


# MOSSY CELLS ---> GRANULE CELLS
# The NMDA/AMPA synapses @ granule proximal dendrite (dendrite 2)
# 1st branch
nmda_eqs_mg1 = '''
dj_mg1/dt = -j_mg1 / t_nmda_decay_g + alpha_nmda_g * x_mg1 * (1 - j_mg1) : 1
dx_mg1/dt = -x_mg1 / t_nmda_rise_g                                       : 1
wNMDA_mg1                                                                : 1
'''
synNMDA_mg1 = Synapses(mossy, granule, model = nmda_eqs_mg1, pre = 'x_mg1 += wNMDA_mg1', implicit=True, freeze=True)
granule.s_nmda_mg_dend02 = synNMDA_mg1.j_mg1
synNMDA_mg1.load_connectivity('syn_mg1.txt')
synNMDA_mg1.wNMDA_mg1[:, :] = 1.0
synNMDA_mg1.delay[:, :]     = 3.0 * ms

ampa_eqs_mg1 = '''
dy_mg1/dt = -y_mg1 / t_ampa_decay_g + h_mg1*alpha_ampa*(1 - y_mg1) : 1
dh_mg1/dt = -h_mg1 / t_ampa_rise_g                                 : 1
wAMPA_mg1                                                          : 1
'''
synAMPA_mg1 = Synapses(mossy, granule, model = ampa_eqs_mg1, pre = 'h_mg1 += wAMPA_mg1', implicit=True, freeze=True)
granule.s_ampa_mg_dend02 = synAMPA_mg1.y_mg1
synAMPA_mg1.load_connectivity('syn_mg1.txt')
synAMPA_mg1.wAMPA_mg1[:, :] = 1.0
synAMPA_mg1.delay[:, :]     = 3.0 * ms

# 2nd branch
nmda_eqs_mg2 = '''
dj_mg2/dt = -j_mg2 / t_nmda_decay_g + alpha_nmda_g * x_mg2 * (1 - j_mg2) : 1
dx_mg2/dt = -x_mg2 / t_nmda_rise_g                                       : 1
wNMDA_mg2                                                                : 1
'''
synNMDA_mg2 = Synapses(mossy, granule, model = nmda_eqs_mg2, pre = 'x_mg2 += wNMDA_mg2', implicit=True, freeze=True)
granule.s_nmda_mg_dend12 = synNMDA_mg2.j_mg2
synNMDA_mg2.load_connectivity('syn_mg2.txt')
synNMDA_mg2.wNMDA_mg2[:, :] = 1.0
synNMDA_mg2.delay[:, :]     = 3.0 * ms

ampa_eqs_mg2 = '''
dy_mg2/dt = -y_mg2 / t_ampa_decay_g + h_mg2*alpha_ampa*(1 - y_mg2) : 1
dh_mg2/dt = -h_mg2 / t_ampa_rise_g                                 : 1
wAMPA_mg2                                                          : 1
'''
synAMPA_mg2 = Synapses(mossy, granule, model = ampa_eqs_mg2, pre = 'h_mg2 += wAMPA_mg2', implicit=True, freeze=True)
granule.s_ampa_mg_dend12 = synAMPA_mg2.y_mg2
synAMPA_mg2.load_connectivity('syn_mg2.txt')
synAMPA_mg2.wAMPA_mg2[:, :] = 1.0
synAMPA_mg2.delay[:, :]     = 3.0 * ms

# 3rd branch
nmda_eqs_mg3 = '''
dj_mg3/dt = -j_mg3 / t_nmda_decay_g + alpha_nmda_g * x_mg3 * (1 - j_mg3) : 1
dx_mg3/dt = -x_mg3 / t_nmda_rise_g                                       : 1
wNMDA_mg3                                                                : 1
'''
synNMDA_mg3 = Synapses(mossy, granule, model = nmda_eqs_mg3, pre = 'x_mg3 += wNMDA_mg3', implicit=True, freeze=True)
granule.s_nmda_mg_dend22 = synNMDA_mg3.j_mg3
synNMDA_mg3.load_connectivity('syn_mg3.txt')
synNMDA_mg3.wNMDA_mg3[:, :] = 1.0
synNMDA_mg3.delay[:, :]     = 3.0 * ms

ampa_eqs_mg3 = '''
dy_mg3/dt = -y_mg3 / t_ampa_decay_g + h_mg3*alpha_ampa*(1 - y_mg3) : 1
dh_mg3/dt = -h_mg3 / t_ampa_rise_g                                 : 1
wAMPA_mg3                                                          : 1
'''
synAMPA_mg3 = Synapses(mossy, granule, model = ampa_eqs_mg3, pre = 'h_mg3 += wAMPA_mg3', implicit=True, freeze=True)
granule.s_ampa_mg_dend22 = synAMPA_mg3.y_mg3
synAMPA_mg3.load_connectivity('syn_mg3.txt')
synAMPA_mg3.wAMPA_mg3[:, :] = 1.0
synAMPA_mg3.delay[:, :]     = 3.0 * ms

# MOSSY CELL ---> BASKET CELLS
# The NMDA/AMPA synapses @ basket cell
nmda_eqs_mb = '''
dj_mb/dt = -j_mb / t_nmda_decay_b + alpha_nmda * x_mb * (1 - j_mb) : 1
dx_mb/dt = -x_mb / t_nmda_rise_b                                   : 1
wNMDA_mb                                                           : 1
'''
synNMDA_mb = Synapses(mossy, basket, model = nmda_eqs_mb, pre = 'x_mb += wNMDA_mb', implicit=True, freeze=True)
basket.s_nmda_mb = synNMDA_mb.j_mb
synNMDA_mb.connect_random(mossy, basket, sparseness = 1.0)
synNMDA_mb.wNMDA_mb[:, :] = 1.0
synNMDA_mb.delay[:, :]    = 3.0 * ms

ampa_eqs_mb = '''
dy_mb/dt = -y_mb / t_ampa_decay_b + h_mb*alpha_ampa*(1 - y_mb) : 1
dh_mb/dt = -h_mb / t_ampa_rise_b                               : 1
wAMPA_mb                                                       : 1
'''
synAMPA_mb = Synapses(mossy, basket, model = ampa_eqs_mb, pre = 'h_mb += wAMPA_mb', implicit=True, freeze=True)
basket.s_ampa_mb = synAMPA_mb.y_mb
synAMPA_mb.connect_random(mossy, basket, sparseness = 1.0)
synAMPA_mb.wAMPA_mb[:, :] = 1.0
synAMPA_mb.delay[:, :]    = 3.0 * ms

# BASKET CELLS ----> GRANULE CELLS (INHIBITION @ soma)
# Synapses @ granule cell (soma)
syn_bg = {}
for btog in xrange(N_cl):
    gaba_eqs_bg = '''
    dz_bg/dt = -z_bg / t_gaba_decay_g + r_bg*alpha_gaba*(1 - z_bg) : 1
    dr_bg/dt = -r_bg / t_gaba_rise_g                               : 1
    w_bg                                                           : 1
    '''
    syn_bg[btog] = Synapses(basket_cl[btog], granule_cl[btog], model = gaba_eqs_bg, pre = 'r_bg += w_bg', implicit=True, freeze=True)
    granule_cl[btog].s_gaba_bg_soma = syn_bg[btog].z_bg
    syn_bg[btog].connect_random(basket_cl[btog], granule_cl[btog], sparseness = 1.0)
    syn_bg[btog].w_bg[:, :]  = 1.0
    syn_bg[btog].delay[:, :] = 0.85 * ms

# HIPP CELLS ----> GRANULE CELLS (INHIBITION @ distal dendrite)

# Synapses at granule cell distal dendrite (0)
# Synapses @ 1st dendrite
gaba_eqs_hg1 = '''
dz_hg1/dt = -z_hg1 / t_gaba_decay_g + r_hg1*alpha_gaba*(1 - z_hg1) : 1
dr_hg1/dt = -r_hg1 / t_gaba_rise_g                                 : 1
w_hg1                                                              : 1
'''
syn_hg1 = Synapses(hipp, granule, model = gaba_eqs_hg1, pre = 'r_hg1 += w_hg1', implicit=True, freeze=True)
granule.s_gaba_hg_dend001 = syn_hg1.z_hg1
syn_hg1.load_connectivity('syn_hg1.txt')
syn_hg1.w_hg1[:, :] = 1.0
syn_hg1.delay[:, :] = 1.6 * ms

# Synapses @ 2nd dendrite
gaba_eqs_hg2 = '''
dz_hg2/dt = -z_hg2 / t_gaba_decay_g + r_hg2*alpha_gaba*(1 - z_hg2) : 1
dr_hg2/dt = -r_hg2 / t_gaba_rise_g                                 : 1
w_hg2                                                              : 1
'''
syn_hg2 = Synapses(hipp, granule, model = gaba_eqs_hg2, pre = 'r_hg2 += w_hg2', implicit=True, freeze=True)
granule.s_gaba_hg_dend002 = syn_hg2.z_hg2
syn_hg2.load_connectivity('syn_hg2.txt')
syn_hg2.w_hg2[:, :] = 1.0
syn_hg2.delay[:, :] = 1.6 * ms

# Synapses @ 3rd dendrite
gaba_eqs_hg3 = '''
dz_hg3/dt = -z_hg3 / t_gaba_decay_g + r_hg3*alpha_gaba*(1 - z_hg3) : 1
dr_hg3/dt = -r_hg3 / t_gaba_rise_g                                 : 1
w_hg3                                                              : 1
'''
syn_hg3 = Synapses(hipp, granule, model = gaba_eqs_hg3, pre = 'r_hg3 += w_hg3', implicit=True, freeze=True)
granule.s_gaba_hg_dend101 = syn_hg3.z_hg3
syn_hg3.load_connectivity('syn_hg3.txt')
syn_hg3.w_hg3[:, :] = 1.0
syn_hg3.delay[:, :] = 1.6 * ms

# Synapses @ 4th dendrite
gaba_eqs_hg4 = '''
dz_hg4/dt = -z_hg4 / t_gaba_decay_g + r_hg4*alpha_gaba*(1 - z_hg4) : 1
dr_hg4/dt = -r_hg4 / t_gaba_rise_g                                 : 1
w_hg4                                                              : 1
'''
syn_hg4 = Synapses(hipp, granule, model = gaba_eqs_hg4, pre = 'r_hg4 += w_hg4', implicit=True, freeze=True)
granule.s_gaba_hg_dend102 = syn_hg4.z_hg4
syn_hg4.load_connectivity('syn_hg4.txt')
syn_hg4.w_hg4[:, :] = 1.0
syn_hg4.delay[:, :] = 1.6 * ms

# Synapses @ 5th dendrite
gaba_eqs_hg5 = '''
dz_hg5/dt = -z_hg5 / t_gaba_decay_g + r_hg5*alpha_gaba*(1 - z_hg5) : 1
dr_hg5/dt = -r_hg5 / t_gaba_rise_g                                 : 1
w_hg5                                                              : 1
'''
syn_hg5 = Synapses(hipp, granule, model = gaba_eqs_hg5, pre = 'r_hg5 += w_hg5', implicit=True, freeze=True)
granule.s_gaba_hg_dend201 = syn_hg5.z_hg5
syn_hg5.load_connectivity('syn_hg5.txt')
syn_hg5.w_hg5[:, :] = 1.0
syn_hg5.delay[:, :] = 1.6 * ms

# Synapses @ 6th dendrite
gaba_eqs_hg6 = '''
dz_hg6/dt = -z_hg6 / t_gaba_decay_g + r_hg6*alpha_gaba*(1 - z_hg6) : 1
dr_hg6/dt = -r_hg6 / t_gaba_rise_g                                 : 1
w_hg6                                                              : 1
'''
syn_hg6 = Synapses(hipp, granule, model = gaba_eqs_hg6, pre = 'r_hg6 += w_hg6', implicit=True, freeze=True)
granule.s_gaba_hg_dend202 = syn_hg6.z_hg6
syn_hg6.load_connectivity('syn_hg6.txt')
syn_hg6.w_hg6[:, :] = 1.0
syn_hg6.delay[:, :] = 1.6 * ms


############################################# N O I S E ################################################################
# GRANULE CELLS
# NOISE
noise_g = PoissonGroup(500, 0.1*Hz)
# DISTAL
# Synapses at dend001
nmda_eqs_gn001 = '''
dj_gn001/dt = -j_gn001 / t_nmda_decay_g + alpha_nmda_g * x_gn001 * (1 - j_gn001) : 1
dx_gn001/dt = -x_gn001 / t_nmda_rise_g                                           : 1
dy_gn001/dt = -y_gn001 / t_ampa_decay_g + alpha_ampa * h_gn001 * (1 - y_gn001)   : 1
dh_gn001/dt = -h_gn001 / t_ampa_rise_g                                           : 1
w_gn001                                                                          : 1
'''
syn_gn001 = Synapses(noise_g, granule, model = nmda_eqs_gn001,
                pre = 'x_gn001 = w_gn001; h_gn001 = w_gn001', implicit=True, freeze=True)
granule.s_nmda_gn_dend001 = syn_gn001.j_gn001
granule.s_ampa_gn_dend001 = syn_gn001.y_gn001
syn_gn001.connect_random(noise_g, granule, sparseness = 1.0/Nseg)
syn_gn001.w_gn001[:, :] = 1.0
syn_gn001.delay[:, :]   = '10 * rand() * ms'

# Synapses at dend002
nmda_eqs_gn002 = '''
dj_gn002/dt = -j_gn002 / t_nmda_decay_g + alpha_nmda_g * x_gn002 * (1 - j_gn002) : 1
dx_gn002/dt = -x_gn002 / t_nmda_rise_g                                           : 1
dy_gn002/dt = -y_gn002 / t_ampa_decay_g + alpha_ampa * h_gn002 * (1 - y_gn002)   : 1
dh_gn002/dt = -h_gn002 / t_ampa_rise_g                                           : 1
w_gn002                                                                          : 1
'''
syn_gn002 = Synapses(noise_g, granule, model = nmda_eqs_gn002,
                pre = 'x_gn002 = w_gn002; h_gn002 = w_gn002', implicit=True, freeze=True)
granule.s_nmda_gn_dend002 = syn_gn002.j_gn002
granule.s_ampa_gn_dend002 = syn_gn002.y_gn002
syn_gn002.connect_random(noise_g, granule, sparseness = 1.0/Nseg)
syn_gn002.w_gn002[:, :] = 1.0
syn_gn002.delay[:, :]   = '10 * rand() * ms'

# Synapses at dend101
nmda_eqs_gn101 = '''
dj_gn101/dt = -j_gn101 / t_nmda_decay_g + alpha_nmda_g * x_gn101 * (1 - j_gn101) : 1
dx_gn101/dt = -x_gn101 / t_nmda_rise_g                                           : 1
dy_gn101/dt = -y_gn101 / t_ampa_decay_g + alpha_ampa * h_gn101 * (1 - y_gn101)   : 1
dh_gn101/dt = -h_gn101 / t_ampa_rise_g                                           : 1
w_gn101                                                                          : 1
'''
syn_gn101 = Synapses(noise_g, granule, model = nmda_eqs_gn101,
                pre = 'x_gn101 = w_gn101; h_gn101 = w_gn101', implicit=True, freeze=True)
granule.s_nmda_gn_dend101 = syn_gn101.j_gn101
granule.s_ampa_gn_dend101 = syn_gn101.y_gn101
syn_gn101.connect_random(noise_g, granule, sparseness = 1.0/Nseg)
syn_gn101.w_gn101[:, :] = 1.0
syn_gn101.delay[:, :]   = '10 * rand() * ms'

# Synapses at dend102
nmda_eqs_gn102 = '''
dj_gn102/dt = -j_gn102 / t_nmda_decay_g + alpha_nmda_g * x_gn102 * (1 - j_gn102) : 1
dx_gn102/dt = -x_gn102 / t_nmda_rise_g                                           : 1
dy_gn102/dt = -y_gn102 / t_ampa_decay_g + alpha_ampa * h_gn102 * (1 - y_gn102)   : 1
dh_gn102/dt = -h_gn102 / t_ampa_rise_g                                           : 1
w_gn102                                                                          : 1
'''
syn_gn102 = Synapses(noise_g, granule, model = nmda_eqs_gn102,
                pre = 'x_gn102 = w_gn102; h_gn102 = w_gn102', implicit=True, freeze=True)
granule.s_nmda_gn_dend102 = syn_gn102.j_gn102
granule.s_ampa_gn_dend102 = syn_gn102.y_gn102
syn_gn102.connect_random(noise_g, granule, sparseness = 1.0/Nseg)
syn_gn102.w_gn102[:, :] = 1.0
syn_gn102.delay[:, :]   = '10 * rand() * ms'

# Synapses at dend201
nmda_eqs_gn201 = '''
dj_gn201/dt = -j_gn201 / t_nmda_decay_g + alpha_nmda_g * x_gn201 * (1 - j_gn201) : 1
dx_gn201/dt = -x_gn201 / t_nmda_rise_g                                           : 1
dy_gn201/dt = -y_gn201 / t_ampa_decay_g + alpha_ampa * h_gn201 * (1 - y_gn201)   : 1
dh_gn201/dt = -h_gn201 / t_ampa_rise_g                                           : 1
w_gn201                                                                          : 1
'''
syn_gn201 = Synapses(noise_g, granule, model = nmda_eqs_gn201,
                pre = 'x_gn201 = w_gn201; h_gn201 = w_gn201', implicit=True, freeze=True)
granule.s_nmda_gn_dend201 = syn_gn201.j_gn201
granule.s_ampa_gn_dend201 = syn_gn201.y_gn201
syn_gn201.connect_random(noise_g, granule, sparseness = 1.0/Nseg)
syn_gn201.w_gn201[:, :] = 1.0
syn_gn201.delay[:, :]   = '10 * rand() * ms'

# Synapses at dend202
nmda_eqs_gn202 = '''
dj_gn202/dt = -j_gn202 / t_nmda_decay_g + alpha_nmda_g * x_gn202 * (1 - j_gn202) : 1
dx_gn202/dt = -x_gn202 / t_nmda_rise_g                                           : 1
dy_gn202/dt = -y_gn202 / t_ampa_decay_g + alpha_ampa * h_gn202 * (1 - y_gn202)   : 1
dh_gn202/dt = -h_gn202 / t_ampa_rise_g                                           : 1
w_gn202                                                                          : 1
'''
syn_gn202 = Synapses(noise_g, granule, model = nmda_eqs_gn202,
                pre = 'x_gn202 = w_gn202; h_gn202 = w_gn202', implicit=True, freeze=True)
granule.s_nmda_gn_dend202 = syn_gn202.j_gn202
granule.s_ampa_gn_dend202 = syn_gn202.y_gn202
syn_gn202.connect_random(noise_g, granule, sparseness = 1.0/Nseg)
syn_gn202.w_gn202[:, :] = 1.0
syn_gn202.delay[:, :]   = '10 * rand() * ms'

# MEDIAL
# Synapses at dend011
nmda_eqs_gn011 = '''
dj_gn011/dt = -j_gn011 / t_nmda_decay_g + alpha_nmda_g * x_gn011 * (1 - j_gn011) : 1
dx_gn011/dt = -x_gn011 / t_nmda_rise_g                                           : 1
dy_gn011/dt = -y_gn011 / t_ampa_decay_g + alpha_ampa * h_gn011 * (1 - y_gn011)   : 1
dh_gn011/dt = -h_gn011 / t_ampa_rise_g                                           : 1
w_gn011                                                                          : 1
'''
syn_gn011 = Synapses(noise_g, granule, model = nmda_eqs_gn011,
                pre = 'x_gn011 = w_gn011; h_gn011 = w_gn011', implicit=True, freeze=True)
granule.s_nmda_gn_dend011 = syn_gn011.j_gn011
granule.s_ampa_gn_dend011 = syn_gn011.y_gn011
syn_gn011.connect_random(noise_g, granule, sparseness = 1.0/Nseg)
syn_gn011.w_gn011[:, :] = 1.0
syn_gn011.delay[:, :]   = '10 * rand() * ms'

# Synapses at dend002
nmda_eqs_gn012 = '''
dj_gn012/dt = -j_gn012 / t_nmda_decay_g + alpha_nmda_g * x_gn012 * (1 - j_gn012) : 1
dx_gn012/dt = -x_gn012 / t_nmda_rise_g                                           : 1
dy_gn012/dt = -y_gn012 / t_ampa_decay_g + alpha_ampa * h_gn012 * (1 - y_gn012)   : 1
dh_gn012/dt = -h_gn012 / t_ampa_rise_g                                           : 1
w_gn012                                                                          : 1
'''
syn_gn012 = Synapses(noise_g, granule, model = nmda_eqs_gn012,
                pre = 'x_gn012 = w_gn012; h_gn012 = w_gn012', implicit=True, freeze=True)
granule.s_nmda_gn_dend012 = syn_gn012.j_gn012
granule.s_ampa_gn_dend012 = syn_gn012.y_gn012
syn_gn012.connect_random(noise_g, granule, sparseness = 1.0/Nseg)
syn_gn012.w_gn012[:, :] = 1.0
syn_gn012.delay[:, :]   = '10 * rand() * ms'

# Synapses at dend111
nmda_eqs_gn111 = '''
dj_gn111/dt = -j_gn111 / t_nmda_decay_g + alpha_nmda_g * x_gn111 * (1 - j_gn111) : 1
dx_gn111/dt = -x_gn111 / t_nmda_rise_g                                           : 1
dy_gn111/dt = -y_gn111 / t_ampa_decay_g + alpha_ampa * h_gn111 * (1 - y_gn111)   : 1
dh_gn111/dt = -h_gn111 / t_ampa_rise_g                                           : 1
w_gn111                                                                          : 1
'''
syn_gn111 = Synapses(noise_g, granule, model = nmda_eqs_gn111,
                pre = 'x_gn111 = w_gn111; h_gn111 = w_gn111', implicit=True, freeze=True)
granule.s_nmda_gn_dend111 = syn_gn111.j_gn111
granule.s_ampa_gn_dend111 = syn_gn111.y_gn111
syn_gn111.connect_random(noise_g, granule, sparseness = 1.0/Nseg)
syn_gn111.w_gn111[:, :] = 1.0
syn_gn111.delay[:, :]   = '10 * rand() * ms'

# Synapses at dend112
nmda_eqs_gn112 = '''
dj_gn112/dt = -j_gn112 / t_nmda_decay_g + alpha_nmda_g * x_gn112 * (1 - j_gn112) : 1
dx_gn112/dt = -x_gn112 / t_nmda_rise_g                                           : 1
dy_gn112/dt = -y_gn112 / t_ampa_decay_g + alpha_ampa * h_gn112 * (1 - y_gn112)   : 1
dh_gn112/dt = -h_gn112 / t_ampa_rise_g                                           : 1
w_gn112                                                                          : 1
'''
syn_gn112 = Synapses(noise_g, granule, model = nmda_eqs_gn112,
                pre = 'x_gn112 = w_gn112; h_gn112 = w_gn112', implicit=True, freeze=True)
granule.s_nmda_gn_dend112 = syn_gn112.j_gn112
granule.s_ampa_gn_dend112 = syn_gn112.y_gn112
syn_gn112.connect_random(noise_g, granule, sparseness = 1.0/Nseg)
syn_gn112.w_gn112[:, :] = 1.0
syn_gn112.delay[:, :]   = '10 * rand() * ms'

# Synapses at dend211
nmda_eqs_gn211 = '''
dj_gn211/dt = -j_gn211 / t_nmda_decay_g + alpha_nmda_g * x_gn211 * (1 - j_gn211) : 1
dx_gn211/dt = -x_gn211 / t_nmda_rise_g                                           : 1
dy_gn211/dt = -y_gn211 / t_ampa_decay_g + alpha_ampa * h_gn211 * (1 - y_gn211)   : 1
dh_gn211/dt = -h_gn211 / t_ampa_rise_g                                           : 1
w_gn211                                                                          : 1
'''
syn_gn211 = Synapses(noise_g, granule, model = nmda_eqs_gn211,
                pre = 'x_gn211 = w_gn211; h_gn211 = w_gn211', implicit=True, freeze=True)
granule.s_nmda_gn_dend211 = syn_gn211.j_gn211
granule.s_ampa_gn_dend211 = syn_gn211.y_gn211
syn_gn211.connect_random(noise_g, granule, sparseness = 1.0/Nseg)
syn_gn211.w_gn211[:, :] = 1.0
syn_gn211.delay[:, :]   = '10 * rand() * ms'

# Synapses at dend212
nmda_eqs_gn212 = '''
dj_gn212/dt = -j_gn212 / t_nmda_decay_g + alpha_nmda_g * x_gn212 * (1 - j_gn212) : 1
dx_gn212/dt = -x_gn212 / t_nmda_rise_g                                           : 1
dy_gn212/dt = -y_gn212 / t_ampa_decay_g + alpha_ampa * h_gn212 * (1 - y_gn212)   : 1
dh_gn212/dt = -h_gn212 / t_ampa_rise_g                                           : 1
w_gn212                                                                          : 1
'''
syn_gn212 = Synapses(noise_g, granule, model = nmda_eqs_gn212,
                pre = 'x_gn212 = w_gn212; h_gn212 = w_gn212', implicit=True, freeze=True)
granule.s_nmda_gn_dend212 = syn_gn212.j_gn212
granule.s_ampa_gn_dend212 = syn_gn212.y_gn212
syn_gn212.connect_random(noise_g, granule, sparseness = 1.0/Nseg)
syn_gn212.w_gn212[:, :] = 1.0
syn_gn212.delay[:, :]   = '10 * rand() * ms'

# PROXIMAL
# Synapses at dend02
nmda_eqs_gn02 = '''
dj_gn02/dt = -j_gn02 / t_nmda_decay_g + alpha_nmda_g * x_gn02 * (1 - j_gn02) : 1
dx_gn02/dt = -x_gn02 / t_nmda_rise_g                                         : 1
dy_gn02/dt = -y_gn02 / t_ampa_decay_g + alpha_ampa * h_gn02 * (1 - y_gn02)   : 1
dh_gn02/dt = -h_gn02 / t_ampa_rise_g                                         : 1
w_gn02                                                                       : 1
'''
syn_gn02 = Synapses(noise_g, granule, model = nmda_eqs_gn02,
                pre = 'x_gn02 = w_gn02; h_gn02 = w_gn02', implicit=True, freeze=True)
granule.s_nmda_gn_dend02 = syn_gn02.j_gn02
granule.s_ampa_gn_dend02 = syn_gn02.y_gn02
syn_gn02.connect_random(noise_g, granule, sparseness = 1.0/Nseg)
syn_gn02.w_gn02[:, :] = 1.0
syn_gn02.delay[:, :]  = '10 * rand() * ms'

# Synapses at dend12
nmda_eqs_gn12 = '''
dj_gn12/dt = -j_gn12 / t_nmda_decay_g + alpha_nmda_g * x_gn12 * (1 - j_gn12) : 1
dx_gn12/dt = -x_gn12 / t_nmda_rise_g                                         : 1
dy_gn12/dt = -y_gn12 / t_ampa_decay_g + alpha_ampa * h_gn12 * (1 - y_gn12)   : 1
dh_gn12/dt = -h_gn12 / t_ampa_rise_g                                         : 1
w_gn12                                                                       : 1
'''
syn_gn12 = Synapses(noise_g, granule, model = nmda_eqs_gn12,
                pre = 'x_gn12 = w_gn12; h_gn12 = w_gn12', implicit=True, freeze=True)
granule.s_nmda_gn_dend12 = syn_gn12.j_gn12
granule.s_ampa_gn_dend12 = syn_gn12.y_gn12
syn_gn12.connect_random(noise_g, granule, sparseness = 1.0/Nseg)
syn_gn12.w_gn12[:, :] = 1.0
syn_gn12.delay[:, :]  = '10 * rand() * ms'

# Synapses at dend22
nmda_eqs_gn22 = '''
dj_gn22/dt = -j_gn22 / t_nmda_decay_g + alpha_nmda_g * x_gn22 * (1 - j_gn22) : 1
dx_gn22/dt = -x_gn22 / t_nmda_rise_g                                         : 1
dy_gn22/dt = -y_gn22 / t_ampa_decay_g + alpha_ampa * h_gn22 * (1 - y_gn22)   : 1
dh_gn22/dt = -h_gn22 / t_ampa_rise_g                                         : 1
w_gn22                                                                       : 1
'''
syn_gn22 = Synapses(noise_g, granule, model = nmda_eqs_gn22,
                pre = 'x_gn22 = w_gn22; h_gn22 = w_gn22', implicit=True, freeze=True)
granule.s_nmda_gn_dend22 = syn_gn22.j_gn22
granule.s_ampa_gn_dend22 = syn_gn22.y_gn22
syn_gn22.connect_random(noise_g, granule, sparseness = 1.0/Nseg)
syn_gn22.w_gn22[:, :] = 1.0
syn_gn22.delay[:, :]  = '10 * rand() * ms'


# BASKET CELLS
noise_b = PoissonGroup(20*N_basket, 3*Hz)
noise_b_cl = {}
for no in xrange(N_basket):
    noise_b_cl[no] = noise_b.subgroup(20)

# Synapses at basket cell (noise role)
syn_bn = {}
for cell0 in xrange(N_basket):
    nmda_eqs_bn = '''
    dj_bn/dt = -j_bn / t_nmda_decay_bn + alpha_nmda * x_bn * (1 - j_bn) : 1
    dx_bn/dt = -x_bn / t_nmda_rise_bn                                   : 1
    dy_bn/dt = -y_bn / t_ampa_decay_bn + alpha_ampa * h_bn * (1 - y_bn) : 1
    dh_bn/dt = -h_bn / t_ampa_rise_bn                                   : 1
    w_bn                                                                : 1
    '''
    syn_bn[cell0] = Synapses(noise_b_cl[cell0], basket_cl[cell0], model = nmda_eqs_bn,
                    pre = 'x_bn = w_bn; h_bn = w_bn')
    basket_cl[cell0].s_nmda_bn = syn_bn[cell0].j_bn
    basket_cl[cell0].s_ampa_bn = syn_bn[cell0].y_bn
    syn_bn[cell0].connect_random(noise_b_cl[cell0], basket_cl[cell0], sparseness = 1.0)
    syn_bn[cell0].w_bn[:, :]  = 1.0
    syn_bn[cell0].delay[:, :] = '10 * rand() * ms'

# MOSSY CELLS
noise = PoissonGroup(30*N_mossy, 3.8*Hz)
noise_cl = {}
mossy_cl = {}
for no in xrange(N_mossy):
    noise_cl[no] = noise.subgroup(20)
    mossy_cl[no] = mossy[no]

# Synapses at mossy cell (noise role)
syn_mn = {}
for kk in xrange(N_mossy):
    nmda_eqs_mn = '''
    dj_mn/dt = -j_mn / t_nmda_decay_mn + alpha_nmda * x_mn * (1 - j_mn) : 1
    dx_mn/dt = -x_mn / t_nmda_rise_mn                                   : 1
    dy_mn/dt = -y_mn / t_ampa_decay_mn + alpha_ampa * h_mn * (1 - y_mn) : 1
    dh_mn/dt = -h_mn / t_ampa_rise_mn                                   : 1
    w_mn                                                                : 1
    '''
    syn_mn[kk] = Synapses(noise_cl[kk], mossy_cl[kk], model = nmda_eqs_mn,
                    pre = 'x_mn = w_mn; h_mn = w_mn')
    mossy_cl[kk].s_nmda_mn = syn_mn[kk].j_mn
    mossy_cl[kk].s_ampa_mn = syn_mn[kk].y_mn
    syn_mn[kk].connect_random(noise_cl[kk], mossy_cl[kk], sparseness = 1.0)
    syn_mn[kk].w_mn[:, :]  = 1.0
    syn_mn[kk].delay[:, :] = '10 * rand() * ms'


# HIPP Cells
noise_h = PoissonGroup(20*N_hipp, 3*Hz)
noise_h_cl = {}
hipp_cl = {}
for no in xrange(N_hipp):
    noise_h_cl[no] = noise_h.subgroup(20)
    hipp_cl[no] = hipp[no]

# Synapses at hipp cell (noise role)
syn_hn = {}
for cell2 in xrange(N_hipp):
    nmda_eqs_hn = '''
    dj_hn/dt = -j_hn / t_nmda_decay_hn + alpha_nmda * x_hn * (1 - j_hn) : 1
    dx_hn/dt = -x_hn / t_nmda_rise_hn                                   : 1
    dy_hn/dt = -y_hn / t_ampa_decay_hn + alpha_ampa * h_hn * (1 - y_hn) : 1
    dh_hn/dt = -h_hn / t_ampa_rise_hn                                   : 1
    w_hn                                                                : 1
    '''
    syn_hn[cell2] = Synapses(noise_h_cl[cell2], hipp_cl[cell2], model = nmda_eqs_hn,
                    pre = 'x_hn = w_hn; h_hn = w_hn')
    hipp_cl[cell2].s_nmda_hn = syn_hn[cell2].j_hn
    hipp_cl[cell2].s_ampa_hn = syn_hn[cell2].y_hn
    syn_hn[cell2].connect_random(noise_h_cl[cell2], hipp_cl[cell2], sparseness = 1.0)
    syn_hn[cell2].w_hn[:, :]  = 1.0
    syn_hn[cell2].delay[:, :] = '10 * rand() * ms'
#=======================================================================================================================

#=======================================================================================================================
# MONITORING
I_S = SpikeMonitor(Input_ec)
G_S = SpikeMonitor(granule)

#=======================================================================================================================

#=======================================================================================================================
# ***************************************  S  I  M  U  L  A  T  I  O  N  S  ********************************************
#=======================================================================================================================
#Simulation run

start_timestamp = time.time()

run(t1+simtime+t2, report='text', report_period = 1 *second)

sim_duration = time.time() - start_timestamp
print "\nDuration of simulation: " + str(sim_duration)

os.chdir('../')
output_pattern = []
for spikes in xrange(N_granule):
    output_pattern.append(len(G_S[spikes]))
np.save('output_pattern6d_'+str(Trial), output_pattern)

input_pattern = []
for spikes_i in xrange(len(Input_ec)):
    input_pattern.append(len(I_S[spikes_i]))
np.save('input_pattern6d_'+str(Trial), input_pattern)

