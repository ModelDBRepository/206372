#==============================================================================
# Point neuron (exponential I&F model) with 6 dendrites.
# This script is used for the experiment: peak dendritic voltage per number of
# synapses!
#==============================================================================

from brian import *
from brian.library.ionic_currents import *
from brian.library.synapses import *
from brian.library.IF import *
import numpy as np
import time
import math
import random


def constrained_sum_sample_pos(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(random.sample(xrange(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]


def constrained_sum_sample_nonneg(n, total):
    """Return a randomly chosen list of n nonnegative
    integers summing to total.
    Each such list is equally likely to occur."""

    return [x - 1 for x in constrained_sum_sample_pos(n, total + n)]


def synchronous_input(A, N_pre):
    "function to built a synchronous input"
    C = []
    for jj in range(len(A)):
        for ll in xrange(N_pre):
            B = list(A[jj])
            B[0] = ll
            B = tuple(B)
            C.append(B)
    C.sort(key = lambda tup: tup[1])
    return C


start_time = time.time()


#Initialisation of the simulator
clear(erase = True, all = True)
reinit(states = True)
defaultclock.dt = 0.1*ms

N_granule = 1
NumberOfInputs = 1
# Parameters
gl      =   0.00003 * siemens/(cm**2) # leakage conductance
gl_dend =   0.00001 * siemens/(cm**2) # leakage conductance
El_soma = -87.0     * mV               # reversal-resting potential
El_dend = -82.0     * mV               # reversal-resting potential
Cm      =   1.0     * uF/(cm**2)       # membrane capacitance
Cm_dend =   2.5     * uF/(cm**2)       # membrane capacitance
v_th    = -56.0     * mV               # threshold potential
v_reset = -74.0     * mV               # reset potential

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
length_dend  = [distal_l, distal_l, distal_l, distal_l, medial_l, medial_l, proximal_l]
length_dend *= Nbranch
distal_d   = 0.80 * um
medial_d   = 0.90 * um
proximal_d = 1.00 * um
diam_dend    = [distal_d, distal_d, distal_d, distal_d, medial_d, medial_d, proximal_d]
diam_dend   *= Nbranch
area_dend    = [math.pi*x*y for x,y in zip(length_dend, diam_dend)]

# Synaptic Reversal Potentials
E_nmda =  0.0  * mV  # NMDA reversal potential
E_ampa =  0.0  * mV  # AMPA reversal potential
E_gaba = -70.0 * mV  # GABA reversal potential

# AMPA/NMDA/GABA Model Parameters
gama       = 0.04 * mV**-1 # the steepness of Mg sensitivity of Mg unblock
Mg         = 2.0  # [mM]--mili Molar - the extracellular Magnesium concentration
eta        = 0.2 # [mM**-1] -1- mili Molar **(-1) - Magnesium sensitivity of unblock
alpha_nmda = 2.0  * ms**-1
alpha_ampa = 1.0  * ms**-1
alpha_gaba = 1.0  * ms**-1


# EC Synapses
g_ampa = 0.8066 * nS  # AMPA maximum conductance
g_nmda = 1.0800*g_ampa
t_nmda_decay = 50.0  * ms  # NMDA decay time constant
t_nmda_rise  =  0.33 * ms  # NMDA rise time constant
t_ampa_decay =  2.5  * ms  # AMPA decay time constant
t_ampa_rise  =  0.1  * ms  # AMPA rise time constant

# GABAergic Input from basket cells/hipp cells
g_gaba       = 2.8  * nS  # GABA maximum conductance
t_gaba_decay = 6.8  * ms  # GABA decay time constant
t_gaba_rise  = 0.9  * ms  # GABA rise time constant

# Axial resistances
Ri   = 210.0 * ohm * cm
ra0  = Ri * 4 / (pi * distal_d ** 2)
ra1  = Ri * 4 / (pi * medial_d ** 2)
ra2  = Ri * 4 / (pi * proximal_d ** 2)
Ra_0 = ra0 * distal_l
Ra_1 = ra1 * medial_l
Ra_2 = ra2 * proximal_l
# AHP patrameters
tau_ahp = 45*ms
g_ahp   = 2*nS
# Synaptic current equations @ SOMA
eq_soma = Equations('''
I_synS = I_gaba_g - I_inj + I_Sahp               : amp
I_Sahp                                           : amp
dI_Sahp/dt = (g_ahp*(vm-El_soma)-I_Sahp)/tau_ahp : amp
I_gaba_g = g_gaba*(vm - E_gaba)*s_gaba_g         : amp
s_gaba_g                                         : 1
I_inj                                            : amp
''')

# Synaptic current equations @ dendrites
eq_dend = Equations('''
I_synD = I_nmda_g + I_ampa_g + I_gaba_g - I_inj                       : amp
I_nmda_g = g_nmda*(vm - E_nmda)*s_nmda_g/(1.0 + eta*Mg*exp(-gama*vm)) : amp
s_nmda_g                                                              : 1
I_ampa_g = g_ampa*(vm - E_ampa)*s_ampa_g                              : amp
s_ampa_g                                                              : 1
I_gaba_g = g_gaba*(vm - E_gaba)*s_gaba_g                              : amp
s_gaba_g                                                              : 1
I_inj                                                                 : amp
''')

# Soma equation
eqs_soma  = MembraneEquation(Cm * area_soma)
eqs_soma += leak_current(gl * area_soma, El_soma)
eqs_soma += IonicCurrent('I = I_synS : amp')
eqs_soma += eq_soma

# Dendrite equations
eqs_dendrite = {}
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
granule.vm_soma   = El_soma
# 1st branch
granule.vm_dend001 = El_dend
granule.vm_dend002 = El_dend
granule.vm_dend011 = El_dend
granule.vm_dend012 = El_dend
granule.vm_dend02 = El_dend

# 2nd branch
granule.vm_dend101 = El_dend
granule.vm_dend102 = El_dend
granule.vm_dend111 = El_dend
granule.vm_dend112 = El_dend
granule.vm_dend12 = El_dend

# 3rd branch
granule.vm_dend201 = El_dend
granule.vm_dend202 = El_dend
granule.vm_dend211 = El_dend
granule.vm_dend212 = El_dend
granule.vm_dend22 = El_dend

#==============================================================================
# ***************************** I N P U T S ***********************************
#==============================================================================
N_pre = NumberOfInputs

#==========================================================================
tspike = 500
spiketimes = [(0, tspike * ms)]
spiketimes = synchronous_input(spiketimes, N_pre)
#==========================================================================
# Only 1st tip synapse! Reproduce results of Remy, 2009
P001 = SpikeGeneratorGroup(N_pre, spiketimes)

#==============================================================================
# ****************************S Y N A P S E S**********************************
#==============================================================================
# The NMDA/AMPA synapses
# DISTAL SYNAPSES
# Synapses at 0 - branch
nmda_eqs_001 = '''
dj_001/dt = -j_001 / t_nmda_decay + alpha_nmda * x_001 * (1 - j_001) : 1
dx_001/dt = -x_001 / t_nmda_rise                                     : 1
wNMDA_001                                                            : 1
'''
synNMDA_001 = Synapses(P001, granule, model = nmda_eqs_001, pre = 'x_001 = wNMDA_001', implicit=True, freeze=True)
granule.s_nmda_g_dend001 = synNMDA_001.j_001
synNMDA_001[:,:] = True
synNMDA_001.wNMDA_001[:, :] = 1.0
synNMDA_001.delay[:, :]     = 3 * ms

ampa_eqs_001 = '''
dy_001/dt = -y_001 / t_ampa_decay + alpha_ampa * h_001 * (1 - y_001) : 1
dh_001/dt = -h_001 / t_ampa_rise                                     : 1
wAMPA_001                                                            : 1
'''
synAMPA_001 = Synapses(P001, granule, model = ampa_eqs_001, pre = 'h_001 = wAMPA_001', implicit=True, freeze=True)
granule.s_ampa_g_dend001 = synAMPA_001.y_001
synAMPA_001[:,:] = True
synAMPA_001.wAMPA_001[:, :] = 1.0
synAMPA_001.delay[:, :]     = 3 * ms

#==============================================================================


#==============================================================================
#***************************** M O N I T O R S ********************************
#==============================================================================
M0 = MultiStateMonitor(granule, record = True)
#==============================================================================

#==============================================================================
#**************************S I M U L A T I O N ********************************
#==============================================================================

#Simulation run
total_simt  = 1000 * ms
print "\nSimulation running... "

run(total_simt, report='text')
rest = int(tspike/(defaultclock.dt/ms))
V_soma  = (1000*(max(M0['vm_soma'][0][3000:]) - M0['vm_soma'][0][rest]))
print V_soma

V_peak001  = (1000*(max(M0['vm_dend001'][0][3000:]) - M0['vm_dend001'][0][rest]))
V_peak002  = (1000*(max(M0['vm_dend002'][0][3000:]) - M0['vm_dend002'][0][rest]))
V_peak011  = (1000*(max(M0['vm_dend011'][0][3000:]) - M0['vm_dend011'][0][rest]))
V_peak012  = (1000*(max(M0['vm_dend012'][0][3000:]) - M0['vm_dend012'][0][rest]))
V_peak02  = (1000*(max(M0['vm_dend02'][0][3000:]) - M0['vm_dend02'][0][rest]))

V_peak101  = (1000*(max(M0['vm_dend101'][0][3000:]) - M0['vm_dend101'][0][rest]))
V_peak102  = (1000*(max(M0['vm_dend102'][0][3000:]) - M0['vm_dend102'][0][rest]))
V_peak111  = (1000*(max(M0['vm_dend111'][0][3000:]) - M0['vm_dend111'][0][rest]))
V_peak112  = (1000*(max(M0['vm_dend112'][0][3000:]) - M0['vm_dend112'][0][rest]))
V_peak12  = (1000*(max(M0['vm_dend12'][0][3000:]) - M0['vm_dend12'][0][rest]))

V_peak201  = (1000*(max(M0['vm_dend201'][0][3000:]) - M0['vm_dend201'][0][rest]))
V_peak202  = (1000*(max(M0['vm_dend202'][0][3000:]) - M0['vm_dend202'][0][rest]))
V_peak211  = (1000*(max(M0['vm_dend211'][0][3000:]) - M0['vm_dend211'][0][rest]))
V_peak212  = (1000*(max(M0['vm_dend212'][0][3000:]) - M0['vm_dend212'][0][rest]))
V_peak22  = (1000*(max(M0['vm_dend22'][0][3000:]) - M0['vm_dend22'][0][rest]))

V_peak = [V_soma, V_peak001, V_peak002, V_peak011, V_peak012, V_peak02,
          V_peak101, V_peak102, V_peak111, V_peak112, V_peak12,
          V_peak201, V_peak202, V_peak211, V_peak212, V_peak22]


np.save('Vpeak_6d_supralinear_single_'+str(NumberOfInputs), V_peak)
