#==============================================================================
# Point neuron (exponential I&F model) with 12 dendrites - Control model.
#==============================================================================

from brian import *
from brian.library.ionic_currents import *
from brian.library.IF import *
import numpy as np
import time
import math
from brian.library.electrophysiology import *
import matplotlib.pyplot as plt
import scipy.io

doVisualise = 0
doData = 0
doSpikes = 0

#Iinjected = range(-50, 90, 10)
Iinjected = [-100]
#Iinjected = range(0, 260, 25)

Vpeak  = np.zeros(len(Iinjected))
Spikes = np.zeros(len(Iinjected))

for jj in range(len(Iinjected)):
    reinit(states = True)
    clear(erase = True, all = True)

    # Number of presynaptic neurons
    N_pre  = 1
    N_granule = 1

    # Parameters
    gl      =   0.00003  * siemens/(cm**2) # leakage conductance
    gl_dend =   0.00001  * siemens/(cm**2) # leakage conductance
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
    Nseg    = 21 # Number of dendritic compartments
    Nbranch =  3 # Number of main branches
    Ntips   = 12 # Number of distal dendritic compartments
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
    E_gaba = -86.0 * mV  # GABA reversal potential

    # AMPA/NMDA/GABA Model Parameters
    gamma      = 0.04 * mV**-1 # the steepness of Mg sensitivity of Mg unblock
    Mg         = 2.0  # [mM]--mili Molar - the extracellular Magnesium concentration
    eta        = 0.2 # [mM**-1] -1- mili Molar **(-1) - Magnesium sensitivity of unblock
    alpha_nmda = 1.0  * ms**-1
    alpha_ampa = 1.0  * ms**-1
    alpha_gaba = 1.0  * ms**-1

    # Input to Granule cells from Perforant Path(EC)/Mossy Cells (Remy model & experiments)

    # Supralinear dendrites - working good, discussion with Yiota
    g_ampa = 0.8066 * nS  # AMPA maximum conductance
    g_nmda = 1.08*g_ampa

    t_nmda_decay = 50.0  * ms  # NMDA decay time constant
    t_nmda_rise  =  0.33 * ms  # NMDA rise time constant
    t_ampa_decay =  2.5  * ms  # AMPA decay time constant
    t_ampa_rise  =  0.1  * ms  # AMPA rise time constant

    # GABAergic Input from basket cells/hipp cells
    # Basket
    g_gaba       = 10.0  * nS  # GABA maximum conductance
    t_gaba_decay = 6.4  * ms  # GABA decay time constant
    t_gaba_rise  = 0.7  * ms  # GABA rise time constant

    # HIPP
    g_gaba_d       = 2.2  * nS  # GABA maximum conductance
    t_gaba_decay_d = 40.8 * ms  # GABA decay time constant
    t_gaba_rise_d  = 2.5  * ms  # GABA rise time constant

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
    g_ahp   = 2.0*nS
    # Synaptic current equations @ SOMA
    eq_soma = Equations('''
    I_synS = I_gaba_g - I_inj + I_Sahp               : amp
    I_Sahp                                           : amp
    dI_Sahp/dt = (g_ahp*(vm-El_soma)-I_Sahp)/tau_ahp : amp
    I_gaba_g = g_gaba*(vm - E_gaba)*s_gaba_g         : amp
    s_gaba_g                                         : 1
    I_inj                                            : amp
    clamp : 1
    ''')

    # Synaptic current equations @ dendrites
    eq_dend = Equations('''
    I_synD = I_nmda_g + I_ampa_g + I_gaba_g - I_inj                        : amp
    I_nmda_g = g_nmda*(vm - E_nmda)*s_nmda_g/(1.0 + eta*Mg*exp(-gamma*vm)) : amp
    s_nmda_g                                                               : 1
    I_ampa_g = g_ampa*(vm - E_ampa)*s_ampa_g                               : amp
    s_ampa_g                                                               : 1
    I_gaba_g = g_gaba_d*(vm - E_gaba)*s_gaba_g                             : amp
    s_gaba_g                                                               : 1
    I_inj                                                                  : amp
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
                               'dend00': eqs_dendrite[4],
                               'dend01': eqs_dendrite[5],
                               'dend0': eqs_dendrite[6],
                               'dend101': eqs_dendrite[7],
                               'dend102': eqs_dendrite[8],
                               'dend111': eqs_dendrite[9],
                               'dend112': eqs_dendrite[10],
                               'dend10': eqs_dendrite[11],
                               'dend11': eqs_dendrite[12],
                               'dend1': eqs_dendrite[13],
                               'dend201': eqs_dendrite[14],
                               'dend202': eqs_dendrite[15],
                               'dend211': eqs_dendrite[16],
                               'dend212': eqs_dendrite[17],
                               'dend20': eqs_dendrite[18],
                               'dend21': eqs_dendrite[19],
                               'dend2': eqs_dendrite[20]})

    granule_eqs.connect('dend001', 'dend00', Ra_0)
    granule_eqs.connect('dend002', 'dend00', Ra_0)
    granule_eqs.connect('dend011', 'dend01', Ra_0)
    granule_eqs.connect('dend012', 'dend01', Ra_0)
    granule_eqs.connect('dend00', 'dend0', Ra_1)
    granule_eqs.connect('dend01', 'dend0', Ra_1)
    granule_eqs.connect('dend0', 'soma', Ra_2)

    granule_eqs.connect('dend101', 'dend10', Ra_0)
    granule_eqs.connect('dend102', 'dend10', Ra_0)
    granule_eqs.connect('dend111', 'dend11', Ra_0)
    granule_eqs.connect('dend112', 'dend11', Ra_0)
    granule_eqs.connect('dend10', 'dend1', Ra_1)
    granule_eqs.connect('dend11', 'dend1', Ra_1)
    granule_eqs.connect('dend1', 'soma', Ra_2)

    granule_eqs.connect('dend201', 'dend20', Ra_0)
    granule_eqs.connect('dend202', 'dend20', Ra_0)
    granule_eqs.connect('dend211', 'dend21', Ra_0)
    granule_eqs.connect('dend212', 'dend21', Ra_0)
    granule_eqs.connect('dend20', 'dend2', Ra_1)
    granule_eqs.connect('dend21', 'dend2', Ra_1)
    granule_eqs.connect('dend2', 'soma', Ra_2)

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
    granule.vm_dend00  = El_dend
    granule.vm_dend01  = El_dend
    granule.vm_dend0   = El_dend

    # 2nd branch
    granule.vm_dend101 = El_dend
    granule.vm_dend102 = El_dend
    granule.vm_dend111 = El_dend
    granule.vm_dend112 = El_dend
    granule.vm_dend10  = El_dend
    granule.vm_dend11  = El_dend
    granule.vm_dend1   = El_dend

    # 3rd branch
    granule.vm_dend201 = El_dend
    granule.vm_dend202 = El_dend
    granule.vm_dend211 = El_dend
    granule.vm_dend212 = El_dend
    granule.vm_dend20  = El_dend
    granule.vm_dend21  = El_dend
    granule.vm_dend2   = El_dend

#======================================================================


#======================================================================
    # Monitors
    S    = SpikeMonitor(granule)

    M0 = MultiStateMonitor(granule, record = True)
#    M2 = MultiStateMonitor(granule, {'I_Sahp_soma'}, record = True)
    M3 = MultiStateMonitor(granule, {'I_ampa_g_dend00'+str(1),\
    'I_nmda_g_dend00'+str(1), 'I_gaba_g_soma', 'I_synD_dend00'+str(1),\
    'I_inj_soma', 'I_Sahp_soma','I_gaba_g_dend00'+str(1)}, record = True)

#======================================================================
#==============================================================================
#**************************S I M U L A T I O N *************************
#==============================================================================

    #Simulation run
    simt  = 200 * ms
    total_simt = 10*simt
    rateA =  0 * Hz
    rateB =  20 * Hz

#    @network_operation
#    def apply_soma_injections():
#        if (defaultclock.t>300 * ms) and (defaultclock.t<=1300*ms):
#            granule.I_inj_soma = Iinjected[jj]*pA
#        else:
#            granule.I_inj_soma = 0* pA


    print "Simulation running..."
    start_time = time.time()
    run(300*ms, report='text')
    granule.I_inj_soma = Iinjected[jj]*pA
    run(1000*ms, report='text')
    granule.I_inj_soma = 0*pA
    run(200*ms, report='text')
    duration = time.time() - start_time

    print "Number of somatic spikes: " + str(S.nspikes)
    print
    print "Simulation time: ", duration, "seconds"

    if Iinjected[jj] >= 0:
        Vpeak[jj] = 1000*(max(M0['vm_soma'][0][10000:15000])  - M0['vm_soma'][0][2999]) # in mvolts
        print
        print Vpeak
    else:
        sag_ratio = (El_soma/volt - M0['vm_soma'][0][12000])/(El_soma/volt - min(M0['vm_soma'][0]))
        R_input   = ((El_soma/volt - M0['vm_soma'][0][12000])*volt) / (-Iinjected[jj] * pA)
        R_input_d = ((El_dend/volt - M0['vm_dend00'][0][12000])*volt) / (-Iinjected[jj] * pA)
        Vpeak[jj] = 1000*(min(M0['vm_soma'][0][10000:15000])  - M0['vm_soma'][0][2999]) # in mvolt
        print
        print "Sag ratio: "+str(sag_ratio), "Rin: " + str(R_input), "Rind: " + str(R_input_d)


    Spikes[jj] = S.nspikes


#==============================================================================
#**************************V I S U A L I Z A T I O N *************************
#==============================================================================
total_simt = 1400 * ms
step = 500

if doData == 1:
    np.save('Granule_Voltage', [Vpeak, Iinjected])
    fig = figure(1)
    plot(Iinjected, Vpeak, '.')
    xlabel('Iinj (pA)')
    ylabel('Peak voltage from rest (mV)')

    ax = gca()
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_label_text('I (pA)')
    ax.xaxis.set_label_coords(1, .26)
    ax.yaxis.set_label_text('V (mV)')
    ax.yaxis.set_label_coords(0.22, 0.8)
    ax.yaxis.get_label().set_rotation('horizontal')
    savefig('granuleCell_12dendrites.eps', format = 'eps', bbox_inches = 'tight', dpi = 1200)
#    savefig('granuleCell_12dendrites.svg', format = 'svg', bbox_inches = 'tight', dpi = 1200)

if doSpikes == 1:
    np.save ('Granule_Spikes', [Spikes, Iinjected])
    fig = figure(2)
    ax = plt.subplot(111)
    ax.plot(Iinjected, Spikes, '-o')
    xlabel('Current injection (pA)', fontsize = 16)
    ylabel('Firing frequency (Hz)', fontsize = 16)
    xticks(range(0, 260, 50))

    zed = [tick.label.set_fontsize(14) for tick in ax.yaxis.get_major_ticks()]
    zed = [tick.label.set_fontsize(14) for tick in ax.xaxis.get_major_ticks()]
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    savefig('granule_FiringRates.eps', format = 'eps', bbox_inches='tight', dpi = 1200)
#    savefig('granule_FiringRates.svg', format = 'svg', bbox_inches='tight', dpi = 1200)

    scipy.io.savemat('granule_Spikes.mat', dict(x=Iinjected, y=Spikes))

    plt.show()

matplotlib.rcParams['pdf.fonttype'] = 42

if doVisualise == 1:


    fig = figure(1)

    vm_soma = M0['vm_soma'][0]
    for _,t in S.spikes:
        i_spike = int(t / defaultclock.dt)
        vm_soma[i_spike] = (v_th + 80*mV)
    plot(M0['vm_soma'].times / ms, vm_soma / mV)
    ylabel('V [mV]', fontsize = 16)
    xlabel('Time [ms]', fontsize = 16)
    ax = gca()
    zed = [tick.label.set_fontsize(14) for tick in ax.yaxis.get_major_ticks()]
    zed = [tick.label.set_fontsize(14) for tick in ax.xaxis.get_major_ticks()]

    ax.plot([1550, 1550], [0,20], linewidth = 1.6, color = 'blue')
    ax.plot([1550,2050], [0, 0], linewidth = 1.6, color = 'blue')

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    title('Iinjected = '+ str(Iinjected[jj])+' pA', fontsize = 18)
    scipy.io.savemat('granule_voltageTrace.mat', dict(x=M0.times / ms, y=vm_soma / mV))

    if Iinjected[jj] > 0:
        savefig('Granule12d_voltageTrace.eps', format='eps', bbox_inches='tight', dpi = 1200)
    else:
        savefig('Granule12d_voltageTrace_negative.eps', format='eps', bbox_inches='tight', dpi = 1200)

#    savefig('Granule12d_voltageTrace.svg', format='svg', bbox_inches='tight', dpi = 1200)

    fig = figure(2)
    for i in range(size(granule)):
        plot(M3['I_inj_soma'].times / ms, M3['I_inj_soma'][i] / pA)
    ylabel('I_inj (pA)')
    xlabel('time (ms)')

    show()
##==============================================================================
