#==============================================================================
# Point neuron (exponential I&F model) Basket cell.
#==============================================================================

from brian import *
from brian.library.ionic_currents import *
from brian.library.IF import *
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.io

doVisualise = 1
doData = 0
doSpikes = 0

#Iinjected = range(-100, 220, 20)
Iinjected = [210]
#Iinjected = range(0, 1000, 100)

Vpeak  = np.zeros(len(Iinjected))
Spikes = np.zeros(len(Iinjected))

for jj in range(len(Iinjected)):
    reinit(states = True)
    clear(erase = True, all = True)
    defaultclock.dt = 0.1*ms

    # Parameters
    gl_b         =  18.054  * nS # leakage conductance
    El_b         = -52      * mV # reversal-resting potential
    Cm_b         =   0.1793 * nF # membrane capacitance
    v_th_b       = -39      * mV # threshold potential
    v_reset_b    = -45      * mV # reset potential
    DeltaT_b     =   2      * mV # slope factor

    # Synaptic Reversal Potentials
    E_nmda       =  0  * mV  # NMDA reversal potential
    E_ampa       =  0  * mV  # AMPA reversal potential
    E_gaba       = -75 * mV  # GABA reversal potential

    # AMPA/NMDA/GABA Model Parameters
    gamma        = 0.072 * mV**-1
    alpha_nmda   = 0.5   * ms**-1
    alpha_ampa   = 1     * ms**-1
    alpha_gaba   = 1     * ms**-1

    # Input to basket cell (from Granule Cells or/and Mossy Cells)
    g_nmda_b     =   0.5 * nS  # NMDA maximum conductance
    g_ampa_b     =   0.7 * nS  # AMPA maximum conductance
    t_nmda_decay = 130   * ms  # NMDA decay time constant
    t_nmda_rise  =  10   * ms  # NMDA rise time constant
    t_ampa_decay =   4.2 * ms  # AMPA decay time constant
    t_ampa_rise  =   1.2 * ms  # AMPA rise time constant

    # GABAergic Input
    g_gaba_b     = 1.6  * nS  # GABA maximum conductance
    t_gaba_decay = 5.5  * ms  # GABA decay time constant
    t_gaba_rise  = 0.26 * ms  # GABA rise time constant

#    # Morphology
#    # Soma
#    length_basket   =  (600) * um
#    diam_basket     =  4 * um
#    area_basket     = pi * diam_basket * length_basket

    # Synaptic current equations
    eq_soma_b = Equations('''
    I_syn  = - I_inj : amp
    I_inj            : amp
    ''')
    # Synaptic current equations
    eq_soma_b = Equations('''
    I_syn  = I_nmda + I_ampa + I_gaba - I_inj                           : amp
    I_nmda = g_nmda_b*(vm - E_nmda)*s_nmda*1./(1 + exp(-gamma*vm)/3.57) : amp
    I_ampa = g_ampa_b*(vm - E_ampa)*s_ampa                              : amp
    I_gaba = g_gaba_b*(vm - E_gaba)*s_gaba                              : amp
    I_inj                                                               : amp
    s_nmda                                                              : 1
    s_ampa                                                              : 1
    s_gaba                                                              : 1
    ''')

    # Brette-Gerstner
    basket_eqs  = Brette_Gerstner(Cm_b, gl_b, El_b, v_th_b, DeltaT_b, tauw = 100 * ms, a = .1 * nS)
    basket_eqs += IonicCurrent('I = I_syn : amp')
    basket_eqs += eq_soma_b

    basket = NeuronGroup(1, model = basket_eqs, threshold = 'vm > v_th_b',
                         reset = AdaptiveReset(Vr=v_reset_b, b = 0.0205*nA),
                         refractory = 2 * ms, compile = True)

    # Initialization of membrane potential
    basket.vm = El_b

    #==============================================================================
    N_pre = 100
    # Input stochastic for each run
    P0 = PoissonGroup(N_pre)
    P1 = PoissonGroup(500)

    # The NMDA/AMPA synapses

    # Synapses at 1st branch
    nmda_eqs0 = '''
    dj/dt = -j / t_nmda_decay + alpha_nmda * x * (1 - j)   : 1
    dx/dt = -x / t_nmda_rise                               : 1
    dy/dt = -y / t_ampa_decay + h*alpha_ampa*(1 - y)       : 1
    dh/dt = -h / t_ampa_rise                               : 1
    w                                                      : 1
    '''

    syn_0 = Synapses(P0, basket, model = nmda_eqs0,
                    pre = 'x += 1; h += 1')
    basket.s_nmda = syn_0.j
    basket.s_ampa = syn_0.y
    syn_0.connect_random(P0, basket, sparseness = 0.6)
    syn_0.w[:, :]           = 1.0
    syn_0.delay[:, :]       = 0.8 * ms

    # Synapses at 1st branch
    nmda_eqs1 = '''
    dj/dt = -j / t_nmda_decay + alpha_nmda * x * (1 - j)   : 1
    dx/dt = -x / t_nmda_rise                               : 1
    dy/dt = -y / t_ampa_decay + h*alpha_ampa*(1 - y)       : 1
    dh/dt = -h / t_ampa_rise                               : 1
    w                                                      : 1
    '''

    syn_1 = Synapses(P1, basket, model = nmda_eqs1,
                    pre = 'x = w; h = w')
    basket.s_nmda = syn_1.j
    basket.s_ampa = syn_1.y
    syn_1.connect_random(P1, basket, sparseness = 0.9)
    syn_1.w[:, :]           = 1.0
    syn_1.delay[:, :]       = 0.8 * ms
    #==============================================================================
    # # Monitors
    S  = SpikeMonitor(basket)
    M0 = StateMonitor(basket, 'vm', record = True)
    M1 = StateMonitor(basket, 'I_inj', record = True)

    #======================================================================
    # SIMULATION
    #======================================================================

    #Simulation run
    simt  = 200 * ms
    total_simt = 10*simt
    rateA =  0 * Hz
    rateB =  0 * Hz
    rateC =  0 * Hz

    @network_operation
    def apply_soma_injections():
        if (defaultclock.t>300 * ms) and (defaultclock.t<=1300*ms):
            basket.I_inj = Iinjected[jj]*pA
        else:
            basket.I_inj = 0 * pA



    print "Simulation running..."
    start_time = time.time()
    run(total_simt, report='text')
    duration = time.time() - start_time

    print "Number of somatic spikes: " + str(S.nspikes)
    print

    print "Simulation time: ", duration, "seconds"
    if Iinjected[jj] >= 0:
        Vpeak[jj] = 1000*(max(M0[0])  - El_b / volt) # in mvolts
    else:
        sag_ratio = (El_b/volt - M0[0][12000])/(El_b/volt - min(M0[0]))
        R_input   = ((El_b/volt - M0[0][12000])*volt) / (-Iinjected[jj] * pA)
        Vpeak[jj] = 1000*(min(M0[0])  - El_b / volt) # in mvolts
        print
        print "Sag ratio: "+str(sag_ratio)
        print
        print "Rin: " + str(R_input)
        print
        print "time constant: " +str(R_input*Cm_b)

    Spikes[jj] = S.nspikes


##=============================================================================
# Visualization
if doData == 1:
    np.save('basket_Voltage', [Vpeak, Iinjected])
    fig = figure(1)
    plot(Iinjected, Vpeak, '.')
    m, b = np.polyfit(Iinjected, Vpeak, 1)
    plot(Iinjected, [m*x + b for x in Iinjected], '--')
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
    ax.xaxis.set_ticks(range(-200, 0, 50) + range(50, 260, 50))
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_ticks(range(-10, 0, 5) + range(5, 16, 5))
    ax.xaxis.set_label_text('I (pA)')
    ax.xaxis.set_label_coords(0.85, .3)
    ax.yaxis.set_label_text('V (mV)')
    ax.yaxis.set_label_coords(0.35, 0.9)
    ax.yaxis.get_label().set_rotation('horizontal')

    savefig('basketCell.eps', format = 'eps', bbox_inches = 'tight', dpi = 1200)

if doSpikes == 1:
    np.save ('basket_Spikes', [Spikes, Iinjected])
    fig = figure(2)
    ax = plt.subplot(111)
    ax.plot(Iinjected, Spikes, '-o')
    xlabel('Current injection (pA)', fontsize = 16)
    ylabel('Firing frequency (Hz)', fontsize = 16)
    xticks(range(0, 1000, 200))

    zed = [tick.label.set_fontsize(14) for tick in ax.yaxis.get_major_ticks()]
    zed = [tick.label.set_fontsize(14) for tick in ax.xaxis.get_major_ticks()]
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    savefig('basket_FiringRates.eps', format = 'eps', bbox_inches='tight', dpi = 1200)

    scipy.io.savemat('basket_Spikes.mat', dict(x=Iinjected, y=Spikes))
    plt.show()

step = 100
if doVisualise == 1:

    ax = plt.subplot(111)
    vm_soma = M0[0]
    for _,t in S.spikes:
        i = int(t / defaultclock.dt)
        vm_soma[i] = (v_th_b + 78*mV)
    ax.plot(M0.times / ms, vm_soma / mV)
    ylabel('V [mV]', fontsize = 16)
    xlabel('Time [ms]', fontsize = 16)
    zed = [tick.label.set_fontsize(14) for tick in ax.yaxis.get_major_ticks()]
    zed = [tick.label.set_fontsize(14) for tick in ax.xaxis.get_major_ticks()]

    ax.plot([1550, 1550], [20,40], linewidth = 1.6, color = 'blue')
    ax.plot([1550,2050], [20, 20], linewidth = 1.6, color = 'blue')

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    title('Iinjected = '+ str(Iinjected[jj])+' pA', fontsize = 18)
#    scipy.io.savemat('basket_voltageTrace.mat', dict(x=M0.times / ms, y=vm_soma / mV))


    if Iinjected[jj] > 0:
        savefig('basket_voltageTrace.eps', format='eps', dpi = 1200)
    else:
        savefig('basket_voltageTrace_negative.eps', format='eps', dpi = 1200)
#    savefig('basket_voltageTrace.svg', format='svg', dpi = 1200)


    fig = figure(2)
    for i in range(size(basket)):
        plot(M1.times / ms, M1[i] / pA)
    ylabel('I_inj (pA)')
    xlabel('time (ms)')

    plt.show()
##==============================================================================