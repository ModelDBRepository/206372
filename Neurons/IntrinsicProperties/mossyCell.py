#==============================================================================
# Point neuron (exponential I&F model) mossy cell.
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

#Iinjected = range(-150, -40, 50) + range(-40, 50, 10) + range(50, 150, 50)
Iinjected = [250]
#Iinjected = range(0,1100, 100)

Vpeak  = np.zeros(len(Iinjected))
Spikes = np.zeros(len(Iinjected))

for jj in range(len(Iinjected)):
    reinit(states = True)
    clear(erase = True, all = True)

    # Number of presynaptic neurons
    N_pre  = 1
    N_mossy = 1
    # Parameters
    gl_m           =   4.53   * nS      # leakage conductance
    El_m           = -64      * mV      # reversal-resting potential
    Cm_m           =   0.2521 * nfarad  # membrane capacitance
    v_th_m         = -42      * mV      # threshold potential
    v_reset_m      = -49      * mV      # reset potential
    DeltaT_m       =   2      * mV      # slope factor

    # Soma
#    length_mossy   = (20 + 18400)   * um
#    diam_mossy     = 1.8   * um
#    area_mossy     = pi * diam_mossy * length_mossy

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
    g_nmda_m     =   1.1 * nS  # NMDA maximum conductance
    g_ampa_m     =   0.7 * nS  # AMPA maximum conductance
    t_nmda_decay = 100   * ms  # NMDA decay time constant
    t_nmda_rise  =   4   * ms  # NMDA rise time constant
    t_ampa_decay =   6.2 * ms  # AMPA decay time constant
    t_ampa_rise  =   0.5 * ms  # AMPA rise time constant

    # GABAergic Input
    g_gaba_m     = 1.6  * nS  # GABA maximum conductance
    t_gaba_decay = 5.5  * ms  # GABA decay time constant
    t_gaba_rise  = 0.26 * ms  # GABA rise time constant

    # Synaptic current equations
    eq_soma_m = Equations('''
    I_syn_m  = - I_inj : amp
    I_inj            : amp
    ''')
    # Synaptic current equations
    eq_soma_m = Equations('''
    I_syn_m  = I_nmda + I_ampa + I_gaba - I_inj                         : amp
    I_nmda = g_nmda_m*(vm - E_nmda)*s_nmda*1./(1 + exp(-gamma*vm)/3.57) : amp
    I_ampa = g_ampa_m*(vm - E_ampa)*s_ampa                              : amp
    I_gaba = g_gaba_m*(vm - E_gaba)*s_gaba                              : amp
    I_inj                                                               : amp
    s_nmda                                                              : 1
    s_ampa                                                              : 1
    s_gaba                                                              : 1
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

    #==============================================================================

    #==============================================================================
    # Monitors
    S  = SpikeMonitor(mossy)
    M0 = StateMonitor(mossy, 'vm', record = True)
    M1 = StateMonitor(mossy, 'I_inj', record = True)

    #======================================================================
    # SIMULATION
    #======================================================================

    #Simulation run
    simt  = 200 * ms
    total_simt = 10*simt
    rateA =  0 * Hz
    rateB =  20 * Hz

    @network_operation
    def apply_soma_injections():
        if (defaultclock.t>300 * ms) and (defaultclock.t<=1300*ms):
            mossy.I_inj = Iinjected[jj]*pA
        else:
            mossy.I_inj = 0* pA


    print "Simulation running..."
    start_time = time.time()
    run(total_simt, report='text')
    duration = time.time() - start_time

    print "Number of somatic spikes: " + str(S.nspikes)
    print

    print "Simulation time: ", duration, "seconds"
    if Iinjected[jj] >= 0:
        Vpeak[jj] = 1000*(max(M0[0])  - El_m / volt) # in mvolts
    else:
        sag_ratio = (El_m/volt - M0[0][12000])/(El_m/volt - min(M0[0]))
        R_input   = ((El_m/volt - M0[0][12000])*volt) / (-Iinjected[jj] * pA)
        Vpeak[jj] = 1000*(min(M0[0])  - El_m / volt) # in mvolts
        print
        print "Sag ratio: "+str(sag_ratio)
        print
        print "Rin: " + str(R_input)
        print
        print "time constant: " +str(R_input*Cm_m)

    Spikes[jj] = S.nspikes


##=============================================================================
# Visualization
if doData == 1:
    np.save('mossy_Voltage', [Vpeak, Iinjected])
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
    ax.xaxis.set_ticks(range(-200, 0, 50) + range(50, 160, 50))
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_ticks(range(-20, 0, 5) + range(5, 21, 5))
    ax.xaxis.set_label_text('I (pA)')
    ax.xaxis.set_label_coords(1, .4)
    ax.yaxis.set_label_text('V (mV)')
    ax.yaxis.set_label_coords(0.50, 0.9)
    ax.yaxis.get_label().set_rotation('horizontal')

    savefig('mossyCell.eps', format = 'eps', bbox_inches = 'tight', dpi = 1200)
#    savefig('mossyCell.svg', format = 'svg', bbox_inches = 'tight', dpi = 1200)

if doSpikes == 1:
    np.save ('mossy_Spikes', [Spikes, Iinjected])
    fig = figure(2)
    ax = plt.subplot(111)
    ax.plot(Iinjected, Spikes, '-o')
    xlabel('Current injection (pA)', fontsize = 16)
    ylabel('Firing frequency (Hz)', fontsize = 16)
    xticks(range(0, 1100, 200))

    zed = [tick.label.set_fontsize(14) for tick in ax.yaxis.get_major_ticks()]
    zed = [tick.label.set_fontsize(14) for tick in ax.xaxis.get_major_ticks()]
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.savefig('mossy_FiringRates.eps')
#    savefig('mossy_FiringRates.svg', format = 'svg', bbox_inches='tight', dpi = 1200)

    scipy.io.savemat('mossy_Spikes.mat', dict(x=Iinjected, y=Spikes))
    plt.show()


step = 100
if doVisualise == 1:

    ax = plt.subplot(111)
    vm_soma = M0[0]
    for _,t in S.spikes:
        i = int(t / defaultclock.dt)
        vm_soma[i] = (v_th_m + 88*mV)
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
#    scipy.io.savemat('mossy_voltageTrace.mat', dict(x=M0.times / ms, y=vm_soma / mV))

#    ax.set_rasterized(True)
    if Iinjected[jj] > 0:
        savefig('mossy_voltageTrace.eps', format='eps', dpi = 1200)
    else:
        savefig('mossy_voltageTrace_negative.eps', format='eps', dpi = 1200)
#    savefig('mossy_voltageTrace.svg', format='svg', dpi = 1200)

    fig = figure(2)
    for i in range(size(mossy)):
        plot(M1.times / ms, M1[jj] / pA)
    ylabel('I_inj (pA)')
    xlabel('time (ms)')

    plt.show()
##==============================================================================
