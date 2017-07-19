#==============================================================================
# Point neuron (exponential I&F model) HIPP cell.
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

#Iinjected = range(-20, 25, 5)
Iinjected = [25]
#Iinjected = [-50]
#Iinjected = range(0, 210, 25)

Vpeak  = np.zeros(len(Iinjected))
Spikes = np.zeros(len(Iinjected))

for jj in range(len(Iinjected)):
    reinit(states = True)
    clear(erase = True, all = True)

    # Number of presynaptic neurons
    N_pre  = 1
    N_hipp = 1

    # Parameters
    gl_h      =   1.930  * nS # leakage conductance
    El_h      = -59      * mV # reversal-resting potential
    Cm_h      =  0.0584  * nF # membrane capacitance
    v_th_h    = -50      * mV # threshold potential
    v_reset_h = -56      * mV # reset potential
    DeltaT_h  =   2      * mV # slope factor

    # Synaptic Reversal Potentials
    E_nmda       =  0  * mV  # NMDA reversal potential
    E_ampa       =  0  * mV  # AMPA reversal potential
    E_gaba       = -75 * mV  # GABA reversal potential

    # AMPA/NMDA/GABA Model Parameters
    gamma        = 0.072 * mV**-1
    alpha_nmda   = 0.5   * ms**-1
    alpha_ampa   = 1     * ms**-1
    alpha_gaba   = 1     * ms**-1

    # Input to HIPP from Perforant Path(EC) (AMPA/NMDA kinetics from Kneisler, Dingledine, 1995 Hippocampus)
    g_ampa_h   =   0.24 * nS  # AMPA maximum conductance
    g_nmda_h   =   1.28 * nS  # NMDA maximum conductance

    t_nmda_decay = 110.0  * ms  # NMDA decay time constant
    t_nmda_rise  =   4.8  * ms  # NMDA rise time constant
    t_ampa_decay =  11.0  * ms  # AMPA decay time constant
    t_ampa_rise  =   2.0  * ms  # AMPA rise time constant

    # GABAergic Input
    g_gaba_h       = 1.6  * nS  # GABA maximum conductance
    t_gaba_decay   = 5.5  * ms  # GABA decay time constant
    t_gaba_rise    = 0.26 * ms  # GABA rise time constant

    # Synaptic current equations
    eq_soma_h = Equations('''
    I_syn_h  = I_nmda + I_ampa + I_gaba - I_inj                         : amp
    I_nmda = g_nmda_h*(vm - E_nmda)*s_nmda*1./(1 + exp(-gamma*vm)/3.57) : amp
    I_ampa = g_ampa_h*(vm - E_ampa)*s_ampa                              : amp
    I_inj                                                               : amp
    s_nmda                                                              : 1
    s_ampa                                                              : 1
    ''')

    # Brette-Gerstner
    hipp_eqs  = Brette_Gerstner(Cm_h, gl_h, El_h, v_th_h, DeltaT_h, tauw = 93 * ms, a = .82 * nS)
    hipp_eqs += IonicCurrent('I = I_syn_h : amp')
    hipp_eqs += eq_soma_h

    hipp = NeuronGroup(N_hipp, model = hipp_eqs, threshold = EmpiricalThreshold(threshold = v_th_h,refractory = 3*ms),
                         reset = AdaptiveReset(Vr=v_reset_h, b = 0.015*nA), compile = True, freeze = True)

    # Initialization of membrane potential
    hipp.vm = El_h

    #==============================================================================
    P0 = PoissonGroup(100)
    #==============================================================================
    # Monitors
    S  = SpikeMonitor(hipp)
    M0 = StateMonitor(hipp, 'vm', record = True)
    M1 = StateMonitor(hipp, 'I_inj', record = True)

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
            hipp.I_inj = Iinjected[jj]*pA
        else:
            hipp.I_inj = 0* pA




    print "Simulation running..."
    start_time = time.time()
    run(total_simt)
    duration = time.time() - start_time

    print "Number of somatic spikes: " + str(S.nspikes)
    print
    print "Simulation time: ", duration, "seconds"
    if Iinjected[jj] >= 0:
        Vpeak[jj] = 1000*(max(M0[0])  - El_h / volt) # in mvolts
    else:
        sag_ratio = (El_h/volt - M0[0][12000])/(El_h/volt - min(M0[0]))
        R_input   = ((El_h/volt - M0[0][12000])*volt) / (-Iinjected[jj] * pA)
        Vpeak[jj] = 1000*(min(M0[0])  - El_h / volt) # in mvolts
        print
        print "Sag ratio: "+str(sag_ratio)
        print
        print "Rin: " + str(R_input)
        print
        print "time constant: " +str(R_input*Cm_h)

    Spikes[jj] = S.nspikes


##=============================================================================
# Visualization
if doData == 1:
    np.save('HIPP_Voltage', [Vpeak, Iinjected])
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
    ax.xaxis.set_ticks(range(-25, 0, 5) + range(5, 30, 5))
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_ticks(range(-10, 0, 5) + range(5, 12, 5))
    ax.xaxis.set_label_text('I (pA)')
    ax.xaxis.set_label_coords(.7, .4)
    ax.yaxis.set_label_text('V (mV)')
    ax.yaxis.set_label_coords(0.4, 0.9)
    ax.yaxis.get_label().set_rotation('horizontal')

    savefig('hippCell.eps', format = 'eps', bbox_inches = 'tight', dpi = 1200)
#    savefig('hippCell.svg', format = 'svg', bbox_inches = 'tight', dpi = 1200)

if doSpikes == 1:
    np.save ('HIPP_Spikes', [Spikes, Iinjected])
    fig = figure(2)
    ax = plt.subplot(111)
    ax.plot(Iinjected, Spikes, '-o')
    xlabel('Current injection (pA)', fontsize = 16)
    ylabel('Firing frequency (Hz)', fontsize = 16)
    xticks(range(0, 210, 25))

    zed = [tick.label.set_fontsize(14) for tick in ax.yaxis.get_major_ticks()]
    zed = [tick.label.set_fontsize(14) for tick in ax.xaxis.get_major_ticks()]
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    savefig('HIPP_FiringRates.eps', format = 'eps', bbox_inches='tight', dpi = 1200)
#    savefig('HIPP_FiringRates.svg', format = 'svg', bbox_inches='tight', dpi = 1200)

    scipy.io.savemat('HIPP_Spikes.mat', dict(x=Iinjected, y=Spikes))
    plt.show()


step = 100

if doVisualise == 1:

    ax = plt.subplot(111)
    vm_soma = M0[0]
    for _,t in S.spikes:
        i = int(t / defaultclock.dt)
        vm_soma[i] = (v_th_h + 90*mV)
    ax.plot(M0.times / ms, vm_soma / mV)
    ylabel('V [mV]', fontsize = 16)
    xlabel('Time [ms]', fontsize = 16)
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

#    scipy.io.savemat('HIPP_voltageTrace.mat', dict(x=M0.times / ms, y=vm_soma / mV))
    if Iinjected[jj] > 0:
        savefig('HIPP_voltageTrace.eps', format='eps', dpi = 1200)
    else:
        savefig('HIPP_voltageTrace_negative.eps', format='eps', dpi = 1200)


#    savefig('HIPP_voltageTrace.svg', format='svg', dpi = 1200)

    fig = figure(2)
    for i in range(size(hipp)):
        plot(M1.times / ms, M1[i] / pA)
    ylabel('I_inj (pA)')
    xlabel('time (ms)')

    plt.show()
##==============================================================================
