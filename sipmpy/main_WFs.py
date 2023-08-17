import numpy as np
from Analysis_WFs import *

### In this code, I import all the methods in Analysis_WFs and use them.
### I can uncomment here only the lines that allow to perform specific steps, so I can choose what to do.

if __name__ == "__main__":

    # Here I define the reverse bias voltages
    hvs = np.linspace(26., 28., 5)
    n_hvs = len(hvs)

    # To be run the first time!
    #os.system("mkdir -p Plots")
    #os.system("mkdir -p Peaks")

    ###################################################### QUALITATIVE LOOK AT FEW WAVEFORMS ######################################################

    '''
    id = 2 # --> hv = 27
    ids = [23, 27]
    plot_few_WFs (hvs[id], ids = ids)
    '''

    ###################################################### WAVEFORM SET ANALYSIS ######################################################

    '''
    id = 4
    boolean_amplitude = True
    raggio_intorno = [5, 3, 3, 3, 3]  # raggio (in numero di bin) dell'intorno in cui cercare il massimo nella manina
    wfset_analysis (hvs[id], boolean_amplitude = boolean_amplitude, raggio_intorno = raggio_intorno[id])
    '''

    ###################################################### MULTITHREAD WAVEFORM SET ANALYSIS ######################################################

    '''
    id = 4
    threadList = []
    for i in range(id, id+1):
        thread = WFsetAnalysisThread (
                     hvs[i], boolean_amplitude = boolean_amplitude, raggio_intorno = raggio_intorno[i]
                 )
        threadList.append(thread)

    for thread in threadList:
        thread.start()
    '''

    ###################################################### MANINE PLOTS FOR ALL HVs ######################################################

    '''
    produce_peaks_plot(hvs, type = "charge", same_axis = True)
    produce_peaks_plot(hvs, type = "charge", same_axis = False)
    produce_peaks_plot(hvs, type = "amplitude", same_axis = True)
    produce_peaks_plot(hvs, type = "amplitude", same_axis = False)
    '''

    ###################################################### GAIN AND SNR ANALYSIS ######################################################

    '''
    peaks_analysis(hvs, type = "amplitude", doPlot = True)
    peaks_analysis(hvs, type = "charge", doPlot = True)
    gain_analysis (hvs)
    '''

    ###################################################### VINOGRADOV ANALYSIS ######################################################

    '''
    Vinogradov_analysis(hvs, type = "charge")
    Vinogradov_analysis(hvs, type = "amplitude")
    '''
