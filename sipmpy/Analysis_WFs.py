import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import time
import pickle
from iminuit import Minuit
from iminuit.cost import LeastSquares
from threading import Thread

fontsize = 12
font = {"family":"Dejavu Sans", "weight":"normal", "size":fontsize}
mpl.rc("font", **font)

dir = "/home/cerasole/dottorato/photodetection/Lab_data_2023/WAVEFORMS/"

hvs = np.linspace(26, 28, 5)
number_of_files = [5084, 5224, 6430, 6973, 5656]
nof = {hvs[i]:number_of_files[i] for i in range(5)}

def linear_function(x, q, p):
    return q + p * x

def gaussian_function(x, A, x0, sigma):
    return A * np.exp( -0.5 * np.power(( x - x0 ) / sigma, 2) )

def save_object_to_pkl_file (thing, filename = None):
    if filename is None:
        print ("No filename provided!")
        return
    f = open(filename, "wb")  # write + binary
    pickle.dump(thing, f)
    f.close()
    print ("Saved %s successfully!" % filename)

def laod_objects_from_file (filename = None):
    if filename is None:
        print ("No filename provided!")
        return
    things = []
    f = open(filename, "rb")
    while True:
        try:
            a = pickle.load(f)
            things.append(a)
        except EOFError:
            f.close()
            break
    return things


class WF(object):

    def __init__ (self, hv = None, id = None, filename = None):
        self.hv = hv
        self.id = id
        self.time = None
        self.ampl = None
        self.baseline_sigma = None
        self.tstart = 5.0e-8
        self.integration_time = 1e-7

        if filename is None:
            self.filename = dir + "hv%4d/C2--FBK-NUV_1x1--%05d.txt" % (int(hv*100), id)
        else:
            self.filename = filename

    def read_from_file (self, time = None):
        if time is None:
            self.time, self.ampl = np.loadtxt(self.filename, skiprows = 5, delimiter = ",", unpack = True)
        else:
            self.time = time
            self.ampl = np.loadtxt(self.filename, skiprows = 5, delimiter = ",", unpack = True, usecols = 1)

    def plot_wf (self, fig = None, ax = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(self.time, self.ampl)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (V)")
        return fig, ax

    def compute_baseline (self):
        mask = self.time < 0.5e-7
        baseline = np.average(self.ampl[mask])
        sigma = np.std(self.ampl[mask])
        self.baseline_sigma = sigma
        return baseline, sigma

    def subtract_baseline (self):
        baseline, sigma = self.compute_baseline()
        self.ampl -= baseline

    def find_1st_maximum (self):
        mask = (self.time > 5e-8) * (self.time < 7e-8 )
        min = np.min(self.ampl[mask])
        time_min = self.time[np.where(self.ampl == min)[0][0]]
        return time_min

    def return_ampl_at (self, t):
        index = np.argmin( np.abs(self.time - t) )
        return self.ampl[index]

    def find_optimal_integration_window (self):
        ''' To be implemented '''
        return 0

    def integrate_charge (self):
        istart = np.argmin( np.abs(self.time - self.tstart) )
        istop = int(self.integration_time / (self.time[1] - self.time[0]))
        s = (self.time[1] - self.time[0]) * np.sum(self.ampl[istart : istop])
        return s


class WFset (object):

    def __init__(self, hv = None, dir_path = dir):
        self.hv = hv
        self.dir_path = dir
        self.number_of_wfs = None
        self.file_to_start_with = 0
        self.files_to_be_processed = None
        self.common_time_bins = False
        self.time = None
        self.wfs = None   # vettore degli oggetti di tipo WF
        self.boolean_amplitude = True
        self.t_peak = None
        self.manina_bin_centers = None
        self.manina_events = None
        self.raggio_intorno = 3
        self.n_peaks = None
        self.peaks = None

    def count_wfs (self):
        files = np.array(os.listdir(self.dir_path))
        mask = [files[i].startswith("C2--FBK-NUV_1x1--") for i in range(len(files))]
        files = files[mask]
        return len(files)

    def setup (self, file_to_start_with = None, files_to_be_processed = None, common_time_bins = None):
        """
        Function that reads the time and amplitude of the set of WFs.
        Parameters:
            - file_to_start_with: int, index of waveform file to start with
            - files_to_be_processed: int, number of files to process, starting from file_to_start_with
            - common_time_bins: True/False, to use the same time array as the first file (less time consuming)
        """
        if self.hv is None:
            print ("No HV value was entered. Aborting"); return 0
        self.dir_path = dir + "hv%4d/" % (int(self.hv * 100))
        try:
            self.number_of_wfs = nof[self.hv]
        except:
            self.number_of_wfs = self.count_wfs()

        if file_to_start_with is not None:
            self.file_to_start_with = file_to_start_with
        if files_to_be_processed is not None:
            self.files_to_be_processed = files_to_be_processed
        else:
            self.files_to_be_processed = self.number_of_wfs
            self.file_to_start_with = 0
        if common_time_bins is not None:
            self.common_time_bins = common_time_bins

        self.wfs = [WF(self.hv, id) for id in range(self.file_to_start_with, self.file_to_start_with + self.files_to_be_processed)]

        if self.common_time_bins is True:   # reduction by 30% of computing time
            self.wfs[self.file_to_start_with].read_from_file()
            [self.wfs[i].read_from_file(time = self.wfs[0].time) for i in range(1, self.files_to_be_processed)]
        else:
            [self.wfs[i].read_from_file() for i in range(self.files_to_be_processed)]

    def plot_wfset(self, fig = None, ax = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        for wf in self.wfs:
            wf.plot_wf(fig, ax)
        return fig, ax

    def subtract_baseline(self):
        for wf in self.wfs:
            wf.subtract_baseline()

    def find_1st_maximum(self):   # method to find the optimal instant of time at which considering the amplitudes for the manine
        ### Can be improved
        a = []
        x1, x2 = 5.7e-8, 5.95e-8
        x1fit, x2fit = 5.76e-8, 5.86e-8
        for wf in self.wfs:
            a.append(wf.find_1st_maximum())
        fig, ax = plt.subplots(figsize = (8, 6))
        n, bins, patches = ax.hist (a, bins = 40, density = False, range=(x1, x2))
        bin_centers = (bins[:-1] + bins[1:]) * 0.5
        mask = (bin_centers > x1fit) * (bin_centers < x2fit)
        lsq = LeastSquares(bin_centers[mask], n[mask], np.sqrt(n)[mask], gaussian_function)
        lsq.loss = "soft_l1"
        m = Minuit(lsq, A = 1e2, x0 = 5.805e-8, sigma = 1e-10)
        m.limits = [(1, 1.e4), (5.79e-8, 5.81e-8), (0.1e-10, 1e-8)]
        m.limits = [(1, 1.e4), (5.79e-8, 5.82e-8), (0.1e-10, 1e-8)]
        m.migrad()
        xx = np.linspace(x1fit, x2fit, 100)
        yy = gaussian_function(xx, *m.values)
        self.t_peak = m.values["x0"]
        ax.errorbar(xx, yy, label = "Fit")
        ax.set_xlim(left = x1, right = x2)
        ax.set_xlabel("Time of maximum from trigger (s)")
        ax.set_ylabel("Events")
        ax.grid()
        plt.savefig("Plots/hold_time_%d_hv.eps" % self.hv)
        plt.savefig("Plots/hold_time_%d_hv.png" % self.hv)
        #plt.show()

    def find_optimal_integration_window(self):
        self.wfs[0].find_optimal_integration_window()
        return 0

    def search_peaks(self):
        n_peaks = 0
        v_x_max = []
        raggio_intorno = self.raggio_intorno
        y = self.manina_events
        n = len(y)
        for i in range(raggio_intorno, n - raggio_intorno):
            # Verifico se nel punto i-esimo c'è un massimo
            y0 = y[i]
            max_flag = True
            for j in range(i - raggio_intorno, i + raggio_intorno):
                yj = y[j]
                if yj > y0:
                    max_flag = False
                    break
            if (max_flag is True and (y0 > np.min(y) + 0.05 * (np.max(y) - np.min(y)) ) ):
                v_x_max.append(i)
        nn = 2.3
        def check_equidistant_peaks(v, flag = False):  # algorithm to discard the non equidistant peaks     # can be improved
            i = 0
            if flag == False:
                diff = np.array([ v[j+1] - v[j] for j in range(len(v) - 1)])
                mean = np.mean(diff)
                std = np.std(diff)
                mask = (diff >= mean - nn * std) * (diff <= mean + nn * std)
                if np.sum(mask) == len(mask):
                    flag = True
                else:
                    mask = np.array([True] + list(mask))
                    v = check_equidistant_peaks (np.array(v)[mask], flag = False)
            return v
        #print ("%d peaks indices before treatment:" % len(v_x_max))
        #print (v_x_max)
        v_x_max = check_equidistant_peaks(v_x_max)
        #print ("%d peaks indices after treatment:" % len(v_x_max))
        #print (v_x_max)
        return v_x_max

    def fit_single_peak (self, index_max, value_max):
        i1, i2 = index_max - self.raggio_intorno, index_max + self.raggio_intorno
        x, y = self.manina_bin_centers[i1:i2], self.manina_events[i1:i2]
        y[y == 0] = 0.1
        dy = np.sqrt(y)
        lsq = LeastSquares(x, y, dy, gaussian_function)
        sigma = self.manina_bin_centers[index_max] - self.manina_bin_centers[index_max - 2]
        A = self.manina_events[index_max]
        m = Minuit(lsq, A = A, x0 = value_max, sigma = sigma)
        #m.limits["x0"] = (value_max * 0.5, value_max * 2)
        #m.limits["A"] = (A * 0.5, A * 1.5)  # no good
        #m.limits["sigma"] = (0., 100.)
        m.limits["sigma"] = (sigma*0.3, sigma*3.)
        m.migrad()
        return m

    def fit_multi_peaks (self, minuit_instances):
        # Definisco una multi-gaussiana dalle singole gaussiane e rifitto tutto l'istogramma
        n = self.n_peaks
        vars = ["A", "x0", "sigma"]
        values  = [[minuit_instances[i].values[var] for i in range(n)] for var in vars]
        dvalues = [[minuit_instances[i].errors[var] for i in range(n)] for var in vars]

        def multigaussian_function (x, A0,  A1,  A2,  A3,  A4,  A5,  A6,  A7,  A8,  A9,  \
                                       x00, x01, x02, x03, x04, x05, x06, x07, x08, x09, \
                                       sigma0, sigma1, sigma2, sigma3, sigma4, sigma5, sigma6, sigma7, sigma8, sigma9):
            sum = 0
            A = [A0,  A1,  A2,  A3,  A4,  A5,  A6,  A7,  A8,  A9]
            x0 = [x00, x01, x02, x03, x04, x05, x06, x07, x08, x09]
            sigma = [sigma0, sigma1, sigma2, sigma3, sigma4, sigma5, sigma6, sigma7, sigma8, sigma9]
            nn = len(A)
            for i in range(nn):
                sum += gaussian_function (x, A[i], x0[i], sigma[i])
            return sum

        x = self.manina_bin_centers
        y = self.manina_events
        y[y == 0] = 0.1
        dy = np.sqrt(y)
        lsq = LeastSquares (x, y, dy, multigaussian_function)
        m = Minuit(lsq, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., \
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.)
        #m = Minuit(lsq, np.zeros(30)) # non funge così
        inits = [0., 0., 1.]
        for i in range(10):
            for j in range(len(vars)):
                if i < n:
                    m.values["%s%d" % (vars[j], i)] = values[j][i]
                    m.limits["%s%d" % (vars[j], i)] = (values[j][i] - 2*dvalues[j][i], values[j][i] + 2*dvalues[j][i])
                    m.fixed["%s%d" % (vars[j], i)] = False
                else:
                    m.values["%s%d" % (vars[j], i)] = inits[j]
                    m.fixed["%s%d" % (vars[j], i)] = True
        m.migrad()
        xx = np.linspace(self.manina_bin_centers[0], self.manina_bin_centers[-1], 300)
        yy = multigaussian_function(xx, *m.values)
        return m, xx, yy

    def fit_manina (self, fig, ax):

        # Algoritmo di ricerca dei picchi in n, il vettore delle occorrenze
        peaks_indices_vec = self.search_peaks()
        n_peaks = len(peaks_indices_vec)
        self.n_peaks = n_peaks
        peaks_vec = [ self.manina_bin_centers[peaks_indices_vec[i]] for i in range(n_peaks) ]

        #print ("Found %d peaks, at values of" % n_peaks); print (peaks_vec)
        print ("Found %d peaks" % n_peaks)

        # Trovati i picchi, fitto ciascuno di questi con una gaussiana
        minuit_instances = []
        for i in range(n_peaks):
            m = self.fit_single_peak(peaks_indices_vec[i], peaks_vec[i])
            minuit_instances.append(m)

        # A partire dai risultati dei singoli fit, definisco una multigaussiana e la fitto
        m, xx, yy = self.fit_multi_peaks(minuit_instances)

        # Disegno la multi-gaussiana nel plot
        ax.errorbar(xx, yy, color = "red")

        A = [ m.values["A%d" % i] for i in range(n_peaks) ]
        peaks = [ m.values["x0%d" % i] for i in range(n_peaks) ]
        sigma_peaks = [ m.values["sigma%d" % i] for i in range(n_peaks) ]
        dA = [ m.errors["A%d" % i] for i in range(n_peaks) ]
        dpeaks = [ m.errors["x0%d" % i] for i in range(n_peaks) ]
        dsigma_peaks = [ m.errors["sigma%d" % i] for i in range(n_peaks) ]
        self.peaks = np.array([A, dA, peaks, dpeaks, sigma_peaks, dsigma_peaks])

    def plot_manina (self):
        a = []
        if self.boolean_amplitude is True:
            self.find_1st_maximum()
            for wf in self.wfs:
                a.append( wf.return_ampl_at( self.t_peak ) )
        else:
            self.find_optimal_integration_window()
            for wf in self.wfs:
                a.append( wf.integrate_charge() )
        a = -1. * np.array(a)
        fig, ax = plt.subplots(figsize = (8, 6))
        n, bins, patches = ax.hist (a, bins = 100, density = False)
        bin_centers = (bins[:-1] + bins[1:]) * 0.5
        self.manina_bin_centers, self.manina_events = bin_centers, n
        self.fit_manina (fig, ax)
        if self.boolean_amplitude is True:
            ax.set_xlabel("Amplitude (V)")
        else:
            ax.set_xlabel("Charge (V s)")
        ax.set_ylabel("Events")
        ax.grid()
        if self.boolean_amplitude is True:
            filename = "Plots/Manina_amplitude_%.1f_hv.png" % self.hv
            fig.savefig(filename)
        else:
            filename = "Plots/Manina_charge_%.1f_hv.png" % self.hv
            fig.savefig(filename)
        save_object_to_pkl_file ([fig, ax], filename.replace(".png", ".pkl"))
        #plt.show()

    def save_peaks_to_file (self):
        if self.boolean_amplitude is True:
            filename = "Peaks/peaks_amplitude_%.1f_hv.txt" % self.hv
        else:
            filename = "Peaks/peaks_charge_%.1f_hv.txt" % self.hv
        print ("Saving peaks in %s" % filename)
        np.savetxt(filename, np.transpose(self.peaks), header = "A - dA - mu - dmu - sigma - dsigma")
        return 0



def plot_few_WFs (hv, ids = None):
    # Quick look at the temporal shape of an individual wfs
    fig, ax = plt.subplots(figsize = (12, 6))
    if ids is None:
        ids = [23, 27]
    t_trigger = 5.4e-8 # s
    t_end = t_trigger + 100e-9 # s
    for id in ids:
        wf = WF(hv, id)
        wf.read_from_file()
        baseline, sigma = wf.compute_baseline()
        t_peak = wf.find_1st_maximum()
        wf.plot_wf(fig, ax)
        ax.hlines(baseline, xmin = wf.time[0], xmax = t_trigger, label = "Baseline", ls = "dashdot", color = "indigo")
        #x, y1, y2 = np.linspace(wf.time[0], t_trigger, 100), (baseline-sigma)*np.ones(100), (baseline+sigma)*np.ones(100)
        #ax.fill_between(x, y1, y2, alpha = 0.6)

    windows = [wf.time[0], t_trigger, t_end]
    colors = ["green", "green"]
    labels = ["Integration window start", "Integration window end"]

    for i in range(1, 3):
        ax.axvline(windows[i], ls = "dashed", color = colors[i-1], label = labels[i-1])

    if len(ids) < 2:
        ax.legend()
    ax.grid()
    fig.savefig("Plots/WFs.png")
    fig.savefig("Plots/WFs.eps")
    plt.show()


def wfset_analysis (hv, boolean_amplitude = True, raggio_intorno = 3, \
                    files_to_be_processed = None, file_to_start_with = 0, common_time_bins = False):
    ''' Method performs all the steps of the waveform analysis for a single hv
        Parameters:
          - hv, float type
                Up to now, only data for [26.0, 26.5, 27.0, 27.5, 28.0] are present
          - boolean_amplitude, boolean type True/False
                True -> the manina will be the histogram of the signals amplitudes at a given instant of time.
                        This time instant equals the best-fit Gaussian average of the distribution of the signal peak times
                        in the [50, 70] ns window from the trigger
                False -> the manina will be the histogram of the integrated signals in the 100 ns of the [50, 150] ns window  # can be improved
          - raggio_intorno, int type
                In the manina peaks' finder algorithm, I need to define the radius (in index) of the neighbourhood in which to search for a peaks
          - files_to_be_processed, None type or int type
                If None, all the files at the hv reverse bias voltage will be processed
                If an int number, files_to_be_processed files will be processed, starting from file_to_start_with
          - file_to_start_with, int type
                Index of file to start with during the processing
          - common_time_bins, boolean True/False
                True -> The same time binning is used for all waveforms.
                        It can save a reasonable amount of computing time since the np.loadtxt has to read 2 lines once and only 1 line for all other files
                False -> The time binning is read from each of the waveforms files.
    '''
    start = time.time()
    wfset = WFset(hv)
    print ("################ HV %.1f V" % hv)
    common_time_bins = common_time_bins
    file_to_start_with = file_to_start_with
    files_to_be_processed = files_to_be_processed
    wfset.setup(file_to_start_with, files_to_be_processed, common_time_bins)
    wfset.subtract_baseline()
    wfset.plot_wfset()
    wfset.raggio_intorno = raggio_intorno
    wfset.boolean_amplitude = boolean_amplitude
    wfset.plot_manina()
    wfset.save_peaks_to_file()
    stop = time.time()
    print ("Elapsed %.3f s" % (stop-start) )
    plt.show()

def load_peaks_from_file (filename):
    peaks = np.loadtxt(filename, skiprows = 1, unpack = True)
    return peaks

def peaks_analysis(hvs, type = "amplitude", fig = None, ax = None, doPlot = False):
    filename = "Peaks/peaks_%s_%.1f_hv.txt"
    peaks = []
    for hv in hvs:
        p = load_peaks_from_file (filename % (type, hv))
        peaks.append(p)

    n_peaks = int(np.min([ len(peaks[i][0]) for i in range(len(hvs))]))
    peaks_A = [[peaks[i][0][j] for i in range(len(hvs))] for j in range(n_peaks)]
    peaks_dA = [[peaks[i][1][j] for i in range(len(hvs))] for j in range(n_peaks)]
    peaks_mu = [[peaks[i][2][j] for i in range(len(hvs))] for j in range(n_peaks)]
    peaks_dmu = [[peaks[i][3][j] for i in range(len(hvs))] for j in range(n_peaks)]
    peaks_sigma = [[np.abs(peaks[i][4][j]) for i in range(len(hvs))] for j in range(n_peaks)]
    peaks_dsigma = [[np.abs(peaks[i][5][j]) for i in range(len(hvs))] for j in range(n_peaks)]

    if ax is None:
        fig, ax = plt.subplots(figsize = (8,6))
    for j in range(n_peaks-1, -1, -1):
        ax.errorbar(hvs, peaks_mu[j], yerr = peaks_sigma[j], label = "%d p.e. peaks" % j, capsize = 6)
    ax.set_xlabel("Reverse Bias Voltage (V)")
    if type == "amplitude":
        ax.set_ylabel("Amplitude (V)")
    else:
        ax.set_ylabel("Charge (V s)")
    ax.grid()
    ax.legend()
    if doPlot is True:
        plt.show()
    return n_peaks, peaks_mu, peaks_sigma

def load_figures_from_file (filename = None):
    if filename is None:
        print ("No filename provided!")
        return
    f = open(filename, "rb")
    things = []
    while True:
        try:
            a = pickle.load(f)
            things.append(a)
            print ("Loaded something from %s!" % filename)
        except EOFError:
            f.close()
            break
    return things

def produce_peaks_plot (hvs, type = "amplitude", same_axis = True):
    filename = "Plots/Manina_%s_%.1f_hv.pkl"
    fig, axs = plt.subplots(3, 2, figsize = (14, 14))
    for k in range(len(hvs)):
        i, j = k // 2, k % 2
        things = load_figures_from_file (filename % (type, hvs[k]))
        ax = things[0][1]
        x, y = [], []
        for artist in ax.get_children():
            if isinstance(artist, plt.Rectangle):
                xx = artist.get_x()                 # left extreme of the bin on the x axis
                yy = artist.get_height()            # height of the bin on the y axis
                x.append(xx)
                y.append(yy)
        x, y = x[:-1], y[:-1]
        delta = x[1] - x[0]
        x = x + [x[-1] + delta]
        axs[i][j].stairs(y, x, fill = True)
        for artist in ax.get_children():
            if isinstance(artist, plt.Line2D):
                x_data = artist.get_xdata()
                y_data = artist.get_ydata()
                axs[i][j].errorbar(x_data, y_data, color = "red")

        if type == "amplitude":
            axs[i][j].set_xlabel("Amplitude (V)")
            if same_axis == True:
                axs[i][j].set_xlim(left = -0.008, right = .12)
                axs[i][j].set_ylim(bottom = 0, top = 580)
        else:
            axs[i][j].set_xlabel("Charge (V s)")
            if same_axis == True:
                axs[i][j].set_xlim(left = -0.2e-9, right = 2.8e-9)
                axs[i][j].set_ylim(bottom = 0, top = 580)
        axs[i][j].set_ylabel ("Events")
        axs[i][j].text(0.59, 0.8, "V$_{bias}$ = %.1f V" % hvs[k], fontsize = 15, transform = axs[i][j].transAxes, bbox=dict(facecolor='white'))
        axs[i][j].grid()
    peaks_analysis(hvs, type = type, fig = fig, ax = axs[2][1])
    if same_axis is True:
        st = "same_axes"
    else:
        st = "different_axes"
    fig.savefig("Plots/All_manine_%s_%s.png" % (type, st))
    fig.savefig("Plots/All_manine_%s_%s.eps" % (type, st))
    plt.show()


def gain_analysis (hvs):
    n_peaks, peaks_mu, peaks_sigma = peaks_analysis(hvs, type = "charge", fig = None, ax = None)
    averages, sigmas = np.zeros(len(peaks_mu)), np.zeros(len(peaks_mu))
    for k in range(len(peaks_mu)):
        n, m = [], []
        for i in range(n_peaks):
            for j in range(i):
                a = (peaks_mu[i][k] - peaks_mu[j][k]) / (i-j)
                b = np.sqrt( np.power(peaks_sigma[i][k], 2) + np.power(peaks_sigma[j][k], 2) ) / (i-j)
                n.append(a)
                m.append(b)
        averages[k] = np.average(np.array(n), weights = 1. / np.power(np.array(m), 2))
        sigmas[k] = np.sqrt( 1. / np.sum(1. / np.power(m, 2)) )
    fig, ax = plt.subplots(figsize = (8, 6))
    e = 1.6e-19 # Coulomb
    gain_ampl = 100
    R_oscill = 50 # Ohm
    gain = averages / (gain_ampl * e * R_oscill)
    dgain = sigmas / (gain_ampl * e * R_oscill)
    ax.errorbar(hvs, gain/1.e6, yerr = dgain/1e6, marker = "^", ms = 8, capsize = 6, ls = "None", label = "Data", color = "black")
    # Linear fit
    lsq = LeastSquares(hvs, gain/1.e6, dgain/1e6, linear_function)
    m = Minuit(lsq, p = 0.1, q = 1.)
    m.migrad()
    xx = np.linspace(25.8, 28.2, 100)
    yy = linear_function (xx, *m.values)
    ax.errorbar(xx, yy, color = "red", label = "Fit")
    print ("Best-fit values of Gain = q + p * V_bias\n q = %.2e +/- %.2e \n p = %.2e +/- %.2e V^{-1} " % \
           ( m.values["q"]*1e6, m.errors["q"]*1e6, m.values["p"]*1e6, m.errors["p"]*1e6 ))
    ax.legend()
    ax.set_xlabel("Reverse Bias Voltage (V)")
    ax.set_ylabel("Gain (10$^{6}$)")
    ax.grid()
    fig.savefig("Plots/Gain_vs_Vbias.png")
    fig.savefig("Plots/Gain_vs_Vbias.eps")

    snr = averages / np.array(peaks_sigma[0])
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.errorbar(hvs, snr, marker = "v", ms = 10)
    ax.set_xlabel("Reverse Bias Voltage (V)")
    ax.set_ylabel("SNR")
    ax.grid()
    fig.savefig("Plots/SNR_vs_Vbias.png")
    fig.savefig("Plots/SNR_vs_Vbias.eps")

    q, dq, p, dp = m.values["q"]*1e6, m.errors["q"]*1e6, m.values["p"]*1e6, m.errors["p"]*1e6

    ### Capacitance
    # G = averages / (G_amp * R * e) = q + p * V_bias = (V_bias - V_bd) * (C_mucell + C_q) / e = [- V_bd * (C_mucell + C_q) / e] + [(C_mucell + C_q) / e] * V_bias
    C_sum = p * e
    dC_sum = C_sum * (dp / p)
    print ("Sum of the microcell capacitance and the parasitic capacitance: \n C = %.2e +/- %.2e F" % (C_sum, dC_sum))

    ### Breakdown voltage
    V_bd = - q * e / C_sum
    dV_bd = V_bd * (dq/q + dC_sum/C_sum)
    print ("Estimate of the breakdown voltage: \n V_bd = %.2f +/- %.2f V" % (V_bd, dV_bd))

    ### Depletion layer thickness
    eps_0 = 8.854e-15  # F / mm
    eps_Si = 11.9
    A = 0.6 / 625.     # mm^2
    d = 1e3 * eps_0 * eps_Si * A / C_sum # (F/mm) * mm^2 * F^{-1} = mm  #### micro m
    dd = d * (dC_sum / C_sum)
    print ("Estimate of the depletion layer thickness: \n d = %.2f +/- %.2f mu m" % (d, dd))

    plt.show()


def Vinogradov_analysis (hvs, type = "amplitude"):
    filename = "Peaks/peaks_%s_%.1f_hv.txt"
    peaks = []
    for hv in hvs:
        p = load_peaks_from_file (filename % (type, hv))
        peaks.append(p)

    def B (i, k):
        if i == 0 and k == 0:
            y = 1
        elif i == 0 and k > 0:
            y = 0
        else:
            y = math.factorial(k) * math.factorial(k-1) / \
                ( math.factorial(i) * math.factorial(i-1) * math.factorial(k-i))
        return y

    def compound_Poisson (ks, n_tot, p, L):
        n = len(ks)
        y = np.zeros(n)
        for i_k in range(n):
            sum = 0
            k = int(ks[i_k])
            for i in range(k + 1):
                sum += B(i, k) * np.power(p, k - i) * np.power(L * (1-p), i)
            y[i_k] = n_tot * np.exp(-L) * ( 1 / math.factorial(k) ) * sum
        return y

    n = len(hvs)
    aaa = np.zeros(n)
    N, P, L = np.zeros(n), np.zeros(n), np.zeros(n)
    dN, dP, dL = np.zeros(n), np.zeros(n), np.zeros(n)
    fig, axs = plt.subplots(2, 3, figsize = (18, 10))
    for i_hv in range(n):
        i, j = i_hv // 3, i_hv % 3
        ax = axs[i][j]
        A, dA = peaks[i_hv][0], peaks[i_hv][1]
        n_peaks = len(A)
        bins = np.linspace(-0.5, n_peaks-0.5, n_peaks+1)
        bin_centers = np.linspace(0., n_peaks-1, n_peaks)
        ax.stairs(A, bins, fill = True, alpha = 0.7)
        ax.errorbar(bin_centers, A, xerr = 0.5*np.ones(n_peaks), yerr = dA, capsize = 6, marker = "o", label = "Data")

        aaa[i_hv] = np.sum(A)
        lsq = LeastSquares(bin_centers, A, dA, compound_Poisson)
        m = Minuit (lsq, n_tot = aaa[i_hv], p = 0.10, L = 2.)
        #m.fixed["n_tot"] = True
        #m.limits["n_tot"] = (aaa[i_hv] - 10, aaa[i_hv] + 10)    # Don't know why ma non converge il fit se pongo questi limiti
        m.limits["p"] = (0., 1.)
        m.migrad()
        N[i_hv], P[i_hv], L[i_hv] = m.values["n_tot"], m.values["p"], m.values["L"]
        dN[i_hv], dP[i_hv], dL[i_hv] = m.errors["n_tot"], m.errors["p"], m.errors["L"]

        yy = compound_Poisson(bin_centers, *m.values)
        ax.errorbar(bin_centers, yy, xerr = 0.5*np.ones(n_peaks), capsize = 6, marker = "o", color = "red", label = "Fit")
        ax.text(0.59, 0.72, "V$_{bias}$ = %.1f V" % hvs[i_hv], fontsize = 15, transform = ax.transAxes, bbox=dict(facecolor='white'))
        ax.set_xlabel("Photo-electrons")
        ax.set_ylabel("Events")
        ax.legend()
        ax.grid()

    ax = axs[1, 2]
    ax.set_xlabel("Reverse Bias Voltage (V)")
    color = "blue"
    ax.set_ylabel("p (%)", color = color)
    ax.tick_params(axis='y', labelcolor = color)
    ax.errorbar(hvs, P*100, yerr = dP*100, color = color, label = "p", marker = ">", ms = 6, capsize = 6)

    color = "tab:red"
    ax2 = ax.twinx()
    ax2.set_ylabel("L", color = color)
    ax2.tick_params(axis='y', labelcolor = color)
    ax2.errorbar(hvs, L, yerr = dL, label = "L", color = color, marker = "<", ms = 6, capsize = 6)
    ax.grid()

    print ("### Best-fit results with uncertainties")
    #print ("Values of np.sum(A):", aaa)
    print ("Values of n_tot:", N, dN)
    print ("Values of p:", P, dP)
    print ("Values of L:", L, dL)
    fig.savefig("Plots/Vinogradov_analysis_%s.png" % type)
    fig.savefig("Plots/Vinogradov_analysis_%s.eps" % type)
    plt.show()


####### If you want to use this class, you have to comment all plots!!! #######
### TO BE IMPROVED
class WFsetAnalysisThread (Thread):
    def __init__(self, hv, boolean_amplitude = True, raggio_intorno = 3, \
                       files_to_be_processed = None, file_to_start_with = 0, common_time_bins = False):
        Thread.__init__(self, name = "Analysis waveform set at %.1f V of reverse bias voltage" % (hv))
        self.hv = hv
        self.boolean_amplitude = boolean_amplitude
        self.raggio_intorno = raggio_intorno
        self.files_to_be_processed = files_to_be_processed
        self.file_to_start_with = file_to_start_with
        self.common_time_bins = common_time_bins

    def run (self):
        print ("Starting thread: %s" % self.name)
        wfset_analysis (self.hv, boolean_amplitude = self.boolean_amplitude, raggio_intorno = self.raggio_intorno, \
                        files_to_be_processed = self.files_to_be_processed, file_to_start_with = self.file_to_start_with,\
                        common_time_bins = self.common_time_bins)
        print ("Ending thread: %s" % self.name)
