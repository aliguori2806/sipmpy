import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
import pickle

fontsize = 12
font = {"family":"Dejavu Sans", "weight":"normal", "size":fontsize}
mpl.rc("font", **font)

fw_rv_dict = {"forward" : {"id":0, "xlabel": "Direct Bias Voltage (V)", "ylabel":"Current (A)", "yscale":"linear"}, \
              "inverse" : {"id":1, "xlabel":"Reverse Bias Voltage (V)", "ylabel":"Current (A)", "yscale":"log"} }

def read_temperatures(dir = "."):
    files = os.listdir(dir)
    # tt è il vettore che ha come elemento 0 (1) il vettore delle temperature delle forward (inverse) IV
    tt = [[], []]
    for i in range(len(files)):
        iv_type = files[i].strip(".txt").split("_")[2]
        iv_type_id = fw_rv_dict[iv_type]["id"]
        t = int(files[i].strip(".txt").split("_")[3].strip("TC"))
        tt[iv_type_id].append(t)
    tt[0].sort(); tt[1].sort()
    print ("Direct IV temperatures (°C):", tt[0])
    print ("Inverse IV temperatures (°C):", tt[1])
    return tt

def linear_function(x, q, p):
    return q + p * x

def quadratic_function(x, a0, a1, a2):
    return a0 + a1 * x + a2 * x**2

def logarithmic_derivative_above_Vbd(x, Vbd):
    return 2 / (x - Vbd)


class IV:

    def __init__(self, iv_type, temperature, dir):
        self.iv_type = iv_type
        self.temperature = temperature
        if dir[-1] != "/": dir += "/"
        self.filename = dir + "IV_HD3-4_%s_T%dC.txt" % (iv_type, temperature)
        self.data = None
        self.xlabel, self.ylabel, self.yscale = None, None, None

    def read_iv_from_file(self):
        # leggo con pandas perchè ha la funzionalità di interpretare la virgola come separatore decimale
        print ("Opening %s\t\tIV type = %s\tTemperature = %d °C" % (self.filename, self.iv_type, self.temperature))
        iv_data = pd.read_csv(self.filename, sep = "\t", names = ["Time (?)", "V (V)", "I (A)", "dI (A)"], decimal = ",")
        iv_data = iv_data.drop(columns = 'Time (?)')
        if self.iv_type == "forward":
            #iv_data = pd.DataFrame.abs(iv_data)
            iv_data["V (V)"] = iv_data["V (V)"] * -1.
            iv_data["I (A)"] = iv_data["I (A)"] * -1.
        return iv_data 		# DataFrame type

    def set_xlabel(self, xlabel):
        self.xlabel = xlabel

    def set_ylabel(self, ylabel):
        self.ylabel = ylabel

    def set_yscale(self, yscale):
        self.yscale = yscale

    def __help__(self):
        string = "IV class \n"
        string += "Parameters: \n"
        string += " - iv_type: str type, 'forward'/'inverse' \n"
        string += " - temperature: int type \n"
        string += " - dir = str type, directory where the IV file is stored (e.g. './IV/')"
        return string

    def setup(self):
        self.data = self.read_iv_from_file()
        self.set_xlabel(fw_rv_dict[self.iv_type]["xlabel"])
        self.set_ylabel(fw_rv_dict[self.iv_type]["ylabel"])
        self.set_yscale(fw_rv_dict[self.iv_type]["yscale"])

    def set_plot_style_options(self, ax):
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_yscale(self.yscale)
        plt.subplots_adjust(left = 0.13)
        plt.legend()

    def plot_iv(self, fig = None, ax = None):
        if ax is None: fig, ax = plt.subplots(figsize = (8, 6))
        vv, ii, ii_unc = self.data["V (V)"], self.data["I (A)"], self.data["dI (A)"]
        ax.errorbar(vv, ii, yerr = ii_unc, label = "%d °C" % self.temperature)
        self.set_plot_style_options(ax)
        return fig, ax

    def fit_forward_iv(self, V1, V2):
        mask = (self.data["V (V)"] > V1) * (self.data["V (V)"] < V2)
        V, I, I_unc = self.data["V (V)"][mask], self.data["I (A)"][mask], self.data["dI (A)"][mask]
        lsq = LeastSquares(V, I, I_unc, linear_function)
        m = Minuit(lsq, q=0., p=0.1)
        m.migrad()
        return m

    def save_to_file(self, filename):
        # to save with pickle module, see page 332-334 of Python manual
        return 0

    def load_to_file(self, filename):
        # to load with pickle module, see page 332-334 of Python manual
        return 0

    def logarithmic_derivative_analysis(self, fig = None, ax = None):

        if ax == None:
            fig, ax = plt.subplots(figsize = (8, 6))

        q0, p0 = 25.68, 0.028
        V, dV = self.data["V (V)"], 0.025
        I, dI = self.data["I (A)"], self.data["dI (A)"]
        logI, dlogI = np.log(I), dI/I
        logarithmic_derivative = np.gradient(logI, V) # = np.gradient(np.log(I)) / np.gradient(V)
        #logarithmic_derivative_unc = np.abs( 3 * logarithmic_derivative * ( (dI / I) + dV / V ) )
        dlogarithmic_derivative_dlogI = (np.gradient(logI + dlogI/100, V) - np.gradient(logI - dlogI/100, V)) / (2*dlogI/100)
        dlogarithmic_derivative_dV = (np.gradient(logI, V+dV/100) - np.gradient(logI, V-dV/100)) / (2*dV/100)
        dlogarithmic_derivative = np.sqrt( np.power( dlogarithmic_derivative_dlogI * dlogI, 2) + np.power( dlogarithmic_derivative_dV * dV, 2))
        ax.errorbar(V, logarithmic_derivative, xerr = dV, yerr = dlogarithmic_derivative, \
                    label = "%d C" % self.temperature, marker = "o", ms = 4)
        # V, logarithmic_derivative, dlogarithmic_derivative
        max_logarithmic_derivative = np.max(logarithmic_derivative)
        i_max = np.where(logarithmic_derivative == max_logarithmic_derivative)[0][0]
        i1, i2 = i_max - ((35 - self.temperature) // 12), i_max + 50 - ((35 - self.temperature) // 50) * 25
        # i1 -> at decreasing temperatures, there is a wider peak, so I need to account also for more points before the maximum one
        # i2 -> at -15°C, there is too much noise --> need to select a smaller interval for the peak (from 50 points to 25)
        lsq = LeastSquares(V[i1:i2], logarithmic_derivative[i1:i2], dlogarithmic_derivative[i1:i2], logarithmic_derivative_above_Vbd)
        Vbd0 = q0 + p0 * self.temperature
        m = Minuit(lsq, Vbd = Vbd0)
        m.limits["Vbd"] = [0.95*Vbd0, 1.05*Vbd0]
        m.migrad()
        print (m)
        x = np.linspace(V[i_max], np.max(V), 200)
        y = logarithmic_derivative_above_Vbd(x, *m.values)
        ax.errorbar(x, y, color = "red", ls = "dashed")
        return m, fig, ax


###################################################### DIRECT IV CURVES ANALYSIS ######################################################


def forward_analysis_equivalent_resistance(direct_iv_curves, show_plot = True):
    n = len(direct_iv_curves)
    R_eq, R_eq_stat_unc, R_eq_syst_unc = np.zeros(n), np.zeros(n), np.zeros(n)

    fig, ax = plt.subplots(figsize=(8,6))   # Preparo un plot in cui ci metto le iv dirette con stesso ordine (decr.) e stessi colori del plot fatto all'inizio
    for i in range(n-1, -1, -1):
        direct_iv_curves[i].plot_iv(fig, ax)

    for i in range(n):  # Riempio R_q in ordine crescente, seguendo l'ordine di direct_iv_curves
        R_eq_, R_eq_stat_unc_ = np.zeros(6), np.zeros(6)
        j = 0
        for V1 in np.linspace(1., 1.5, 6):
            V2 = V1 + 0.5
            m = direct_iv_curves[i].fit_forward_iv(V1, V2)  # m è di tipo Minuit
            R_eq_[j] = 1 / m.values["p"]
            R_eq_stat_unc_[j] = (1 / m.values["p"]**2) * m.errors["p"]
            j += 1

            x = np.linspace(1., 2., 100)      # Plotto le rette di best fit in tutto l'intervallo [1.-2.]
            y = linear_function(x, *m.values)
            ax.errorbar(x, y, ls = "dotted") #, label = "linear fit in [%.2f V - %.2f V]" % (V1, V2))

        R_eq[i] = np.average(R_eq_, weights = 1/np.power(R_eq_stat_unc_, 2) )
        R_eq_stat_unc[i] = np.sqrt(1 / np.sum( 1 / np.power(R_eq_stat_unc_, 2)) )
        R_eq_syst_unc[i] = (np.max(R_eq_) - np.min(R_eq_)) * 0.5
        print ("Temperature = %d°C \t R = %.3f +/- %.3f (stat) +/- %.3f (syst)" % \
               (direct_iv_curves[i].temperature, R_eq[i], R_eq_stat_unc[i], R_eq_syst_unc[i]))
    plt.legend()
    plt.grid()
    ax.set_xlim(0.8, 2.2)
    ax.set_ylim(0.005, 0.035)
    if show_plot: plt.show()
    return R_eq, R_eq_stat_unc, R_eq_syst_unc


def plot_resistance_vs_temperature(t, R_eq, R_eq_unc):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.errorbar(t, R_eq, yerr = R_eq_unc, label = "Data", ls = "None", marker = "v", ms = 8, capsize = 6)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("R$_{eq}$ ($\Omega$)")

    # fit
    x = np.linspace(5, 35, 300)
    lsq_lin, lsq_quad = LeastSquares(t, R_eq, R_eq_unc, linear_function), LeastSquares(t, R_eq, R_eq_unc, quadratic_function)
    m_lin, m_quad = Minuit(lsq_lin, q = 70., p = -0.5), Minuit(lsq_quad, a0 = 70., a1 = -0., a2 = -0.1)
    m_quad.fixed["a1"] = True
    m_lin.migrad(); m_quad.migrad()
    print(50 * "#" + " Fit with linear function\n", m_lin); print(50*"#" + " Fit with quadratic function\n", m_quad)
    y_lin, y_quad = linear_function(x, *m_lin.values), quadratic_function(x, *m_quad.values)
    ax.errorbar(x, y_lin, label = "Linear fit"); ax.errorbar(x, y_quad, label = "Quadratic fit")
    ax.legend()
    ax.grid()
    fig.savefig("Plots/R_vs_T_fit.png")
    fig.savefig("Plots/R_vs_T_fit.eps")
    plt.show()


def plot_normalized_forward_IV(direct_iv_curves, ref = 0):  # ref ? --> Plotto le IV normalizzate alla IV a temperatura tt[ref]. ref = 0 --> T = 5
    ### Direct IV analysis: ratio of the IV curves with respect to the curve at 5°C to see the effect of temperature variations
    fig, axs = plt.subplots(2, 1, figsize = (8,8), sharex = True, height_ratios = [0.6, 0.4])
    fig.subplots_adjust(hspace = 0)

    for i in range(len(direct_iv_curves)-1, -1, -1):
        direct_iv_curves[i].plot_iv(fig, axs[0])
    axs[0].set_yscale("log")
    axs[0].set_xlim(left = 0.2)
    axs[0].set_ylim(bottom = 7e-10)
    axs[0].grid()
    #plt.legend()

    x = direct_iv_curves[ref].data["V (V)"]            ## ivs[0] # Curva IV a 5 gradi # ivs[-1] # Curva IV a 35 gradi
    for i in range(len(direct_iv_curves)-1, -1, -1):
        y  = direct_iv_curves[i].data["I (A)"] / direct_iv_curves[ref].data["I (A)"]
        dy = y * \
             ( np.abs( direct_iv_curves[i].data["dI (A)"] / direct_iv_curves[i].data["I (A)"] ) + \
               np.abs( direct_iv_curves[ref].data["dI (A)"] / direct_iv_curves[ref].data["I (A)"] ) )
        axs[1].errorbar(x, y, yerr = dy, label = "%d °C" % direct_iv_curves[i].temperature)
    axs[1].set_ylabel("I(V, T) / I(V, %d °C)" % (direct_iv_curves[ref].temperature))
    axs[1].set_xlabel(direct_iv_curves[ref].xlabel)
    axs[1].set_yscale("log")
    axs[1].set_xlim(left = 0.2)
    axs[1].set_ylim(top = 30, bottom = 0.6)
    axs[1].grid()
    axs[1].legend()
    fig.savefig("Plots/Forward_IV_log_scale_ratio_to_%dC.png" % direct_iv_curves[ref].temperature)
    fig.savefig("Plots/Forward_IV_log_scale_ratio_to_%dC.eps" % direct_iv_curves[ref].temperature)
    plt.show()


###################################################### INVERSE IV CURVES ANALYSIS ######################################################


def logarighmic_derivative_study(inverse_iv_curves):
    n = len(inverse_iv_curves)
    Vbd, dVbd = np.zeros(n), np.zeros(n)
    T = np.zeros(n)
    fig, ax = plt.subplots(figsize = (12, 6))

    for i in range(n-1, -1, -1):
        #fig, ax = plt.subplots(figsize = (8, 6))
        m, fig, ax = inverse_iv_curves[i].logarithmic_derivative_analysis(fig, ax)
        T[i] = inverse_iv_curves[i].temperature
        Vbd[i], dVbd[i] = m.values["Vbd"], m.errors["Vbd"]
        ax.set_ylabel("Logarithmic derivative of I(V) (A/V)")
        ax.set_xlabel(inverse_iv_curves[i].xlabel)
        ax.legend()
        ax.grid()
        #fig.savefig("Plots/fit_Vbd/fit_Vbd_at_%dC.png" % T[i])
        #fig.savefig("Plots/fit_Vbd/fit_Vbd_at_%dC.eps" % T[i])

    ax.set_xlim(left = 25, right = 28)
    ax.set_ylim(bottom = -2., top = 8.5)
    fig.savefig("Plots/fit_Vbd/all.png")
    fig.savefig("Plots/fit_Vbd/all.eps")
    plt.show()

    fig_, ax_ = plt.subplots(figsize = (8,6))
    ax_.errorbar(T, Vbd, xerr = 1, yerr = dVbd, label = "Data", ls = "None", marker = "^", ms = 6, color = "black")
    lsq = LeastSquares(T, Vbd, dVbd, linear_function)
    m = Minuit(lsq, q = 25., p = 0.01)
    m.migrad()
    print (m)
    x = np.linspace(-20, 40, 200)
    y = linear_function(x, *m.values)
    ax_.errorbar(x, y, label = "Fit", color = "red")
    ax_.set_xlabel("Temperature (°C)")
    ax_.set_ylabel("Breakdown voltage (V)")
    ax_.set_ylim(bottom = 25, top = 27.)
    ax_.grid()
    ax_.legend()
    fig_.savefig("Plots/Vbd_vs_T.png")
    fig_.savefig("Plots/Vbd_vs_T.eps")
    plt.show()
    return Vbd, dVbd, m


def exponential_function(x, C, a):
    return C * np.exp(a * x)


def plot_normalized_inverse_IV(inverse_iv_curves, Vbd, ref = 0):  # ref ? --> Plotto le IV normalizzate alla IV a temperatura tt[ref]. ref = 0 --> T = 5
    ### Direct IV analysis: ratio of the IV curves with respect to the curve at 5°C to see the effect of temperature variations
    fig, axs = plt.subplots(2, 2, figsize = (15,10))
    n = len(inverse_iv_curves)
    T = np.zeros(n)

    for i in range(n-1, -1, -1):
        T[i] = inverse_iv_curves[i].temperature
        axs[0][0].errorbar(inverse_iv_curves[i].data["V (V)"] - Vbd[i], inverse_iv_curves[i].data["I (A)"], yerr = inverse_iv_curves[i].data["dI (A)"],\
                        label = "%d °C" % inverse_iv_curves[i].temperature)
    axs[0][0].set_xlabel("V$_{OV}$ (V)")
    axs[0][0].set_ylabel("I(V$_{OV}$, T) (A)")
    axs[0][0].set_yscale("log")
    axs[0][0].set_xlim(right = 7.2)
    axs[0][0].grid()
    axs[0][0].legend()

    # Ora, le curve IV in funzione dell'overvoltage sono definite su array lievemente differenti. Voglio definirle su un array standard X interpolando.
    xmin, xmax = -2., 4.
    X = np.linspace(xmin, xmax, int(xmax - xmin) * 5 + 1)
    y, dy = [], []
    for i in range(n):
        _x = inverse_iv_curves[i].data["V (V)"] - Vbd[i]            ## ivs[0] # Curva IV a 5 gradi # ivs[-1] # Curva IV a 35 gradi
        _y, _dy = inverse_iv_curves[i].data["I (A)"], inverse_iv_curves[i].data["dI (A)"]
        Y, dY = np.interp(X, _x, _y), np.interp(X, _x, _dy)
        print (Y)
        y.append(Y)
        dy.append(dY)
    # X è l'array comune delle tensioni, y[i] è l'interpolazione della corrente della curva IV i-esima sui punti di X (analogo per dy[i])

    # Plot delle curve IV normalizzate a quella a 20 °C
    ref = 7  # -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35 --> Così prendo la curva a 20 gradi
    for i in range(n-1, -1, -1):
        ratio = y[i] / y[ref]
        dratio = ratio * (dy[i]/y[i] + dy[ref]/y[ref])
        axs[0][1].errorbar(X, ratio, yerr = dratio, label = "%d °C" % inverse_iv_curves[i].temperature)
    axs[0][1].set_xlabel("V$_{OV}$ (V)")
    axs[0][1].set_ylabel("I(V$_{OV}$, T) / I(V$_{OV}$, T=20 °C)")
    axs[0][1].set_yscale("log")
    axs[0][1].set_xlim(left = -4.)
    axs[0][1].grid()
    axs[0][1].legend()

    # Plot di I vs T a V_OV fissato
    V_ov_plot = [-1., 0., 1., 2., 3., 4.]
    V_ov = np.linspace(-2., 4., 6 * 5 + 1)
    V_ov = V_ov[::-1]
    n_V_ov = len(V_ov)
    # Vettore delle temperature è T, riempito all'inizio

    I, dI = [], []
    for i in range(n_V_ov):
        index_of_V_OV_in_X = np.where(X == V_ov[i])[0][0]
        curr, dcurr = np.zeros(n), np.zeros(n)
        for j in range(n):
            curr[j], dcurr[j] = y[j][index_of_V_OV_in_X], dy[j][index_of_V_OV_in_X]
        I.append(curr)
        dI.append(dcurr)

    ref = np.where(V_ov == 0)[0][0]
    for i in range(n_V_ov):
        if V_ov[i] in V_ov_plot:
            axs[1][0].errorbar(T, I[i], yerr = dI[i], label = "V$_{OV}$ = %2.1f V" % V_ov[i])
        #ratio = I[i] / I[ref]
        #dratio = ratio * (dI[i]/I[i] + dI[ref]/I[ref])
        #axs[1][1].errorbar(T, ratio, yerr = dratio, label = "V$_{OV}$ = %2.1f V" % V_ov[i])

    # Fit of IT curve with exponential function
    M = []
    C, dC, a, da = np.zeros(n_V_ov), np.zeros(n_V_ov), np.zeros(n_V_ov), np.zeros(n_V_ov)
    for i in range(n_V_ov):
        lsq = LeastSquares(T, I[i], dI[i], exponential_function)
        m = Minuit(lsq, C = 1e-8, a = 82.e-3)
        m.migrad()
        print (m)
        M.append(m)
        if V_ov[i] in V_ov_plot:
            tmin, tmax = -18., 38.
            tt = np.linspace(tmin, tmax, int(tmax - tmin)*2+1)
            yy = exponential_function(tt, *m.values)
            axs[1][0].errorbar(tt, yy, ls = "dotted", color = "black")
        C[i], dC[i], a[i], da[i] = m.values["C"], m.errors["C"], m.values["a"], m.errors["a"]

    axs[1][0].set_ylabel("I(V$_{OV}$, T)) (A)")
    axs[1][0].set_xlabel("T (°C)")
    axs[1][0].set_yscale("log")
    axs[1][0].set_ylim(bottom = 1e-11)
    axs[1][0].grid()
    axs[1][0].legend(fontsize = "small")

    color = 'tab:blue'
    axs[1][1].tick_params(axis='y', labelcolor = color)
    axs[1][1].errorbar(V_ov, C, yerr = dC, label = "C", color = color)
    axs[1][1].set_xlabel("V$_{OV}$ (V)")
    axs[1][1].set_ylabel("C(V$_{OV}$) (A)", color = color)
    axs[1][1].set_yscale("log")
    axs[1][1].grid()
    axs[1][1].legend()

    color = "tab:red"
    ax2 = axs[1][1].twinx()
    ax2.set_ylabel("a(V$_{OV}$) (°C$^{-1}$)", color = color)
    ax2.tick_params(axis='y', labelcolor = color)
    ax2.errorbar(V_ov, a, yerr = da, label = "a", color = color)

    fig.savefig("Plots/Inverse_IV_vs_OV.png")
    fig.savefig("Plots/Inverse_IV_vs_OV.eps")

    plt.show()
