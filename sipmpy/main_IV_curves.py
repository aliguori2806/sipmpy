import matplotlib as mpl
import matplotlib.pyplot as plt
from Analysis_IV_curves import *

fontsize = 12
font = {"family":"Dejavu Sans", "weight":"normal", "size":fontsize}
mpl.rc("font", **font)


if __name__ == '__main__':

    dir = "/home/cerasole/dottorato/photodetection/Lab_data_2023/IV/"

    # Leggo le temperature alle quali sono state prese le IV forward e reverse. tt[0] e tt[1] sono ordinate in modo crescente
    tt = read_temperatures(dir)

    ivs = [[], []]

    # Nice plot of the IV curves, forward and reverse
    for i_iv_type in range(2):
        fig, ax = plt.subplots(figsize = (8, 6))
        for i_tt in range(len(tt[i_iv_type])-1, -1, -1):  # Per chiarezza (IV a T maggiore > IV a T minore), le disegno da quella a T maggiore a quella a T minore
            iv = IV( list(fw_rv_dict)[i_iv_type], tt[i_iv_type][i_tt], dir) # tipo di IV, temperatura e directory
            iv.setup()
            #if i_iv_type == 0: iv.set_yscale("log")
            iv.plot_iv(fig, ax)
            ivs[i_iv_type].append(iv)
        # Ho riempito ivs[i_iv_type] in un ciclo in cui ho spannato le temperature in ordine decrescente
        # Devo dunque invertire ivs[i_iv_type] per ordinare in modo crescente
        ivs[i_iv_type] = ivs[i_iv_type][::-1]
        plt.grid()
        fig.savefig("Plots/%s_%s_IV.png" % (iv.iv_type, iv.yscale))
        fig.savefig("Plots/%s_%s_IV.eps" % (iv.iv_type, iv.yscale))
        plt.show()

    ###################################################### FORWARD IV CURVES ANALYSIS ######################################################

    # ivs[0] è il vettore delle 7 IV dirette, tt[0] è il vettore delle 7 temperature alle quali sono prese quelle IV dirette
    # tt[0] e tt[1] sono ordinate in modo crescente.
    # ivs[0][i] è l'oggetto IV alla temperature tt[0][i]
    # R_eq, R_eq_stat_unc, R_eq_syst_unc sono i vettori delle 7 stime della resistenza (una per ognu temperatura)

    ### Direct IV analysis: evaluation of the equivalent resistance
    i_iv_type = 0
    R_eq, R_eq_stat_unc, R_eq_syst_unc = forward_analysis_equivalent_resistance(ivs[i_iv_type])
    R_eq_unc = np.sqrt(np.power(R_eq_stat_unc, 2) + np.power(R_eq_syst_unc, 2))
    plot_resistance_vs_temperature(tt[0], R_eq, R_eq_unc)

    ### Direct IV analysis: ratio of the IV curves with respect to the curve at 5°C to see the effect of temperature variations
    i_iv_type = 0
    plot_normalized_forward_IV(ivs[i_iv_type])

    ###################################################### INVERSE IV CURVES ANALYSIS ######################################################

    os.system("mkdir -p Plots/fit_Vbd/")

    ### INVERSE IV curves are ivs[i_iv_type]
    i_iv_type = 1
    Vbd, dVbd, m_Vbd_vs_T = logarighmic_derivative_study(ivs[i_iv_type])

    ### Inverse IV analysis: ratio of the IV curves with respect to the curve at 5°C to see the effect of temperature variations
    i_iv_type = 1
    plot_normalized_inverse_IV(ivs[i_iv_type], Vbd)
