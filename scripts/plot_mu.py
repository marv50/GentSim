from scripts.create_plots import plot_elementary_effects

data = {
    "mu_star": [13318.717128, 8031.224568, 1664.675112, 13101.188208, 14287.461216],
    "sigma": [27380.305100, 21294.095533, 3500.080045, 30355.147938, 30610.527789],
}
params = ["epsilon", "p_h", "b", "r_moore", "eta"]

plot_elementary_effects(data, params)