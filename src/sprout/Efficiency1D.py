#!/usr/bin/env python3

import numpy as np
from ROOT import TEfficiency, TH1

class Efficiency1D(object):
    """Class to convert ROOT TEfficiency to NumPy arrays."""
    def __init__(self, teff, label) -> None:
        self.nbins = teff.GetTotalHistogram().GetNbinsX()
        eff = np.zeros(self.nbins)
        eff_err_lo = np.zeros(self.nbins)
        eff_err_hi = np.zeros(self.nbins)

        for bin in range(1, self.nbins + 1):
            eff[bin - 1] = teff.GetEfficiency(bin) * 100
            eff_err_lo[bin - 1] = teff.GetEfficiencyErrorLow(bin) * 100
            eff_err_hi[bin - 1] = teff.GetEfficiencyErrorUp(bin) * 100
        
        self.eff = eff
        self.eff_err_lo = eff_err_lo
        self.eff_err_hi = eff_err_hi
        
        hist = teff.GetTotalHistogram()
        self.bin_edges = np.zeros(self.nbins + 1)
        for bin in range(1, self.nbins + 1):
            self.bin_edges[bin - 1] = hist.GetXaxis().GetBinLowEdge(bin)
        self.bin_edges[self.nbins] = hist.GetXaxis().GetBinLowEdge(self.nbins + 1)
        if np.isclose(self.bin_edges[1] - self.bin_edges[0], self.bin_edges[2] - self.bin_edges[1]):
            self.bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2
        elif np.isclose(self.bin_edges[1] / self.bin_edges[0], self.bin_edges[2] / self.bin_edges[1]):
            self.bin_centers = np.sqrt(self.bin_edges[1:] * self.bin_edges[:-1])
        else:
            raise ValueError("Can't figure out if this efficiency object is log or linear binned...")

        self.xerr_hi = self.bin_edges[1:] - self.bin_centers
        self.xerr_lo = self.bin_centers - self.bin_edges[:-1]
        self.bin_width = self.bin_edges[1:] - self.bin_edges[:-1]
        
        self.label = label

    @classmethod
    def from_TH1(cls, hist_num: TH1, hist_denom: TH1, label: str):
        teff = TEfficiency(hist_num, hist_denom)
        return cls(teff, label)


    def add_to(self, ax, color, scale = 1):
        ax.errorbar(self.bin_centers, self.eff * scale, yerr = [self.eff_err_lo * scale, self.eff_err_hi * scale],
                    xerr = [self.xerr_lo, self.xerr_hi], lw=0, marker='s', elinewidth=0.8, capsize=0, markersize = 0.2,
                    color = color, label = self.label)