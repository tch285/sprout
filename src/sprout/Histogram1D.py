import json
from io import TextIOBase

import fit as ff
import gen_utils as gutils
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import numpy as np
import ROOT
import scipy.optimize as opt
import uproot as ur
from scipy.stats import gmean

class Histogram1D:
    """Class to convert ROOT histogram to NumPy arrays."""
    def __init__(self, name, contents, centers, edges, xerr, yerr, binning, label = None) -> None:
        assert len(centers) == (len(edges) - 1)
        self.nbins = len(centers)
        assert binning in ['lin', 'log']
        self.binning = binning
        self.name = name

        self.contents = contents
        self.edges = edges
        self.xerrlo, self.xerrhi = xerr
        self.yerr = yerr
        self.widths = self.edges[1:] - self.edges[:-1]
        self.centers = centers
        self.label = label
        self.transition = None

    @classmethod
    def load(cls, file, name, label = None):
        if isinstance(file, str):
            if file.endswith(".root"):
                return cls.from_uproot(file, name, label)
            elif file.endswith(".json"):
                return cls.from_json(file, name, label)
            elif file.endswith(".txt"):
                return cls.from_hepdata(file, name, label)
        elif isinstance(file, ur.ReadOnlyDirectory):
            return cls._from_uproot(file, name, label)
        elif isinstance(file, TextIOBase):
            if file.name.endswith(".json"):
                return cls._from_json(file, name, label)
            elif file.name.endswith(".txt"):
                return cls._from_hepdata(file, name, label)
        elif isinstance(file, ROOT.TFile):
            return cls.from_TFile(file, name, label)
        elif isinstance(file, ROOT.TH1):
            return cls.from_TH1(file, name, label)
        raise TypeError(f"Could not parse this type for file: {type(file)}")

    @classmethod
    def _from_arrays(cls, name, heights, edges, errors, label = ""):
        widths = edges[1:] - edges[:-1]
        with np.errstate(divide='raise'):
            try:
                ratios = edges[1:] / edges[:-1]
            except FloatingPointError:
                binning = "lin"
        with np.errstate(divide='raise'):
            try:
                ratios = edges[1:] / edges[:-1]
            except FloatingPointError:
                binning = "lin"

        if np.allclose(widths, widths[0]):
            binning = "lin"
            centers = (edges[:-1] + edges[1:]) / 2
        elif np.allclose(ratios, ratios[0]):
            binning = "log"
            centers = np.sqrt(edges[:-1] * edges[1:])
        else:
            print("Binning style couldn't be verified, defaulting to log.")
            binning = "log"
            centers = np.sqrt(edges[:-1] * edges[1:])
        xerrlo = centers - edges[:-1]
        xerrhi = edges[1:] - centers

        return cls(name, heights, centers, edges, [xerrlo, xerrhi], errors, binning, label)

    @classmethod
    def from_uproot(cls, file, name, label):
        """Extract Histogram1D from file with uproot."""
        with ur.open(file) as f:
            return cls._from_uproot(f, name, label)

    @classmethod
    def _from_uproot(cls, file, name, label):
        """Extract Histogram1D from opened uproot file."""
        h = file[name]
        return cls._from_arrays(name, h.values(), h.axis().edges, h.errors(), label)

    @classmethod
    def from_TH1(cls, h, name, label):
        """Extract Histogram1D from existing TH1 object."""
        hname = name if name else h.GetName()

        nbins = h.GetNbinsX()
        if h.GetXaxis().GetXbins().GetSize() != 0: # variable edges
            edges = np.array(h.GetXaxis().GetXbins())
        else: # fixed edges
            edges = np.linspace(h.GetXaxis().GetXmin(), h.GetXaxis().GetXmax(), nbins + 1)

        heights = np.array([h.GetBinContent(i) for i in range(1, nbins + 1)])
        errors = np.array([h.GetBinErrorLow(i) for i in range(1, nbins + 1)])
        return cls._from_arrays(hname, heights, edges, errors, label)

    @classmethod
    def from_TFile(cls, file, name, label):
        """Extract Histogram1D from open TFile object."""
        h = file.Get(name)
        return cls.from_TH1(h, name, label)

    @classmethod
    def from_hepdata(cls, file, name, label):
        """Extract Histogram1D from HEPData txt file."""
        with open(file) as f:
            return cls._from_hepdata(f, name, label)

    @classmethod
    def _from_hepdata(cls, file, name, label):
        """Extract Histogram1D from opened HEPData txt file."""
        lines = file.read().splitlines()
        hname = name if name else lines[2]
        val_lines = lines[12:-1]
        values = np.asarray([line.split('\t') for line in val_lines], dtype = float).T
        binL, binR, heights, errors, syst = values
        assert len(binL) == len(binR)
        edges = np.append(binL, binR[-1])

        return cls._from_arrays(hname, heights, edges, errors, label), syst

    @classmethod
    def from_json(cls, path, name, label = None):
        """Extract Histogram1D from JSON file."""
        with open(path) as file:
            return cls._from_json(file, name, label)

    @classmethod
    def _from_json(cls, file, name, label):
        """Extract Histogram1D from opened JSON file."""
        info = json.load(file)
        heights = info['val']
        errors = info['stat']
        binL = info['bin_L']
        binR = info['bin_R']
        assert len(binL) == len(binR)
        edges = np.append(binL, binR[-1])
        syst = info['syst']
        hname = name if name else info['name']
        return cls._from_arrays(hname, heights, edges, errors, label), syst

    @classmethod
    def copy(cls, orig):
        return cls("", orig.contents.copy(), orig.centers.copy(), orig.edges.copy(), [orig.xerrlo.copy(), orig.xerrhi.copy()], orig.yerr.copy(), orig.binning, orig.label)

    def _selfcopy(self):
        return Histogram1D(self.name, self.contents.copy(), self.centers.copy(), self.edges.copy(), [self.xerrlo.copy(), self.xerrhi.copy()], self.yerr.copy(), self.binning, self.label)

    def set_binning(self, binning):
        if binning == 'lin':
            self.centers = (self.edges[:-1] + self.edges[1:]) / 2
        elif binning == "log":
            self.centers = np.sqrt(self.edges[:-1] * self.edges[1:])
        else:
            raise ValueError(f"Binning style {binning} is invalid.")
        self.binning = binning

    def norm_max(self, norm = 1):
        f = norm / self.contents.max()
        self.contents *= f
        self.yerr *= f

    def rebin(self, n = 2, width_normed = True):
        if self.nbins % n != 0:
            raise ValueError(f"Rebin number {n} is not a divisor of nbins {self.nbins}")
        old_widths = self.widths

        self.edges = self.edges[::n]
        self.widths = self.edges[1:] - self.edges[:-1]

        if self.binning == "lin":
            self.centers = (self.edges[:-1] + self.edges[1:]) / 2
        else:
            self.centers = np.sqrt(self.edges[:-1] * self.edges[1:])

        self.xerrlo = self.centers - self.edges[:-1]
        self.xerrhi = self.edges[1:] - self.centers

        # Modify contents
        if width_normed:
            self.contents *= old_widths # remove previous width norm
            self.contents = np.sum(self.contents.reshape(-1, n), axis=1)
            self.contents /= self.widths # apply new width norm

            self.yerr *= old_widths # remove previous width norm
            errsq = self.yerr ** 2 # get squares of errors
            self.yerr = np.sqrt(np.sum(errsq.reshape(-1, n), axis=1)) # sum and square root to get errors
            self.yerr /= self.widths # apply new width norm
        else:
            self.contents = np.sum(self.contents.reshape(-1, n), axis=1)
            errsq = self.yerr ** 2
            self.yerr = np.sqrt(np.sum(errsq.reshape(-1, n), axis=1))

        return self

    def add_to(self, ax, color = None, as_step = False, as_marker = False, as_line = False, as_shaded_line = False, **kwargs):
        if as_step:
            return ax.hist(self.edges[:-1], self.edges, weights = self.contents,
                    histtype = 'step', color = color, **kwargs)
        elif as_marker:
            draw_args = {'marker': 'o',
                         'mfc': 'none',
                         'label': self.label,
                         'ls': '',
                         'alpha': 0.5,
                         }
            draw_args.update(kwargs)
            if ('color' in kwargs and 'mec' not in kwargs and 'markeredgecolor' not in kwargs):
                draw_args['mec'] = kwargs['color']
            elif color:
                draw_args['mec'] = color
            return ax.plot(self.centers, self.contents, **draw_args)
        elif as_line:
            draw_args = {'label': self.label,
                         'alpha': 0.8,
                         'color': color,
                         'lw'   : 1.3,
                         }
            draw_args.update(kwargs)
            return ax.plot(self.centers, self.contents, **draw_args)
        elif as_shaded_line:
            draw_args = {'label': self.label,
                         'color': color,
                         }
            draw_args.update(kwargs)
            ax.plot(self.centers, self.contents, alpha = 0.6, lw=1.2, **draw_args)
            return ax.fill_between(self.centers, self.contents - self.yerr, self.contents + self.yerr,
                            alpha = 0.3, lw = 0, **draw_args)
        else:
            draw_args = {'marker': 'o',
                         'ls': '',
                         'ms': 3.5,
                         'label': self.label,
                         'mec': 'none',
                         'mfc': 'none',
                         'alpha': 0.8,
                         'mew': 0,
                         'zorder': 100
                         }
            draw_args.update(kwargs)
            # default_alphas = {'e': 0.8, 'm': 1} # alphas for errorbars (e) vs. markers (m)
            
            if 'color' in kwargs:
                draw_args['mfc'] = mplc.to_rgba(kwargs['color'], 1)
                draw_args['ecolor'] = mplc.to_rgba(kwargs['color'], draw_args['alpha'])
                draw_args['mec'] = mplc.to_rgba(kwargs['color'], draw_args['alpha'])
            elif color:
                if 'mfc' in kwargs:
                    if kwargs['mfc'] in ['none', "white", 'w']:
                        draw_args['mfc'] = kwargs['mfc']
                        draw_args['ecolor'] = mplc.to_rgba(color, draw_args['alpha'])
                        draw_args['mec'] = mplc.to_rgba(color, 1)
                        # draw_args['mec'] = mplc.to_rgba(color, draw_args['alpha'])
                    else:
                        draw_args['mfc'] = mplc.to_rgba(kwargs['mfc'], 1)
                        draw_args['ecolor'] = mplc.to_rgba(kwargs['mfc'], draw_args['alpha'])
                        draw_args['mec'] = mplc.to_rgba(kwargs['mfc'], draw_args['alpha'])
                else:
                    draw_args['mfc'] = mplc.to_rgba(color, 1)
                    draw_args['ecolor'] = mplc.to_rgba(color, draw_args['alpha'])
                    draw_args['mec'] = mplc.to_rgba(color, draw_args['alpha'])
            # print(draw_args['mfc'])
            # print(draw_args['ecolor'])
            # print(draw_args['mec'])
            draw_args.pop('alpha')
            return ax.errorbar(self.centers, self.contents, self.yerr, [self.xerrlo, self.xerrhi], **draw_args)

    def plot(self, filename: str, xlabel = "", ylabel = "", show: bool = False):
        fig, ax = plt.subplots()
        ax.errorbar(self.centers, self.contents, self.yerr, [self.xerrlo, self.xerrhi], marker = 'o', linestyle = '', markersize = 2, label = self.label, color = 'r')
        ax.set_title(self.label) if self.label is not None else self.name
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if self.binning == 'log':
            ax.set_yscale('log')

        fig.savefig(filename)
        if show:
            plt.show()
        fig.close()

    def add_transition_to(self, ax, color, overwrite = False, show_curve = False):
        if self.transition is None or overwrite:
            self.calculate_transition()
        self.transition_line = ax.axvline(self.transition, color = color, label = f"$T \\approx {self.transition:.2f}\\pm {self.transition_err:.3f}$ GeV", linewidth = 0.6)
        if show_curve:
            self.show_fit_curve(ax, color)

    def calculate_fwhm(self, ax, color='r'):
        try:
            abs_contents = np.abs(self.contents)
            bool_arr = abs_contents > np.abs(self.peak / 2)
            lbound = bool_arr.argmax() - 1#, bool_arr.argmax()
            rbound = len(bool_arr) - np.flip(bool_arr).argmax() - 1#, len(bool_arr) - np.flip(bool_arr).argmax() + 1
            left = gmean(self.centers[lbound:lbound + 2])
            right = gmean(self.centers[rbound:rbound + 2])
            fwhm = right - left
            ax.hlines(self.peak / 2, left, right, color = color, label = f"FWHM $\\approx {fwhm:.2f}$", linestyle='--', alpha = 0.4)
            return fwhm
        except AttributeError:
            self.calculate_transition()
            print(self.transition)
            return self.calculate_fwhm(ax)

    def calculate_transition(self, xrange = 0.3, shift = 4, func = 'quad', exp = 1.5, syst = None):
        # print(f"Calculating transition for H1D {self.title}.")
        self.fit_idx_min, self.fit_idx_max = self._find_range(xrange, shift)
        fitx = self.centers[self.fit_idx_min:self.fit_idx_max + 1]
        fity = np.abs(self.contents[self.fit_idx_min:self.fit_idx_max + 1])
        # abs_contents = np.abs(self.contents)
        if syst is not None:
            unc = np.sqrt(self.yerr ** 2 + (self.contents * syst) ** 2)[self.fit_idx_min:self.fit_idx_max + 1]
            # unc = self.yerr[self.fit_idx_min:self.fit_idx_max + 1]
        else:
            unc = self.yerr[self.fit_idx_min:self.fit_idx_max + 1]
        factor = 1
        if self.contents[np.abs(self.contents).argmax()] < 0:
            factor = -1

        if func == 'quad':
            [self.transition, self.peak, self.a], cov = opt.curve_fit(ff.quad_log, fitx, fity, [fitx[fity.argmax()], fity.max(), -10], unc, absolute_sigma = True)
            self.a, self.peak = factor * self.a, factor * self.peak
            self.transition_err, self.peak_err, *rem = np.sqrt(np.diag(cov))
        elif func == 'gaus':
            [self.transition, self.peak, self.sg], cov = opt.curve_fit(ff.gaus_log, fitx, fity, [fitx[fity.argmax()], fity.max(), 3.5], unc, absolute_sigma = True)
            self.peak *= factor
            self.transition_err, self.peak_err, *rem = np.sqrt(np.diag(cov))
        elif func == 'cust_TC':
            init_tr = fitx[fity.argmax()]
            init_pk = fity.max()
            T0  = 2 * init_tr ** 2
            C0 = T0 * init_pk * 1.5 * np.sqrt(3)
            def func(x, T, C):
                return ff.cust_TC(x, T, C, exp)
            [T, C], cov = opt.curve_fit(func, fitx, fity, [T0, C0], unc, absolute_sigma = True)
            T_err, C_err = np.sqrt(np.diag(cov))
            assert cov[0, 1] == cov[1, 0]
            covTC = cov[0, 1]
            self.transition = np.sqrt(T / 2)
            self.transition_err = T_err / (np.sqrt(8 * T))
            self.peak = (2 * C) / (3 * np.sqrt(3) * T)
            self.peak_err = 2 / (3 * np.sqrt(3)) * self.peak * np.sqrt( (C_err / C)**2 + (T_err / T)**2 - 2 * covTC / (T *C) )
            self.peak *= factor
        elif func == 'cust_tp':
            t0 = fitx[fity.argmax()]
            p0 = fity.max()
            def func(x, t, p):
                return ff.cust_tp(x, t, p, exp)
            [self.transition, self.peak], cov = opt.curve_fit(func, fitx, fity, [t0, p0], unc, absolute_sigma = True)
            self.peak *= factor
            self.transition_err, self.peak_err = np.sqrt(np.diag(cov))
        else:
            print("Invalid function name, exiting.")
            return

    def _find_range(self, xrange, shift):
        abs_contents = np.abs(self.contents)
        idx = abs_contents[shift:].argmax() + shift

        if isinstance(xrange, int) and xrange >= 1:
            fit_idx_min = idx - xrange
            fit_idx_max = idx + xrange
        elif isinstance(xrange, float) and xrange < 1:
            max_val = abs_contents[idx]
            cutoff = max_val * (1 - xrange)

            def check(curr_idx, incr, cutoff):
                if abs_contents[curr_idx] > cutoff:
                    return check(curr_idx + incr, incr, cutoff)
                else:
                    return curr_idx

            fit_idx_min = check(idx - 1, -1, cutoff)
            fit_idx_max = check(idx + 1, 1, cutoff)
        elif isinstance(xrange, (tuple, list)) and len(xrange) == 3:
            xmin, xmax, greedy = xrange
            close_min = np.isclose(self.edges, xmin, rtol = 1e-3)
            if sum(close_min) == 1:
                fit_idx_min = np.argwhere(close_min)[0, 0]
            elif greedy:
                fit_idx_min = np.searchsorted(self.edges, xmin, side='left') - 1
            else:
                fit_idx_min = np.searchsorted(self.edges, xmin, side='left')

            close_max = np.isclose(self.edges, xmax, rtol = 1e-3)
            if sum(close_max) == 1:
                fit_idx_max = np.argwhere(close_max)[0, 0]
            elif greedy:
                fit_idx_max = np.searchsorted(self.edges, xmax, side='left') - 1
            else:
                fit_idx_max = np.searchsorted(self.edges, xmax, side='left') - 2
            print(fit_idx_min, fit_idx_max)
        else:
            raise ValueError("X range for fitting invalid.")
        return fit_idx_min, fit_idx_max

    def show_fit_curve(self, ax, color):
        xs = np.linspace(self.centers[self.fit_idx_min], self.centers[self.fit_idx_max])
        ys = self.a * np.square(np.log10(xs / self.transition)) + self.peak
        ax.plot(xs, ys, color = color)

    def save(self, f, name):
        if isinstance(f, np.lib.npyio.NpzFile):
            pass
        # return cls("", orig.contents.copy(), orig.centers.copy(), orig.edges.copy(), [orig.xerrlo.copy(), orig.xerrhi.copy()], orig.yerr.copy(), orig.binning, orig.label)

    def __len__(self): return self.nbins
    def __getitem__(self, idx): return self.contents[idx]
    def __setitem__(self, idx, val): self.contents[idx] = val
    def __str__(self): return f"Histogram1D '{self.name}' ({self.label})"
    def __add__(self, other):
        copy = self._selfcopy()
        if isinstance(other, (float, int)):
            copy.contents = self.contents + other
            copy.yerr = self.yerr.copy()
            copy.label = f"{self.label} + {other}"
        elif isinstance(other, Histogram1D):
            self._check_matched_edges(other, "+")
            copy.contents = self.contents + other.contents
            copy.yerr = np.sqrt(np.square(self.yerr) + np.square(other.yerr))
            copy.label = f"{self.label} + {other.label}"
        else:
            return NotImplemented

        return copy
    def __iadd__(self, other):
        if isinstance(other, (float, int)):
            self.contents += other
            self.label = f"{self.label} + {other}"
        elif isinstance(other, Histogram1D):
            self._check_matched_edges(other, "+=")
            self.contents += other.contents
            self.yerr = np.sqrt(np.square(self.yerr) + np.square(other.yerr))
            self.label = f"{self.label} + {other.label}"
        else:
            return NotImplemented

        return self
    def __sub__(self, other):
        copy = self._selfcopy()
        if isinstance(other, (float, int)):
            copy.contents = self.contents - other
            copy.yerr = self.yerr.copy()
            copy.label = f"{self.label} - {other}"
        elif isinstance(other, Histogram1D):
            self._check_matched_edges(other, "-")
            copy.contents = self.contents - other.contents
            copy.yerr = np.sqrt(np.square(self.yerr) + np.square(other.yerr))
            copy.label = f"{copy.label} - {other.label}"
        else:
            return NotImplemented

        return copy
    def __isub__(self, other):
        if isinstance(other, (float, int)):
            self.contents = self.contents - other
            self.label = f"{self.label} - {other}"
        elif isinstance(other, Histogram1D):
            self._check_matched_edges(other, "-=")
            self.contents -= other.contents
            self.yerr = np.sqrt(np.square(self.yerr) + np.square(other.yerr))
            self.label = f"{self.label} - {other.label}"
        else:
            return NotImplemented

        return self
    def __mul__(self, other):
        copy = self._selfcopy()
        if isinstance(other, (float, int)):
            copy.contents *= float(other)
            copy.yerr *= abs(float(other))
        elif isinstance(other, Histogram1D):
            self._check_matched_edges(other, "*")
            copy.contents = self.contents * other.contents
            copy.yerr = np.sqrt(np.square(self.yerr * other.contents) + np.square(other.yerr * self.contents))
            copy.label = f"{copy.label} \u00D7 {other.label}"
        else:
            return NotImplemented

        return copy
    def __imul__(self, other):
        if isinstance(other, (float, int)):
            self.contents *= float(other)
            self.yerr *= abs(float(other))
        elif isinstance(other, Histogram1D):
            self._check_matched_edges(other, "*=")
            # yerr must be modified before contents
            self.yerr = np.sqrt(np.square(self.yerr * other.contents) + np.square(other.yerr * self.contents))
            self.contents = self.contents * other.contents
            self.label = f"{self.label} \u00D7 {other.label}"
        else:
            return NotImplemented

        return self
    def __truediv__(self, other):
        copy = self._selfcopy()
        if isinstance(other, (float, int)):
            copy *= (1.0 / other)
        elif isinstance(other, Histogram1D):
            self._check_matched_edges(other, "/")
            copy.contents = np.divide(self.contents, other.contents, out = np.zeros(self.nbins), where = other.contents != 0)
            copy.yerr = np.sqrt(np.divide(self.yerr**2 , other.contents**2, out = np.zeros(self.nbins), where = other.contents != 0)
                              + np.divide(self.contents**2 , other.contents**4, out = np.zeros(self.nbins), where = other.contents != 0) * other.yerr**2)
            # copy.yerr = np.abs(np.divide(self.contents, other.contents, out = np.zeros_like(self.contents), where = other.contents != 0)) * np.sqrt(np.square(np.divide(self.yerr, self.contents, out = np.zeros_like(self.yerr), where = self.contents != 0)) + np.square(np.divide(other.yerr, other.contents, out = np.zeros_like(other.yerr), where = other.contents != 0)))
            copy.label = f"{self.label} / {other.label}"
        else:
            return NotImplemented

        return copy
    def __itruediv__(self, other):
        if isinstance(other, (float, int)):
            self *= (1.0 / other)
        elif isinstance(other, Histogram1D):
            self._check_matched_edges(other, "/=")
            self.yerr = np.sqrt(np.divide(self.yerr**2 , other.contents**2, out = np.zeros(self.nbins), where = other.contents != 0)
                              + np.divide(self.contents**2 , other.contents**4, out = np.zeros(self.nbins), where = other.contents != 0) * other.yerr**2)
            # self.yerr = np.divide(self.contents, other.contents, out = np.zeros_like(self.contents), where = other.contents != 0) * np.sqrt(np.square(np.divide(self.yerr, self.contents, out = np.zeros_like(self.yerr), where = self.contents != 0)) + np.square(np.divide(f.yerr, other.contents, out = np.zeros_like(f.yerr), where = other.contents != 0)))
            self.contents = np.divide(self.contents, other.contents, out = np.zeros(self.nbins), where = other.contents != 0)
            self.label = f"{self.label} / {other.label}"
        else:
            return NotImplemented

        return self
    __radd__ = __add__
    __rmul__ = __mul__
    def __rsub__(self, other):
        # called with other - self. Don't need to define for Histogram1D since __sub__ would be called first
        copy = self._selfcopy()
        if isinstance(other, (float, int)):
            copy.contents = other - self.contents
            copy.label = f"{self.label} - {other}"
        else:
            return NotImplemented

        return copy
    def __rtruediv__(self, other):
        # called with other / self. Don't need to define for Histogram1D since __truediv__ would be called first
        copy = self._selfcopy()
        if isinstance(other, (float, int)):
            copy.contents = np.divide(other.contents, self.contents, out = np.zeros(self.nbins), where = other.contents != 0)
            copy.yerr = self.yerr * np.divide(other.contents, self.contents ** 2, out = np.zeros(self.nbins), where = self.contents != 0)
            copy.label = f"{self.label} / {other.label}"
        else:
            return NotImplemented

        return copy
    def __neg__(self):
        copy = self._selfcopy()
        copy.contents = - copy.contents
        return copy
    def __mod__(self, other):
        """Check if two Histograms have compatible bin edges."""
        if np.allclose(self.edges, other.edges):
            return True
        else:
            return False
    def __abs__(self):
        copy = self._selfcopy()
        copy.contents = np.abs(copy.contents)
        return copy
    def make_abs(self):
        self.contents = np.abs(self.contents)

    def _check_matched_edges(self, other, op):
        if not self % other:
            raise ValueError(f"Histograms '{self.name}' and '{other.name}' edges do not match - cannot perform '{op}' operation.")

    def combine(self, hist):
        """Combine two histograms with inverse-variance weighting."""
        res = self._selfcopy()
        res.label = f"Combined {self.label} + {hist.label}"
        res.contents = (self.contents / (self.yerr ** 2) + hist.contents / (hist.yerr ** 2)) / ((1 / self.yerr ** 2) + (1 / (hist.yerr ** 2) ))
        res.yerr = 1 / np.sqrt((1 / self.yerr ** 2) + (1 / (hist.yerr ** 2) ))
        return res

    def width_norm(self):
        self.contents /= self.widths
        self.yerr /= self.widths
        return self

    def scalex(self, f: float):
        self.edges *= f
        self.centers *= f
        self.widths *= f
        self.xerrlo *= f
        self.xerrhi *= f
        self.contents /= f
        self.yerr /= abs(f)
        return self

    def to_pdf(self, do_width_norm = False):
        if do_width_norm:
            self.width_norm()
        integral = np.sum(self.contents * self.widths)
        self.contents /= integral
        self.yerr /= integral
        return self
    def do_barlow(self, val = 1, nsig = 2):
        self.contents = gutils.apply_barlow(self.contents, self.yerr, val, nsig)
    def smooth(self, n: int, bin_min: int = 1, bin_max = None):
        if bin_max is None:
            bin_max = self.nbins
        if self.nbins < 3:
            print(f"Smooth only works for histograms with 3 or more bins (nbins = {self.nbins})")
            return
        firstbin = bin_min - 1
        lastbin = bin_max - 1
        nbins = lastbin - firstbin + 1
        xx = np.zeros(nbins)
        for i in range(nbins):
            xx[i] = self.contents[i + firstbin]

        #TODO: xx modified in place? or need to return new np array
        xx_smoothed = gutils.apply_smoothing(nbins, xx, n)
        for i in range(nbins):
            self.contents[i+firstbin] = xx_smoothed[i]