# import ROOT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import scipy.optimize as opt
from scipy.stats import gmean
# from functools import singledispatchmethod

import cEEC_utils.gen_utils as gutils
from cEEC_utils.Quantity import toQuant
import uproot
# import uproot

class Histogram1D:
    """Class to convert ROOT histogram to NumPy arrays."""
    def __init__(self, title, contents, centers, edges, xerr, yerr, binning, custom_label = None) -> None:
        assert len(centers) == (len(edges) - 1)
        self.nbins = len(centers)
        self.binning = binning

        self.contents = contents
        self.edges = edges
        self.xerrlow, self.xerrup = xerr
        self.yerr = yerr
        self.widths = self.edges[1:] - self.edges[:-1]
        self.centers = centers
        self.binning = binning
        self.transition = None
        self.label = None
        self._parse_title(title, custom_label)
    
    def set_title(self, title):
        self._parse_title(title, self.label)

    def norm_max(self, norm = 1):
        f = norm / self.contents.max()
        self.contents *= f
        self.yerr *= f
    
    def _parse_title(self, title, custom_label):
        spl = title.split(";")
        self.title = self.xlabel = self.ylabel = ""
        try:
            self.title = spl[0]
            self.xlabel = spl[1]
            self.ylabel = spl[2]
        except IndexError:
            pass

        self.label = self.title if custom_label is None else custom_label

    @classmethod
    def fromROOT(cls, hist, custom_label = None):
        label = gutils.convert_from_name(hist.GetName()) if custom_label is None else custom_label
        nbins = hist.GetNbinsX()
        title = hist.GetTitle()
        edges = np.zeros(nbins + 1)
        heights = np.zeros(nbins)
        yerr = np.zeros(nbins)
        for i in range(nbins):
            edges[i] = hist.GetXaxis().GetXbins().At(i)
            # print('got edge')
            heights[i] = hist.GetBinContent(i+1)
            # print('got height')
            yerr[i] = hist.GetBinErrorLow(i+1)
            assert yerr[i] == hist.GetBinErrorUp(i+1)
        edges[nbins] = hist.GetXaxis().GetXbins().At(nbins)
        widths = edges[1:] - edges[:-1]
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
        
        xerrlow = centers - edges[:-1]
        xerrup = edges[1:] - centers
        
        return cls(title, heights, centers, edges, [xerrlow, xerrup], yerr, binning, label)

    @classmethod
    def fromROOTfixed(cls, hist, custom_label = None):
        label = gutils.convert_from_name(hist.GetName()) if custom_label is None else custom_label
        nbins = hist.GetNbinsX()
        edges = np.linspace(hist.GetXaxis().GetXmin(), hist.GetXaxis().GetXmax(), nbins + 1)
        title = hist.GetTitle()
        heights = np.zeros(nbins)
        yerr = np.zeros(nbins)
        for i in range(nbins):
            heights[i] = hist.GetBinContent(i+1)
            yerr[i] = hist.GetBinErrorLow(i+1)
            assert yerr[i] == hist.GetBinErrorUp(i+1)

        binning = "lin"
        centers = (edges[:-1] + edges[1:]) / 2
        xerrlow = centers - edges[:-1]
        xerrup = edges[1:] - centers
        
        return cls(title, heights, centers, edges, [xerrlow, xerrup], yerr, binning, label)
    
    @classmethod
    def fromUproot(cls, *args):
        # other_args = args[1:]
        if isinstance(args[0], uproot.reading.ReadOnlyDirectory):
            # file given but not hist, hist name is next arg
            # args[1] is the filename
            pass
        elif isinstance(args[0], uproot.behaviors.TH1.TH1):
            # it's already opened and the hist is give
            pass
        elif isinstance(args[0], uproot.behaviors.TH1.TH1):
            pass

    # @singledispatchmethod
    # @classmethod
    # def fromUproot(cls, arg):
    #     raise NotImplementedError("Unknown type")
    
    # # Register an implementation for strings
    # # The _ name is just a convention; could be any name
    # @fromUproot.register(str)
    # @classmethod
    # def _(cls, arg):
    #     return cls(f"From string: {arg}")
    
    # # Register an implementation for integers
    # # Could use a different name instead of _
    # @fromUproot.register(int)
    # @classmethod
    # def handle_int(cls, arg):  # name doesn't matter
    #     return cls(f"From int: {arg}")
    
    def add_to(self, ax, color, as_marker = False, **kwargs):
        if as_marker:
            ax.plot(self.centers, self.contents, color = color, marker = as_marker, mfc = 'none', label = self.label, linestyle = '', markeredgecolor = color, alpha = 0.5, **kwargs)
        else:
            ax.errorbar(self.centers, self.contents, self.yerr, [self.xerrlow, self.xerrup], marker = 'o', linestyle = '', markersize = 3.5, label = self.label, mec='none', mfc=mplc.to_rgba(color, 1), ecolor = mplc.to_rgba(color, 0.5), **kwargs)

    def save(self, filename: str, show: bool = False, title = None):
        fig, ax = plt.subplots()
        ax.errorbar(self.centers, self.contents, self.yerr, [self.xerrlow, self.xerrup], marker = 'o', linestyle = '', markersize = 2, label = self.label, color = 'r')
        ax.set_title(self.title) if title is None else ax.set_title(title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        if self.binning == 'log':
            ax.set_yscale('log')

        fig.savefig(filename)
        if show:
            plt.show()

    def add_transition_to(self, ax, color, overwrite = False, show_curve = False):
        if self.transition is None or overwrite:
            self.calculate_transition()
        self.transition_line = ax.axvline(self.transition, color = color, label = f"$T \\approx {self.transition:.2f}\pm {self.transition_err:.3f}$ GeV", linewidth = 0.6)
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
            # print(left)
            # print(right)
            fwhm = right - left
            ax.hlines(self.peak / 2, left, right, color = color, label = f"FWHM $\\approx {fwhm:.2f}$", linestyle='--', alpha = 0.4)
            return fwhm
        except AttributeError:
            self.calculate_transition()
            print(self.transition)
            return self.calculate_fwhm(ax)
    
    def calculate_transition(self, pT_scope = 0.3, shift = 4):
        # print(f"Calculating transition for H1D {self.title}.")
        self.fit_idx_min, self.fit_idx_max = self._find_range(pT_scope, shift)
        fitx = self.centers[self.fit_idx_min:self.fit_idx_max + 1]
        fity = np.abs(self.contents[self.fit_idx_min:self.fit_idx_max + 1])
        # abs_contents = np.abs(self.contents)
        unc = self.yerr[self.fit_idx_min:self.fit_idx_max + 1]

        def quad_log(x, x0, y0, a):
            return a * (x - x0) * (x - x0) + y0
        
        # print("fitx", fitx)
        # print("fity", fity)
        [self.transition, self.peak, self.a], cov = opt.curve_fit(quad_log, np.log10(fitx), fity, [np.log10(fitx[fity.argmax()]), fity.max(), -10], unc, absolute_sigma = True)
        factor = 1
        if self.contents[np.abs(self.contents).argmax()] < 0:
            factor = -1
        self.a, self.peak = factor * self.a, factor * self.peak
        self.transition_err = np.sqrt(np.diag(cov))[0]
        self.transition = np.power(10, self.transition)
        # print(f"Transition found for H1D {self.title} at Q={self.transition:.2f} GeV.")
        # print('')
    
    def _find_range(self, pT_scope, shift):
        abs_contents = np.abs(self.contents)
        idx = abs_contents[shift:].argmax() + shift
        
        if isinstance(pT_scope, int) and pT_scope >= 1:
            fit_idx_min = idx - pT_scope
            fit_idx_max = idx + pT_scope
        elif isinstance(pT_scope, float) and pT_scope < 1:
            max_val = abs_contents[idx]
            cutoff = max_val * (1 - pT_scope)

            def check(curr_idx, incr, cutoff):
                # print(f'cutoff is {cutoff}, idx to check is {curr_idx} and current value {abs_contents[curr_idx]}')
                if abs_contents[curr_idx] > cutoff:
                    # print('moving on')
                    return check(curr_idx + incr, incr, cutoff)
                else:
                    # print('stopped, returning idx', curr_idx - incr, "with value", abs_contents[curr_idx - incr])
                    # return curr_idx - incr
                    return curr_idx
            
            fit_idx_min = check(idx - 1, -1, cutoff)
            fit_idx_max = check(idx + 1, 1, cutoff)
            # print(fit_idx_min, fit_idx_max)
            # print('')
        else:
            raise ValueError("pT scope for fitting invalid.")
        return fit_idx_min, fit_idx_max

    def show_fit_curve(self, ax, color):
        xs = np.linspace(self.centers[self.fit_idx_min], self.centers[self.fit_idx_max])
        ys = self.a * np.square(np.log10(xs / self.transition)) + self.peak
        ax.plot(xs, ys, color = color)

    @classmethod
    def copy(cls, orig):
        return cls("", orig.contents.copy(), orig.centers.copy(), orig.edges.copy(), [orig.xerrlow.copy(), orig.xerrup.copy()], orig.yerr.copy(), orig.binning, orig.label)

    def __len__(self): return self.nbins
    def __getitem__(self, idx): return self.contents[idx]
    def __setitem__(self, idx, val): self.contents[idx] = val
    def __str__(self): return f"{self.label};{self.title};{self.xlabel};{self.ylabel}"
    def __add__(self, hist):
        if not np.allclose(self.edges, hist.edges):
            raise ValueError(f"Histograms '{self.title}' and '{hist.title}' edges do not match - cannot add.")
        cls = self.__class__
        copy = cls.copy(self)
        copy.yerr = np.sqrt(np.square(self.yerr) + np.square(hist.yerr))
        copy.contents = self.contents + hist.contents
        copy.label = f"${gutils.clean(copy.label)} + {gutils.clean(hist.label)}$"
        return copy
    def __iadd__(self, hist):
        if not np.allclose(self.edges, hist.edges):
            raise ValueError(f"Histograms '{self.title}' and '{hist.title}' edges do not match - cannot add.")
        self.yerr = np.sqrt(np.square(self.yerr) + np.square(hist.yerr))
        self.contents += hist.contents
        self.label = f"${gutils.clean(self.label)} + {gutils.clean(hist.label)}$"
        return self
    def __sub__(self, hist):
        if not np.allclose(self.edges, hist.edges):
            raise ValueError(f"Histograms '{self.title}' and '{hist.title}' edges do not match - cannot add.")
        cls = self.__class__
        copy = cls.copy(self)
        copy.yerr = np.sqrt(np.square(self.yerr) + np.square(hist.yerr))
        copy.contents = self.contents - hist.contents
        copy.label = f"${gutils.clean(copy.label)} - {gutils.clean(hist.label)}$"
        return copy
    def __isub__(self, hist):
        if not np.allclose(self.edges, hist.edges):
            raise ValueError(f"Histograms '{self.title}' and '{hist.title}' edges do not match - cannot add.")
        self.yerr = np.sqrt(np.square(self.yerr) + np.square(hist.yerr))
        self.contents -= hist.contents
        self.label = f"${gutils.clean(self.label)} - {gutils.clean(hist.label)}$"
        return self
    def __mul__(self, f):
        cls = self.__class__
        copy = cls.copy(self)
        if isinstance(f, (float, int)):
            copy.yerr *= abs(float(f))
            copy.contents *= float(f)
        elif isinstance(f, Histogram1D):
            # if not(np.all(self.contents)):
            #     print(f"{self.label} contains zeroes")
            # if not(np.all(f.contents)):
            #     print(f"{f.label} contains zeroes")
            # copy.yerr = np.abs(self.contents * f.contents) * np.sqrt(np.square(self.yerr / self.contents) + np.square(f.yerr / f.contents))
            copy.yerr = np.sqrt(np.square(self.yerr * f.contents) + np.square(f.yerr * self.contents))
            copy.contents = self.contents * f.contents
            copy.label = f"${gutils.clean(copy.label)} \\times {gutils.clean(f.label)}$"
        else:
            raise TypeError(f"Histogram1D cannot be multiplied by type {type(f)}.")
        return copy
    def __imul__(self, f):
        if isinstance(f, (float, int)):
            self.yerr *= abs(float(f))
            self.contents *= float(f)
        elif isinstance(f, Histogram1D):
            self.yerr = self.contents * f.contents * np.sqrt(np.square(self.yerr / self.contents) + np.square(f.yerr / f.contents))
            self.contents = self.contents * f.contents
            self.label = f"${gutils.clean(self.label)} \\times {gutils.clean(f.label)}$"

        else:
            raise TypeError(f"Histogram1D cannot be multiplied by type {type(f)}.")
        return self
    def __truediv__(self, f):
        if isinstance(f, (float, int)):
            return self.__mul__(1.0 / f)
        elif isinstance(f, Histogram1D):
            cls = self.__class__
            copy = cls.copy(self)
            # print('self cont', self.contents)
            # print('f cont', f.contents)
            copy.yerr = np.abs(np.divide(self.contents, f.contents, out = np.zeros_like(self.contents), where = f.contents != 0)) * np.sqrt(np.square(np.divide(self.yerr, self.contents, out = np.zeros_like(self.yerr), where = self.contents != 0)) + np.square(np.divide(f.yerr, f.contents, out = np.zeros_like(f.yerr), where = f.contents != 0)))
            copy.contents = np.divide(self.contents, f.contents, out = np.zeros_like(self.contents), where = f.contents != 0)
            copy.label = f"$\\frac{{ {gutils.clean(copy.label)} }}{{ {gutils.clean(f.label)} }}$"

            return copy
        else:
            raise TypeError(f"Histogram1D cannot be divided by type f{type(f)}.")
    def __itruediv__(self, f):
        if isinstance(f, (float, int)):
            return self.__imul__(1.0 / f)
        elif isinstance(f, Histogram1D):
            self.yerr = np.divide(self.contents, f.contents, out = np.zeros_like(self.contents), where = f.contents != 0) * np.sqrt(np.square(np.divide(self.yerr, self.contents, out = np.zeros_like(self.yerr), where = self.contents != 0)) + np.square(np.divide(f.yerr, f.contents, out = np.zeros_like(f.yerr), where = f.contents != 0)))
            self.contents = np.divide(self.contents, f.contents, out = np.zeros_like(self.contents), where = f.contents != 0)
            self.label = f"$\\frac{{ {gutils.clean(self.label)} }}{{ {gutils.clean(f.label)} }}$"
            return self
        else:
            raise TypeError(f"Histogram1D cannot be divided by type f{type(f)}.")
    def width_norm(self):
        self.contents /= self.widths
        self.yerr /= self.widths
    def scalex(self, f: float):
        self.edges *= f
        self.centers *= f
        self.widths *= f
        self.xerrlow *= f
        self.xerrup *= f
        self.contents /= f
        self.yerr /= abs(f)
        return self
    # def 

class HistogramENC(Histogram1D):
    def __init__(self, title, contents, centers, edges, xerr, yerr, binning, custom_label, order = 2, xquant = None, yquant = None, ) -> None:
        super(Histogram1D, self).__init__(title, contents, centers, edges, xerr, yerr, binning, custom_label)
        self.order = order
        self.xquant, self.yquant = toQuant(xquant), toQuant(yquant)
    
    def to_virtuality(self, pT: int):
        self.edges *= pT
        self.centers *= pT
        self.widths *= pT
        self.xerrlow *= pT
        self.xerrup *= pT
        self /= pT