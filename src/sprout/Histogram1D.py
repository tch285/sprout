# import ROOT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import scipy.optimize as opt
from scipy.stats import gmean
import json
# from functools import singledispatchmethod

import cEEC_utils.gen_utils as gutils
import cEEC_utils.fit as ff
import uproot

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
    def fromHepdata(cls, path, custom_label = None, return_syst = True):
        with open(path) as file:
            all_lines = file.read().splitlines()
        name = all_lines[2]
        val_lines = all_lines[12:-1]
        values = np.asarray([line.split('\t') for line in val_lines], dtype = float).T
        binL, binR, heights, yerr, syst = values
        label = name if custom_label is None else custom_label
        assert len(binL) == len(binR)
        title = name
        edges = np.append(binL, binR[-1])
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
        if return_syst:
            return cls(title, heights, centers, edges, [xerrlow, xerrup], yerr, binning, label), syst
        else:
            return cls(title, heights, centers, edges, [xerrlow, xerrup], yerr, binning, label)

    @classmethod
    def fromJson(cls, path, custom_label = None, return_syst = True):
        with open(path) as file:
            info = json.load(file)
        binL = info['bin_L']
        binR = info['bin_R']
        heights = info['val']
        yerr = info['stat']
        syst = info['syst']
        label = info['name'] if custom_label is None else custom_label
        assert len(binL) == len(binR)
        title = info['name']
        edges = np.append(binL, binR[-1])
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
        if return_syst:
            return cls(title, heights, centers, edges, [xerrlow, xerrup], yerr, binning, label), syst
        else:
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
    def rebin(self, n = 2, width_normed = True):
        if self.nbins % n != 0:
            raise ValueError(f"Rebin number {n} is not a divisor of nbins {self.nbins}")
        old_widths = self.widths
        
        self.edges = self.edges[::n]
        self.widths = self.edges[1:] - self.edges[:-1]
        ratios = self.edges[1:] / self.edges[:-1]

        if np.allclose(self.widths, self.widths[0]):
            self.binning = "lin"
            self.centers = (self.edges[:-1] + self.edges[1:]) / 2
        elif np.allclose(ratios, ratios[0]):
            self.binning = "log"
            self.centers = np.sqrt(self.edges[:-1] * self.edges[1:])
        else:
            print("Binning style couldn't be verified, centers defaulting to log.")
            self.binning = "log"
            self.centers = np.sqrt(self.edges[:-1] * self.edges[1:])
        
        self.xerrlow = self.centers - self.edges[:-1]
        self.xerrup = self.edges[1:] - self.centers

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
            return ax.errorbar(self.centers, self.contents, self.yerr, [self.xerrlow, self.xerrup], **draw_args)

    def plot(self, filename: str, show: bool = False, title = None):
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
        # return cls("", orig.contents.copy(), orig.centers.copy(), orig.edges.copy(), [orig.xerrlow.copy(), orig.xerrup.copy()], orig.yerr.copy(), orig.binning, orig.label)
    @classmethod
    def load(cls, fname):
        pass
        # return cls("", orig.contents.copy(), orig.centers.copy(), orig.edges.copy(), [orig.xerrlow.copy(), orig.xerrup.copy()], orig.yerr.copy(), orig.binning, orig.label)
    @classmethod
    def copy(cls, orig):
        return cls("", orig.contents.copy(), orig.centers.copy(), orig.edges.copy(), [orig.xerrlow.copy(), orig.xerrup.copy()], orig.yerr.copy(), orig.binning, orig.label)
    def selfcopy(self):
        return Histogram1D("", self.contents.copy(), self.centers.copy(), self.edges.copy(), [self.xerrlow.copy(), self.xerrup.copy()], self.yerr.copy(), self.binning, self.label)

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
    def __rmul__(self, f):
        return self.__mul__(f)
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
            raise TypeError(f"Histogram1D cannot be divided by type {type(f)}.")
    def __itruediv__(self, f):
        if isinstance(f, (float, int)):
            return self.__imul__(1.0 / f)
        elif isinstance(f, Histogram1D):
            self.yerr = np.divide(self.contents, f.contents, out = np.zeros_like(self.contents), where = f.contents != 0) * np.sqrt(np.square(np.divide(self.yerr, self.contents, out = np.zeros_like(self.yerr), where = self.contents != 0)) + np.square(np.divide(f.yerr, f.contents, out = np.zeros_like(f.yerr), where = f.contents != 0)))
            self.contents = np.divide(self.contents, f.contents, out = np.zeros_like(self.contents), where = f.contents != 0)
            self.label = f"$\\frac{{ {gutils.clean(self.label)} }}{{ {gutils.clean(f.label)} }}$"
            return self
        else:
            raise TypeError(f"Histogram1D cannot be divided by type {type(f)}.")
    __radd__ = __add__
    def __abs__(self):
        copy = self.selfcopy()
        copy.contents = np.abs(copy.contents)
        return copy
    # __rsub__ = __sub__
    # __rmul__ = __mul__
    def make_abs(self):
        self.contents = np.abs(self.contents)

    def combine(self, hist):
        res = self.selfcopy()
        res.label = ""
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
        self.xerrlow *= f
        self.xerrup *= f
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
    def smooth(self, n, bin_min = 1, bin_max = None):
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