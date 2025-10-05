import json
import os
from functools import singledispatchmethod
from io import TextIOBase

import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import numpy as np
import ROOT
import scipy.optimize as opt
import uproot as ur
from scipy.stats import gmean

import sprout.fit as ff
import sprout.operations as ops
import sprout.utils as utils


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

    @singledispatchmethod
    @classmethod
    def load(cls, file, *args, **kwargs):
        raise NotImplementedError(f'Cannot extract histogram from type {type(file)}.')

    @load.register
    @classmethod
    def _load(cls, h2: ROOT.TH2, axis, proj_range, name = "", label = ""):
        xmin, xmax = proj_range
        if axis in [0, 'x']:
            # project onto x axis
            nbins = h2.GetNbinsY()
            if h2.GetYaxis().GetYbins().GetSize() != 0: # variable edges
                edges = np.array(h2.GetYaxis().GetYbins())
            else: # fixed edges
                edges = np.linspace(h2.GetYaxis().GetYmin(), h2.GetYaxis().GetYmax(), nbins + 1)
            imin, imax = utils.find_bins(edges, xmin, xmax, True, True, True)
            # ROOT bins are 1-indexed, so add one
            imin += 1
            imax += 1
            h1 = h2.ProjectionX(name, imin, imax, "e")
        elif axis in [1, 'y']:
            # project on y axis
            nbins = h2.GetNbinsX()
            if h2.GetXaxis().GetXbins().GetSize() != 0: # variable edges
                edges = np.array(h2.GetXaxis().GetXbins())
            else: # fixed edges
                edges = np.linspace(h2.GetXaxis().GetXmin(), h2.GetXaxis().GetXmax(), nbins + 1)
            imin, imax = utils.find_bins(edges, xmin, xmax, True, True, True)
            # ROOT bins are 1-indexed, so add one
            imin += 1
            imax += 1
            h1 = h2.ProjectionY(name, imin, imax, "e")
        else:
            raise ValueError(f"Cannot parse axis '{axis}'.")
        return cls.load(h1, name, label)
    @load.register
    @classmethod
    def _load(cls, file: str, name = "", label = ""):
        if file.endswith(".root"):
            with ur.open(file) as f:
                return cls._from_uproot(f, name, label)
        elif file.endswith(".json"):
            with open(file) as f:
                return cls._from_json(f, name, label)
        elif file.endswith(".txt"):
            with open(file) as f:
                return cls._from_hepdata(f, name, label)
        elif file.endswith(".npz"):
            with np.load(file) as f:
                return cls._from_npz(f, name, label)
    @load.register
    @classmethod
    def _load(cls, file: TextIOBase, name = "", label = ""):
        if file.name.endswith(".json"):
            return cls._from_json(file, name, label)
        elif file.name.endswith(".txt"):
            return cls._from_hepdata(file, name, label)
    @load.register
    @classmethod
    def _load(cls, file: np.lib.npyio.NpzFile, name, label):
        """Extract Histogram1D from opened npz file."""
        pass
    @load.register
    @classmethod
    def _load(cls, file: ur.ReadOnlyDirectory, name, label = ""):
        """Extract Histogram1D from opened uproot file."""
        h = file[name]
        return cls._from_arrays(name, h.values(), h.axis().edges(), h.errors(), label)
    @load.register
    @classmethod
    def _load(cls, h: ROOT.TH1, name = "", label = ""):
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
    @load.register
    @classmethod
    def _load(cls, file: ROOT.TFile, name, label = ""):
        """Extract Histogram1D from open TFile object."""
        return cls.load(file.Get(name), name, label)

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

    def _selfcopy(self, imin = None, imax = None):
        if imin is None:
            imin = 0
        if imax is None:
            imax = self.nbins - 1
        return Histogram1D(self.name, self.contents[imin:imax+1].copy(), self.centers[imin:imax+1].copy(), self.edges[imin:imax+2].copy(), [self.xerrlo[imin:imax+1].copy(), self.xerrhi[imin:imax+1].copy()], self.yerr[imin:imax+1].copy(), self.binning, self.label)

    def set_binning(self, binning):
        if binning == 'lin':
            self.centers = (self.edges[:-1] + self.edges[1:]) / 2
        elif binning == "log":
            self.centers = np.sqrt(self.edges[:-1] * self.edges[1:])
        else:
            raise ValueError(f"Binning style {binning} is invalid.")
        self.binning = binning

    def integral(self, xmin, xmax, greedy = True, err = False):
        imin, imax = utils.find_bins(self.edges, xmin, xmax, greedy, greedy, True)
        # ROOT bins are 1-indexed, so add one
        # imin += 1
        # imax += 1
        intg = np.sum(self.contents[imin : imax + 1])
        if not err:
            return intg
        else:
            return intg, np.sqrt(np.sum(self.yerr[imin: imax + 1] ** 2))

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
        self.nbins //= n
        return self

    def add_to(self, ax, color = None, as_step = False, as_marker = False, as_line = False, as_shaded_line = False, adjust = None, **kwargs):
        if adjust is None:
            centers = self.centers
            xerrlo = self.xerrlo
            xerrhi = self.xerrhi
        else:
            if isinstance(adjust, (float, int, np.number)):
                if self.binning == "log":
                    centers = self.centers * adjust
                else:
                    centers = self.centers + adjust
                if as_step or as_marker or as_line or as_shaded_line:
                    print("[w] asked for adjust on non-histogram display, ignoring...")
                xerrlo = centers - self.edges[:-1]
                xerrhi = self.edges[1:] - centers
            else:
                print(f"[w] invalid type '{type(adjust)}' for adjust value: {adjust}, ignoring...")
                centers = self.centers
                xerrlo = self.xerrlo
                xerrhi = self.xerrhi

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
            return ax.errorbar(centers, self.contents, self.yerr, [xerrlo, xerrhi], **draw_args)

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

    def calculate_transition(self, xrange = 0.3, shift = 4, func = 'quad', exp = 1.5, syst = None, return_pts = False):
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
            self.transition_err, self.peak_err, self.a_err = np.sqrt(np.diag(cov))
            if return_pts:
                xs = np.logspace(np.log10(fitx[0]), np.log10(fitx[-1]))
                ys = ff.quad_log(xs, self.transition, self.peak, self.a)
                return xs, ys
        elif func == 'gaus':
            [self.transition, self.peak, self.sg], cov = opt.curve_fit(ff.gaus_log, fitx, fity, [fitx[fity.argmax()], fity.max(), 0.9], unc, absolute_sigma = True)
            self.peak *= factor
            self.transition_err, self.peak_err, self.sg_err = np.sqrt(np.diag(cov))
            if return_pts:
                xs = np.logspace(np.log10(fitx[0]), np.log10(fitx[-1]))
                ys = ff.gaus_log(xs, self.transition, self.peak, self.sg)
                return xs, ys
        elif func == 'gaus_log2':
            [self.mu, self.peak, self.sg], cov = opt.curve_fit(ff.gaus_log2, fitx, fity, [np.log(fitx[fity.argmax()]), fity.max(), 0.9], unc, absolute_sigma = True)
            self.peak *= factor
            self.mu_err, self.peak_err, self.sg_err = np.sqrt(np.diag(cov))
            if return_pts:
                xs = np.logspace(np.log10(fitx[0]), np.log10(fitx[-1]))
                ys = ff.gaus_log2(xs, self.mu, self.peak, self.sg)
                return xs, ys
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
            if return_pts:
                xs = np.logspace(np.log10(fitx[0]), np.log10(fitx[-1]))
                ys = func(xs, T, C)
                return xs, ys
        elif func == 'cust_tp':
            t0 = fitx[fity.argmax()]
            p0 = fity.max()
            def func(x, t, p):
                return ff.cust_tp(x, t, p, exp)
            [self.transition, self.peak], cov = opt.curve_fit(func, fitx, fity, [t0, p0], unc, absolute_sigma = True)
            self.peak *= factor
            self.transition_err, self.peak_err = np.sqrt(np.diag(cov))
            if return_pts:
                xs = np.logspace(np.log10(fitx[0]), np.log10(fitx[-1]))
                ys = func(xs, self.transition, self.peak)
                return xs, ys
        else:
            print("Invalid function name, exiting.")
            return

    def _find_range(self, xrange, shift):
        abs_contents = np.abs(self.contents)
        idx = abs_contents[shift:].argmax() + shift

        if isinstance(xrange, int) and xrange >= 1:
            # set range to be all bins within +- xrange of the extremum
            fit_idx_min = idx - xrange
            fit_idx_max = idx + xrange
        elif isinstance(xrange, float) and xrange < 1:
            # set range to be all bins that has height within a fraction xrange
            # of the max height
            max_val = abs_contents[idx]
            cutoff = max_val * (1 - xrange)

            def check(curr_idx, incr, cutoff):
                if abs_contents[curr_idx] > cutoff:
                    return check(curr_idx + incr, incr, cutoff)
                else:
                    return curr_idx

            fit_idx_min = check(idx - 1, -1, cutoff)
            fit_idx_max = check(idx + 1, 1, cutoff)
        elif isinstance(xrange, (tuple, list, np.ndarray)):
            if len(xrange) == 3:
                # set range of xaxis values. If the edge is within a bin, greedy = True will
                # take the bin, and greedy = False will not
                xmin, xmax, greedy = xrange
                fit_idx_min, fit_idx_max = utils.find_bins(self.edges, xmin, xmax, greedy, greedy, False)
            elif len(xrange) == 2:
                fit_idx_min, fit_idx_max = xrange
        else:
            raise ValueError("X range for fitting invalid.")
        return fit_idx_min, fit_idx_max

    def show_fit_curve(self, ax, color):
        xs = np.linspace(self.centers[self.fit_idx_min], self.centers[self.fit_idx_max])
        ys = self.a * np.square(np.log10(xs / self.transition)) + self.peak
        ax.plot(xs, ys, color = color)

    def save(self, f, name = None, compressed = False):
        nm = name if name is not None else self.name
        if f.endswith(".npz"):
            h = {
                f"{nm}_cts": self.contents,
                f"{nm}_edg": self.edges,
                f"{nm}_err": self.edges,
            }
            if compressed:
                np.savez_compressed(f, **h)
            else:
                np.savez(f, **h)
        else:
            ext = os.path.splitext(f)
            raise RuntimeError(f"Filetype {ext} is not supported yet!")

    @property
    def yerr_rel(self):
        return np.abs(ops.safediv(self.yerr, self.contents))

    def __len__(self): return self.nbins
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            if isinstance(idx.start, int) and isinstance(idx.stop, int): # indices passed
                return self._selfcopy(idx.start, idx.stop)
            else: # xrange passed
                imin, imax = utils.find_bins(self.edges, idx.start, idx.stop, False, False, False)
                return self._selfcopy(imin, imax)
        else:
            return self.contents[idx]
    def __setitem__(self, idx, val): self.contents[idx] = val
    def __str__(self): return f"Histogram1D '{self.name}' ({self.label})"
    def __eq__(self, other):
        if isinstance(other, Histogram1D):
            return self.nbins == other.nbins and np.allclose(self.contents, other.contents) \
                and self % other and np.allclose(self.centers, other.centers) \
                and np.allclose(self.yerr, other.yerr)
        return False
    def __add__(self, other):
        copy = self._selfcopy()
        if isinstance(other, (float, int, np.number)):
            copy.contents = self.contents + other
        elif isinstance(other, Histogram1D):
            self._check_matched_edges(other, "+")
            copy.contents = self.contents + other.contents
            copy.yerr = ops.qadd(self.yerr, other.yerr)
        else:
            return NotImplemented

        return copy
    def __iadd__(self, other):
        return self + other # want H1D to be immutable (older refs point to original H1D)
    def __sub__(self, other):
        copy = self._selfcopy()
        if isinstance(other, (float, int, np.number)):
            copy.contents = self.contents - other
        elif isinstance(other, Histogram1D):
            self._check_matched_edges(other, "-")
            copy.contents = self.contents - other.contents
            copy.yerr = ops.qadd(self.yerr, other.yerr)
        else:
            return NotImplemented

        return copy
    def __isub__(self, other):
        return self - other # want H1D to be immutable (older refs point to original H1D)
    def __mul__(self, other):
        copy = self._selfcopy()
        if isinstance(other, (float, int, np.number)):
            copy.contents *= float(other)
            copy.yerr *= abs(float(other))
        elif isinstance(other, Histogram1D):
            self._check_matched_edges(other, "*")
            copy.contents = self.contents * other.contents
            copy.yerr = np.abs(copy.contents) * ops.qadd(self.yerr_rel, other.yerr_rel)
            # copy.yerr = ops.qadd(self.yerr * other.contents, other.yerr * self.contents)
        else:
            return NotImplemented

        return copy
    def __imul__(self, other):
        return self * other
    def __truediv__(self, other):
        if isinstance(other, (float, int, np.number)):
            return self * (1.0 / other)
        elif isinstance(other, Histogram1D):
            copy = self._selfcopy()
            self._check_matched_edges(other, "/")
            copy.contents = ops.safediv(self.contents, other.contents)
            copy.yerr = np.abs(copy.contents) * ops.qadd(self.yerr_rel, other.yerr_rel)
            # copy.yerr = np.sqrt(ops.safediv(self.yerr**2 , other.contents**2) + ops.safediv(self.contents**2 , other.contents**4) * other.yerr**2)
            # copy.yerr = np.abs(np.divide(self.contents, other.contents, out = np.zeros_like(self.contents), where = other.contents != 0)) * np.sqrt(np.square(np.divide(self.yerr, self.contents, out = np.zeros_like(self.yerr), where = self.contents != 0)) + np.square(np.divide(other.yerr, other.contents, out = np.zeros_like(other.yerr), where = other.contents != 0)))
            return copy
        else:
            return NotImplemented
    def __itruediv__(self, other):
        return self / other
    def __floordiv__(self, other):
        """Return self / other, assuming fully correlated uncertainties."""
        if isinstance(other, (float, int, np.number)):
            return self * (1.0 / other)
        elif isinstance(other, Histogram1D):
            copy = self._selfcopy()
            self._check_matched_edges(other, "/")
            copy.contents = ops.safediv(self.contents, other.contents)
            copy.yerr = np.abs(copy.contents * (self.yerr_rel - other.yerr_rel))
            return copy
        else:
            return NotImplemented
    def __ifloordiv__(self, other):
        return self // other
    __radd__ = __add__
    __rmul__ = __mul__
    def __rsub__(self, other):
        # called with other - self. Don't need to define for Histogram1D since __sub__ would be called first
        if isinstance(other, (float, int, np.number)):
            copy = self._selfcopy()
            copy.contents = other - self.contents
            return copy
        else:
            return NotImplemented
    def __rtruediv__(self, other):
        # called with other / self. Don't need to define for Histogram1D since __truediv__ would be called first
        if isinstance(other, (float, int, np.number)):
            copy = self._selfcopy()
            copy.contents = np.divide(other, self.contents, out = np.zeros(self.nbins), where = self.contents != 0)
            copy.yerr = self.yerr * np.divide(other, self.contents ** 2, out = np.zeros(self.nbins), where = self.contents != 0)
            return copy
        else:
            return NotImplemented
    def __neg__(self):
        copy = self._selfcopy()
        copy.contents = - copy.contents
        return copy
    def __mod__(self, other):
        """Check if two Histograms have compatible bin edges."""
        return np.allclose(self.edges, other.edges)
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
        self.contents = utils.apply_barlow(self.contents, self.yerr, val, nsig)
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
        xx_smoothed = utils.apply_smoothing(nbins, xx, n)
        for i in range(nbins):
            self.contents[i+firstbin] = xx_smoothed[i]

H1D = Histogram1D