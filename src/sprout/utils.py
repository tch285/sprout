import math
import os
import re
from array import array
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import ROOT
from matplotlib.patches import Rectangle as rect

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
markers = ['s', 'o', 'D', 'v']
cEEC_types = ["T", "Q", "P", "M"]
cEEC_ratio_types = ["PM/TR", "P+M/TR", "Q/TR"]

def rmedian(n, a):
    return ROOT.TMath.Median(n, array('d', a))

def nested_dict():
    return defaultdict(nested_dict)

def dict_to_ndict(d):
    ndict = nested_dict()
    for k, v in d.items():
        if isinstance(v, dict):
            ndict[k] = dict_to_ndict(v)
        else:
            ndict[k] = v
    return ndict

def ndict_to_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ndict_to_dict(v)
    return dict(d)

def linbins(xmin, xmax, nbins):
    arr = np.linspace(xmin, xmax, nbins+1)
    centers = (arr[:-1] + arr[1:]) / 2
    return arr, centers

def logbins(xmin, xmax, nbins):
    arr = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
    centers = np.sqrt(arr[:-1] * arr[1:])
    lowerr = centers - arr[:-1]
    uperr = arr[1:] - centers
    bin_widths = arr[1:] - arr[:-1]
    return arr, centers, lowerr, uperr, bin_widths

def to_extrema(edges):
    edges_arr = np.array(edges)
    assert len(edges_arr) >= 2
    assert np.all((edges_arr[1:] - edges_arr[:-1]) > 0)
    return edges_arr[:-1], edges_arr[1:], (edges_arr[:-1] + edges_arr[1:]) / 2

def get_pT_bins(pTmin, pTmax, bin_lo = 0, bin_hi = 200, nbins = 200):
    width = (bin_hi - bin_lo) / nbins
    return round((pTmin - bin_lo) / width) + 1, round((pTmax - bin_lo) / width)

def get_kT_bins(kTmin, kTmax, bin_lo = 0, bin_hi = 10, nbins = 100):
    # edges = np.linspace(bin_lo, bin_hi, nbins+1)
    width = (bin_hi - bin_lo) / nbins
    # print(width)
    return round((kTmin - bin_lo) / width) + 1, round((kTmax - bin_lo) / width)

def find_latest_job(dir):
    return max([int(f.name) for f in os.scandir(dir) if f.is_dir()])

def clean(istr):
    return istr.replace("$", "")

def convert_from_name(code: str):
    match = re.fullmatch(r"^h1_proj_ENC([0-9]*)_([TQPM]+)_([0-9]*)_([0-9]*)$", code)
    if match:
        ipoint = int(match.group(1))
        cEEC_type = match.group(2)
        # pTmin = int(match.group(3))
        # pTmax = int(match.group(4))
        # print(ipoint)
        # print(cEEC_type)
        if len(cEEC_type) == ipoint:
            return "$\\langle " + " ".join([convert_to_op(letter) for letter in cEEC_type]) + " \\rangle$"
        elif len(cEEC_type) == 1:
            return "$\\langle " + (convert_to_op(cEEC_type) * ipoint) + " \\rangle$"
        else:
            raise ValueError(f"cEEC type '{cEEC_type}' with point {ipoint} couldn't be parsed")
    else:
        return code
        # raise ValueError(f"Histogram name {code} couldn't be matched")
    
def convert_to_op(letter):
    if letter == "P":
        op = "{+}"
    elif letter == "M":
        op = "{-}"
    elif letter == 'T':
        op = "\\mathregular{tr}"
    elif letter == "Q":
        op = "\\mathcal{Q}"
    else:
        return letter
    return f"\\mathcal{{E}}_{op}"
    # return f"$\\langle \mathcal{{E}}_{op1} \mathcal{{E}}_{op2} \\rangle$"

def convert_from_info(ipoint: int, cEEC_type: str):
    if cEEC_type == "PM" and ipoint == 2:
        return r"$\langle \\mathcal{E}_{+} \\mathcal{E}_{-} \rangle$"
    elif len(cEEC_type) == 1:
        operator = convert_to_op(cEEC_type)
        return "$\\langle " + (operator * ipoint) + " \\rangle$"
    else:
        raise ValueError(f"Given cEEC type '{cEEC_type}' not recognized.")

def convert_from_info_sigma(ipoint: int, cEEC_type: str):
    if cEEC_type == "PM" and ipoint == 2:
        return r"$\Sigma^\text{+" + u"\u2212" + "}_\\mathregular{EEC}$"
    elif cEEC_type == "P":
        operator = "+"
        return r"$\Sigma^\text{" + (operator * ipoint) + "}_\\mathregular{EEC}$"
    elif cEEC_type == "M":
        operator = u"\u2212"
        return r"$\Sigma^\text{" + (operator * ipoint) + "}_\\mathregular{EEC}$"
    elif cEEC_type == "Q" and ipoint == 2:
        return "$\\Sigma^\\mathcal{Q}_\\mathregular{EEC}$"
    elif cEEC_type == "T":
        return "$\\Sigma_\\mathregular{EEC}$"
    elif cEEC_type == "Q/T":
        return "$\\Sigma^\\mathcal{Q}_\\mathregular{EEC} / \\Sigma_\\mathregular{EEC}$"
    elif cEEC_type == "L/T":
        return r"$\\Sigma^{\\pm\\pm}_\\mathregular{EEC} / \\Sigma_\\mathregular{EEC}$"
    elif cEEC_type == "UL/T":
        return r"$\\Sigma^\text{+" + u"\u2212" + "}_\\mathregular{EEC} / \\Sigma_\\mathregular{EEC}$"
    else:
        raise ValueError(f"Given cEEC type '{cEEC_type}' not recognized.")

def center(ax, val = 0):
    yabs_max = abs(max(np.array(ax.get_ylim()) - val, key=abs))
    ax.set_ylim(bottom = val -yabs_max, top = val + yabs_max)
    if val - yabs_max > val + yabs_max:
        print(val - yabs_max)
        print(val + yabs_max)
        
def round_down(num: float, to: float) -> float:
    if num < 0:
        return -round_up(-num, to)
    mod = math.fmod(num, to)
    return num if math.isclose(mod, to) else num - mod

def round_up(num: float, to: float) -> float:
    if num < 0:
        return -round_down(-num, to)
    down = round_down(num, to)
    return num if num == down else down + to

def get_parton_name(id: int):
    if id == 0:
        return "All"
    elif id == 1:
        return "Down-type"
    elif id == 2:
        return "Up-type"
    elif id == 3:
        return "Strange-type"
    elif id == 4:
        return "Charm-type"
    elif id == 5:
        return "Bottom-type"
    elif id == 6:
        return "Top-type"
    elif id == 21:
        return "Gluon-type"
    else:
        raise ValueError(f"PDG ID '{id}' not recognized.")
    
def check_projection(filename: str, pTmin: int, pTmax: int):
    with ROOT.TFile(filename, "READ") as infile:
        titer = ROOT.TIter(infile.GetListOfKeys())
        proj_hists = [key.GetName() for key in titer if "ENC" in key.GetName()]
    proj_hists_match = [hist_name for hist_name in proj_hists if hist_name.endswith(f"{pTmin}_{pTmax}")]
    if len(proj_hists_match) > 0:
        return True
    else:
        return False
    
def get_aspect(ax):
    # NOTE: doesn't work with semi log scales lmfao
    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    data_ratio = (ax.get_ylim()[0] - ax.get_ylim()[1]) / (ax.get_xlim()[0] - ax.get_xlim()[1])

    return disp_ratio / data_ratio

def draw_syst(ax, hist, syst, color, alpha, flat_syst = False, **kwargs):
    edges = hist.edges
    for bin_idx in range(len(edges) - 1):
        # bin_idx = 20
        xlo = edges[bin_idx]
        xhi = edges[bin_idx + 1]
        systematic = syst[bin_idx]
        height = hist.contents[bin_idx]
        err = systematic if flat_syst else height * systematic / 100
        fill = ax.fill_between([xlo, xhi], height - err, height + err, color = color, alpha = alpha, ec = 'none', **kwargs)
    return fill

def align_to_rlcorner(text, ax = None, buffer = 0.04):
    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()
    bb = text.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
    x0, y0 = text.get_position()
    return x0 + bb.width + buffer, y0

def adjust_syst_patch(leg, hl = 2):
    for legobj in leg.legend_handles:
        if isinstance(legobj, rect):
            oldx = legobj.get_x()
            oldwidth = legobj.get_width()
            legobj.set(x = oldx + oldwidth * 0.5 * (1 - 1 / hl), width = oldwidth / hl)

def enumzip(*args):
    return enumerate(zip(*args))

def get_syst(h, do_barlow = False, do_smooth = False, n = 1, bin_min = 1, bin_max = None, nsig = 1.5):
    # assumes h is already a ratio
    # calculate relevant range
    # returns systematic as a percentage (not fraction)
    if bin_max is None:
        bin_max = h.nbins
    firstbin = bin_min - 1
    lastbin = bin_max - 1
    nbins = lastbin - firstbin + 1
    # extract relevant slice
    full_syst = np.abs(h.contents - 1)
    xx = full_syst[firstbin:lastbin + 1].copy()

    if do_barlow:
        yerr = h.yerr[firstbin:lastbin + 1]
        xx = apply_barlow(xx, yerr, 0, nsig)
    if do_smooth:
        if nbins < 3:
            print(f"Smooth only works for histograms with 3 or more bins (nbins = {nbins})")
            return
        xx = apply_smoothing(nbins, xx, n)

    full_syst[firstbin:lastbin + 1] = xx[:] * 100
    return full_syst

def apply_barlow(contents, err, val = 1, nsig = 2):
    for i, (content, yerr) in enumerate(zip(contents, err)):
            if yerr == 0 or abs(content - val) / yerr < nsig:
                contents[i] = val
    return contents

def get_syst_wavg(h, bin_min = 1, bin_max = None):
    if bin_max is None:
        bin_max = h.nbins
    firstbin = bin_min - 1
    lastbin = bin_max - 1
    # assume h is already a ratio
    dev = np.abs(h.contents - 1)[firstbin:lastbin + 1]
    w = 1 / (h.yerr[firstbin:lastbin + 1]**2)
    return np.average(dev, weights = w) * 100

def get_syst_smoothed(contents, n = 1, bin_min = 1, bin_max = None):
    # assumes h is already a ratio
    # calculate relevant range
    if bin_max is None:
        bin_max = len(contents)
    firstbin = bin_min - 1
    lastbin = bin_max - 1
    nbins = lastbin - firstbin + 1
    # extract relevant slice
    xx = contents[firstbin:lastbin + 1].copy()

    if nbins < 3:
        print(f"Smooth only works for histograms with 3 or more bins (nbins = {nbins})")
        return
    xx = apply_smoothing(nbins, xx, n)

    contents[firstbin:lastbin + 1] = xx[:]
    return contents

def apply_smoothing(nn, xx, ntimes):
    if nn < 3:
        print(f"Smoothing need at least 3 points for smoothing: n = {nn}")
        return
    hh = np.zeros(3)
    
    yy = np.zeros(nn)
    zz = np.zeros(nn)
    rr = np.zeros(nn)
    
    for iteration in range(ntimes):
        zz = xx.copy()
    
        for noent in range(2):
            # do 353 i.e. running median 3, 5, and 3 in a single loop
            for kk in range(3):
                yy = zz.copy()
                medianType = 3 if kk != 1 else 5
                ifirst = 1 if kk != 1 else 2
                ilast = nn - 1 if kk != 1 else nn - 2
                # nn2 = nn - ik - 1;
                # do all elements beside the first and last point for median 3
                #  and first two and last 2 for median 5
                for ii in range(ifirst, ilast):
                    zz[ii] = rmedian(medianType, yy[ii - ifirst:])

                if kk == 0: # first median 3
                    # first point
                    hh[0] = zz[1]
                    hh[1] = zz[0]
                    hh[2] = 3*zz[1] - 2*zz[2]
                    zz[0] = rmedian(3, hh[:])
                    # last point
                    hh[0] = zz[nn - 2]
                    hh[1] = zz[nn - 1]
                    hh[2] = 3*zz[nn - 2] - 2*zz[nn - 3]
                    zz[nn - 1] = rmedian(3, hh[:])

                if kk == 1: # median 5
                    # second point with window length 3
                    zz[1] = rmedian(3, yy)
                    # second-to-last point with window length 3
                    zz[nn - 2] = rmedian(3, yy[nn - 3:])

                # In the third iteration (kk == 2), the first and last point stay
                # the same (see paper linked in the documentation).
    
            yy = zz.copy()

            # quadratic interpolation for flat segments
            for ii in range(2, nn - 2):
                if zz[ii - 1] != zz[ii]:
                    continue
                if zz[ii] != zz[ii + 1]:
                    continue
                tmp0 = zz[ii - 2] - zz[ii]
                tmp1 = zz[ii + 2] - zz[ii]
                if tmp0 * tmp1 <= 0:
                    continue
                jk = 1
                if np.abs(tmp1) > np.abs(tmp0):
                    jk = -1
                yy[ii] = -0.5*zz[ii - 2*jk] + zz[ii]/0.75 + zz[ii + 2*jk] /6.0
                yy[ii + jk] = 0.5*(zz[ii + 2*jk] - zz[ii - 2*jk]) + zz[ii]
    
            # running means
            # std::copy(zz.begin(), zz.end(), yy.begin());
            for ii in range(1, nn - 1):
                zz[ii] = 0.25*yy[ii - 1] + 0.5*yy[ii] + 0.25*yy[ii + 1]
            zz[0] = yy[0]
            zz[nn - 1] = yy[nn - 1]
    
            if noent == 0:
                # save computed values
                rr = zz.copy()

                # COMPUTE residuals
                for ii in range(nn):
                    zz[ii] = xx[ii] - zz[ii]
        # end loop on noent
    
    
        # xmin = ROOT.TMath.MinElement(nn, array('d', xx))
        xmin = np.min(xx[:nn])
        for ii in range(nn):
            if xmin < 0:
                xx[ii] = rr[ii] + zz[ii]
            # make smoothing defined positive - not better using 0 ?
            else:
                xx[ii] = max((rr[ii] + zz[ii]),0.0 )
    # TODO: necessary to return?
    return xx