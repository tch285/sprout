import numpy as np
import ROOT
import os
import re
import math
from collections import defaultdict

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
markers = ['s', 'o', 'D', 'v']
cEEC_types = ["T", "Q", "P", "M"]
cEEC_ratio_types = ["PM/TR", "P+M/TR", "Q/TR"]

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
        op = "\mathregular{tr}"
    elif letter == "Q":
        op = "\mathcal{Q}"
    else:
        return letter
    return f"\mathcal{{E}}_{op}"
    # return f"$\\langle \mathcal{{E}}_{op1} \mathcal{{E}}_{op2} \\rangle$"

def convert_from_info(ipoint: int, cEEC_type: str):
    if cEEC_type == "PM" and ipoint == 2:
        return r"$\langle \mathcal{E}_{+} \mathcal{E}_{-} \rangle$"
    elif len(cEEC_type) == 1:
        operator = convert_to_op(cEEC_type)
        return "$\\langle " + (operator * ipoint) + " \\rangle$"
    else:
        raise ValueError(f"Given cEEC type '{cEEC_type}' not recognized.")

def convert_from_info_sigma(ipoint: int, cEEC_type: str):
    if cEEC_type == "PM" and ipoint == 2:
        return r"$\Sigma^\text{+" + u"\u2212" + "}_\mathregular{EEC}$"
    elif cEEC_type == "P":
        operator = "+"
        return r"$\Sigma^\text{" + (operator * ipoint) + "}_\mathregular{EEC}$"
    elif cEEC_type == "M":
        operator = u"\u2212"
        return r"$\Sigma^\text{" + (operator * ipoint) + "}_\mathregular{EEC}$"
    elif cEEC_type == "Q" and ipoint == 2:
        return "$\Sigma^\mathcal{Q}_\mathregular{EEC}$"
    elif cEEC_type == "T":
        return "$\Sigma_\mathregular{EEC}$"
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