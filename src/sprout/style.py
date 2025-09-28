import matplotlib.pyplot as plt
import json
from pathlib import Path

def use(name = "style.mplstyle"):
    if not name.endswith(".mpstyle"):
        name += ".mplstyle"
    path = Path(__file__).parent / "style" / name
    if path.exists():
        plt.style.use(path)
    else:
        print(f"Unable to find style file {path}!")

def colord(tag):
    if tag in [6, 8, 10, '6', '8', '10']:
        name = f"petroff{tag}.json"
    elif tag in ['p6', 'p8', 'p10']:
        name = f"petroff{tag[1:]}.json"
    elif tag in ['oi', 'okabe', 'okabe_ito']:
        name = "okabe_ito.json"
    else:
        if tag.endswith(".json"):
            name = tag
        else:
            name = f"{tag}.json"
    path = Path(__file__).parent / "style" / name
    if path.exists():
        with open(path) as f:
            colord = json.load(f)
        return colord
    else:
        print(f"Unable to find color file {path}!")
