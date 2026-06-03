import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "gelu": "#4C78A8",
    "tanh": "#F58518",
    "softsign": "#54A24B",
    "lee": "#B279A2",
    "tanh_sin": "#E45756",
    "tanh_cos": "#72B7B2",
    "tanh_rand": "#9D755D",
    "negative": "#E45756",
    "positive": "#54A24B",
    "neutral": "#8C8C8C",
}


def save_fig(fig, path):
    fig.savefig(path)
    print(f"Saved {path}")
