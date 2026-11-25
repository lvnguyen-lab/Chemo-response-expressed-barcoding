"""House style for the paper's figure panels, shared by the notebooks that draw them.

Rules:
- All text Helvetica, never under 7 pt at the size the panel is placed on the page. Every panel is therefore
  drawn at its placed size and saved without bbox_inches="tight", so the saved file is exactly that size and goes
  into the deck at 100%.
- p-values are shown only as asterisks, and only where p < 0.05: * p < 0.05, ** p < 0.01, *** p < 0.001,
  **** p < 0.0001. Non-significant comparisons are not marked.
- Every significant comparison is drawn the same way: a line between the two comparators, with short end ticks
  down towards them, and the asterisks centred above it (significance_bracket).
- Axis titles in sentence case; "clones" are "cell clones".

Panels are saved as SVG plus a PNG
of the same size into visualizations/paper_panels/.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

FONT = 7                                   
STAR_FONT = 9                             
FONT_FAMILY = "Helvetica"
LINE = 0.6                                
PANEL_DIR = Path(__file__).resolve().parent.parent / "visualizations" / "paper_panels"
STAR_CUTOFFS = [(0.0001, "****"), (0.001, "***"), (0.01, "**"), (0.05, "*")]
SIG_ALPHA = 0.05

HELVETICA_TTC = Path("/System/Library/Fonts/Helvetica.ttc")
_faces_registered = False

def register_helvetica_faces():
    global _faces_registered
    if _faces_registered or not HELVETICA_TTC.exists():
        return
    import logging
    from fontTools.ttLib import TTCollection
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    out_dir = Path(mpl.get_cachedir()) / "helvetica_faces"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, face in enumerate(TTCollection(str(HELVETICA_TTC)).fonts):
        path = out_dir / f"Helvetica_{i}.ttf"
        if not path.exists():
            face.save(str(path))
        mpl.font_manager.fontManager.addfont(str(path))
    _faces_registered = True

def apply_style():
    register_helvetica_faces()
    sizes = ["font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize",
             "legend.fontsize", "legend.title_fontsize", "figure.titlesize", "figure.labelsize"]
    mpl.rcParams.update({key: FONT for key in sizes})
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_FAMILY,
        "mathtext.it": f"{FONT_FAMILY}:italic",
        "mathtext.bf": f"{FONT_FAMILY}:bold",
        "mathtext.sf": FONT_FAMILY,
        "mathtext.default": "regular",
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
        "axes.linewidth": LINE,
        "xtick.major.width": LINE,
        "ytick.major.width": LINE,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "axes.titlepad": 2,
        "axes.labelpad": 2,
        "lines.linewidth": 0.8,
        "patch.linewidth": 0.5,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

def stars(p):
    if p is None or not np.isfinite(p):
        return None
    return next((s for cutoff, s in STAR_CUTOFFS if p < cutoff), None)

def significance_bracket(ax, x_left, x_right, y, p, tick=None, transform=None, color="#333333", clip_on=False):
    label = stars(p)
    if label is None:
        return None
    if tick is None:
        lo, hi = ax.get_ylim()
        tick = 0.03 * (hi - lo)
    kw = dict(color=color, linewidth=LINE, clip_on=clip_on, solid_capstyle="butt")
    if transform is not None:
        kw["transform"] = transform
    ax.plot([x_left, x_left, x_right, x_right], [y - tick, y, y, y - tick], **kw)
    return stars_text(ax, (x_left + x_right) / 2, y, label, color, clip_on, transform)

def stars_text(ax, x, y, label, color="#333333", clip_on=False, transform=None):
    return ax.annotate(label, xy=(x, y), xycoords=transform if transform is not None else "data",
                       xytext=(0, -STAR_FONT * 0.42), textcoords="offset points", ha="center", va="bottom",
                       fontsize=STAR_FONT, color=color, annotation_clip=False, clip_on=clip_on)

def log_significance_bracket(ax, x_left, x_right, y, p, tick_factor=1.35, color="#333333"):
    label = stars(p)
    if label is None:
        return None
    ax.plot([x_left, x_left, x_right, x_right], [y / tick_factor, y, y, y / tick_factor], color=color,
            linewidth=LINE, clip_on=False, solid_capstyle="butt")
    return stars_text(ax, (x_left + x_right) / 2, y, label, color)

def new_panel(width, height):
    apply_style()
    return plt.figure(figsize=(width, height))

def inch_axes(fig, left, top, width, height, **kwargs):
    fig_w, fig_h = fig.get_size_inches()
    return fig.add_axes([left / fig_w, 1 - (top + height) / fig_h, width / fig_w, height / fig_h], **kwargs)

def inch_grid(fig, left, top, nrows, ncols, width, height, wgap, hgap, **kwargs):
    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            axes[r, c] = inch_axes(fig, left + c * (width + wgap), top + r * (height + hgap), width, height, **kwargs)
    return axes

def fig_text(fig, x, y, s, **kwargs):
    fig_w, fig_h = fig.get_size_inches()
    kwargs.setdefault("fontsize", FONT)
    return fig.text(x / fig_w, 1 - y / fig_h, s, **kwargs)

def text_overlaps(fig):
    texts = drawn_texts(fig)
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in texts]
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if boxes[i].overlaps(boxes[j]):
                pairs.append((texts[i].get_text(), texts[j].get_text()))
    return pairs

def drawn_texts(fig):
    fig.canvas.draw()
    all_tick_labels, drawn_tick_labels = set(), set()
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for axis in ax._axis_map.values():
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                all_tick_labels.update({id(tick.label1), id(tick.label2)})
            for tick in axis._update_ticks():
                drawn_tick_labels.update({id(tick.label1), id(tick.label2)})
    texts = []
    for text in fig.findobj(mpl.text.Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        if id(text) in all_tick_labels and id(text) not in drawn_tick_labels:
            continue
        axes = text.axes
        if axes is not None and not axes.get_visible():
            continue
        texts.append(text)
    return texts

def check_fits(fig, name, tolerance=0.01):
    texts = drawn_texts(fig)
    renderer = fig.canvas.get_renderer()
    width, height = fig.get_size_inches() * fig.dpi
    problems = []
    for text in texts:
        if text.get_fontsize() < FONT - 1e-6:
            problems.append(f"{text.get_text()!r} is {text.get_fontsize():.1f} pt")
        family = text.get_fontfamily()
        if family and family[0] not in (FONT_FAMILY,):
            problems.append(f"{text.get_text()!r} is in {family[0]}")
        box = text.get_window_extent(renderer)
        pad = tolerance * fig.dpi
        if box.x0 < -pad or box.y0 < -pad or box.x1 > width + pad or box.y1 > height + pad:
            problems.append(f"{text.get_text()!r} runs off the panel ({box.x0:.0f},{box.y0:.0f})-({box.x1:.0f},"
                            f"{box.y1:.0f}) of {width:.0f}x{height:.0f} px")
    if problems:
        raise ValueError(f"{name}: " + "; ".join(problems))

def save_panel(fig, name, check=True, transparent=False):
    if check:
        check_fits(fig, name)
        overlaps = text_overlaps(fig)
        if overlaps:
            print(f"{name}: overlapping text: {overlaps[:8]}{' ...' if len(overlaps) > 8 else ''}")
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    # dpi matters to the SVG only for rasterized artists (dense heatmaps); the PNG is the deck's fallback image
    for ext, kw in [("svg", {"dpi": 600}), ("png", {"dpi": 300})]:
        fig.savefig(PANEL_DIR / f"{name}.{ext}", transparent=transparent, **kw)
    w, h = fig.get_size_inches()
    print(f"saved {name}: {w:.3f} x {h:.3f} in")

def legend_panel(name, width, height, handles, ncol=1, title=None, loc="upper left", pad_top=0.0, **kwargs):
    fig = new_panel(width, height)
    kw = dict(loc=loc, ncol=ncol, frameon=False, fontsize=FONT, borderpad=0, borderaxespad=0, handlelength=1.0,
              handleheight=0.7, handletextpad=0.4, labelspacing=0.3, columnspacing=0.8)
    kw.update(kwargs)
    anchor = {"upper left": (0, 1), "center left": (0, 0.5), "upper center": (0.5, 1), "center": (0.5, 0.5),
              "lower left": (0, 0)}[loc]
    if loc.startswith("upper"):
        anchor = (anchor[0], 1 - pad_top / height)
    fig.legend(handles=handles, title=title, bbox_to_anchor=anchor, **kw)
    save_panel(fig, name, transparent=True)
    return fig
