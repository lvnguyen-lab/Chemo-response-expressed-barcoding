"""
Expected % emerging clones from detection alone: a Poisson allocation and sampling model  

WHAT IT ASKS
    A clone is "emerging" when a secondary xenograft calls it but the primary it came from did not. A clone can be
    missing from the primary simply because the primary had too few of its cells in the sequenced material to call it.
    This script predicts, for every secondary mouse, the % of its clones that would look emerging for that reason
    alone, if every clone grew at the same rate. It then tests the observed % against it.
    The model is never fitted to the observed emerging clones.

HOW ONE CLONE IS FOLLOWED FROM THE PRIMARY INTO A SECONDARY MOUSE
    A clone with s cells in the primary's sequenced material:
      1. Allocation  The secondary receives Poisson(s x % of primary injected / % of primary sampled) of its cells
      2. Growth      Every founder cell grows by the same factor: cell yield at harvest / cells injected
      3. Sampling    The secondary's sequenced material holds Poisson(founders x growth x % sampled) of its cells
      4. Detection   The clone is called with its sequencing batch's chance for that many cells, measured with the
                     batch's spike-in controls (see section 4)

    The primary's clones are the ones it called, plus the ones it missed: those its batch misses at >= 20 cells, and
    a power law of smaller clones continuing its own size distribution below 20 cells.

    Expected % emerging = expected number of the secondary's called clones that are not on the primary's list
                          / expected number of the secondary's called clones

TESTS
    Per mouse:   observed emerging count vs binomial(clones called, expected %), allowing for uncertainty in the
                 slope of the primary's size distribution
    p < 0.05, no multiple-comparison correction

LAYOUT
    The script sits in Code/ and needs nothing else there; its inputs, the Appendix Figure S3D Source Data (named
    there with a figS3D_ prefix), should sit in data_files/poisson_model/, with data_files/ next to Code/:
        Code/amplicon_emergence_poisson_model.py
        data_files/poisson_model/model_inputs_xenografts.csv ...
    It needs Python 3 with numpy, pandas, scipy, statsmodels, matplotlib and openpyxl (for the results workbook)

INPUTS  (all in data_files/poisson_model/, or the folder given by --inputs-dir)
    model_inputs_xenografts.csv                 each xenograft: model, subtype, cohort,
                                                replicate, sequencing run, cells injected, % GFP, barcoded cells
                                                injected, % of the primary injected, % sampled per sequencing run,
                                                number of sequencing runs, % sequenced in all, cell yield at harvest
    model_inputs_clones.csv                     every clone called in every mouse, with its cells in the sequenced
                                                material (multiple integrations merged, Appendix Table S5)
    spike_in_rates_per_sequencing_run.csv       each sequencing run's spike-in controls, by size: added, with
                                                reads, and called above the run's noise threshold (as Appendix
                                                Figure S10B)
    Everything about the mice, sequencing runs, spike-ins and clones comes from these files. The only thing typed in
    below is the cohort colours

CROSS-CHECK  (optional, --notebook-clone-types)
    amplicon_pct_clones_per_mouse_untreated_vs_treated.xlsx, the analysis notebooks' clone-type table. If it is given,
    the observed emerging counts must match it

OUTPUTS  (in results/, a folder at the same level as Code/, unless given)
    --out-xlsx  results workbook (default ../results/amplicon_emerging_poisson_model.xlsx)
    --out-svg   figure of observed vs expected % emerging per mouse
                (default ../results/amplicon_emerging_poisson_model_observed_vs_expected.svg)
    --panel-dir also save the figure, as SVG and PNG, as the paper's panel in this folder (optional)

USAGE
    python Code/amplicon_emergence_poisson_model.py
    python Code/amplicon_emergence_poisson_model.py --inputs-dir "path/to/folder with the input files"
    python Code/amplicon_emergence_poisson_model.py --out-xlsx my_results.xlsx --out-svg my_figure.svg
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.multitest import multipletests


# =====================================================================================================================
# 1. SETTINGS
# =====================================================================================================================

# Folders: this script is in Code/; its inputs should be in data_files/poisson_model/ and its outputs go to results/,
# with data_files/ and results/ both next to Code/
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
INPUTS_DIR = REPO_DIR / "data_files" / "poisson_model"
XENOGRAFTS_FILE = "model_inputs_xenografts.csv"              
CLONES_FILE = "model_inputs_clones.csv"                       
SPIKE_IN_RATES_FILE = "spike_in_rates_per_sequencing_run.csv" 
DEFAULT_OUT_XLSX = REPO_DIR / "results" / "amplicon_emerging_poisson_model.xlsx"
DEFAULT_OUT_SVG = REPO_DIR / "results" / "amplicon_emerging_poisson_model_observed_vs_expected.svg"

SIG_ALPHA = 0.05                                    

# The project's colours for each cohort in the xenograft table 
COHORT_COLOURS = {"Untreated": "dimgrey", "Paclitaxel": "darkturquoise", "Carboplatin": "darkviolet"}

# Clone sizes are always "cells in the primary's sequenced material". The model is evaluated on this grid of sizes
SIZE_GRID = np.logspace(-4, 5, 721)

# The primary's size distribution is fitted on its clones of at least this many cells (inside the spike-in
# calibration, where most clones are called) and extended below it as a power law
TAIL_MIN_CELLS = 20.0

# Slopes at which the expected % emerging is also reported, to show how much it depends on the slope
SLOPE_SENSITIVITY = [0.5, 1.0, 1.5, 2.0, 3.0]


# =====================================================================================================================
# 2. INPUT COLUMNS
# =====================================================================================================================

# The columns of each input file this script uses, and the shorter names it gives them
XENOGRAFT_COLUMNS = {
    "Model": "Tumor",
    "Subtype": "Subtype",
    "Cohort": "Cohort",
    "Replicate": "Replicate",
    "Mouse ID": "Mouse",
    "Sequencing run": "SLX",
    "Cells injected (n)": "Cells injected",
    "% GFP": "% GFP",
    "Barcoded cells injected (n)": "Barcoded cells injected",
    "% of primary xenograft injected": "% of primary injected",
    "% of xenograft sampled per sequencing run": "% sampled",
    "Sequencing runs (n)": "Sequencing runs",
    "% of xenograft sequenced (all runs)": "% sequenced",
    "Cell yield at harvest (n)": "Cell yield",
}
CLONE_COLUMNS = {
    "Model": "Tumor",
    "Cohort": "Cohort",
    "Mouse ID": "Mouse",
    "Clone": "var_seq",
    "Cells in the sequenced material (n)": "cell",
}
SPIKE_IN_RATE_COLUMNS = ["SLX", "Cells", "Spike-ins", "With reads", "Called above threshold", "Rate"]


# =====================================================================================================================
# 3. LOAD AND CHECK THE DATA
# =====================================================================================================================

def load_spike_in_rates(path):
    rates = pd.read_csv(path)
    assert list(rates.columns) == SPIKE_IN_RATE_COLUMNS, f"{path.name} should have the columns {SPIKE_IN_RATE_COLUMNS}"
    rate = rates["Called above threshold"] / rates["Spike-ins"]
    assert np.allclose(rates["Rate"], rate), f"{path.name}: Rate is not Called above threshold / Spike-ins"
    rates["Rate"] = rate
    return rates

def pooled_spike_in_rates(rates):
    pooled = rates.groupby("Cells")[["Spike-ins", "With reads", "Called above threshold"]].sum()
    pooled["Rate"] = pooled["Called above threshold"] / pooled["Spike-ins"]
    return pooled.reset_index()

def load_clone_table(path):
    raw = pd.read_csv(path)
    missing = set(CLONE_COLUMNS) - set(raw.columns)
    assert not missing, f"{path.name} has no column {missing}"
    return raw[list(CLONE_COLUMNS)].rename(columns=CLONE_COLUMNS)

def build_xenograft_table(path, clone_table):
    raw = pd.read_csv(path)
    missing = set(XENOGRAFT_COLUMNS) - set(raw.columns)
    assert not missing, f"{path.name} has no column {missing}"
    table = raw[list(XENOGRAFT_COLUMNS)].rename(columns=XENOGRAFT_COLUMNS)
    table["SLX"] = table["SLX"].str.replace("SLX-", "").astype(int)
    table["Xenograft"] = np.where(table["Cohort"] == "Primary", "Primary",
                                  table["Cohort"] + " " + table["Replicate"].astype("Int64").astype(str))

    # Sequencing runs. "% sampled" is the % of the xenograft sampled in one run; a mouse sequenced more than once has
    # its runs added together in "% sequenced" (e.g. two runs of 4% = 8%). Each run is called on its own
    for _, mouse in table.loc[table["Sequencing runs"] > 1].iterrows():
        print(f"{mouse['Tumor']} {mouse['Xenograft']} ({mouse['Mouse']}) was sequenced in {mouse['Sequencing runs']} "
              f"runs of {mouse['% sampled']:g}%")

    # Growth from injection to harvest (used by the model). % GFP is on both sides of the ratio, so this is simply
    # cell yield / cells injected
    barcoded_cells_at_harvest = table["Cell yield"] * table["% GFP"] / 100
    table["Growth, fold (cell yield)"] = barcoded_cells_at_harvest / table["Barcoded cells injected"]

    # Growth from the mouse's own sequencing (a comparison only): barcoded cells in its called clones at harvest
    cells_sequenced = clone_table.groupby("Mouse")["cell"].sum()
    table["Barcoded cells sequenced (called clones)"] = table["Mouse"].map(cells_sequenced)
    cells_in_called_clones_at_harvest = cells_sequenced / (table.set_index("Mouse")["% sequenced"] / 100)
    barcoded_cells_in_called_clones = table["Mouse"].map(cells_in_called_clones_at_harvest)
    table["Growth, fold (sequencing)"] = barcoded_cells_in_called_clones / table["Barcoded cells injected"]
    return table

def check_inputs(xenograft_table, clone_table, rates):
    t = xenograft_table
    per_clone = clone_table.join(t.set_index("Mouse")[["Tumor", "Cohort"]], on="Mouse", rsuffix=" of its mouse")
    secondary = t["Cohort"] != "Primary"

    assert t["Mouse"].is_unique, "a mouse is in the xenograft table more than once"
    assert set(t["Mouse"]) == set(clone_table["Mouse"]), "the xenograft and clone tables hold different mice"
    assert not clone_table.duplicated(["Mouse", "var_seq"]).any(), "a clone is listed more than once in a mouse"
    assert (per_clone["Tumor"] == per_clone["Tumor of its mouse"]).all(), "a clone's model differs from its mouse's"
    assert (per_clone["Cohort"] == per_clone["Cohort of its mouse"]).all(), "a clone's cohort differs from its mouse's"
    assert (t.groupby("Tumor")["Subtype"].nunique() == 1).all(), "a model has more than one subtype"
    assert (t.groupby("Tumor")["Cohort"].apply(lambda c: (c == "Primary").sum()) == 1).all(), "a model needs 1 primary"
    assert t.loc[secondary, "Replicate"].notna().all(), "a secondary has no replicate number"
    missing_colours = set(t.loc[secondary, "Cohort"]) - set(COHORT_COLOURS)
    assert not missing_colours, f"no colour in COHORT_COLOURS for {missing_colours}"
    assert (t["Barcoded cells injected"] == t["Cells injected"] * t["% GFP"] / 100).all()
    assert (t["Sequencing runs"] >= 1).all(), "a mouse has no sequencing run"
    total_pct_sampled = t["% sampled"] * t["Sequencing runs"]
    assert ((total_pct_sampled - t["% sequenced"]).abs() <= 0.06).all(), "% sampled differs (beyond rounding)"
    assert t.loc[secondary, ["% of primary injected", "Cell yield"]].notna().all().all(), \
        "a secondary has no % of primary injected or cell yield"
    assert set(t["SLX"]) <= set(rates["SLX"]), "a mouse's sequencing run has no spike-in controls"

def count_observed_emerging(clone_table, xenograft_table, notebook_clone_types_path=None):
    """For each secondary mouse: how many clones it called, and how many of them its primary did not call."""
    replicates = xenograft_table.set_index("Mouse")["Replicate"]
    rows = []
    for model, model_clones in clone_table.groupby("Tumor"):
        primary_barcodes = set(model_clones.loc[model_clones["Cohort"] == "Primary", "var_seq"])
        secondaries = model_clones.loc[model_clones["Cohort"] != "Primary"]
        for (cohort, mouse), mouse_clones in secondaries.groupby(["Cohort", "Mouse"]):
            barcodes = set(mouse_clones["var_seq"])
            rows.append({"Tumor": model, "Cohort": cohort, "Mouse": mouse,
                         "Replicate": f"Replicate {int(replicates[mouse])}",
                         "Clones called": len(barcodes), "Emerging": len(barcodes - primary_barcodes)})
    observed = pd.DataFrame(rows).set_index("Mouse")

    # Cross-check against the notebooks' own clone-type table, if it was given
    if notebook_clone_types_path is not None:
        notebook = pd.read_excel(notebook_clone_types_path, sheet_name="Clone type % per mouse", index_col=[0, 1, 2])
        keys = list(zip(observed["Tumor"], observed["Cohort"], observed["Replicate"]))
        assert len(keys) == len(notebook), "different mice from the notebooks' clone-type table"
        notebook_called = notebook.loc[keys, "Unique clones per replicate"].to_numpy()
        notebook_emerging = notebook.loc[keys, "Emerging"].to_numpy()
        assert (observed["Clones called"].to_numpy() == notebook_called).all(), "clones called differ from notebook"
        assert (observed["Emerging"].to_numpy() == notebook_emerging).all(), "emerging counts differ from notebook"
        print(f"Observed counts match {notebook_clone_types_path.name}")
    return observed


# =====================================================================================================================
# 4. DETECTION: THE CHANCE A CLONE IS CALLED, PER SEQUENCING BATCH
# =====================================================================================================================
#
# Each batch's chance of calling a clone with c true cells in its sequenced material is measured with its own spike-ins:
# at each size the batch ran, the share of its spike-ins of that size called above its red line. In between sizes,
# straight-line interpolation on log10(cells); above the largest size, 100%.
# Below the smallest size a batch ran, its spike-ins say nothing. Every batch is taken to call nothing at or below the
# largest spike-in size that no batch ever called (10 cells: SLX-20723's 30 ten-cell spike-ins), and the curve rises
# from there to the batch's own smallest size.

class BatchDetection:

    def __init__(self, rates):
        self.rates = rates                                               # one row per batch and spike-in size
        called_per_size = rates.groupby("Cells")["Called above threshold"].sum()
        never_called = called_per_size[called_per_size == 0]
        assert len(never_called), "every spike-in size was called at least once: no floor for detection"
        self.never_called_cells = never_called.index.max()

        table = pd.DataFrame(index=pd.Index(list(dict.fromkeys(rates["SLX"])), name="SLX"))
        table["Spike-in sizes (cells)"] = [", ".join(str(c) for c in rates.loc[rates["SLX"] == batch, "Cells"])
                                           for batch in table.index]
        table["Called at 20 cells"] = [float(self.chance_called_with(20, batch)) for batch in table.index]
        table["LOD50 (cells)"] = [self._cells_called_half_the_time(batch) for batch in table.index]
        self.table = table

        # Look-up table per batch: chance of being called when the clone's cells in the sequenced material are
        # Poisson-distributed with a given mean
        self.mean_cells_grid = np.logspace(-3, np.log10(800), 1500)
        cell_counts = np.arange(0, 1200)
        poisson = scipy.stats.poisson.pmf(cell_counts[None, :], self.mean_cells_grid[:, None])
        self.lookup = {batch: (poisson * self.chance_called_with(cell_counts, batch)[None, :]).sum(axis=1)
                       for batch in table.index}

    def _rate_points(self, batch):
        """(cells, chance called) points the detection curve passes through, for one batch."""
        own = self.rates.loc[(self.rates["SLX"] == batch) & (self.rates["Cells"] > self.never_called_cells),
                             ["Cells", "Rate"]]
        floor = pd.DataFrame({"Cells": [self.never_called_cells], "Rate": [0.0]})
        return pd.concat([floor, own]).sort_values("Cells")

    def _cells_called_half_the_time(self, batch):
        points = self._rate_points(batch)
        log_cells, rate = np.log10(points["Cells"].to_numpy(dtype=float)), points["Rate"].to_numpy()
        first_half = np.argmax(rate >= 0.5)                     # the first size called at least half the time
        segment = slice(first_half - 1, first_half + 1)
        return 10 ** np.interp(0.5, rate[segment], log_cells[segment])

    def chance_called_with(self, cells, batch):
        """Chance a clone with exactly this many cells in the sequenced material is called."""
        points = self._rate_points(batch)
        log_cells = np.log10(np.maximum(np.asarray(cells, dtype=float), 1e-9))
        return np.interp(log_cells, np.log10(points["Cells"]), points["Rate"], left=0.0, right=1.0)

    def chance_called_on_average(self, mean_cells, batch):
        """Chance a clone is called when its cells in the sequenced material are Poisson(mean_cells)."""
        mean_cells = np.asarray(mean_cells, dtype=float)
        grid = self.mean_cells_grid
        from_lookup = np.interp(np.log(np.maximum(mean_cells, grid[0])), np.log(grid), self.lookup[batch])
        return np.where(mean_cells >= grid[-1], 1.0, np.where(mean_cells < grid[0], 0.0, from_lookup))


# =====================================================================================================================
# 5. THE MODEL
# =====================================================================================================================

def secondary_detection_curve(detection, founders_per_primary_cell, sequenced_cells_per_founder, runs, batch):
    """For every clone size on SIZE_GRID: the chance this secondary mouse calls the clone.

    Steps 1-4: founders ~ Poisson(size x founders_per_primary_cell); cells in the sequenced material
    ~ Poisson(founders x sequenced_cells_per_founder), split evenly over the sequencing runs, each run called on its
    own; the clone is called if any run calls it.
    """
    curve = np.empty_like(SIZE_GRID)
    for i, size in enumerate(SIZE_GRID):
        mean_founders = size * founders_per_primary_cell
        spread = 10 * np.sqrt(mean_founders)
        founders = np.arange(max(0, int(mean_founders - spread - 10)), int(mean_founders + spread + 15))
        chance_one_run = detection.chance_called_on_average(founders * sequenced_cells_per_founder / runs, batch)
        chance_any_run = 1 - (1 - chance_one_run) ** runs
        curve[i] = np.sum(scipy.stats.poisson.pmf(founders, mean_founders) * chance_any_run)
    return curve

def clones_below_20_cells(clones_at_20_or_more, slope, primary_pct_sampled, primary_barcoded_cells_injected):
    """The primary's clones smaller than 20 cells, as a power law continuing its larger clones.

    Number of clones of at least s cells = clones_at_20_or_more x (s / 20) ^ -slope. The extension stops at 1 cell in
    the whole primary, or where the primary would hold as many clones as barcoded cells were injected into it,
    whichever comes first. Returns the size of each size bin, the number of clones in it, and the smallest size.
    """
    one_cell_in_primary = primary_pct_sampled / 100
    clones_per_cell_injected = primary_barcoded_cells_injected / clones_at_20_or_more
    as_many_clones_as_cells_injected = TAIL_MIN_CELLS * clones_per_cell_injected ** (-1 / slope)
    smallest = max(one_cell_in_primary, as_many_clones_as_cells_injected)

    def clones_at_least(size):
        return clones_at_20_or_more * (size / TAIL_MIN_CELLS) ** -slope

    edges = np.r_[smallest, SIZE_GRID[(SIZE_GRID > smallest) & (SIZE_GRID < TAIL_MIN_CELLS)], TAIL_MIN_CELLS]
    clones_per_bin = clones_at_least(edges[:-1]) - clones_at_least(edges[1:])
    bin_sizes = np.sqrt(edges[:-1] * edges[1:])
    return bin_sizes, clones_per_bin, smallest

def analyse_one_pdx(model, clone_table, xenograft_table, observed, detection, rng, n_slope_draws):
    # Expected % emerging, and the per-mouse test, for every secondary mouse of one PDX model
    pdx = xenograft_table.loc[xenograft_table["Tumor"] == model]
    primary = pdx.loc[pdx["Cohort"] == "Primary"].iloc[0]
    secondaries = pdx.loc[pdx["Cohort"] != "Primary"]
    primary_batch = primary["SLX"]
    primary_clones = clone_table.loc[(clone_table["Tumor"] == model) & (clone_table["Cohort"] == "Primary")]

    # --- The primary's called clones -------------------------------------------------------------------------------
    measured_sizes = primary_clones["cell"].to_numpy()
    # For the chance a secondary calls one of them, a called clone had at least as many cells as its batch calls half
    # the time; a smaller `cell` estimate comes from the standard curve extrapolated below the spike-ins
    called_sizes = np.maximum(measured_sizes, detection.table.loc[primary_batch, "LOD50 (cells)"])
    # The clones measured at >= 20 cells, inside the spike-in calibration, set the size distribution's slope. Each
    # stands for 1 / (chance called) clones; the rest were missed
    large_clones = measured_sizes[measured_sizes >= TAIL_MIN_CELLS]
    clones_represented = 1 / detection.chance_called_with(large_clones, primary_batch)

    # --- The slope of the primary's size distribution, and its uncertainty ----------------------------------------
    # Maximum-likelihood power-law slope of the clones of >= 20 cells, each weighted by the clones it stands for: the
    # primary's batch misses some clones near 20 cells, so unweighted, its called clones would make the distribution
    # look shallower than it is
    #     slope = sum(weights) / sum(weights x log(size / 20))
    # Its uncertainty: Gamma(n_eff, rate = n_eff / slope), where n_eff = sum(weights)^2 / sum(weights^2) is the clones'
    # effective number (Kish). With equal weights, this is the exact Gamma(number of clones, rate = sum of log(size / 20))
    log_excess = np.log(large_clones / TAIL_MIN_CELLS)
    fitted_slope = clones_represented.sum() / (clones_represented * log_excess).sum()
    effective_clones = clones_represented.sum() ** 2 / (clones_represented ** 2).sum()
    slope_draws = rng.gamma(effective_clones, fitted_slope / effective_clones, n_slope_draws)
    slope_low, slope_high = np.quantile(slope_draws, [0.025, 0.975])
    unweighted_slope = len(large_clones) / log_excess.sum()        # a comparison only

    # --- Each secondary mouse's detection curve (steps 1-4) -------------------------------------------------------
    curves, curves_sequencing_growth, relative_depth = {}, {}, {}
    for _, mouse in secondaries.iterrows():
        founders_per_primary_cell = mouse["% of primary injected"] / primary["% sampled"]
        total_pct_sampled = mouse["% sampled"] * mouse["Sequencing runs"]
        sequenced_cells_per_founder = mouse["Growth, fold (cell yield)"] * total_pct_sampled / 100
        curves[mouse["Mouse"]] = secondary_detection_curve(
            detection, founders_per_primary_cell, sequenced_cells_per_founder, mouse["Sequencing runs"], mouse["SLX"])
        relative_depth[mouse["Mouse"]] = founders_per_primary_cell * sequenced_cells_per_founder

        # Comparison only: growth from the mouse's own sequencing instead of its cell yield
        sequenced_cells_per_founder = (mouse["Barcoded cells sequenced (called clones)"]
                                       / mouse["Barcoded cells injected"])
        curves_sequencing_growth[mouse["Mouse"]] = secondary_detection_curve(
            detection, founders_per_primary_cell, sequenced_cells_per_founder, mouse["Sequencing runs"], mouse["SLX"])

    def expected_emerging(slope, curve_set):
        """{mouse: (expected % emerging, expected clones called)} for one slope."""
        bin_sizes, clones_per_bin, _ = clones_below_20_cells(clones_represented.sum(), slope, primary["% sampled"],
                                                             primary["Barcoded cells injected"])
        missed_small_clones = clones_per_bin * (1 - detection.chance_called_on_average(bin_sizes, primary_batch))
        result = {}
        for mouse_id, curve in curve_set.items():
            def chance_secondary_calls(sizes):
                return np.interp(np.log(sizes), np.log(SIZE_GRID), curve)
            on_primary_list = chance_secondary_calls(called_sizes).sum()
            missed_large = ((clones_represented - 1) * chance_secondary_calls(large_clones)).sum()
            missed_small = (missed_small_clones * chance_secondary_calls(bin_sizes)).sum()
            not_on_primary_list = missed_large + missed_small
            total = on_primary_list + not_on_primary_list
            result[mouse_id] = (100 * not_on_primary_list / total, total)
        return result

    at_fitted_slope = expected_emerging(fitted_slope, curves)
    at_slope_low = expected_emerging(slope_low, curves)
    at_slope_high = expected_emerging(slope_high, curves)
    at_each_draw = [expected_emerging(slope, curves) for slope in slope_draws]
    at_fixed_slopes = {slope: expected_emerging(slope, curves) for slope in SLOPE_SENSITIVITY}
    with_sequencing_growth = expected_emerging(fitted_slope, curves_sequencing_growth)

    # --- One row describing the primary ---------------------------------------------------------------------------
    bin_sizes, clones_per_bin, smallest = clones_below_20_cells(
        clones_represented.sum(), fitted_slope, primary["% sampled"], primary["Barcoded cells injected"])
    chance_primary_calls_small = detection.chance_called_on_average(bin_sizes, primary_batch)
    primary_row = {
        "Tumor": model,
        "Called, >= 20 cells": len(large_clones),
        "Called, < 20 cells": int((measured_sizes < TAIL_MIN_CELLS).sum()),
        "Called, < 20 cells, expected": (clones_per_bin * chance_primary_calls_small).sum(),
        "Slope α": fitted_slope, "α 95% low": slope_low, "α 95% high": slope_high,
        "Effective clones, >= 20 cells": effective_clones, "Slope α, unweighted": unweighted_slope,
        "Size-distribution stop (cells in sequenced material)": smallest,
        "Stopped by": ("1 cell in the primary" if np.isclose(smallest, primary["% sampled"] / 100)
                       else "barcoded cells injected"),
        "Clones below 20 cells, not called (expected)": (clones_per_bin * (1 - chance_primary_calls_small)).sum(),
    }

    # --- One row per secondary mouse ------------------------------------------------------------------------------
    mouse_rows, sensitivity_rows = [], []
    for _, mouse in secondaries.iterrows():
        mouse_id = mouse["Mouse"]
        n_called = int(observed.loc[mouse_id, "Clones called"])
        n_emerging = int(observed.loc[mouse_id, "Emerging"])
        expected_fraction_draws = np.array([draw[mouse_id][0] for draw in at_each_draw]) / 100

        # Two-sided p: the observed emerging count against binomial(clones called, expected fraction), averaged over
        # the slope draws
        p_this_many_or_more = np.mean(scipy.stats.binom.sf(n_emerging - 1, n_called, expected_fraction_draws))
        p_this_many_or_fewer = np.mean(scipy.stats.binom.cdf(n_emerging, n_called, expected_fraction_draws))
        p_value = min(1.0, 2 * min(p_this_many_or_more, p_this_many_or_fewer))

        # The range of % emerging the model allows for a mouse that called this many clones: the same binomial mixture,
        # from its 2.5th to its 97.5th percentile. An observed count outside it is the same as p < 0.05 above
        possible_counts = np.arange(n_called + 1)
        cdf_per_draw = scipy.stats.binom.cdf(possible_counts[:, None], n_called, expected_fraction_draws[None, :])
        mixture_cdf = cdf_per_draw.mean(axis=1)
        range_low = possible_counts[np.argmax(mixture_cdf >= 0.025)]
        range_high = possible_counts[np.argmax(mixture_cdf >= 0.975)]

        mouse_rows.append({
            "Tumor": model, "Cohort": mouse["Cohort"], "Xenograft": mouse["Xenograft"], "Mouse": mouse_id,
            "Sequencing runs": mouse["Sequencing runs"],
            "Cell yield": mouse["Cell yield"],
            "Growth, fold (cell yield)": mouse["Growth, fold (cell yield)"],
            "Relative depth (recipient / primary)": relative_depth[mouse_id],
            "Clones called": n_called,
            "Clones called, expected": at_fitted_slope[mouse_id][1],
            "Emerging": n_emerging,
            "Emerging, expected": at_fitted_slope[mouse_id][0] / 100 * at_fitted_slope[mouse_id][1],
            "Observed % emerging": 100 * n_emerging / n_called,
            "Expected % emerging": at_fitted_slope[mouse_id][0],
            "Expected 95% low (α only)": 100 * np.quantile(expected_fraction_draws, 0.025),
            "Expected 95% high (α only)": 100 * np.quantile(expected_fraction_draws, 0.975),
            "Expected 95% low (α and binomial)": 100 * range_low / n_called,
            "Expected 95% high (α and binomial)": 100 * range_high / n_called,
            "Expected at α 95% low": at_slope_low[mouse_id][0],
            "Expected at α 95% high": at_slope_high[mouse_id][0],
            "p (binomial, α uncertainty)": p_value,
            "Growth, fold (sequencing)": mouse["Growth, fold (sequencing)"],
            "Expected % emerging, growth from sequencing": with_sequencing_growth[mouse_id][0],
        })
        at_each_fixed_slope = {f"α = {slope:g}": at_fixed_slopes[slope][mouse_id][0] for slope in SLOPE_SENSITIVITY}
        sensitivity_rows.append({"Tumor": model, "Cohort": mouse["Cohort"], "Mouse": mouse_id,
                                 "Expected % at fitted α": at_fitted_slope[mouse_id][0], **at_each_fixed_slope})
    return primary_row, mouse_rows, sensitivity_rows


# =====================================================================================================================
# 6. FIGURE
# =====================================================================================================================

FONT_SIZE, STAR_FONT_SIZE, LINE_WIDTH = 7, 9, 0.6
FONT_FAMILIES = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]

def apply_figure_style():
    installed = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    family = next((name for name in FONT_FAMILIES if name in installed), "sans-serif")
    sizes = ["font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize",
             "legend.fontsize", "legend.title_fontsize", "figure.titlesize", "figure.labelsize"]
    matplotlib.rcParams.update({key: FONT_SIZE for key in sizes})
    matplotlib.rcParams.update({
        "font.family": family, "svg.fonttype": "path", "pdf.fonttype": 42,
        "axes.linewidth": LINE_WIDTH, "xtick.major.width": LINE_WIDTH, "ytick.major.width": LINE_WIDTH,
        "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "axes.titlepad": 2, "axes.labelpad": 2, "lines.linewidth": 0.8, "patch.linewidth": 0.5,
        "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    return family

def inch_grid(fig, left, top, nrows, ncols, width, height, wgap, hgap):
    fig_w, fig_h = fig.get_size_inches()
    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            x, y = left + c * (width + wgap), top + r * (height + hgap)
            axes[r, c] = fig.add_axes([x / fig_w, 1 - (y + height) / fig_h, width / fig_w, height / fig_h])
    return axes

def fig_text(fig, x, y, s, **kwargs):
    fig_w, fig_h = fig.get_size_inches()
    kwargs.setdefault("fontsize", FONT_SIZE)
    return fig.text(x / fig_w, 1 - y / fig_h, s, **kwargs)

def star_text(ax, x, y, label, color="#333333"):
    return ax.annotate(label, xy=(x, y), xytext=(0, -STAR_FONT_SIZE * 0.42), textcoords="offset points",
                       ha="center", va="bottom", fontsize=STAR_FONT_SIZE, color=color, annotation_clip=False,
                       clip_on=False)

def plot_observed_vs_expected(mouse_results, out_svg, model_order, cohort_order, subtypes, panel_dir=None,
                              width=3.369, height=2.022):
    family = apply_figure_style()
    if family != FONT_FAMILIES[0]:
        print(f"Note: {FONT_FAMILIES[0]} is not installed, so the figure is drawn in {family}")
    n_rows = int(np.ceil(len(model_order) / 2))
    fig = plt.figure(figsize=(width, height))
    left, right, top, bottom, wgap, hgap = 0.36, 0.07, 0.13, 0.27, 0.14, 0.3
    panel_w = (width - left - right - wgap) / 2
    panel_h = (height - top - bottom - (n_rows - 1) * hgap) / n_rows
    axes = inch_grid(fig, left, top, n_rows, 2, panel_w, panel_h, wgap, hgap)
    for ax in axes.flat[len(model_order):]:
        ax.set_visible(False)                                        # an odd number of models leaves one panel empty
    for k, (ax, model) in enumerate(zip(axes.flat, model_order)):
        mice = mouse_results.loc[mouse_results["Tumor"] == model].copy()
        mice["Cohort rank"] = mice["Cohort"].map(cohort_order.index)
        mice = mice.sort_values(["Cohort rank", "Xenograft"]).reset_index(drop=True)
        mice["x"] = mice.index + 0.4 * mice["Cohort rank"]            # leave a gap between cohorts

        for _, m in mice.iterrows():
            x_expected, x_observed = m["x"] - 0.16, m["x"] + 0.16
            ax.plot([x_expected, x_observed], [m["Expected % emerging"], m["Observed % emerging"]],
                    color="#c8c8c8", linewidth=0.5, zorder=1)
            ax.vlines(x_expected, m["Expected 95% low (α and binomial)"], m["Expected 95% high (α and binomial)"],
                      color="#333333", linewidth=0.6, zorder=2)
            ax.plot(x_expected, m["Expected % emerging"], marker="D", markersize=2.4, markerfacecolor="white",
                    markeredgecolor="#333333", markeredgewidth=0.5, linestyle="none", zorder=3,
                    clip_on=False)                                      # a 0% marker is drawn whole
            ax.plot(x_observed, m["Observed % emerging"], marker="o", markersize=3.2, color=COHORT_COLOURS[m["Cohort"]],
                    markeredgecolor="white", markeredgewidth=0.4, linestyle="none", zorder=4, clip_on=False)
            if m["Significant"]:
                top_value = max(m["Observed % emerging"], m["Expected 95% high (α and binomial)"])
                star_text(ax, m["x"], top_value + 3, "*")

        tick_labels = [f"{cohort[0]}{replicate}" for cohort, replicate in mice["Xenograft"].str.split(" ")]
        ax.set_xticks(mice["x"])
        ax.set_xticklabels(tick_labels)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(mice["x"].min() - 0.45, mice["x"].max() + 0.45)
        ax.set_ylim(0, 120)
        ax.set_yticks([0, 50, 100])
        ax.tick_params(axis="y", labelleft=(k % 2 == 0))
        ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{model} ({subtypes[model]})")
    fig_text(fig, left + panel_w + wgap / 2, height - 0.01, "Xenograft replicate", ha="center", va="bottom")
    fig_text(fig, 0.01, top + (height - top - bottom) / 2, "Emerging cell clones (%)", rotation=90, ha="left",
                va="center")
    if panel_dir is not None:
        panel_dir.mkdir(parents=True, exist_ok=True)
        for ext, dpi in (("svg", 600), ("png", 300)):
            fig.savefig(panel_dir / f"figS3D_amplicon_emerging_poisson_model_observed_vs_expected.{ext}", dpi=dpi)
    fig.savefig(out_svg)
    plt.close(fig)


# =====================================================================================================================
# 7. RUN
# =====================================================================================================================

def main():
    parser = argparse.ArgumentParser(description="Expected % emerging clones from detection alone (Poisson model).")
    parser.add_argument("--inputs-dir", type=Path, default=INPUTS_DIR,
                        help="folder holding the input files (see INPUTS above; default ../data_files/poisson_model)")
    parser.add_argument("--notebook-clone-types", type=Path, default=None,
                        help="the notebooks' clone-type table, to cross-check the observed counts (optional)")
    parser.add_argument("--out-xlsx", type=Path, default=DEFAULT_OUT_XLSX, help="results workbook to write")
    parser.add_argument("--out-svg", type=Path, default=DEFAULT_OUT_SVG, help="figure to write")
    parser.add_argument("--panel-dir", type=Path, default=None,
                        help="also save the figure as the paper's panel (SVG and PNG) in this folder (optional)")
    parser.add_argument("--slope-draws", type=int, default=2000, help="draws of the slope for the per-mouse p-values")
    parser.add_argument("--seed", type=int, default=2026, help="random seed for the slope draws")
    args = parser.parse_args()

    # Inputs
    print(f"Reading inputs from {args.inputs_dir}")
    clone_table = load_clone_table(args.inputs_dir / CLONES_FILE)
    xenograft_table = build_xenograft_table(args.inputs_dir / XENOGRAFTS_FILE, clone_table)
    rates_per_batch = load_spike_in_rates(args.inputs_dir / SPIKE_IN_RATES_FILE)
    check_inputs(xenograft_table, clone_table, rates_per_batch)
    observed = count_observed_emerging(clone_table, xenograft_table, args.notebook_clone_types)
    rates_pooled = pooled_spike_in_rates(rates_per_batch)
    detection = BatchDetection(rates_per_batch)

    # Models, their subtypes and the cohorts, in the xenograft table's order
    model_order = list(dict.fromkeys(xenograft_table["Tumor"]))
    cohort_order = list(dict.fromkeys(xenograft_table.loc[xenograft_table["Cohort"] != "Primary", "Cohort"]))
    subtypes = xenograft_table.groupby("Tumor")["Subtype"].first().to_dict()

    # Model, one PDX at a time (in the xenograft table's order, so the slope draws are reproducible)
    rng = np.random.default_rng(args.seed)
    primary_rows, mouse_rows, sensitivity_rows = [], [], []
    for model in model_order:
        primary_row, mice, sensitivity = analyse_one_pdx(model, clone_table, xenograft_table, observed, detection,
                                                         rng, args.slope_draws)
        primary_rows.append(primary_row)
        mouse_rows += mice
        sensitivity_rows += sensitivity

    primary_results = pd.DataFrame(primary_rows)
    mouse_results = pd.DataFrame(mouse_rows)
    mouse_results["Difference (observed - expected, % points)"] = (mouse_results["Observed % emerging"]
                                                                   - mouse_results["Expected % emerging"])
    mouse_results["BH q"] = multipletests(mouse_results["p (binomial, α uncertainty)"], method="fdr_bh")[1]
    mouse_results["Significant"] = mouse_results["p (binomial, α uncertainty)"] < SIG_ALPHA
    sensitivity_results = pd.DataFrame(sensitivity_rows)

    # Outputs
    args.out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    args.out_svg.parent.mkdir(parents=True, exist_ok=True)
    # Only what the model computes, and the spike-in rates it used
    with pd.ExcelWriter(args.out_xlsx) as writer:
        detection.table[["Called at 20 cells", "LOD50 (cells)"]].to_excel(writer, sheet_name="Detection per batch")
        rates_per_batch.to_excel(writer, sheet_name="Spike-in rates per batch", index=False)
        primary_results.to_excel(writer, sheet_name="Primary size distribution", index=False)
        mouse_results.to_excel(writer, sheet_name="Per mouse", index=False)
        sensitivity_results.to_excel(writer, sheet_name="Slope sensitivity", index=False)
    plot_observed_vs_expected(mouse_results, args.out_svg, model_order, cohort_order, subtypes, args.panel_dir)

    # Summary on screen
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 100)
    print("Spike-ins pooled over batches\n", rates_pooled.round(3), "\n")
    print(f"Nothing is called at or below {detection.never_called_cells} cells "
          f"(no batch ever called a spike-in that small)")
    print("Detection per batch\n", detection.table.round(3), "\n")
    print("Primaries\n", primary_results.round(3), "\n")
    print("Per mouse\n", mouse_results[["Tumor", "Xenograft", "Mouse", "Relative depth (recipient / primary)",
                                        "Clones called", "Observed % emerging", "Expected % emerging",
                                        "Expected 95% low (α and binomial)", "Expected 95% high (α and binomial)",
                                        "p (binomial, α uncertainty)",
                                        "BH q", "Significant"]].round(3), "\n")
    print(f"Wrote {args.out_xlsx}\nWrote {args.out_svg}")


if __name__ == "__main__":
    main()
