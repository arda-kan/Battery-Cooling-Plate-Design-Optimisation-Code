"""
=============================================================================
 1D Serpentine Battery Cooling Plate Layout Optimizer
=============================================================================
 University Engineering Project – First-Pass Design Screening Tool
 Author  : [Your Name]
 Version : 4.0  (v4 fixes: pitch/bend-radius geometric consistency enforced in
                  feasibility check, hydraulic length model, and schematic plot.
                  When pitch > 2*R, connector legs are added to the U-bend.
                  When pitch < 2*R, design is rejected as infeasible.)

 PURPOSE
 -------
 This script performs a brute-force parameter sweep over a family of
 hybrid serpentine / parallel-branch liquid-cooling-plate designs.
 For every candidate geometry it applies 1D thermal-fluid analysis to
 estimate:
   • heat removal capacity
   • coolant and battery temperatures
   • pressure drop
   • an aggregate optimization score

 The output is a ranked table (CSV + printed), summary plots, and a 2-D
 schematic of the best design.

 ASSUMPTIONS & LIMITATIONS (stated explicitly)
 ----------------------------------------------
  1.  1-D screening model only – not a substitute for CFD.
  2.  Equal flow split among parallel branches assumed.
  3.  Serpentine enhancement modelled by a scalar phi factor; real
      secondary flows are not resolved.
  4.  Bend pressure loss modelled by a single empirical K_bend coefficient.
  5.  Conduction resistance is a simple 1-D path through the plate
      (no 2-D spreading resistance).
  6.  Battery treated as a uniform volumetric heat source – no cell-level
      thermal gradient.
  7.  Fluid properties evaluated once at the mean film temperature (inlet
      conditions used as initial estimate).
  8.  Steady-state model; transient effects are ignored.
  9.  Manifold pressure losses are neglected.
 10.  Channel cross-section assumed circular throughout.

 STRUCTURE
 ---------
  1.  Imports
  2.  User Configuration  ← EDIT HERE
  3.  Physical Constants
  4.  Dataclasses
  5.  Fluid & Material Properties
  6.  Geometry Generation & Feasibility Checks
  7.  Per-Segment Calculations
  8.  Branch Evaluation
  9.  Full Design Evaluation
 10.  Parameter Sweep
 11.  Ranking & Results
 12.  Plotting
 13.  main()
=============================================================================
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import math
import warnings
import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 2. USER CONFIGURATION  ← EDIT RANGES AND WEIGHTS HERE
# =============================================================================
# ---------------------------------------------------------------------------
# 2a.  Fixed boundary conditions
# ---------------------------------------------------------------------------
BATTERY_LENGTH_M   = 0.420       # m  – battery pack x-dimension
BATTERY_WIDTH_M    = 0.210       # m  – battery pack y-dimension
BATTERY_HEIGHT_M   = 0.070       # m  – battery pack z-dimension (thickness)
Q_VOL_W_M3         = 120_000.0   # W/m³ – volumetric heat generation rate
T_INLET_C          = 25.0        # °C  – coolant inlet temperature
T_BATTERY_INIT_C   = 25.0        # °C  – initial battery temperature
M_DOT_TOTAL_KG_S   = 0.012       # kg/s – total inlet mass flow rate

# ---------------------------------------------------------------------------
# 2b.  Plate geometry
# ---------------------------------------------------------------------------
PLATE_THICKNESS_M  = 0.010       # m  aluminium plate thickness (10 mm)
#                                       Updated from 4 mm to 10 mm to reflect the
#                                       intended thicker cold-plate design.
#                                       Internal circular channels may be up to
#                                       0.75 × 10 mm = 7.5 mm diameter.
#                                       Inlet/outlet port features are external and
#                                       are not limited by this plate-thickness rule.
EDGE_OFFSET_M      = 0.010       # m  minimum distance from channel centreline to plate edge
# Minimum centre-to-centre spacing: must exceed channel diameter + wall thickness.
# For 3 mm channels with 1 mm minimum wall: pitch > 3+1 = 4 mm.
# Set to 5 mm to give comfortable margin.
MIN_CHANNEL_PITCH_M = 0.005      # m  minimum centre-to-centre channel spacing
MANIFOLD_WIDTH_M   = 0.010       # m  width consumed by inlet + outlet manifold (5 mm each side)

# ---------------------------------------------------------------------------
# 2b-ii.  Manifold / header geometry for flow distribution  (Change 2 additions)
# ---------------------------------------------------------------------------
# These parameters control the inlet and outlet header geometry used for the
# manifold distribution penalty.  They do NOT alter the 1-D hydraulic model
# (which continues to assume equal flow split among branches) but they feed
# a geometric proxy penalty that encourages the optimizer to prefer layouts
# that are more likely to distribute flow evenly in real hardware / CFD.
#
# Physical rationale
# ------------------
#   Inlet header:   flow decreases along the header as branches draw fluid off.
#     → Tapered (wider near port, narrower at far end) headers maintain roughly
#       constant header velocity and more uniform branch-inlet static pressure.
#   Outlet header:  flow accumulates along the header as branches rejoin.
#     → Tapered (narrower near first branch, wider toward port) headers similarly
#       equalise static pressure at branch outlets.
#
# Taper model used here:
#   Header width at branch k = W_inlet * (1 - taper_ratio * k / (n_branches-1))
#   where k = 0 … n_branches-1, W_inlet is the header width at the port side.
#   taper_ratio = 0 → uniform (no taper); taper_ratio = 0.5 → 50 % reduction
#   from first to last branch.
#
# NOTE: This is a SCREENING-LEVEL proxy only.  The actual flow distribution
#   depends on branch pressure drops, manifold geometry, and entry conditions
#   which would require a CFD manifold network solver to compute accurately.

# Header width at the port side (the widest point for an inlet tapered header)
MANIFOLD_INLET_WIDTH_M   = 0.015   # m  15 mm  (representative; adjust as needed)

# Taper ratio: fraction by which header width decreases from port to far end
#   0.0 = uniform header (no taper)  ← lower distribution benefit, simpler to make
#   0.5 = header narrows to 50 % of inlet width at far branch
MANIFOLD_TAPER_RATIO     = 0.4     # –   recommended range 0.0 – 0.6

# Realistic header tube inner diameter for hydraulic pressure-drop calculation.
# This is SEPARATE from the visual manifold width above.
# Typical cold-plate header tubes: 8–14 mm ID for this flow rate.
# Smaller headers → higher velocity → stronger manifold pressure gradient → more
# maldistribution.  Larger headers → lower ΔP gradient → better uniformity.
MANIFOLD_TUBE_DIAMETER_M = 0.010   # m  10 mm ID header tube for hydraulic model

# Weight for the manifold distribution penalty in the optimization score
W_MANIFOLD_DIST          = 4.0     # –   weight on manifold uniformity penalty

# ---------------------------------------------------------------------------
# 2c.  Hydraulic model parameters
# ---------------------------------------------------------------------------
K_BEND             = 1.5         # -   bend loss coefficient per U-turn (base value)
#                                       Effective K scaled by bend tightness:
#                                       K_eff = K_BEND × clamp(D / (2×R_b), 0.4, 1.8)
#                                       Tight bends (small R/D) → K up to 1.8×K_BEND
#                                       Gentle bends (large R/D) → K as low as 0.4×K_BEND
#                                       (Idelchik, Handbook of Hydraulic Resistance)
# FIX (v3): Entrance and exit minor loss coefficients.
# Applied once per branch: at the branch inlet and branch outlet.
# K_entry = 0.5 (sharp-edged entry from manifold into channel)
# K_exit  = 1.0 (sudden expansion from channel back into outlet manifold)
K_ENTRY            = 0.5         # -   entrance loss coefficient (per branch, once)
K_EXIT             = 1.0         # -   exit loss coefficient (per branch, once)
# v11: manifold branch junction loss (tee split/merge).
# Applied at each branch take-off (inlet) and merge (outlet) junction.
# K_j = 0.5 is a standard tee branch loss coefficient (Idelchik).
K_JUNCTION_BRANCH  = 0.5         # -   per-branch junction loss coefficient (v11)
PHI_SERPENTINE     = 1.6         # –   Nusselt enhancement factor for serpentine
#                                       mixing (range 1.0–2.5; use 1.0 for no enhancement)
NU_LAM_CONST       = 4.36        # –   baseline Nusselt for fully-developed laminar
#                                       flow (uniform heat flux); alternative: 3.66

# ---------------------------------------------------------------------------
# 2d.  Diameter options
# ---------------------------------------------------------------------------
USE_3ZONE_DIAMETER = False       # True  → D1/D2/D3 taper; False → constant diameter
# For constant-diameter mode the optimizer sweeps D_CONST_OPTIONS.
# For 3-zone mode it sweeps D1/D2/D3 independently.
#
# With PLATE_THICKNESS_M = 10 mm and a 0.75 manufacturing ratio,
# internal channels up to 7.5 mm diameter are feasible.
# Sweep range updated accordingly: 2 / 3 / 4 / 5 / 6 mm channels.
D_CONST_OPTIONS_M  = [0.002, 0.003, 0.004, 0.005, 0.006]  # m  2/3/4/5/6 mm
D1_OPTIONS_M       = [0.006, 0.005]                         # m  inlet-zone (3-zone mode)
D2_OPTIONS_M       = [0.005, 0.004]                         # m  mid-zone   (3-zone mode)
D3_OPTIONS_M       = [0.004, 0.003]                         # m  outlet-zone (3-zone mode)
D_MIN_M            = 0.001       # m  absolute minimum channel diameter (manufacturing)
D_MAX_M            = 0.0075      # m  absolute maximum = 75 % of 10 mm plate thickness

# ---------------------------------------------------------------------------
# 2e.  Parameter sweep ranges
# ---------------------------------------------------------------------------
N_BRANCHES_OPTIONS    = [2, 3, 4]          # number of parallel branches
N_PASSES_OPTIONS      = [3, 4, 5, 6]      # serpentine passes per branch
# Bend radius options scaled for larger channels (2-6 mm diameter).
# Minimum practical bend: ~2x channel diameter. For D=6 mm: R_min ~12 mm.
# For D=3 mm: R_min ~6 mm.  Sweep wider range.
BEND_RADIUS_OPTIONS_M = [0.008, 0.012, 0.016]  # m  U-bend centreline radius

# ---------------------------------------------------------------------------
# 2e-ii.  Topology / turn-style sweep options  (v8: multi-topology extension)
# ---------------------------------------------------------------------------
# These lists control which layout families are included in the sweep.
# Remove entries to narrow the search; add entries to widen it.
#
# TOPOLOGY_OPTIONS  –  plan-view channel routing families
#   "H_SERPENTINE"        straight runs across plate width, stacked along length
#   "V_SERPENTINE"        as above but rotated 90° (runs along length, stacked across width)
#   "MIRRORED_U"          inlet and outlet on the same end; branches go out and return
#   "Z_FLOW"              inlet and outlet on opposite corners; diagonal header routing
#   "CENTRAL_INLET"       central manifold; flow splits symmetrically to both sides
#
# MANIFOLD_OPTIONS  –  how the inlet/outlet ports are placed
#   "LEFT_RIGHT"          inlet on left edge, outlet on right edge (default)
#   "TOP_BOTTOM"          inlet on top edge, outlet on bottom edge
#   "SAME_SIDE"           inlet and outlet on same edge (used with MIRRORED_U)
#   "CENTER_EDGE"         central inlet manifold, outlet at edge (used with CENTRAL_INLET)
#
# TURN_STYLE_OPTIONS  –  how 180° U-bends are modelled
#   "CONNECTOR_SEMICIRCLE"  connector legs + semicircular arc (current behaviour)
#   "PURE_CIRCULAR"         pure semicircle only (only valid when pitch == 2*R_bend)
#   "SMOOTH_SPLINE"         smooth S-curve approximation (cosine blend; slightly longer)

TOPOLOGY_OPTIONS       = [
    "H_SERPENTINE",
    "V_SERPENTINE",
    "MIRRORED_U",
    "Z_FLOW",
    "CENTRAL_INLET",
]

MANIFOLD_OPTIONS       = [
    "LEFT_RIGHT",
    "TOP_BOTTOM",
    "SAME_SIDE",
    "CENTER_EDGE",
]

TURN_STYLE_OPTIONS     = [
    "CONNECTOR_SEMICIRCLE",
    "PURE_CIRCULAR",
    "SMOOTH_SPLINE",
]

# Topology-manifold compatibility table
# Not all combinations are physically meaningful.
# Entries: (topology, manifold) → True = include in sweep
_TOPO_MANIFOLD_COMPAT = {
    ("H_SERPENTINE",  "LEFT_RIGHT"):   True,
    ("H_SERPENTINE",  "TOP_BOTTOM"):   True,
    ("H_SERPENTINE",  "SAME_SIDE"):    False,
    ("H_SERPENTINE",  "CENTER_EDGE"):  False,
    ("V_SERPENTINE",  "LEFT_RIGHT"):   True,
    ("V_SERPENTINE",  "TOP_BOTTOM"):   True,
    ("V_SERPENTINE",  "SAME_SIDE"):    False,
    ("V_SERPENTINE",  "CENTER_EDGE"):  False,
    ("MIRRORED_U",    "LEFT_RIGHT"):   False,
    ("MIRRORED_U",    "TOP_BOTTOM"):   False,
    ("MIRRORED_U",    "SAME_SIDE"):    True,
    ("MIRRORED_U",    "CENTER_EDGE"):  False,
    ("Z_FLOW",        "LEFT_RIGHT"):   True,
    ("Z_FLOW",        "TOP_BOTTOM"):   True,
    ("Z_FLOW",        "SAME_SIDE"):    False,
    ("Z_FLOW",        "CENTER_EDGE"):  False,
    ("CENTRAL_INLET", "LEFT_RIGHT"):   False,
    ("CENTRAL_INLET", "TOP_BOTTOM"):   False,
    ("CENTRAL_INLET", "SAME_SIDE"):    False,
    ("CENTRAL_INLET", "CENTER_EDGE"):  True,
}

# Small topology-complexity penalty weight
# Penalises more complex layouts slightly so the score doesn't blindly prefer
# the topology with the most geometric freedom.
W_TOPOLOGY_COMPLEXITY = 1.0   # –   weight; set to 0.0 to disable

# ---------------------------------------------------------------------------
# 2e-iii.  Velocity control (Change 2) — target velocity band for tapering
# ---------------------------------------------------------------------------
# The optimizer rewards branch designs whose local channel velocity stays
# within this band.  Designs far outside the band receive a penalty.
# Applies in both constant-diameter and 3-zone modes.
V_TARGET_MIN_M_S    = 0.3        # m/s  lower bound of preferred velocity band
V_TARGET_MAX_M_S    = 1.5        # m/s  upper bound of preferred velocity band
W_VELOCITY_PEN      = 6.0        # –    weight on velocity-variation penalty in score

# ---------------------------------------------------------------------------
# 2e-iv.  Branch-count heat-flux scaling (Change 3)
# ---------------------------------------------------------------------------
# Target branch heat-flux range.  Branches that serve heat loads outside this
# range receive a penalty: too few branches → excessive heat per branch,
# too many branches → maldistribution risk / complexity.
Q_BRANCH_TARGET_W_M2_MIN = 1_000.0   # W/m²  lower branch heat-flux target
Q_BRANCH_TARGET_W_M2_MAX = 8_000.0   # W/m²  upper branch heat-flux target
W_BRANCH_HF_PEN          = 4.0       # –     weight on branch-count heat-flux penalty

# ---------------------------------------------------------------------------
# 2f.  Optimization weights  <- TUNE THESE TO CHANGE DESIGN PREFERENCE
# ---------------------------------------------------------------------------
# score = w_T_batt × T_batt_mean   (primary driver: keep battery cool)
#       + w_T_max  × T_batt_max    (penalise worst hot-spot)
#       + w_dP     × dP_total      (pressure drop — secondary, scaled to ~same range)
#       + ...
# Lower score = better design.
#
# v11 rebalance: temperature weights raised, dP weight lowered so that a
# 1°C temperature improvement outweighs a ~500 Pa pressure-drop saving.
# With T_batt typically 35–55°C and dP typically 5–40 kPa:
#   W_T_BATT × 10°C = 100   (10°C improvement = 100 score points)
#   W_DELTA_P × 10000 Pa = 100 → W_DELTA_P = 0.01   (10 kPa = 100 points)
# This gives them roughly equal weight at typical values — but the
# temperature weight is doubled to give it clear priority.
W_T_BATT            = 10.0       # weight on mean battery temperature (°C) — primary
W_DELTA_P           = 0.003      # weight on total pressure drop (Pa) — secondary
W_UNIFORMITY        = 5.0        # weight on thermal uniformity proxy (-)
W_MANUFACTURABILITY = 3.0        # weight on manufacturability penalty (-)

# ---------------------------------------------------------------------------
# 2f-ii.  v9: Physical topology-scoring weights
# ---------------------------------------------------------------------------
# W_BRANCH_UNIFORMITY  – rewards even branch flow distribution
#   Derived from std(branch_flow_fractions) + resistance variation
# W_T_BATT_MAX         – penalises the hottest branch (local worst-case temperature)
#   Set to 0 to use only the mean battery temperature in the score.
# W_MANIFOLD_DP        – penalty on manifold hydraulic pressure loss
#   Adds actual manifold pipe losses to the pressure drop accounting.
W_BRANCH_UNIFORMITY  = 8.0       # weight on branch-flow non-uniformity
W_T_BATT_MAX         = 5.0       # weight on max-branch battery temperature (°C)
W_MANIFOLD_DP        = 0.003     # weight on manifold pressure drop contribution (Pa)

# ---------------------------------------------------------------------------
# 2g.  Output options
# ---------------------------------------------------------------------------
CSV_OUTPUT_FILE     = "cooling_plate_results.csv"
TOP_N_PRINT         = 10         # number of top designs to print to console
FIG_DPI             = 120        # matplotlib figure DPI


# =============================================================================
# 3. PHYSICAL CONSTANTS
# =============================================================================
# Water properties (evaluated at ~30 °C as first-order approximation)
RHO_WATER_KG_M3     = 996.0      # kg/m³   density
CP_WATER_J_KGK      = 4179.0     # J/(kg·K) specific heat capacity
MU_WATER_PA_S       = 7.98e-4    # Pa·s    dynamic viscosity  (30 °C ≈ 7.98e-4)
K_WATER_W_MK        = 0.615      # W/(m·K) thermal conductivity

# Aluminium 6061-T4 properties
K_ALU_W_MK          = 154.0      # W/(m·K) thermal conductivity of aluminium
RHO_ALU_KG_M3       = 2700.0     # kg/m³   density (for reference; not used in calcs)


# =============================================================================
# 4. DATACLASSES
# =============================================================================

@dataclass
class FluidProperties:
    """
    Container for coolant thermophysical properties.
    All values in SI units.
    """
    rho:  float = RHO_WATER_KG_M3     # kg/m³
    cp:   float = CP_WATER_J_KGK      # J/(kg·K)
    mu:   float = MU_WATER_PA_S       # Pa·s
    k:    float = K_WATER_W_MK        # W/(m·K)

    @property
    def Pr(self) -> float:
        """Prandtl number (dimensionless)."""
        return (self.mu * self.cp) / self.k


@dataclass
class MaterialProperties:
    """
    Container for solid plate material properties.
    """
    name:          str   = "Aluminium 6061-T4"
    k_solid_W_mK:  float = K_ALU_W_MK    # W/(m·K)


@dataclass
class LayoutSection:
    """
    Represents one segment of the channel routing in a generic way.

    All topology generators return a list of LayoutSection objects that
    describe the full centreline routing.  The hydraulic model consumes
    these sections independent of which topology produced them.

    Fields
    ------
    name        : human-readable label  (e.g. "B1 P2 straight")
    sec_type    : "straight" | "bend" | "connector" | "header" | "port"
    branch_id   : which branch (0-based); -1 for manifold/port sections
    pass_id     : which pass within the branch (0-based)
    length_m    : centreline arc length  [m]
    diameter_m  : hydraulic diameter  [m]
    bend_radius_m : bend centreline radius [m]; 0.0 for straight sections
    x0, y0      : start point  (mm, in plate coordinates)
    x1, y1      : end point    (mm, in plate coordinates)
    """
    name:         str   = ""
    sec_type:     str   = "straight"   # straight / bend / connector / header / port
    branch_id:    int   = -1
    pass_id:      int   = -1
    length_m:     float = 0.0
    diameter_m:   float = 0.0
    bend_radius_m:float = 0.0
    x0:           float = 0.0
    y0:           float = 0.0
    x1:           float = 0.0
    y1:           float = 0.0


@dataclass
class DesignVariables:
    """
    All independent design parameters for one candidate layout.
    Lengths in metres.
    """
    n_branches:    int    = 3            # number of parallel branches
    n_passes:      int    = 4            # serpentine straight passes per branch
    bend_radius_m: float  = 0.010        # U-bend centreline radius [m]
    # Diameter specification:
    #   use_3zone=False → single constant diameter
    #   use_3zone=True  → inlet / mid / outlet zone diameters
    use_3zone:     bool   = False
    D_const_m:     float  = 0.008        # m  constant channel diameter
    D1_m:          float  = 0.010        # m  inlet-zone diameter (3-zone mode)
    D2_m:          float  = 0.008        # m  mid-zone diameter (3-zone mode)
    D3_m:          float  = 0.006        # m  outlet-zone diameter (3-zone mode)
    # --- v8: topology / routing style variables ---
    topology:      str    = "H_SERPENTINE"         # see TOPOLOGY_OPTIONS
    manifold:      str    = "LEFT_RIGHT"            # see MANIFOLD_OPTIONS
    turn_style:    str    = "CONNECTOR_SEMICIRCLE"  # see TURN_STYLE_OPTIONS


@dataclass
class SegmentResult:
    """
    Thermal-hydraulic results for a single straight channel segment.
    """
    seg_id:        int    = 0
    length_m:      float  = 0.0
    diameter_m:    float  = 0.0
    area_m2:       float  = 0.0         # cross-sectional area
    wetted_area_m2:float  = 0.0         # lateral wetted surface area
    velocity_m_s:  float  = 0.0
    Re:            float  = 0.0
    Nu:            float  = 0.0
    h_W_m2K:       float  = 0.0         # convective heat transfer coefficient
    R_conv_K_W:    float  = 0.0         # convective resistance
    R_cond_K_W:    float  = 0.0         # plate conduction resistance
    R_total_K_W:   float  = 0.0         # total resistance
    dT_coolant_K:  float  = 0.0         # coolant temperature rise in this segment
    dP_friction_Pa:float  = 0.0         # frictional pressure drop
    dP_bend_Pa:    float  = 0.0         # bend pressure drop (applied at end of seg)
    flag:          str    = ""           # warning / info flag


@dataclass
class BranchResult:
    """
    Aggregated results for one parallel branch.
    """
    branch_id:         int             = 0
    segments:          List[SegmentResult] = field(default_factory=list)
    m_dot_branch_kg_s: float           = 0.0
    T_in_C:            float           = T_INLET_C
    T_out_C:           float           = 0.0
    dP_total_Pa:       float           = 0.0
    Q_absorbed_W:      float           = 0.0
    total_length_m:    float           = 0.0
    feasible:          bool            = True
    flag:              str             = ""
    # v9: connector geometry and flow-split fields
    connector_length_m: float          = 0.0   # header-to-first-pass connector length
    branch_resistance:  float          = 0.0   # approx hydraulic resistance [Pa·s/kg]
    flow_fraction:      float          = 0.0   # fraction of total flow this branch receives
    T_batt_branch_C:    float          = 0.0   # estimated local battery temperature for this branch


@dataclass
class DesignResult:
    """
    Full evaluation result for one complete cooling plate design.
    """
    design_id:              int    = 0
    dv:                     Optional[DesignVariables] = None
    feasible:               bool   = True
    infeasibility_reason:   str    = ""
    # Thermal
    Q_total_W:              float  = 0.0
    T_coolant_avg_C:        float  = 0.0
    T_out_C:                float  = 0.0
    delta_T_batt_coolant_K: float  = 0.0
    T_batt_est_C:           float  = 0.0
    R_total_K_W:            float  = 0.0
    # Hydraulic
    dP_total_Pa:            float  = 0.0
    Re_avg:                 float  = 0.0
    # Geometry
    coverage_ratio:         float  = 0.0
    straight_pass_length_m: float  = 0.0
    branch_pitch_m:         float  = 0.0
    total_channel_length_m: float  = 0.0
    # Penalties
    uniformity_penalty:     float  = 0.0
    manuf_penalty:          float  = 0.0
    manifold_dist_penalty:  float  = 0.0   # Change 2: manifold distribution proxy (legacy)
    topology_complexity_pen:float  = 0.0   # v8: topology complexity proxy
    branch_uniformity_pen:  float  = 0.0   # v9: physical branch-flow uniformity penalty
    # Score (lower = better)
    score:                  float  = 1e9
    # Branches
    branches:               List[BranchResult] = field(default_factory=list)
    # v8: generated layout sections (for plotting)
    layout_sections:        List["LayoutSection"] = field(default_factory=list)
    # v9: topology-physical metrics
    T_batt_max_C:           float  = 0.0   # hottest branch battery temperature
    T_batt_mean_C:          float  = 0.0   # mean battery temperature (branch-weighted)
    dP_manifold_Pa:         float  = 0.0   # manifold+connector pressure loss contribution
    mean_branch_flow_kg_s:  float  = 0.0
    std_branch_flow_kg_s:   float  = 0.0
    min_branch_flow_kg_s:   float  = 0.0
    max_branch_flow_kg_s:   float  = 0.0
    mean_branch_resistance: float  = 0.0
    std_branch_resistance:  float  = 0.0
    manifold_length_inlet_m: float = 0.0
    manifold_length_outlet_m:float = 0.0
    # Change 2: velocity-control metrics
    branch_velocity_min_m_s: float = 0.0   # minimum local channel velocity across all passes
    branch_velocity_max_m_s: float = 0.0   # maximum local channel velocity across all passes
    velocity_variation_pen:  float = 0.0   # [0,1] penalty for velocity outside target band
    # Change 3: branch-count / heat-flux scaling metrics
    q_planar_W_m2:           float = 0.0   # overall battery planform heat flux  [W/m²]
    Q_per_branch_W:          float = 0.0   # total heat load per branch  [W]
    A_per_branch_m2:         float = 0.0   # plate footprint area served per branch  [m²]
    q_branch_W_m2:           float = 0.0   # branch heat-flux metric  [W/m²]
    branch_hf_penalty:       float = 0.0   # [0,1] penalty if q_branch outside target band


# =============================================================================
# 5. FLUID & MATERIAL PROPERTY FUNCTIONS
# =============================================================================

def get_fluid_properties(T_C: float = 30.0) -> FluidProperties:
    """
    Return water properties.  Currently returns fixed values evaluated at ~30 °C.
    Can be extended to interpolate from a look-up table for better accuracy.

    Parameters
    ----------
    T_C : float
        Mean coolant temperature in °C (unused in current fixed-property version).

    Returns
    -------
    FluidProperties
    """
    # NOTE: For a production model, replace with polynomial fits or a look-up
    # table keyed on temperature.  The values below are accurate to ±2 % for
    # water between 20 °C and 40 °C.
    return FluidProperties(
        rho = RHO_WATER_KG_M3,
        cp  = CP_WATER_J_KGK,
        mu  = MU_WATER_PA_S,
        k   = K_WATER_W_MK,
    )


def get_material_properties() -> MaterialProperties:
    """Return aluminium 6061-T4 properties."""
    return MaterialProperties(
        name         = "Aluminium 6061-T4",
        k_solid_W_mK = K_ALU_W_MK,
    )


# =============================================================================
# 6. GEOMETRY GENERATION & FEASIBILITY CHECKS
# =============================================================================

def compute_battery_heat_load() -> float:
    """
    Compute total battery heat generation from volumetric rate and pack volume.

    Q_total = q_vol  [W/m³]  ×  V_battery  [m³]

    Returns
    -------
    Q_total : float
        Total heat generation in Watts.
    """
    V_battery = BATTERY_LENGTH_M * BATTERY_WIDTH_M * BATTERY_HEIGHT_M
    Q_total   = Q_VOL_W_M3 * V_battery
    return Q_total


def compute_straight_pass_length(dv: DesignVariables) -> float:
    """
    Compute the length of one straight serpentine run for the given design.

    For H_SERPENTINE / Z_FLOW / MIRRORED_U / CENTRAL_INLET:
        Runs along the plate WIDTH axis.
        L_pass = battery_width − 2×edge_offset − manifold_width

    For V_SERPENTINE:
        Runs along the plate LENGTH axis.
        L_pass = battery_length − 2×edge_offset − manifold_width

    Parameters
    ----------
    dv : DesignVariables

    Returns
    -------
    L_pass : float  [m]
    """
    if dv.topology == "V_SERPENTINE":
        L_available = BATTERY_LENGTH_M - 2.0 * EDGE_OFFSET_M - MANIFOLD_WIDTH_M
    else:
        L_available = BATTERY_WIDTH_M  - 2.0 * EDGE_OFFSET_M - MANIFOLD_WIDTH_M
    return max(L_available, 0.0)


def compute_branch_pitch(dv: DesignVariables) -> float:
    """
    Compute the centre-to-centre pitch between adjacent serpentine passes
    so that channels fill the full available plate dimension.

    v11 fix: The pitch is derived from the stacking axis length *after*
    subtracting edge offsets and one bend-diameter clearance at each end
    (needed so the outermost U-turn arcs don't fall outside the plate).

    For H_SERPENTINE, Z_FLOW, MIRRORED_U, CENTRAL_INLET:
        Passes stacked along the plate LENGTH axis.
        L_stack = BATTERY_LENGTH_M - 2×EDGE - 2×D_mean
        Pitch   = L_stack / (n_branches × n_passes - 1)

    For V_SERPENTINE:
        Passes stacked along the plate WIDTH axis.
        L_stack = BATTERY_WIDTH_M - 2×EDGE - 2×D_mean
        Pitch   = L_stack / (n_branches × n_passes - 1)

    The -1 is because N passes need N-1 gaps between their centrelines.
    Falls back to equal spacing if only 1 total pass.

    Parameters
    ----------
    dv : DesignVariables

    Returns
    -------
    pitch : float  [m]
    """
    N_total_passes = dv.n_branches * dv.n_passes
    if N_total_passes <= 1:
        return 0.0

    D_mean = float(np.mean(get_diameter_profile(dv))) if get_diameter_profile(dv) else 0.004

    if dv.topology == "V_SERPENTINE":
        L_stack = BATTERY_WIDTH_M - 2.0 * EDGE_OFFSET_M - 2.0 * D_mean
    else:
        L_stack = BATTERY_LENGTH_M - 2.0 * EDGE_OFFSET_M - 2.0 * D_mean

    L_stack = max(L_stack, 0.0)
    return L_stack / (N_total_passes - 1)


def get_diameter_profile(dv: DesignVariables) -> List[float]:
    """
    Return a list of channel diameters, one per straight pass.

    For constant-diameter mode: all passes get D_const.
    For 3-zone mode:
        - first  n_passes//3 passes use D1
        - middle n_passes//3 passes use D2
        - last   remaining passes use D3
    This represents a tapered design where the inlet zone has a larger
    diameter (lower velocity → lower pressure drop at entry), the middle
    zone is intermediate, and the outlet zone is smaller (higher velocity
    → better heat transfer).

    Parameters
    ----------
    dv : DesignVariables

    Returns
    -------
    diameters : List[float]
        Diameter for each pass in order, length = n_passes.
    """
    if not dv.use_3zone:
        return [dv.D_const_m] * dv.n_passes

    n = dv.n_passes
    zone1 = n // 3
    zone2 = n // 3
    zone3 = n - zone1 - zone2   # absorb remainder into outlet zone

    diameters = (
        [dv.D1_m] * zone1 +
        [dv.D2_m] * zone2 +
        [dv.D3_m] * zone3
    )
    return diameters


# =============================================================================
# 6b.  TOPOLOGY GEOMETRY GENERATORS  (v8)
# =============================================================================
# Each function below generates a list of LayoutSection objects that describe
# the full centreline routing for one topology family.
#
# COORDINATE CONVENTION (all in mm, plan view)
#   x  →  plate width direction   (0 … BATTERY_WIDTH_M*1e3)
#   y  ↑  plate length direction  (0 … BATTERY_LENGTH_M*1e3)
#
# SIMPLIFICATIONS (acceptable for 1D screening)
#   • Manifold sections are described but their hydraulic loss is currently
#     absorbed into K_ENTRY / K_EXIT (consistent with the existing model).
#   • Bend sections are represented by their centreline arc length.
#   • All coordinates are approximate centreline positions suitable for
#     schematic plotting; they are not precision CAD coordinates.
#
# TURN-STYLE HELPER
# -----------------
# The turn style modifies:
#   (a) the arc length of each 180° U-bend  (affects hydraulic L_turn_total)
#   (b) the visual rendering in schematics
#
# CONNECTOR_SEMICIRCLE (default):
#   L_turn = π·R_bend + max(0, pitch – 2·R_bend)
#   Valid for any pitch ≥ 2·R_bend.
#
# PURE_CIRCULAR:
#   Only allowed when abs(pitch – 2·R_bend) < tolerance.
#   L_turn = π·R_bend  (no connector legs).
#   Feasibility check rejects this turn style when pitch ≠ 2·R_bend.
#
# SMOOTH_SPLINE:
#   Approximated as a cosine-blend S-curve.
#   L_turn ≈ 1.1 × π·R_bend  (empirical factor for a smooth S transition).
#   Valid for any pitch ≥ 2·R_bend.  Slightly longer than CONNECTOR_SEMICIRCLE.

_PURE_CIRCULAR_TOL_M = 0.001   # 1 mm tolerance for pure-circular check


def _turn_length_m(
    pitch_m:      float,
    bend_radius_m: float,
    turn_style:   str,
) -> float:
    """
    Return the hydraulic centreline length of one 180-degree U-turn [m].

    Parameters
    ----------
    pitch_m       : centre-to-centre pass spacing [m]
    bend_radius_m : U-bend centreline radius [m]
    turn_style    : "CONNECTOR_SEMICIRCLE" | "PURE_CIRCULAR" | "SMOOTH_SPLINE"

    Returns
    -------
    L_turn : float   [m]
    """
    L_arc = math.pi * bend_radius_m        # semicircle arc
    L_leg = max(0.0, pitch_m - 2.0 * bend_radius_m)  # connector legs

    if turn_style == "PURE_CIRCULAR":
        # Only valid when pitch ≈ 2R; connector leg is zero.
        # Feasibility already ensures pitch ≈ 2R for this style.
        return L_arc
    elif turn_style == "SMOOTH_SPLINE":
        # Smooth S-curve approximation: 10 % longer than a pure semicircle,
        # plus the same connector legs as CONNECTOR_SEMICIRCLE.
        return 1.10 * L_arc + L_leg
    else:
        # CONNECTOR_SEMICIRCLE (default)
        return L_arc + L_leg


def _check_turn_style_feasibility(
    pitch_m:      float,
    bend_radius_m: float,
    turn_style:   str,
) -> Tuple[bool, str]:
    """
    Check whether the turn style is compatible with the given pitch/radius.

    PURE_CIRCULAR requires pitch ≈ 2·R_bend (within 1 mm).
    Other styles require pitch ≥ 2·R_bend (already checked elsewhere).
    """
    if turn_style == "PURE_CIRCULAR":
        if abs(pitch_m - 2.0 * bend_radius_m) > _PURE_CIRCULAR_TOL_M:
            return False, (
                f"PURE_CIRCULAR turn requires pitch ≈ 2×R_bend "
                f"({2*bend_radius_m*1e3:.1f} mm); "
                f"actual pitch = {pitch_m*1e3:.1f} mm.  "
                f"Use CONNECTOR_SEMICIRCLE or adjust bend radius."
            )
    return True, "OK"


# ---------------------------------------------------------------------------
# Topology 1: Horizontal serpentine  (H_SERPENTINE)
# ---------------------------------------------------------------------------
# Straight runs along the plate WIDTH (x-axis).
# Passes stacked along the plate LENGTH (y-axis).
# Manifold orientation: LEFT_RIGHT → ports on left/right edges
#                       TOP_BOTTOM  → ports on top/bottom edges (geometry rotated)

def generate_h_serpentine(
    dv: DesignVariables,
    pitch_m: float,
    L_pass_m: float,
    diameters: List[float],
) -> List[LayoutSection]:
    """
    Generate LayoutSections for a Horizontal Serpentine layout.
    Runs along x; stacked in y.
    Manifold LEFT_RIGHT: inlet port at left, outlet at right.
    Manifold TOP_BOTTOM: same routing, but plate is conceptually rotated,
                         so pass direction is along y and stacks in x.

    Each branch includes explicit inlet and outlet connector sections
    that bridge from the manifold centreline to the first/last straight pass.
    """
    sections: List[LayoutSection] = []
    PW_mm   = BATTERY_WIDTH_M  * 1e3
    PL_mm   = BATTERY_LENGTH_M * 1e3
    EDGE_mm = EDGE_OFFSET_M    * 1e3
    HINSET  = 25.0   # mm horizontal header inset
    CONN_MM = 10.0   # connector length (manifold edge → branch start)
    pitch_mm = pitch_m * 1e3

    if dv.manifold == "TOP_BOTTOM":
        run_len_mm = PL_mm - 2.0 * EDGE_mm - 25.0
        x_left  = EDGE_mm
        x_right = EDGE_mm + run_len_mm
    else:
        # LEFT_RIGHT default
        x_left  = HINSET + EDGE_mm
        x_right = PW_mm - HINSET - EDGE_mm
        run_len_mm = x_right - x_left

    # Manifold centreline x-positions (used for connector endpoints)
    x_in_ctr  = HINSET                # inlet vertical header x-centreline
    x_out_ctr = PW_mm - HINSET        # outlet vertical header x-centreline

    for b in range(dv.n_branches):
        D_first = diameters[0]
        D_last  = diameters[min(dv.n_passes - 1, len(diameters) - 1)]
        pitch_mm_b = pitch_m * 1e3
        y_first = EDGE_mm + b * dv.n_passes * pitch_mm
        y_last  = EDGE_mm + (b * dv.n_passes + dv.n_passes - 1) * pitch_mm

        # ── Inlet connector: manifold centreline → first pass start ──
        sections.append(LayoutSection(
            name       = f"B{b+1} inlet connector",
            sec_type   = "connector",
            branch_id  = b, pass_id = 0,
            length_m   = CONN_MM * 1e-3,
            diameter_m = D_first,
            x0 = x_in_ctr, y0 = y_first,
            x1 = x_left,   y1 = y_first,
        ))

        for p in range(dv.n_passes):
            D   = diameters[p]
            y_mm = EDGE_mm + (b * dv.n_passes + p) * pitch_mm

            going_right = (p % 2 == 0)
            x_from = x_left  if going_right else x_right
            x_to   = x_right if going_right else x_left

            sections.append(LayoutSection(
                name      = f"B{b+1} P{p+1} straight",
                sec_type  = "straight",
                branch_id = b, pass_id = p,
                length_m  = run_len_mm * 1e-3,
                diameter_m = D,
                x0 = x_from, y0 = y_mm,
                x1 = x_to,   y1 = y_mm,
            ))

            if p < dv.n_passes - 1:
                y_next = EDGE_mm + (b * dv.n_passes + p + 1) * pitch_mm
                rb_mm  = dv.bend_radius_m * 1e3
                L_turn = _turn_length_m(pitch_m, dv.bend_radius_m, dv.turn_style)
                bx = x_right if going_right else x_left
                sections.append(LayoutSection(
                    name      = f"B{b+1} P{p+1}→P{p+2} U-turn",
                    sec_type  = "bend",
                    branch_id = b, pass_id = p,
                    length_m  = L_turn,
                    diameter_m = D,
                    bend_radius_m = dv.bend_radius_m,
                    x0 = bx, y0 = y_mm,
                    x1 = bx, y1 = y_next,
                ))

        # ── Outlet connector: last pass end → outlet manifold centreline ──
        going_right_last = ((dv.n_passes - 1) % 2 == 0)
        x_last_end = x_right if going_right_last else x_left
        sections.append(LayoutSection(
            name       = f"B{b+1} outlet connector",
            sec_type   = "connector",
            branch_id  = b, pass_id = dv.n_passes - 1,
            length_m   = CONN_MM * 1e-3,
            diameter_m = D_last,
            x0 = x_last_end, y0 = y_last,
            x1 = x_out_ctr,  y1 = y_last,
        ))

    return sections


# ---------------------------------------------------------------------------
# Topology 2: Vertical serpentine  (V_SERPENTINE)
# ---------------------------------------------------------------------------
# Straight runs along the plate LENGTH (y-axis).
# Passes stacked along the plate WIDTH (x-axis).
# This is the 90° rotation of H_SERPENTINE.

def generate_v_serpentine(
    dv: DesignVariables,
    pitch_m: float,
    L_pass_m: float,
    diameters: List[float],
) -> List[LayoutSection]:
    """
    Generate LayoutSections for a Vertical Serpentine layout.
    Runs along y; stacked in x.
    Manifold LEFT_RIGHT: headers on left/right (short side).
    Manifold TOP_BOTTOM: headers on top/bottom (long side).

    Each branch includes explicit inlet/outlet connector sections
    bridging the gap between the manifold line and the straight-pass endpoints.
    The manifold line is drawn at y = PL_mm - VINSET (inlet, top)
    and y = VINSET (outlet, bottom).  The straight passes run between
    y_bot = VINSET + EDGE_mm and y_top = PL_mm - VINSET - EDGE_mm,
    leaving a 10 mm gap (= EDGE_mm) on each side that the connector fills.
    """
    sections: List[LayoutSection] = []
    PW_mm   = BATTERY_WIDTH_M  * 1e3
    PL_mm   = BATTERY_LENGTH_M * 1e3
    EDGE_mm = EDGE_OFFSET_M    * 1e3
    VINSET  = 25.0   # mm — vertical header inset (manifold drawn here)
    pitch_mm = pitch_m * 1e3

    # Manifold line y-coordinates (must match _draw_design_onto_ax exactly)
    y_mfld_inlet  = PL_mm - VINSET   # 395 mm — inlet manifold horizontal line
    y_mfld_outlet = VINSET            # 25  mm — outlet manifold horizontal line

    # Straight-pass y extents
    y_bot = VINSET + EDGE_mm          # 35 mm — bottom of straight run
    y_top = PL_mm - VINSET - EDGE_mm  # 385 mm — top of straight run
    run_mm = y_top - y_bot

    N_total    = dv.n_branches * dv.n_passes
    pitch_x_mm = (PW_mm - 2.0 * EDGE_mm) / N_total

    for b in range(dv.n_branches):
        D_first = diameters[0]
        D_last  = diameters[min(dv.n_passes - 1, len(diameters) - 1)]
        # x-position of the FIRST pass of this branch
        x_b0 = EDGE_mm + (b * dv.n_passes + 0) * pitch_x_mm + pitch_x_mm / 2.0
        # x-position of the LAST pass of this branch
        x_b_last = EDGE_mm + (b * dv.n_passes + dv.n_passes - 1) * pitch_x_mm + pitch_x_mm / 2.0

        # ── Inlet connector: manifold line → first straight pass start ──
        # First pass goes UP (p=0, even → going_up), so it starts at y_bot.
        # Connector drops from the inlet manifold (y_mfld_inlet) down to y_top,
        # then the straight continues from y_top up (wait — p=0 goes UP from y_bot).
        # Actually p=0 even → going_up=True → y_from=y_bot, y_to=y_top.
        # So the first straight END (y_top=385) is near the inlet manifold (395).
        # The inlet connector should be: (x_b0, y_mfld_inlet) → (x_b0, y_top)
        sections.append(LayoutSection(
            name       = f"B{b+1} inlet connector",
            sec_type   = "connector",
            branch_id  = b, pass_id = 0,
            length_m   = (y_mfld_inlet - y_top) * 1e-3,
            diameter_m = D_first,
            x0 = x_b0, y0 = y_mfld_inlet,
            x1 = x_b0, y1 = y_top,
        ))

        for p in range(dv.n_passes):
            D    = diameters[p]
            x_mm = EDGE_mm + (b * dv.n_passes + p) * pitch_x_mm + pitch_x_mm / 2.0

            going_up = (p % 2 == 0)
            y_from = y_bot if going_up else y_top
            y_to   = y_top if going_up else y_bot

            sections.append(LayoutSection(
                name      = f"B{b+1} P{p+1} straight",
                sec_type  = "straight",
                branch_id = b, pass_id = p,
                length_m  = run_mm * 1e-3,
                diameter_m = D,
                x0 = x_mm, y0 = y_from,
                x1 = x_mm, y1 = y_to,
            ))

            if p < dv.n_passes - 1:
                x_next = EDGE_mm + (b * dv.n_passes + p + 1) * pitch_x_mm + pitch_x_mm / 2.0
                L_turn = _turn_length_m(pitch_x_mm * 1e-3, dv.bend_radius_m, dv.turn_style)
                by = y_top if going_up else y_bot
                sections.append(LayoutSection(
                    name      = f"B{b+1} P{p+1}→P{p+2} U-turn",
                    sec_type  = "bend",
                    branch_id = b, pass_id = p,
                    length_m  = L_turn,
                    diameter_m = D,
                    bend_radius_m = dv.bend_radius_m,
                    x0 = x_mm,    y0 = by,
                    x1 = x_next,  y1 = by,
                ))

        # ── Outlet connector: last straight pass end → outlet manifold ──
        # Last pass index = n_passes-1.  going_up = (n_passes-1) % 2 == 0.
        # If going_up → last pass ends at y_top (near inlet manifold, wrong end).
        # If not going_up → last pass ends at y_bot (near outlet manifold, correct).
        # n_passes=5 (odd): last p=4, going_up=True → ends at y_top.
        #   So outlet connector: (x_b_last, y_bot) → (x_b_last, y_mfld_outlet)
        #   (the outlet side is the y_bot end, where the last pass *started*)
        # n_passes=4 (even): last p=3, going_up=False → ends at y_bot.
        #   Outlet connector: (x_b_last, y_bot) → (x_b_last, y_mfld_outlet)
        # In both cases the outlet is always at the y_bot end:
        sections.append(LayoutSection(
            name       = f"B{b+1} outlet connector",
            sec_type   = "connector",
            branch_id  = b, pass_id = dv.n_passes - 1,
            length_m   = (y_bot - y_mfld_outlet) * 1e-3,
            diameter_m = D_last,
            x0 = x_b_last, y0 = y_bot,
            x1 = x_b_last, y1 = y_mfld_outlet,
        ))

    return sections


# ---------------------------------------------------------------------------
# Topology 3: Mirrored U-flow  (MIRRORED_U)
# ---------------------------------------------------------------------------
# Inlet and outlet on the same end (SAME_SIDE manifold).
# First half of branches go OUT from inlet side;
# second half RETURN from far end back to the outlet side.
# In practice: all branches run parallel, full plate length,
# with a connecting U at the far end.  Inlet and outlet headers
# are both at the same (near) end, separated laterally.

def generate_mirrored_u_flow(
    dv: DesignVariables,
    pitch_m: float,
    L_pass_m: float,
    diameters: List[float],
) -> List[LayoutSection]:
    """
    Generate LayoutSections for Mirrored U-flow.
    All branches run from the inlet side to the far end and back.
    Inlet and outlet manifolds are both on the same edge.
    """
    sections: List[LayoutSection] = []
    PW_mm   = BATTERY_WIDTH_M  * 1e3
    PL_mm   = BATTERY_LENGTH_M * 1e3
    EDGE_mm = EDGE_OFFSET_M    * 1e3
    pitch_mm = pitch_m * 1e3

    # All branches span full plate length (minus edge offsets)
    run_mm = PL_mm - 2.0 * EDGE_mm - 20.0   # 20 mm for far-end turn space
    # Branches are distributed across width
    x_step = (PW_mm - 2.0 * EDGE_mm) / max(dv.n_branches, 1)
    # Each branch has n_passes passes in two legs: OUT and RETURN
    # Simple model: treat the full branch as n_passes straight sections
    # separated by U-turns, laid out along y.
    half_passes = dv.n_passes // 2
    rem_passes  = dv.n_passes - half_passes

    for b in range(dv.n_branches):
        x_mm = EDGE_mm + (b + 0.5) * x_step

        # OUT leg: passes going from y=EDGE to y=run+EDGE
        for p in range(half_passes):
            D    = diameters[p]
            y_off = p * pitch_mm
            sections.append(LayoutSection(
                name      = f"B{b+1} P{p+1} out",
                sec_type  = "straight",
                branch_id = b, pass_id = p,
                length_m  = run_mm * 1e-3,
                diameter_m = D,
                x0 = x_mm - pitch_mm * 0.25, y0 = EDGE_mm + y_off,
                x1 = x_mm - pitch_mm * 0.25, y1 = EDGE_mm + y_off + run_mm,
            ))
            if p < half_passes - 1:
                L_turn = _turn_length_m(pitch_m, dv.bend_radius_m, dv.turn_style)
                sections.append(LayoutSection(
                    name      = f"B{b+1} P{p+1}→P{p+2} U-turn",
                    sec_type  = "bend",
                    branch_id = b, pass_id = p,
                    length_m  = L_turn, diameter_m = D,
                    bend_radius_m = dv.bend_radius_m,
                    x0 = x_mm - pitch_mm*0.25, y0 = EDGE_mm + y_off + run_mm,
                    x1 = x_mm + pitch_mm*0.25, y1 = EDGE_mm + y_off + run_mm,
                ))

        # Far-end U-turn connecting OUT to RETURN
        D_mid = diameters[half_passes - 1]
        L_far_turn = _turn_length_m(pitch_m, dv.bend_radius_m, dv.turn_style)
        sections.append(LayoutSection(
            name      = f"B{b+1} far-end U-turn",
            sec_type  = "bend",
            branch_id = b, pass_id = half_passes - 1,
            length_m  = L_far_turn, diameter_m = D_mid,
            bend_radius_m = dv.bend_radius_m,
            x0 = x_mm - pitch_mm*0.25, y0 = EDGE_mm + run_mm,
            x1 = x_mm + pitch_mm*0.25, y1 = EDGE_mm + run_mm,
        ))

        # RETURN leg
        for p2 in range(rem_passes):
            p_idx = half_passes + p2
            D     = diameters[min(p_idx, len(diameters) - 1)]
            y_off = p2 * pitch_mm
            sections.append(LayoutSection(
                name      = f"B{b+1} P{p_idx+1} return",
                sec_type  = "straight",
                branch_id = b, pass_id = p_idx,
                length_m  = run_mm * 1e-3,
                diameter_m = D,
                x0 = x_mm + pitch_mm*0.25, y0 = EDGE_mm + run_mm - y_off,
                x1 = x_mm + pitch_mm*0.25, y1 = EDGE_mm - y_off,
            ))

    return sections


# ---------------------------------------------------------------------------
# Topology 4: Z-flow  (Z_FLOW)
# ---------------------------------------------------------------------------
# Inlet and outlet on opposite corners.
# Branches run from one end to the other in the same direction.
# The Z-shape comes from: inlet at bottom-left, outlet at top-right,
# with header routing at each end creating the Z form.

def generate_z_flow(
    dv: DesignVariables,
    pitch_m: float,
    L_pass_m: float,
    diameters: List[float],
) -> List[LayoutSection]:
    """
    Generate LayoutSections for Z-flow.
    All branches run parallel in the same direction (no U-turns between branches).
    Inlet port at bottom-left, outlet at top-right.
    The 'turns' in Z-flow are within each branch (n_passes serpentine passes).
    """
    sections: List[LayoutSection] = []
    PW_mm   = BATTERY_WIDTH_M  * 1e3
    PL_mm   = BATTERY_LENGTH_M * 1e3
    EDGE_mm = EDGE_OFFSET_M    * 1e3
    pitch_mm = pitch_m * 1e3
    INSET_mm = 20.0

    # Z-flow: all branches run along x (width direction).
    # Inlet header along bottom, outlet header along top.
    # Branches stacked in y.  Each branch has n_passes serpentine passes in x.
    run_mm  = PW_mm - 2.0 * INSET_mm - 2.0 * EDGE_mm
    N_total = dv.n_branches * dv.n_passes

    for b in range(dv.n_branches):
        for p in range(dv.n_passes):
            D    = diameters[p]
            y_mm = EDGE_mm + (b * dv.n_passes + p) * pitch_mm

            going_right = (p % 2 == 0)
            x_from = EDGE_mm + INSET_mm if going_right else PW_mm - EDGE_mm - INSET_mm
            x_to   = PW_mm - EDGE_mm - INSET_mm if going_right else EDGE_mm + INSET_mm

            sections.append(LayoutSection(
                name      = f"B{b+1} P{p+1} straight",
                sec_type  = "straight",
                branch_id = b, pass_id = p,
                length_m  = run_mm * 1e-3,
                diameter_m = D,
                x0 = x_from, y0 = y_mm,
                x1 = x_to,   y1 = y_mm,
            ))

            if p < dv.n_passes - 1:
                y_next = EDGE_mm + (b * dv.n_passes + p + 1) * pitch_mm
                L_turn = _turn_length_m(pitch_m, dv.bend_radius_m, dv.turn_style)
                bx = x_to
                sections.append(LayoutSection(
                    name      = f"B{b+1} P{p+1}→P{p+2} U-turn",
                    sec_type  = "bend",
                    branch_id = b, pass_id = p,
                    length_m  = L_turn, diameter_m = D,
                    bend_radius_m = dv.bend_radius_m,
                    x0 = bx, y0 = y_mm,
                    x1 = bx, y1 = y_next,
                ))

    return sections


# ---------------------------------------------------------------------------
# Topology 5: Central Inlet / Edge Outlet  (CENTRAL_INLET)
# ---------------------------------------------------------------------------
# Inlet manifold runs along the centreline of the plate (mid-length).
# Branches split symmetrically: half go toward the top edge, half toward bottom.
# Both outlet manifolds collect at the top and bottom edges respectively.
# CENTER_EDGE manifold: inlet along centre, outlets at both edges.

def generate_central_inlet(
    dv: DesignVariables,
    pitch_m: float,
    L_pass_m: float,
    diameters: List[float],
) -> List[LayoutSection]:
    """
    Generate LayoutSections for Central Inlet layout.
    Branches split symmetrically from a central manifold toward both edges.
    """
    sections: List[LayoutSection] = []
    PW_mm   = BATTERY_WIDTH_M  * 1e3
    PL_mm   = BATTERY_LENGTH_M * 1e3
    EDGE_mm = EDGE_OFFSET_M    * 1e3
    pitch_mm = pitch_m * 1e3

    y_centre_mm = PL_mm / 2.0   # central manifold y-position
    # Each branch runs from the centre outward, then serpentines back/forward
    # Upper half: branches go toward y = PL_mm
    # Lower half: branches go toward y = 0
    nb_upper = dv.n_branches // 2
    nb_lower = dv.n_branches - nb_upper

    x_step = (PW_mm - 2.0 * EDGE_mm) / max(dv.n_branches, 1)
    run_to_edge = PL_mm / 2.0 - EDGE_mm - 10.0   # run length from centre to edge

    b_global = 0
    for side, n_side in [(+1, nb_upper), (-1, nb_lower)]:
        for bs in range(n_side):
            x_mm = EDGE_mm + (b_global + 0.5) * x_step

            for p in range(dv.n_passes):
                D    = diameters[p]
                p_offset = p * pitch_mm
                y_start  = y_centre_mm + side * p_offset
                y_end    = y_centre_mm + side * (p_offset + run_to_edge / max(dv.n_passes, 1))

                # Clamp to plate
                y_start = max(EDGE_mm, min(PL_mm - EDGE_mm, y_start))
                y_end   = max(EDGE_mm, min(PL_mm - EDGE_mm, y_end))

                sections.append(LayoutSection(
                    name      = f"B{b_global+1} P{p+1} {'upper' if side>0 else 'lower'}",
                    sec_type  = "straight",
                    branch_id = b_global, pass_id = p,
                    length_m  = abs(y_end - y_start) * 1e-3,
                    diameter_m = D,
                    x0 = x_mm, y0 = y_start,
                    x1 = x_mm, y1 = y_end,
                ))

                if p < dv.n_passes - 1:
                    L_turn = _turn_length_m(pitch_m, dv.bend_radius_m, dv.turn_style)
                    sections.append(LayoutSection(
                        name      = f"B{b_global+1} P{p+1}→P{p+2} turn",
                        sec_type  = "bend",
                        branch_id = b_global, pass_id = p,
                        length_m  = L_turn, diameter_m = D,
                        bend_radius_m = dv.bend_radius_m,
                        x0 = x_mm, y0 = y_end,
                        x1 = x_mm + pitch_mm * 0.5, y1 = y_end,
                    ))

            b_global += 1

    return sections


# ---------------------------------------------------------------------------
# Dispatcher: generate_layout_sections
# ---------------------------------------------------------------------------

def generate_layout_sections(
    dv:        DesignVariables,
    pitch_m:   float,
    L_pass_m:  float,
    diameters: List[float],
) -> List[LayoutSection]:
    """
    Dispatch to the appropriate topology generator based on dv.topology.

    Returns a list of LayoutSection objects describing the full routing.
    """
    t = dv.topology
    if t == "H_SERPENTINE":
        return generate_h_serpentine(dv, pitch_m, L_pass_m, diameters)
    elif t == "V_SERPENTINE":
        return generate_v_serpentine(dv, pitch_m, L_pass_m, diameters)
    elif t == "MIRRORED_U":
        return generate_mirrored_u_flow(dv, pitch_m, L_pass_m, diameters)
    elif t == "Z_FLOW":
        return generate_z_flow(dv, pitch_m, L_pass_m, diameters)
    elif t == "CENTRAL_INLET":
        return generate_central_inlet(dv, pitch_m, L_pass_m, diameters)
    else:
        # Fallback to horizontal serpentine
        return generate_h_serpentine(dv, pitch_m, L_pass_m, diameters)


def topology_complexity_penalty(topology: str, turn_style: str) -> float:
    """
    Return a small complexity penalty in [0, 1] for a topology/turn-style combo.

    Simpler layouts get 0; more complex routing gets a small positive value.
    This prevents the optimizer blindly favouring the most geometrically
    unconstrained topology unless it genuinely performs better thermally.

    Penalty table (approximate)
    ---------------------------
    H_SERPENTINE  + CONNECTOR_SEMICIRCLE  → 0.00  (baseline)
    H_SERPENTINE  + PURE_CIRCULAR         → 0.05
    H_SERPENTINE  + SMOOTH_SPLINE         → 0.08
    V_SERPENTINE  + any                   → 0.05  (90° reorientation)
    MIRRORED_U    + any                   → 0.12  (same-side manifold, U-far)
    Z_FLOW        + any                   → 0.10  (diagonal header routing)
    CENTRAL_INLET + any                   → 0.15  (bifurcated manifold)
    """
    base = {
        "H_SERPENTINE":  0.00,
        "V_SERPENTINE":  0.05,
        "MIRRORED_U":    0.12,
        "Z_FLOW":        0.10,
        "CENTRAL_INLET": 0.15,
    }.get(topology, 0.10)

    style_add = {
        "CONNECTOR_SEMICIRCLE": 0.00,
        "PURE_CIRCULAR":        0.05,
        "SMOOTH_SPLINE":        0.08,
    }.get(turn_style, 0.05)

    return min(1.0, base + style_add)


# ============================================================
# (existing code continues below)
# ============================================================

def check_geometry_feasibility(
    dv:       DesignVariables,
    pitch:    float,
    L_pass:   float,
    diameters: List[float],
) -> Tuple[bool, str]:
    """
    Check whether the candidate geometry satisfies all packaging and
    manufacturing constraints.  Returns (feasible, reason_string).

    Checks performed
    ----------------
    0.  Topology–manifold compatibility.
    1.  Straight pass length must be positive and fit within battery.
    2.  Channel pitch must exceed minimum spacing.
    3.  Bend radius must be positive and ≥ 0.5 × max diameter (min bend rule).
    4.  All diameters within [D_MIN_M, D_MAX_M].
    5.  In 3-zone mode, diameter must be non-increasing (physical taper).
    6.  Diameter ≤ pitch (channel must fit within its allocated pitch).
    7.  Total footprint of all passes must fit within battery.
    8.  Pitch vs bend-radius geometric consistency.
    9.  Turn-style compatibility with pitch/radius.  (v8)
    """
    # 0. Topology–manifold compatibility  (v8)
    compat = _TOPO_MANIFOLD_COMPAT.get((dv.topology, dv.manifold), False)
    if not compat:
        return False, (
            f"Topology '{dv.topology}' is incompatible with manifold "
            f"placement '{dv.manifold}'."
        )

    # 1. Pass length — relaxed for topologies that use a different active axis
    # V_SERPENTINE and CENTRAL_INLET use the plate length axis as runs,
    # so the pass-length limit is against BATTERY_LENGTH_M, not WIDTH.
    if dv.topology in ("V_SERPENTINE", "CENTRAL_INLET", "MIRRORED_U"):
        max_run = BATTERY_LENGTH_M - 2.0 * EDGE_OFFSET_M
    else:
        max_run = BATTERY_WIDTH_M - 2.0 * EDGE_OFFSET_M
    if L_pass <= 0.0:
        return False, "Straight pass length ≤ 0"
    if L_pass > max_run:
        return False, "Pass length exceeds available plate dimension"

    # 2. Minimum pitch
    if pitch < MIN_CHANNEL_PITCH_M:
        return False, f"Channel pitch {pitch*1e3:.1f} mm < min {MIN_CHANNEL_PITCH_M*1e3:.1f} mm"

    # 3a. Channel diameter must fit physically inside the plate thickness.
    #     Manufacturing rule: D <= 0.75 * t_plate (applies to INTERNAL branch channels;
    #     inlet/outlet port fittings are external features and are not limited here).
    #     With PLATE_THICKNESS_M = 10 mm → max internal channel D = 7.5 mm.
    DIAM_PLATE_RATIO = 0.75   # manufacturability fraction; edit here to relax/tighten
    max_D_allowed = DIAM_PLATE_RATIO * PLATE_THICKNESS_M
    max_D_given   = max(diameters)
    if max_D_given > max_D_allowed:
        return False, (
            f"Channel diameter {max_D_given*1e3:.1f} mm exceeds "
            f"{DIAM_PLATE_RATIO*100:.0f}% of plate thickness "
            f"({max_D_allowed*1e3:.1f} mm).  Reduce D or increase plate thickness."
        )

    # 3b. Bend-radius check
    if dv.bend_radius_m <= 0.0:
        return False, "Bend radius <= 0"
    max_D = max(diameters)
    if dv.bend_radius_m < 0.5 * max_D:
        return False, (f"Bend radius {dv.bend_radius_m*1e3:.1f} mm < 0.5 x D = "
                       f"{0.5*max_D*1e3:.1f} mm")

    # 4. Diameter bounds
    for D in diameters:
        if D < D_MIN_M:
            return False, f"Diameter {D*1e3:.1f} mm < D_min {D_MIN_M*1e3:.1f} mm"
        if D > D_MAX_M:
            return False, f"Diameter {D*1e3:.1f} mm > D_max {D_MAX_M*1e3:.1f} mm"

    # 5. Non-increasing diameter in 3-zone mode (optional physical constraint)
    if dv.use_3zone:
        if not (dv.D1_m >= dv.D2_m >= dv.D3_m):
            return False, "3-zone diameters not non-increasing (D1≥D2≥D3 required)"

    # 6. Diameter vs pitch
    for D in diameters:
        if D > pitch:
            return False, (f"Diameter {D*1e3:.1f} mm > pitch {pitch*1e3:.1f} mm "
                           f"(channels would overlap)")

    # 7. Total footprint — axis depends on topology
    # v11: N passes share N-1 gaps, so span = (N-1)*pitch.
    # Outermost passes sit at EDGE + D_mean from plate edge.
    N_total_passes = dv.n_branches * dv.n_passes
    D_mean_fp      = float(np.mean(diameters)) if diameters else 0.004
    n_gaps         = max(N_total_passes - 1, 1)
    total_footprint = n_gaps * pitch + 2.0 * EDGE_OFFSET_M + 2.0 * D_mean_fp
    if dv.topology in ("V_SERPENTINE",):
        limit = BATTERY_WIDTH_M
    elif dv.topology in ("CENTRAL_INLET",):
        limit = BATTERY_LENGTH_M
    else:
        limit = BATTERY_LENGTH_M
    if total_footprint > limit * 1.02:   # 2 % tolerance for rounding
        return False, (f"Total channel footprint {total_footprint*1e3:.0f} mm > "
                       f"plate limit {limit*1e3:.0f} mm")

    # 8. Pitch vs bend-radius geometric consistency check.
    if pitch < 2.0 * dv.bend_radius_m:
        return False, (
            f"Bend radius {dv.bend_radius_m*1e3:.1f} mm too large for pass pitch "
            f"{pitch*1e3:.1f} mm.  Required: pitch >= 2 x R_bend = "
            f"{2.0*dv.bend_radius_m*1e3:.1f} mm.  "
            f"Reduce bend radius or increase n_passes."
        )

    # 9. Turn-style compatibility  (v8)
    ts_ok, ts_reason = _check_turn_style_feasibility(pitch, dv.bend_radius_m, dv.turn_style)
    if not ts_ok:
        return False, ts_reason

    return True, "OK"


def compute_coverage_ratio(dv: DesignVariables, pitch: float, L_pass: float) -> float:
    """
    Compute a simple thermal coverage ratio:
        coverage = (total wetted footprint area) / (battery plan area)

    This is a proxy for how uniformly the coolant covers the battery surface.
    Values close to 1 indicate good coverage; values < 0.5 suggest poor cooling.

    Parameters
    ----------
    dv      : DesignVariables
    pitch   : float   channel pitch [m]
    L_pass  : float   straight pass length [m]

    Returns
    -------
    coverage : float  (dimensionless, 0 to 1)
    """
    battery_area = BATTERY_LENGTH_M * BATTERY_WIDTH_M
    # Assume each channel influences a strip of width = pitch along its length
    covered_area = dv.n_branches * dv.n_passes * pitch * L_pass
    coverage = min(covered_area / battery_area, 1.0)
    return coverage


# =============================================================================
# 7. PER-SEGMENT CALCULATIONS
# =============================================================================

def friction_factor(Re: float) -> float:
    """
    Darcy–Weisbach friction factor f.

    Regimes
    -------
    Re < 2300   → laminar:       f = 64 / Re
    Re ≥ 2300   → Blasius (smooth pipe):
                                  f = 0.316 / Re^0.25   (valid for Re ≤ 1e5)
    Re > 1e5    → Swamee-Jain approximation with assumed roughness ε = 1e-5 m
                  (mild turbulence, kept for robustness)

    Parameters
    ----------
    Re : float
        Reynolds number (must be > 0).

    Returns
    -------
    f : float
        Darcy–Weisbach friction factor.
    """
    if Re <= 0:
        return 64.0   # fall-back; prevents divide-by-zero
    if Re < 2300:
        # Laminar Hagen-Poiseuille
        return 64.0 / Re
    elif Re <= 1e5:
        # Blasius correlation for smooth pipes
        return 0.316 / (Re ** 0.25)
    else:
        # Approximation for fully turbulent smooth pipe
        return 0.0032 + 0.221 / (Re ** 0.237)


def nusselt_number(Re: float, Pr: float, phi: float = PHI_SERPENTINE) -> float:
    """
    Effective Nusselt number for internal channel flow.

    Model
    -----
    Laminar  (Re < 2300):
        Nu_base = NU_LAM_CONST (default 4.36, uniform heat-flux fully developed)
        Nu_eff  = phi × Nu_base

    Turbulent (Re ≥ 10 000):
        Dittus-Boelter:  Nu_base = 0.023 × Re^0.8 × Pr^0.4
        Nu_eff = phi × Nu_base   (phi should be close to 1 in turbulent regime
                                   because turbulent mixing already dominates)

    Transitional (2300 ≤ Re < 10 000):
        Linear interpolation between laminar and turbulent values.

    The serpentine enhancement factor phi accounts for the secondary flows
    induced by U-bends that increase near-wall mixing.  A value of phi = 1.6
    is a reasonable engineering estimate; calibrate against CFD later.

    Parameters
    ----------
    Re  : float  Reynolds number
    Pr  : float  Prandtl number
    phi : float  Enhancement factor (default = PHI_SERPENTINE)

    Returns
    -------
    Nu_eff : float
    """
    if Re <= 0:
        return phi * NU_LAM_CONST

    # Laminar Nu
    Nu_lam = phi * NU_LAM_CONST

    # Turbulent Dittus-Boelter Nu (heating the fluid)
    # Apply phi at reduced strength in turbulent regime (already well mixed).
    phi_turb = 1.0 + (phi - 1.0) * 0.3   # partial enhancement in turbulent regime
    Nu_turb  = phi_turb * 0.023 * (Re ** 0.8) * (Pr ** 0.4)

    if Re < 2300:
        return Nu_lam
    elif Re >= 10_000:
        return Nu_turb
    else:
        # Linear blend in transition zone
        alpha = (Re - 2300.0) / (10_000.0 - 2300.0)
        return (1.0 - alpha) * Nu_lam + alpha * Nu_turb


def compute_segment_flow(
    seg_id:       int,
    length_m:     float,
    diameter_m:   float,
    m_dot:        float,
    fluid:        FluidProperties,
    n_bends_after: int = 0,
) -> SegmentResult:
    """
    Compute all thermal-hydraulic quantities for a single straight segment.

    Parameters
    ----------
    seg_id          : int    segment index (0-based)
    length_m        : float  segment length [m]
    diameter_m      : float  channel hydraulic diameter [m]
    m_dot           : float  mass flow rate in this segment [kg/s]
    fluid           : FluidProperties
    n_bends_after   : int    number of U-bends at the *end* of this segment
                             (typically 0 or 1 for a serpentine)

    Returns
    -------
    seg : SegmentResult
    """
    seg = SegmentResult(seg_id=seg_id, length_m=length_m, diameter_m=diameter_m)

    # ------------------------------------------------------------------
    # Cross-sectional and wetted areas (circular channel)
    # A_cs    = π/4 × D²
    # A_wet   = π × D × L   (lateral surface area = perimeter × length)
    # ------------------------------------------------------------------
    if diameter_m <= 0:
        seg.flag = "WARN: diameter ≤ 0; segment skipped"
        return seg

    A_cs  = math.pi / 4.0 * diameter_m ** 2           # m²
    A_wet = math.pi * diameter_m * length_m             # m²

    seg.area_m2        = A_cs
    seg.wetted_area_m2 = A_wet

    # ------------------------------------------------------------------
    # Velocity:  V = m_dot / (ρ × A)
    # ------------------------------------------------------------------
    if A_cs <= 0 or fluid.rho <= 0:
        seg.flag = "WARN: zero area or density"
        return seg

    V = m_dot / (fluid.rho * A_cs)   # m/s
    seg.velocity_m_s = V

    # ------------------------------------------------------------------
    # Reynolds number:  Re = ρ V D / μ
    # ------------------------------------------------------------------
    Re = fluid.rho * V * diameter_m / fluid.mu
    seg.Re = Re

    if Re < 1.0:
        seg.flag = "WARN: Re < 1 (very low flow)"

    # ------------------------------------------------------------------
    # Nusselt number (enhanced by phi for serpentine mixing)
    # ------------------------------------------------------------------
    Nu = nusselt_number(Re, fluid.Pr, phi=PHI_SERPENTINE)
    seg.Nu = Nu

    # ------------------------------------------------------------------
    # Heat transfer coefficient:  h = Nu × k_fluid / D
    # ------------------------------------------------------------------
    h = Nu * fluid.k / diameter_m     # W/(m²·K)
    seg.h_W_m2K = h

    # ------------------------------------------------------------------
    # Convective resistance:  R_conv = 1 / (h × A_wet)
    # ------------------------------------------------------------------
    if A_wet > 0 and h > 0:
        R_conv = 1.0 / (h * A_wet)   # K/W
    else:
        R_conv = 1e6
        seg.flag += " WARN: zero h or A_wet"

    seg.R_conv_K_W = R_conv

    # ------------------------------------------------------------------
    # Plate conduction resistance:
    #     R_cond = t_plate / (k_plate × A_effective)
    # A_effective = footprint strip of width = diameter centred on channel
    # This is a simple 1-D normal path through the plate.
    # ------------------------------------------------------------------
    mat = get_material_properties()
    A_effective = diameter_m * length_m   # m²  effective conduction area
    if A_effective > 0:
        R_cond = PLATE_THICKNESS_M / (mat.k_solid_W_mK * A_effective)  # K/W
    else:
        R_cond = 1e6

    seg.R_cond_K_W = R_cond

    # ------------------------------------------------------------------
    # Total segment resistance:
    #     R_total = R_cond + R_conv   (series path: battery → plate → fluid)
    # ------------------------------------------------------------------
    R_total = R_cond + R_conv
    seg.R_total_K_W = R_total

    # ------------------------------------------------------------------
    # Coolant temperature rise in this segment:
    #     Estimated as: ΔT = Q_seg / (m_dot × cp)
    # Here Q_seg is estimated from the segment convective resistance:
    #     We'll compute this in the branch evaluator using the real Q split.
    # Store placeholder 0; updated in evaluate_branch().
    # ------------------------------------------------------------------
    seg.dT_coolant_K = 0.0   # updated later

    # ------------------------------------------------------------------
    # Frictional pressure drop (Darcy–Weisbach):
    #     ΔP_f = f × (L/D) × (ρ V² / 2)
    # ------------------------------------------------------------------
    f_factor   = friction_factor(Re)
    dP_friction = f_factor * (length_m / diameter_m) * (fluid.rho * V ** 2 / 2.0)
    seg.dP_friction_Pa = dP_friction

    # ------------------------------------------------------------------
    # Bend pressure drop at end of segment:
    #     ΔP_bend = n_bends × K_bend_eff × (ρ V² / 2)
    # K_bend_eff is scaled by bend tightness: tighter bends have higher K.
    #     K_bend_eff = K_BEND × clamp(D / (2 × R_bend), 0.4, 1.8)
    # When R_bend is not available in segment context, use K_BEND directly.
    # Note: R_bend is passed via the branch evaluator below, not here.
    # This term captures the 180° U-turn minor loss only.
    # ------------------------------------------------------------------
    dP_bend = n_bends_after * K_BEND * (fluid.rho * V ** 2 / 2.0)
    seg.dP_bend_Pa = dP_bend

    return seg


# =============================================================================
# 8. BRANCH EVALUATION
# =============================================================================

def evaluate_branch(
    branch_id:  int,
    dv:         DesignVariables,
    m_dot_total: float,
    Q_total_W:  float,
    fluid:      FluidProperties,
    L_pass:     float,
    diameters:  List[float],
    best_pitch: float,
    m_dot_override: Optional[float] = None,
) -> BranchResult:
    """
    Evaluate one parallel branch of the serpentine cooling plate.

    v9: accepts optional m_dot_override so each branch can carry its
    physically allocated flow rate from allocate_branch_flows().
    When m_dot_override is None, falls back to equal split (original behaviour).

    Parameters
    ----------
    branch_id      : int
    dv             : DesignVariables
    m_dot_total    : float   total system mass flow [kg/s]
    Q_total_W      : float   total battery heat load [W]
    fluid          : FluidProperties
    L_pass         : float   straight pass length [m]
    diameters      : List[float]   diameter per pass [m]
    best_pitch     : float   centre-to-centre pass pitch [m]
    m_dot_override : Optional[float]  branch-specific flow [kg/s] (v9 unequal split)

    Returns
    -------
    result : BranchResult
    """
    result = BranchResult(branch_id=branch_id)

    # v9: use provided flow override if given, otherwise equal split.
    if m_dot_override is not None:
        m_dot_branch = m_dot_override
    else:
        m_dot_branch = m_dot_total / dv.n_branches
    result.m_dot_branch_kg_s = m_dot_branch

    # Each branch removes 1/n_branches of total heat (uniform assumption).
    Q_branch = Q_total_W / dv.n_branches

    # ------------------------------------------------------------------
    # Pre-compute the U-bend arc length.
    # A 180-degree U-bend with centreline radius R_b traces a semicircle:
    #     L_bend = pi * R_bend
    # This contributes to total channel length and frictional pressure drop.
    # There are (n_passes - 1) bends per branch.
    #
    # FIX (v4): Full U-turn length = arc length + connector leg length.
    #
    #   Geometry of a U-turn connecting two pass centrelines at pitch p, radius R:
    #
    #   Case pitch == 2*R  (pure semicircle):
    #     L_arc       = pi * R
    #     L_connector = 0
    #     L_turn      = pi * R
    #
    #   Case pitch > 2*R  (semicircle + two straight legs):
    #     L_arc       = pi * R         (the 180-deg semicircle)
    #     L_connector = pitch - 2*R    (total connector leg, split evenly either side)
    #     L_turn      = pi * R + (pitch - 2*R)
    #
    #   Case pitch < 2*R:  infeasible – rejected in check_geometry_feasibility().
    #
    #   Both L_arc and L_connector contribute to frictional pressure drop.
    #   The minor loss K_bend is applied once for the 180-deg turn regardless.
    # ------------------------------------------------------------------
    pitch_m         = best_pitch   # passed in from evaluate_design; see below
    # v8: Use _turn_length_m() to handle all turn styles consistently.
    # CONNECTOR_SEMICIRCLE: L_arc + L_leg  (original behaviour)
    # PURE_CIRCULAR:        L_arc only
    # SMOOTH_SPLINE:        1.10 * L_arc + L_leg
    L_turn_total    = _turn_length_m(pitch_m, dv.bend_radius_m, dv.turn_style)

    # ------------------------------------------------------------------
    # FIX (v3): Area-weighted heat distribution per segment.
    #
    # In v2 each segment received an equal share: Q_seg = Q_branch / n_passes.
    # However, segments may have different diameters (3-zone mode) and
    # therefore different wetted areas.  A more physically consistent
    # approach is to weight the heat removal by wetted area:
    #
    #     A_wet_seg = pi * D_seg * L_pass
    #     Q_seg     = Q_branch * (A_wet_seg / A_wet_total_branch)
    #
    # For constant-diameter designs this is identical to equal splitting.
    # For 3-zone designs it correctly puts more heat removal in the
    # larger-area (larger-diameter) segments.
    # ------------------------------------------------------------------
    # Compute wetted area for each pass first
    A_wet_per_pass = [math.pi * D * L_pass for D in diameters]   # m²
    A_wet_total    = sum(A_wet_per_pass)

    if A_wet_total > 0:
        Q_per_pass = [Q_branch * (A / A_wet_total) for A in A_wet_per_pass]
    else:
        # Fallback: equal split
        Q_per_pass = [Q_branch / dv.n_passes] * dv.n_passes

    T_coolant    = T_INLET_C   # coolant bulk temperature tracks along branch
    total_dP     = 0.0
    total_length = 0.0
    segments     = []

    # ------------------------------------------------------------------
    # v11 FIX 1: Bend-radius-scaled K for U-turns.
    # K_bend_eff = K_BEND × clamp(D_mean / (2 × R_bend), 0.4, 1.8)
    # Tighter bends (small R/D) → higher loss.
    # Applied once per 180° U-turn.
    # ------------------------------------------------------------------
    D_mean_branch = float(np.mean(diameters)) if diameters else dv.D_const_m
    R_bend        = max(dv.bend_radius_m, 1e-4)
    tightness     = min(max(D_mean_branch / (2.0 * R_bend), 0.4), 1.8)
    K_bend_eff    = K_BEND * tightness

    # ------------------------------------------------------------------
    # v11 FIX 1: Branch junction losses.
    # At the inlet: flow splits from manifold into each branch.
    # At the outlet: flow merges from each branch into manifold.
    # ΔP_junction = K_JUNCTION_BRANCH × (ρ V_branch² / 2)  (applied twice: in + out)
    # Computed using the inlet-pass velocity.
    # Added to total_dP before and after the pass loop.
    # ------------------------------------------------------------------
    D_inlet = diameters[0] if diameters else dv.D_const_m
    A_cs_inlet = math.pi / 4.0 * D_inlet ** 2
    V_inlet    = m_dot_branch / (fluid.rho * max(A_cs_inlet, 1e-12))
    dP_junc_in = K_JUNCTION_BRANCH * (fluid.rho * V_inlet ** 2 / 2.0)
    total_dP  += dP_junc_in

    # ------------------------------------------------------------------
    # FIX (v3): Entrance loss at branch inlet.
    # Applied once, using the velocity in the first straight segment.
    # K_entry = 0.5 for a sharp-edged entry from manifold header.
    # ΔP_entry = K_entry * (rho * V^2 / 2)
    # ------------------------------------------------------------------
    dP_entry = K_ENTRY * (fluid.rho * V_inlet ** 2 / 2.0)
    total_dP += dP_entry

    for pass_idx in range(dv.n_passes):
        D        = diameters[pass_idx]   # local diameter for this pass (tapered or const)
        Q_seg    = Q_per_pass[pass_idx]

        # A U-bend follows every pass except the last
        n_bends = 1 if pass_idx < (dv.n_passes - 1) else 0

        seg = compute_segment_flow(
            seg_id        = pass_idx,
            length_m      = L_pass,
            diameter_m    = D,           # Fix 3: local D for Re, h, dP
            m_dot         = m_dot_branch,
            fluid         = fluid,
            n_bends_after = 0,           # bend losses added explicitly below
        )

        # ------------------------------------------------------------------
        # v11: Coolant temperature rise uses the correct local Q_seg per pass
        # (wetted-area weighted).  This is Fix 2+3: each pass gets its own
        # h, Re, Nu from local D, and its own dT from local Q_seg.
        # ------------------------------------------------------------------
        if fluid.cp > 0 and m_dot_branch > 0:
            dT_coolant = Q_seg / (m_dot_branch * fluid.cp)
        else:
            dT_coolant = 0.0
        seg.dT_coolant_K = dT_coolant
        T_coolant += dT_coolant

        # Accumulate straight-segment friction pressure drop
        total_dP    += seg.dP_friction_Pa
        total_length += L_pass

        # ------------------------------------------------------------------
        # U-turn losses after this pass (if not last pass).
        # v11: use radius-scaled K_bend_eff; also add Darcy friction in arc.
        # ΔP_bend_total = K_bend_eff × dyn   (minor loss for 180° turn)
        #               + f × (L_turn/D) × dyn   (Darcy friction in arc)
        # ------------------------------------------------------------------
        if n_bends > 0 and L_turn_total > 0 and D > 0:
            dyn_p        = fluid.rho * seg.velocity_m_s ** 2 / 2.0
            # Minor loss (K-based)
            dP_bend_minor = K_bend_eff * dyn_p
            # Darcy friction in arc pipe
            f_bend        = friction_factor(seg.Re)
            dP_turn_darcy = f_bend * (L_turn_total / D) * dyn_p
            total_dP    += dP_bend_minor + dP_turn_darcy
            total_length += L_turn_total

        segments.append(seg)

    # ------------------------------------------------------------------
    # FIX (v3): Exit loss at branch outlet.
    # Applied once, using the velocity in the last straight segment.
    # K_exit = 1.0 for sudden expansion into outlet manifold (Borda-Carnot).
    # ΔP_exit = K_exit * (rho * V_last^2 / 2)
    #
    # v11: Also add outlet junction loss (merge from branch into manifold).
    # ΔP_junc_out = K_JUNCTION_BRANCH × (ρ V_last² / 2)
    # ------------------------------------------------------------------
    if segments:
        last_seg = segments[-1]
        if last_seg.velocity_m_s > 0:
            dyn_out     = fluid.rho * last_seg.velocity_m_s ** 2 / 2.0
            dP_exit     = K_EXIT             * dyn_out
            dP_junc_out = K_JUNCTION_BRANCH  * dyn_out
            total_dP   += dP_exit + dP_junc_out

    result.segments        = segments
    result.T_in_C          = T_INLET_C
    result.T_out_C         = T_coolant
    result.dP_total_Pa     = total_dP
    result.Q_absorbed_W    = Q_branch
    result.total_length_m  = total_length

    return result


# =============================================================================
# 9. FULL DESIGN EVALUATION
# =============================================================================

# =============================================================================
# 9a-NEW.  PHYSICAL TOPOLOGY LAYER  (v9)
# =============================================================================
# This layer replaces the old geometric-proxy manifold penalty with a
# physically grounded set of functions that:
#   (1) estimate topology-specific manifold lengths and connector lengths
#   (2) compute per-branch hydraulic resistances
#   (3) allocate unequal flow among branches
#   (4) evaluate per-branch thermal performance
#   (5) compute a physically grounded uniformity penalty
#
# The model is still 1D and screening-level.  It does NOT solve a full
# hydraulic network.  The key modelling choices are:
#
#   • Manifold pipe resistance  = f × (L_manifold / D_hdr) × ρ V² / 2
#     where D_hdr = sqrt(4 × A_manifold / π)   (hydraulic diameter)
#
#   • Branch connector resistance = f × (L_connector / D) × ρ V² / 2
#     expressed as a linear resistance coeff  R_conn = f × L_conn / D_ch
#     (scaled to mass-flow units: [Pa·s/m³] × A² / ρ)
#
#   • Position bias along the manifold creates a pressure gradient
#     Δp_bias_i = ρ V_hdr² / 2 × position_factor_i
#     Near branches (i=0) see full header dynamic pressure; far branches see less.
#
#   • Flow allocation:  m_dot_i ∝ 1 / R_eff_i
#     where R_eff_i = R_branch + R_connector_i + R_bias_i
#
#   Simplifications:
#   • Manifold diameter is fixed from MANIFOLD_INLET_WIDTH_M (used as D_hdr proxy).
#   • Connector diameter equals branch channel diameter.
#   • No iterative solver — one-pass allocation.
# =============================================================================


@dataclass
class ManifoldGeometry:
    """
    Estimated manifold and connector lengths for a given topology/manifold combo.

    All lengths in metres.  These are routing estimates, not precision CAD.
    They are used to compute manifold friction losses and branch connector
    resistances that feed the unequal flow-split surrogate.

    Fields
    ------
    inlet_manifold_length_m  : length of inlet header pipe
    outlet_manifold_length_m : length of outlet header pipe
    connector_lengths_m      : list of branch connector lengths (one per branch)
    n_junctions              : total header branch-off junction count (inlet + outlet)
    K_junction               : loss coefficient per junction (tee / branch take-off)
    manifold_diameter_m      : hydraulic diameter of header pipe
    topology_imbalance_factor: extra position bias scaling for this topology
                               (0.0 = uniform; 1.0 = strong gradient)
    description              : short human-readable label
    """
    inlet_manifold_length_m:   float      = 0.0
    outlet_manifold_length_m:  float      = 0.0
    connector_lengths_m:       List[float] = field(default_factory=list)
    n_junctions:               int        = 0
    K_junction:                float      = 0.3    # tee branch-off loss coefficient
    manifold_diameter_m:       float      = 0.012  # m  header hydraulic diameter
    topology_imbalance_factor: float      = 0.3
    description:               str        = ""


def estimate_manifold_geometry(
    dv:    DesignVariables,
    pitch: float,
) -> ManifoldGeometry:
    """
    Estimate manifold and connector geometry for a given topology/manifold combo.

    PHYSICAL RATIONALE
    ------------------
    The manifold geometry determines:
      1. Header tube length → how much pressure is lost along the header.
      2. Connector length from header tap to each branch first-pass.
      3. Topology-specific imbalance factor (asymmetry in flow paths).

    All lengths are in metres.  The header hydraulic diameter used here is
    MANIFOLD_TUBE_DIAMETER_M (10 mm), NOT the visual MANIFOLD_INLET_WIDTH_M.
    This gives a realistic header velocity and meaningful manifold ΔP.

    Topology-specific rules
    -----------------------
    H_SERPENTINE LEFT_RIGHT:
        Header runs along the FULL plate length (420 mm).
        Each branch taps off at pitch intervals along the length.
        The port is at mid-length → half-header length ≈ 210 mm.
        Connector = fixed inset (25 mm) for all branches.
        Moderate imbalance: far branches lose ~20–30 Pa header pressure.

    H_SERPENTINE TOP_BOTTOM:
        Header runs along the plate WIDTH (210 mm).
        Shorter header → lower header pressure gradient → better uniformity.
        Connector from header to branch ≈ inset + extra run along width.

    V_SERPENTINE LEFT_RIGHT:
        Passes run along length; branches stacked across width.
        Header on short (210 mm) sides — short header → good uniformity.

    V_SERPENTINE TOP_BOTTOM:
        Header on long (420 mm) sides — long header → worse uniformity.

    MIRRORED_U SAME_SIDE:
        Both headers on the same short edge (width side).
        Branches distributed across width → connectors increase linearly.
        Short header (~half width), but asymmetric connector paths.

    Z_FLOW LEFT_RIGHT / TOP_BOTTOM:
        Cross-flow; branches span from bottom to top (or left to right).
        Inlet at one corner, outlet at opposite corner.
        The inlet header runs along the FULL plate length on one side.
        Connector lengths increase for branches farther from the inlet port.
        This gives a pronounced linear maldistribution — large imbalance.

    CENTRAL_INLET CENTER_EDGE:
        Central manifold; flow splits symmetrically to both halves.
        Each half-header is short (quarter-plate-length ≈ 105 mm).
        Connectors are shortest for centre branches, longest for edge branches.
        Good intra-half uniformity, but possible inter-half mismatch.

    Parameters
    ----------
    dv    : DesignVariables
    pitch : float   pass pitch [m]

    Returns
    -------
    mg : ManifoldGeometry
    """
    nb     = dv.n_branches
    PL     = BATTERY_LENGTH_M
    PW     = BATTERY_WIDTH_M
    INSET  = 0.025      # m  fixed header-to-first-pass connector inset
    D_hdr  = MANIFOLD_TUBE_DIAMETER_M   # realistic hydraulic header diameter

    t = dv.topology
    m = dv.manifold

    if t in ("H_SERPENTINE", "Z_FLOW"):
        if m == "LEFT_RIGHT":
            # Header runs along the LONG axis (BATTERY_LENGTH_M).
            # Total branch span = n_branches * pitch (branches stacked along length).
            branch_span = nb * pitch           # occupied length for all branches
            # Port at one end of header (not mid) → full branch span is traversed.
            # Inlet header: from port at y=0 to last branch at y=branch_span.
            inlet_L  = branch_span
            outlet_L = branch_span   # symmetric outlet header on opposite side
            # All branches have the same x-connector length (fixed inset).
            # But their y-position along the header differs → pressure gradient effect
            # captured in estimate_branch_hydraulic_resistance() position bias.
            connectors = [INSET] * nb
            # Z_FLOW has a more pronounced pressure gradient because the cross-flow
            # path adds Z-path asymmetry (different total path for each branch).
            imbalance = 0.40 if t == "Z_FLOW" else 0.30
            desc = f"{t} LEFT_RIGHT (long-axis header, port at corner)"

        else:  # TOP_BOTTOM
            # Header runs along the SHORT axis (BATTERY_WIDTH_M).
            # Branches are stacked along the width.
            inlet_L  = PW * 0.5    # port at centre of short edge → half-span
            outlet_L = PW * 0.5
            # Branches tap off at pitch intervals; connector is the inset only.
            connectors = [INSET] * nb
            imbalance  = 0.20 if t == "H_SERPENTINE" else 0.28
            desc = f"{t} TOP_BOTTOM (short-axis header, better uniformity)"

    elif t == "V_SERPENTINE":
        if m == "LEFT_RIGHT":
            # Headers on SHORT sides (BATTERY_WIDTH_M).
            # Branches stacked across width; header spans the branch extent.
            branch_span = nb * pitch   # span across width
            inlet_L  = min(branch_span, PW * 0.5)
            outlet_L = inlet_L
            connectors = [INSET] * nb
            imbalance  = 0.15   # short header → good uniformity
            desc = "V_SERPENTINE LEFT_RIGHT (short header, good uniformity)"

        else:  # TOP_BOTTOM
            # Headers on LONG sides (BATTERY_LENGTH_M).
            inlet_L  = PL * 0.5   # port at centre, half-length traversed
            outlet_L = inlet_L
            connectors = [INSET] * nb
            imbalance  = 0.35   # long header → more maldistribution
            desc = "V_SERPENTINE TOP_BOTTOM (long-axis header, more imbalance)"

    elif t == "MIRRORED_U":
        # Both headers on the SAME short edge.
        # Inlet header left half; outlet header right half of that edge.
        # Branches are distributed across width → connectors grow with branch index.
        x_step = (PW - 2.0 * EDGE_OFFSET_M) / max(nb, 1)
        # Connector: inset + fractional branch x-position (branch 0 is near port)
        connectors = [INSET + b * x_step for b in range(nb)]
        inlet_L  = PW * 0.45    # inlet uses ~half the edge width
        outlet_L = PW * 0.45    # outlet uses the other half
        imbalance  = 0.28   # lateral branch spacing gives unequal connectors
        desc = "MIRRORED_U SAME_SIDE (split short-edge headers, variable connectors)"

    elif t == "CENTRAL_INLET":
        # Central manifold runs along plate midline.
        # Flow splits symmetrically to upper and lower halves.
        # Each half-header spans from centre to edge → PL/4 in each direction.
        nb_upper = nb // 2
        nb_lower = nb - nb_upper
        x_step   = (PW - 2.0 * EDGE_OFFSET_M) / max(nb, 1)
        # Connectors: shortest for central branches, longer for edge branches.
        # Upper half: branches 0..nb_upper-1, furthest from port = edge branch
        conn_upper = [INSET + i * x_step * 0.40 for i in range(nb_upper)]
        # Lower half: symmetric
        conn_lower = [INSET + i * x_step * 0.40 for i in range(nb_lower)]
        connectors = conn_upper + conn_lower
        inlet_L    = PL * 0.25   # each half-arm of central manifold is PL/4
        outlet_L   = PL * 0.50   # outlet collects along both edges → full-length manifold
        imbalance  = 0.18   # symmetric split → low intra-half imbalance
        desc = "CENTRAL_INLET CENTER_EDGE (central split manifold, short half-arms)"

    else:
        connectors = [INSET] * nb
        inlet_L    = PL * 0.25
        outlet_L   = PL * 0.25
        imbalance  = 0.30
        desc = "fallback"

    n_junctions = nb * 2   # one inlet tap + one outlet tap per branch

    return ManifoldGeometry(
        inlet_manifold_length_m   = max(inlet_L, 0.0),
        outlet_manifold_length_m  = max(outlet_L, 0.0),
        connector_lengths_m       = [max(c, 0.002) for c in connectors],
        n_junctions               = n_junctions,
        K_junction                = 0.35,   # tee branch-off loss
        manifold_diameter_m       = D_hdr,
        topology_imbalance_factor = imbalance,
        description               = desc,
    )


def estimate_branch_hydraulic_resistance(
    branch_id:    int,
    dv:           DesignVariables,
    fluid:        FluidProperties,
    L_pass:       float,
    diameters:    List[float],
    pitch:        float,
    mg:           ManifoldGeometry,
) -> float:
    """
    Estimate the hydraulic resistance of branch `branch_id` for the
    relative flow-split calculation.

    PHYSICAL MODEL
    --------------
    The effective driving pressure available to branch i is:
        ΔP_drive_i = ΔP_system − ΔP_manifold_bias_i

    where ΔP_manifold_bias_i is the pressure loss along the inlet header
    from the port to branch i's tap-off point (a linear gradient model).
    Branches closer to the port (lower bias) see higher driving pressure
    and receive more flow; far branches get less.

    This is expressed as an effective resistance:
        R_eff_i = R_channel_i + R_connector_i + R_entry_exit_i + R_manifold_bias_i

    where R_manifold_bias_i is derived from the manifold pressure gradient.

    MANIFOLD PRESSURE GRADIENT
    --------------------------
    For a header of length L_hdr carrying total flow M_DOT at the inlet end
    and tapping off nb branches uniformly:

        V_hdr_entry = M_DOT / (ρ × A_hdr)    [velocity at port]
        Mean velocity ≈ V_hdr_entry / 2       [linear taper by extraction]

        ΔP_hdr_full = f × (L_hdr / D_hdr) × ρ V_hdr_entry² / 2

    Branch i at position s_i along the header sees a pressure bias:
        ΔP_bias_i = ΔP_hdr_full × (s_i / L_hdr)
                  × (1 − (1 − s_i/L_hdr) × velocity_recovery_factor)

    Expressed as additional resistance:
        R_bias_i = ΔP_bias_i / (m_dot_branch_ref²) × 2ρ

    where m_dot_branch_ref = M_DOT / nb  (reference equal-split flow)

    This produces a MEANINGFUL spread of 10–40 % in R_eff across branches
    for headers with high imbalance factors, and <5 % for good designs.

    Parameters
    ----------
    branch_id  : which branch (0-based)
    dv, fluid  : design and fluid parameters
    L_pass     : straight pass length [m]
    diameters  : per-pass diameter list [m]
    pitch      : pass pitch [m]
    mg         : ManifoldGeometry for this design

    Returns
    -------
    R_eff : float   relative resistance [Pa·s²/kg²], used for flow-split ratio
    """
    D_mean = float(np.mean(diameters))
    if D_mean <= 0:
        return 1e6

    A_ch  = math.pi / 4.0 * D_mean ** 2   # m²
    nb    = dv.n_branches

    # --- (a) Channel resistance ---
    L_turn = _turn_length_m(pitch, dv.bend_radius_m, dv.turn_style)
    L_ch_total = dv.n_passes * L_pass + (dv.n_passes - 1) * L_turn

    m_dot_ref = M_DOT_TOTAL_KG_S / max(nb, 1)   # equal-split reference
    Re_ref    = fluid.rho * m_dot_ref / (fluid.mu * A_ch) if (fluid.mu * A_ch) > 0 else 1e3
    f_ref     = friction_factor(Re_ref)
    R_ch      = f_ref * (L_ch_total / D_mean) / (A_ch ** 2)

    # --- (b) Connector resistance ---
    L_conn = mg.connector_lengths_m[branch_id % len(mg.connector_lengths_m)]
    R_conn = f_ref * (L_conn / D_mean) / (A_ch ** 2)

    # --- (c) Entry / exit minor losses ---
    R_entry_exit = (K_ENTRY + K_EXIT) / (A_ch ** 2)

    # --- (d) Manifold pressure-gradient bias  [KEY PHYSICS] ---
    #
    # Linear model: inlet header loses pressure from port to far end.
    # Branch i at normalized position s_i ∈ [0, 1] (0 = closest to port)
    # sees a reduced driving pressure proportional to s_i.
    #
    # s_i: branches are ordered 0..nb-1 from near-port to far-port.
    # For symmetric headers (port at centre), s_i = |i - (nb-1)/2| / ((nb-1)/2).
    # For single-end-port headers, s_i = i / (nb-1).
    #
    # The manifold topology determines which model applies:
    topo = dv.topology
    mfld = dv.manifold

    if topo in ("H_SERPENTINE", "Z_FLOW"):
        if mfld == "LEFT_RIGHT":
            # Port at one end (index 0 = near port, nb-1 = far)
            s_i = branch_id / max(nb - 1, 1)
        else:  # TOP_BOTTOM: port at mid-width
            s_i = abs(branch_id - (nb - 1) / 2.0) / max((nb - 1) / 2.0, 0.5)
    elif topo == "V_SERPENTINE":
        if mfld == "LEFT_RIGHT":
            # Short header, port at one end; small gradient
            s_i = branch_id / max(nb - 1, 1)
        else:  # TOP_BOTTOM: long header, port at one end
            s_i = branch_id / max(nb - 1, 1)
    elif topo == "MIRRORED_U":
        # Branches distributed laterally; inlet port at x=0
        s_i = branch_id / max(nb - 1, 1)
    elif topo == "CENTRAL_INLET":
        # Port at centre; branches fan out to both sides
        # Upper half (0..nb//2-1): near-centre branches have s_i=0, edge s_i=1
        # Lower half (nb//2..nb-1): same pattern
        nb_half = nb // 2
        if branch_id < nb_half:
            s_i = branch_id / max(nb_half - 1, 1) if nb_half > 1 else 0.0
        else:
            pos_in_half = branch_id - nb_half
            n_lower_half = nb - nb_half
            s_i = pos_in_half / max(n_lower_half - 1, 1) if n_lower_half > 1 else 0.0
    else:
        s_i = branch_id / max(nb - 1, 1)

    # Manifold pressure drop over full header length
    D_hdr  = mg.manifold_diameter_m
    A_hdr  = math.pi / 4.0 * D_hdr ** 2
    V_hdr  = M_DOT_TOTAL_KG_S / (fluid.rho * max(A_hdr, 1e-9))
    Re_hdr = fluid.rho * V_hdr * D_hdr / max(fluid.mu, 1e-9)
    f_hdr  = friction_factor(Re_hdr)
    dyn_hdr = 0.5 * fluid.rho * V_hdr ** 2

    dP_hdr_full = (
        f_hdr * (mg.inlet_manifold_length_m / max(D_hdr, 1e-9)) * dyn_hdr
        + mg.n_junctions * mg.K_junction * dyn_hdr
    )

    # Pressure bias at branch i = fraction s_i of full header ΔP
    # Scale by topology imbalance factor (physics-to-model fidelity adjustment)
    dP_bias_i = s_i * dP_hdr_full * mg.topology_imbalance_factor

    # Convert to resistance term in [Pa·s²/kg²]:
    # R_bias_i = dP_bias_i / m_dot_ref²
    R_bias = dP_bias_i / max(m_dot_ref ** 2, 1e-18)

    R_eff = R_ch + R_conn + R_entry_exit + R_bias
    return max(R_eff, 1e-9)


def allocate_branch_flows(
    dv:        DesignVariables,
    fluid:     FluidProperties,
    L_pass:    float,
    diameters: List[float],
    pitch:     float,
    mg:        ManifoldGeometry,
) -> List[float]:
    """
    Allocate total mass flow among branches based on available driving pressure.

    METHOD — Pressure-Budget Model
    --------------------------------
    Each branch i receives a driving pressure:
        ΔP_drive_i = ΔP_system − ΔP_header_to_i

    where ΔP_header_to_i is the cumulative manifold pressure drop from the
    inlet port to branch i's tap-off point.

    The branch flow is then:
        m_dot_i ∝ sqrt(ΔP_drive_i / R_channel_i)

    since for turbulent-ish flow: ΔP ≈ R × m_dot².

    This produces PHYSICALLY MEANINGFUL differences:
      - Short-header topologies (V_SERP LEFT_RIGHT) → nearly equal flows
      - Long-header topologies (H_SERP LEFT_RIGHT, Z_FLOW) → noticeably unequal
      - Variable-connector topologies (MIRRORED_U) → gradient from connector ΔP

    MANIFOLD PRESSURE GRADIENT
    ---------------------------
    A simplified linear model is used:
        ΔP_header(s) = ΔP_header_full × s     where s ∈ [0, 1] = normalised position
        ΔP_header_full = f × (L_header / D_header) × (ρ V²/2) + junction losses

    The velocity at the start of the header = M_DOT / (ρ × A_header).
    Since flow is extracted along the header, the mean velocity is ≈ V_entry/2.
    We use V_entry for a conservative (slightly over-estimated) gradient.

    REFERENCE SYSTEM ΔP
    -------------------
    ΔP_system is estimated from the equal-split branch pressure drop
    (evaluated analytically via Darcy-Weisbach at m_dot = M_DOT/nb).
    This gives a reference without calling the full evaluate_branch().

    Parameters
    ----------
    dv, fluid  : design and fluid parameters
    L_pass     : straight run length [m]
    diameters  : per-pass diameters [m]
    pitch      : pass pitch [m]
    mg         : ManifoldGeometry

    Returns
    -------
    m_dots : List[float]   one mass flow per branch [kg/s], sum = M_DOT_TOTAL_KG_S
    """
    nb     = dv.n_branches
    D_mean = float(np.mean(diameters))
    A_ch   = math.pi / 4.0 * D_mean ** 2

    if D_mean <= 0 or A_ch <= 0:
        return [M_DOT_TOTAL_KG_S / nb] * nb

    # --- Reference branch pressure drop at equal split (Darcy-Weisbach) ---
    m_dot_ref = M_DOT_TOTAL_KG_S / max(nb, 1)
    V_ref     = m_dot_ref / (fluid.rho * A_ch)
    Re_ref    = fluid.rho * V_ref * D_mean / max(fluid.mu, 1e-9)
    f_ref     = friction_factor(Re_ref)
    L_turn    = _turn_length_m(pitch, dv.bend_radius_m, dv.turn_style)
    L_total   = dv.n_passes * L_pass + (dv.n_passes - 1) * L_turn
    dyn_ch    = 0.5 * fluid.rho * V_ref ** 2
    dP_system = f_ref * (L_total / max(D_mean, 1e-9)) * dyn_ch   # reference branch ΔP

    # --- Header manifold pressure gradient ---
    D_hdr  = mg.manifold_diameter_m
    A_hdr  = math.pi / 4.0 * D_hdr ** 2
    V_hdr  = M_DOT_TOTAL_KG_S / (fluid.rho * max(A_hdr, 1e-9))
    Re_hdr = fluid.rho * V_hdr * D_hdr / max(fluid.mu, 1e-9)
    f_hdr  = friction_factor(Re_hdr)
    dyn_hdr = 0.5 * fluid.rho * V_hdr ** 2

    # Total manifold ΔP (friction along full header + junction losses)
    dP_hdr_total = (
        f_hdr * (mg.inlet_manifold_length_m / max(D_hdr, 1e-9)) * dyn_hdr
        + mg.n_junctions * mg.K_junction * dyn_hdr
    )

    # Scale by imbalance factor (topology-specific physics fidelity)
    dP_hdr_effective = dP_hdr_total * mg.topology_imbalance_factor

    # --- Connector pressure drop per branch ---
    # Varies by branch for topologies with variable connector lengths.
    dP_connector = []
    for b in range(nb):
        L_conn = mg.connector_lengths_m[b % len(mg.connector_lengths_m)]
        # Connector velocity ≈ branch velocity (same diameter)
        V_conn = m_dot_ref / (fluid.rho * max(A_ch, 1e-9))
        dyn_c  = 0.5 * fluid.rho * V_conn ** 2
        Re_c   = fluid.rho * V_conn * D_mean / max(fluid.mu, 1e-9)
        f_c    = friction_factor(Re_c)
        dP_c   = f_c * (L_conn / max(D_mean, 1e-9)) * dyn_c
        dP_connector.append(dP_c)

    # --- Position fraction s_i along the header for each branch ---
    # s_i = 0 → closest to port (sees full system ΔP)
    # s_i = 1 → furthest from port (sees reduced ΔP by dP_hdr_effective)
    topo = dv.topology
    mfld = dv.manifold

    if topo in ("H_SERPENTINE", "Z_FLOW"):
        if mfld == "LEFT_RIGHT":
            s = [b / max(nb - 1, 1) for b in range(nb)]
        else:  # TOP_BOTTOM: port at mid, two symmetric halves
            s = [abs(b - (nb - 1) / 2.0) / max((nb - 1) / 2.0, 0.5) for b in range(nb)]
    elif topo == "V_SERPENTINE":
        s = [b / max(nb - 1, 1) for b in range(nb)]
    elif topo == "MIRRORED_U":
        # Connector path grows linearly; no position gradient in header (short header)
        s = [0.0] * nb   # header bias negligible (short header); connector length handles it
    elif topo == "CENTRAL_INLET":
        nb_half = nb // 2
        s_upper = [i / max(nb_half - 1, 1) if nb_half > 1 else 0.0 for i in range(nb_half)]
        n_lower = nb - nb_half
        s_lower = [i / max(n_lower - 1, 1) if n_lower > 1 else 0.0 for i in range(n_lower)]
        s = s_upper + s_lower
    else:
        s = [b / max(nb - 1, 1) for b in range(nb)]

    # --- Available driving pressure for each branch ---
    dP_avail = []
    for b in range(nb):
        header_loss_b = s[b] * dP_hdr_effective
        conn_loss_b   = dP_connector[b]
        dP_b = max(dP_system - header_loss_b - conn_loss_b, dP_system * 0.01)
        dP_avail.append(dP_b)

    # --- Flow split: m_dot_i ∝ sqrt(dP_avail_i) (quadratic resistance model) ---
    sqrt_dP = [math.sqrt(max(dp, 1e-9)) for dp in dP_avail]
    total   = sum(sqrt_dP)
    if total <= 0:
        return [M_DOT_TOTAL_KG_S / nb] * nb

    m_dots = [M_DOT_TOTAL_KG_S * sp / total for sp in sqrt_dP]
    return m_dots


def evaluate_topology_specific_losses(
    dv:     DesignVariables,
    fluid:  FluidProperties,
    mg:     ManifoldGeometry,
    m_dots: List[float],
) -> float:
    """
    Estimate the additional pressure drop due to manifold pipe friction and
    branch take-off junction losses.  This is added to the system ΔP.

    COMPONENTS
    ----------
    (a) Inlet manifold friction:
        ΔP_inlet = f × (L_inlet / D_hdr) × (ρ V_hdr² / 2)
        where V_hdr = (M_DOT_TOTAL / ρ) / A_hdr

    (b) Outlet manifold friction (same geometry, symmetric):
        ΔP_outlet = f × (L_outlet / D_hdr) × (ρ V_hdr² / 2)

    (c) Junction losses at each branch take-off:
        ΔP_junction = n_junctions × K_junction × (ρ V_hdr² / 2)

    (d) Mean connector friction:
        ΔP_conn = f × (mean_connector / D_ch) × (ρ V_branch² / 2)
        averaged across branches.

    Parameters
    ----------
    dv     : DesignVariables
    fluid  : FluidProperties
    mg     : ManifoldGeometry
    m_dots : List[float]  per-branch mass flows [kg/s]

    Returns
    -------
    dP_manifold : float   additional manifold + connector ΔP [Pa]
    """
    D_hdr   = mg.manifold_diameter_m
    A_hdr   = math.pi / 4.0 * D_hdr ** 2
    V_hdr   = (M_DOT_TOTAL_KG_S / fluid.rho) / max(A_hdr, 1e-9)
    dyn_hdr = 0.5 * fluid.rho * V_hdr ** 2

    # Reynolds number in header
    Re_hdr  = fluid.rho * V_hdr * D_hdr / max(fluid.mu, 1e-9)
    f_hdr   = friction_factor(Re_hdr)

    # (a) Inlet manifold friction
    dP_inlet = f_hdr * (mg.inlet_manifold_length_m / max(D_hdr, 1e-9)) * dyn_hdr

    # (b) Outlet manifold friction
    dP_outlet = f_hdr * (mg.outlet_manifold_length_m / max(D_hdr, 1e-9)) * dyn_hdr

    # (c) Junction losses
    dP_junction = mg.n_junctions * mg.K_junction * dyn_hdr

    # (d) Mean connector friction (across branches)
    nb = dv.n_branches
    D_ch_mean = float(np.mean(get_diameter_profile(dv)))
    A_ch_mean = math.pi / 4.0 * D_ch_mean ** 2
    m_dot_mean = M_DOT_TOTAL_KG_S / max(nb, 1)
    V_conn     = m_dot_mean / (fluid.rho * max(A_ch_mean, 1e-9))
    dyn_conn   = 0.5 * fluid.rho * V_conn ** 2
    Re_conn    = fluid.rho * V_conn * D_ch_mean / max(fluid.mu, 1e-9)
    f_conn     = friction_factor(Re_conn)
    L_conn_mean = float(np.mean(mg.connector_lengths_m)) if mg.connector_lengths_m else 0.025
    dP_connector = f_conn * (L_conn_mean / max(D_ch_mean, 1e-9)) * dyn_conn

    dP_manifold = dP_inlet + dP_outlet + dP_junction + dP_connector
    return max(dP_manifold, 0.0)


def compute_branch_uniformity_penalty(
    dv:        DesignVariables,
    m_dots:    List[float],
    mg:        ManifoldGeometry,
    branch_R:  List[float],
) -> float:
    """
    Compute a physically grounded branch-flow uniformity penalty.

    This penalty replaces and supersedes compute_manifold_uniformity_penalty()
    for the main score.  The old function is kept for backward compatibility
    but its weight (W_MANIFOLD_DIST) is effectively zeroed out.

    COMPONENTS
    ----------
    (a) Flow fraction coefficient of variation:
        cv_flow = std(m_dot_i) / mean(m_dot_i)
        Normalised to [0,1] by reference CV of 0.10.
        (CV=0.10 = 10% spread across branches)

    (b) Connector length variation:
        cv_conn = std(L_conn_i) / mean(L_conn_i)
        Normalised to [0,1] by reference CV of 0.40.

    (c) Resistance variation:
        cv_R = std(R_eff_i) / mean(R_eff_i)
        Normalised to [0,1] by reference CV of 0.30.

    (d) Topology imbalance factor (direct contribution from ManifoldGeometry).

    Parameters
    ----------
    dv       : DesignVariables
    m_dots   : per-branch mass flows [kg/s]
    mg       : ManifoldGeometry
    branch_R : per-branch R_eff values

    Returns
    -------
    penalty : float  in [0, 1], lower is better (0 = perfectly uniform)
    """
    if not m_dots or len(m_dots) < 2:
        return float(mg.topology_imbalance_factor)

    # (a) Flow CV
    arr_m  = np.array(m_dots, dtype=float)
    mean_m = float(np.mean(arr_m))
    if mean_m <= 0 or not np.isfinite(mean_m):
        pen_flow = 1.0
    else:
        std_m   = float(np.std(arr_m))
        cv_flow = std_m / mean_m
        pen_flow = min(1.0, cv_flow / 0.10)

    # (b) Connector length CV
    conn = np.array(mg.connector_lengths_m, dtype=float) if mg.connector_lengths_m else np.array([0.025])
    mean_c = float(np.mean(conn))
    if mean_c <= 0 or not np.isfinite(mean_c):
        pen_conn = 0.0
    else:
        cv_conn = float(np.std(conn)) / mean_c
        pen_conn = min(1.0, cv_conn / 0.40)

    # (c) Resistance CV
    if branch_R and len(branch_R) >= 2:
        arr_R  = np.array(branch_R, dtype=float)
        mean_R = float(np.mean(arr_R))
        if mean_R > 0 and np.isfinite(mean_R):
            cv_R  = float(np.std(arr_R)) / mean_R
            pen_R = min(1.0, cv_R / 0.30)
        else:
            pen_R = 0.0
    else:
        pen_R = 0.0

    # (d) Topology imbalance factor
    pen_topo = float(mg.topology_imbalance_factor)

    penalty = (
        0.40 * pen_flow  +
        0.15 * pen_conn  +
        0.20 * pen_R     +
        0.25 * pen_topo
    )
    return max(0.0, min(1.0, float(penalty)))


def aggregate_design_thermal_metrics(
    dv:          DesignVariables,
    fluid:       FluidProperties,
    L_pass:      float,
    diameters:   List[float],
    pitch:       float,
    m_dots:      List[float],
    Q_total_W:   float,
) -> Tuple[List[float], float, float, float]:
    """
    Compute per-branch outlet temperatures and battery temperature estimates.

    v11 replacement: discretized 1-D channel model.
    -----------------------------------------------
    The old model used a single energy balance T_out = T_in + Q/(m_dot×cp)
    with a lumped resistance, making T_out nearly constant across designs
    and insensitive to diameter or pass count.

    This version integrates along the channel pass-by-pass, computing
    local Re, Nu, h and dT_coolant for each straight segment.  For tapered
    (3-zone) channels each pass zone uses its own local diameter, giving
    a physically meaningful difference between constant and tapered designs.

    For each branch b and each pass p:
        D_p   = diameters[p]                      (local diameter for this pass)
        A_cs  = π/4 × D_p²
        V_p   = m_dot_b / (ρ × A_cs)
        Re_p  = ρ × V_p × D_p / μ
        Nu_p  = nusselt_number(Re_p, Pr, PHI_SERPENTINE)
        h_p   = Nu_p × k_fluid / D_p
        A_wet = π × D_p × L_pass
        Q_p   = Q_branch × A_wet_p / sum(A_wet)   (wetted-area weighting)
        ΔT_coolant_p = Q_p / (m_dot_b × cp)
        T_coolant tracks cumulatively along the branch

    Battery temperature for each pass strip:
        R_conv_p = 1 / (h_p × A_wet_p)
        R_cond_p = t_plate / (k_solid × D_p × L_pass)
        R_p      = R_conv_p + R_cond_p
        T_batt_p = T_coolant_mid_p + Q_p × R_p

    Branch battery temperature = max over all pass strips (worst strip sets
    the local peak).  Mean is average over all (branch × pass) strips.

    Parameters
    ----------
    dv        : DesignVariables
    fluid     : FluidProperties
    L_pass    : float   straight pass length [m]
    diameters : List[float]  per-pass diameter [m]
    pitch     : float   pass-to-pass pitch [m]
    m_dots    : List[float]  per-branch mass flows [kg/s]
    Q_total_W : float   total heat load [W]

    Returns
    -------
    T_batt_branches : List[float]  per-branch estimated peak battery temperature [°C]
    T_batt_mean     : float        mean of all strip temperatures [°C]
    T_batt_max      : float        worst-case strip temperature [°C]
    T_out_mean      : float        flow-weighted mean branch outlet temperature [°C]
    """
    nb  = dv.n_branches
    np_ = dv.n_passes
    mat = get_material_properties()

    Q_branch = Q_total_W / nb   # equal heat per branch

    # Wetted-area weighting for heat distribution across passes
    A_wet_passes = [math.pi * D * L_pass for D in diameters]
    A_wet_total  = sum(A_wet_passes)
    if A_wet_total > 0:
        Q_per_pass = [Q_branch * (A / A_wet_total) for A in A_wet_passes]
    else:
        Q_per_pass = [Q_branch / max(np_, 1)] * np_

    T_batt_branches = []
    T_out_list      = []

    for b in range(nb):
        m_dot_b = m_dots[b]
        T_cool  = T_INLET_C   # coolant bulk temperature, tracks along branch

        strip_T_batt = []   # per-pass battery strip temperature

        for p in range(np_):
            D_p = diameters[p]
            if D_p <= 0:
                continue

            # ── local geometry ───────────────────────────────────────────
            A_cs_p  = math.pi / 4.0 * D_p ** 2
            A_wet_p = math.pi * D_p * L_pass

            # ── local velocity, Re, Nu, h ────────────────────────────────
            V_p  = m_dot_b / (fluid.rho * max(A_cs_p, 1e-12))
            Re_p = fluid.rho * V_p * D_p / max(fluid.mu, 1e-12)
            Nu_p = nusselt_number(Re_p, fluid.Pr, phi=PHI_SERPENTINE)
            h_p  = Nu_p * fluid.k / max(D_p, 1e-12)

            # ── coolant temperature rise in this pass ────────────────────
            Q_p  = Q_per_pass[p]
            if fluid.cp > 0 and m_dot_b > 0:
                dT_p = Q_p / (m_dot_b * fluid.cp)
            else:
                dT_p = 0.0

            T_cool_mid = T_cool + 0.5 * dT_p   # mid-pass coolant temperature
            T_cool     += dT_p                  # exit of this pass

            # ── thermal resistances for this pass strip ──────────────────
            R_conv_p = 1.0 / max(h_p * A_wet_p, 1e-9)
            A_eff_p  = D_p * L_pass   # 1-D conduction footprint
            R_cond_p = PLATE_THICKNESS_M / max(mat.k_solid_W_mK * A_eff_p, 1e-9)
            R_p      = R_conv_p + R_cond_p

            # ── battery surface temperature for this strip ───────────────
            T_batt_strip = T_cool_mid + Q_p * R_p
            strip_T_batt.append(T_batt_strip)

        # Branch outlet temperature
        T_out_list.append(T_cool)

        # Per-branch battery temperature: worst strip in this branch
        if strip_T_batt:
            T_batt_branches.append(max(strip_T_batt))
        else:
            T_batt_branches.append(T_INLET_C + 30.0)

    T_batt_mean = float(np.mean(T_batt_branches)) if T_batt_branches else T_INLET_C + 30.0
    T_batt_max  = float(np.max(T_batt_branches))  if T_batt_branches else T_INLET_C + 30.0

    # Flow-weighted mean outlet temperature
    total_m = sum(m_dots)
    if total_m > 0 and T_out_list:
        T_out_mean = sum(m_dots[b] * T_out_list[b] for b in range(nb)) / total_m
    else:
        T_out_mean = T_INLET_C + Q_total_W / max(M_DOT_TOTAL_KG_S * fluid.cp, 1e-9)

    return T_batt_branches, T_batt_mean, T_batt_max, T_out_mean


# =============================================================================
# 9b.  Manifold distribution penalty  (original geometric proxy, kept for ref)
# =============================================================================
# NOTE (v9): The physical topology layer above supersedes this function for the
# main score.  W_MANIFOLD_DIST is now effectively zeroed out to avoid double-
# counting the same physical effect.  This function is kept for backward
# compatibility and comparison.
# =============================================================================

def compute_manifold_uniformity_penalty(
    dv:    "DesignVariables",
    pitch: float,
) -> float:
    """
    Compute a geometric proxy penalty for manifold flow maldistribution.

    IMPORTANT — SCREENING PROXY ONLY
    ---------------------------------
    This function does NOT solve the manifold fluid network.  It uses
    geometric heuristics to score how well the header layout is likely
    to distribute flow evenly among branches.  A full CFD manifold
    network solver would be needed for accurate maldistribution prediction.

    Physical motivation
    -------------------
    In a real parallel-branch cold plate, uneven flow distribution arises
    mainly from two sources:
      1.  Pressure gradient along the inlet/outlet headers (more flow
          enters branches near the port, less flow enters far branches).
      2.  Asymmetric connector lengths (branches with longer connectors
          have higher hydraulic resistance and receive less flow).

    Proxy penalties used here
    -------------------------
    (a) Connector length asymmetry penalty
        Branch connections are evenly spaced (y-positions computed from
        pitch).  The connector length from the inlet vertical header to
        each branch first-pass centreline is assumed equal for all
        branches (symmetric layout).  No penalty from this alone.
        However, if total branch span >> MANIFOLD_INLET_WIDTH_M, the
        horizontal header connecting port to vertical header becomes
        disproportionately short, reducing the ability to equalize
        pressure along the vertical section.
        Penalty: max(0, branch_span / manifold_length - 1)
        normalised to ~0–1.

    (b) Header velocity ratio penalty
        A simple estimate of the header velocity: V_hdr ~ Q_branch / A_hdr.
        If header velocity is comparable to branch velocity, dynamic
        pressure effects drive maldistribution.
        Penalty: min(1, header_velocity / channel_velocity) with the ideal
        being header_velocity << channel_velocity.
        Uses the inlet manifold cross-section estimated from
        MANIFOLD_INLET_WIDTH_M × MANIFOLD_INLET_WIDTH_M (square section proxy).

    (c) Taper benefit bonus (reduces penalty)
        A tapered header that reduces in width from port to far end
        maintains roughly constant fluid velocity along the header
        and reduces pressure gradient along the header, improving
        distribution.
        Bonus: MANIFOLD_TAPER_RATIO × 0.4 (subtracted from penalty).
        A taper_ratio of 0.0 gives no benefit; 0.5 gives maximum benefit.

    (d) Branch count penalty
        More branches with a single inlet port means each branch receives
        a smaller fraction of total flow.  With no header widening this
        increases maldistribution risk.
        Penalty: max(0, (n_branches - 2) / 4)  (normalised, 0 for ≤2 branches)

    The combined penalty is clamped to [0, 1].

    Parameters
    ----------
    dv    : DesignVariables
    pitch : float   pass pitch [m]

    Returns
    -------
    penalty : float   in [0, 1], lower is better (more even distribution)
    """
    nb = dv.n_branches
    np_ = dv.n_passes

    # Total vertical span of all branches (m)
    branch_span_m = (nb * np_ - 1) * pitch   # distance from first to last pass

    # Manifold reference length: horizontal header (≈ HEADER_INSET from schematic)
    manifold_h_length_m = 0.025   # 25 mm horizontal header depth (fixed geometry)

    # ------------------------------------------------------------------
    # (a) Span-to-header ratio penalty
    # If all branches span a large y-range but the horizontal header is
    # short, the vertical distribution header must carry very unequal
    # static pressures from top to bottom.
    # Penalty is proportional to how much the branch span exceeds a
    # "comfortable" multiple of the horizontal header length.
    COMFORT_RATIO = 4.0   # span up to 4× manifold h-length is acceptable
    span_ratio    = branch_span_m / max(manifold_h_length_m, 1e-6)
    span_pen      = min(1.0, max(0.0, (span_ratio - COMFORT_RATIO) / COMFORT_RATIO))

    # ------------------------------------------------------------------
    # (b) Header velocity ratio penalty
    # Estimate header cross-section from MANIFOLD_INLET_WIDTH_M (square proxy)
    A_hdr = MANIFOLD_INLET_WIDTH_M ** 2   # m²  (representative square section)
    # Total volumetric flow through the manifold header
    rho   = RHO_WATER_KG_M3
    Q_vol = M_DOT_TOTAL_KG_S / rho       # m³/s
    V_hdr = Q_vol / max(A_hdr, 1e-9)     # m/s  header bulk velocity

    # Branch channel velocity proxy: Q / (n_branches × A_channel)
    D_ch  = dv.D_const_m if not dv.use_3zone else dv.D1_m
    A_ch  = math.pi * (D_ch / 2.0) ** 2   # m²
    Q_branch = Q_vol / max(nb, 1)
    V_ch  = Q_branch / max(A_ch, 1e-9)    # m/s

    # Ideal: V_hdr << V_ch (so dynamic pressure in header ≈ negligible)
    vel_ratio = V_hdr / max(V_ch, 1e-9)
    vel_pen   = min(1.0, vel_ratio)   # 0 = perfect; 1 = header as fast as channel

    # ------------------------------------------------------------------
    # (c) Taper benefit (reduces penalty)
    # A well-tapered header reduces header velocity gradient and improves
    # static pressure uniformity along the header.
    # The taper ratio acts as a direct reduction in the combined penalty.
    taper_benefit = MANIFOLD_TAPER_RATIO * 0.4   # max reduction of 0.4 × penalty

    # ------------------------------------------------------------------
    # (d) Branch count penalty
    # Each additional branch beyond 2 adds maldistribution risk.
    branch_pen = max(0.0, (nb - 2) / 4.0)        # 0 for nb≤2; 0.25 for nb=3; 0.5 for nb=4

    # ------------------------------------------------------------------
    # Combine: equal weight on each component, then subtract taper benefit
    combined = (
        0.35 * span_pen  +
        0.35 * vel_pen   +
        0.30 * branch_pen
    )
    penalty = max(0.0, min(1.0, combined - taper_benefit))
    return penalty


def compute_velocity_variation_penalty(
    dv:     DesignVariables,
    fluid:  FluidProperties,
    m_dots: List[float],
    diameters: List[float],
) -> Tuple[float, float, float]:
    """
    Change 2: Velocity-control penalty.

    Computes the minimum and maximum local channel velocity across all
    branch passes and returns a penalty that is high when velocities fall
    outside the target band [V_TARGET_MIN_M_S, V_TARGET_MAX_M_S].

    Physical motivation
    -------------------
    In a tapered (3-zone) or constant-diameter branch the local velocity
    changes with diameter.  We want:
      - Velocity ≥ V_TARGET_MIN_M_S  to maintain turbulence / heat transfer
      - Velocity ≤ V_TARGET_MAX_M_S  to limit pressure drop and erosion

    Penalty formula
    ---------------
    For each unique pass diameter D_k:
        V_k = m_dot_mean / (rho × pi/4 × D_k²)
    Penalty contribution for pass k:
        p_k = max(0, (V_TARGET_MIN - V_k) / V_TARGET_MIN)     if V_k < V_min
            + max(0, (V_k - V_TARGET_MAX) / V_TARGET_MAX)     if V_k > V_max
    velocity_variation_pen = mean(p_k) clamped to [0, 1].

    Additionally, if there are multiple diameters (3-zone mode) we add
    a variation penalty proportional to the coefficient of variation of
    velocities across zones (high CV → uneven zone heat transfer).

    Parameters
    ----------
    dv        : DesignVariables
    fluid     : FluidProperties
    m_dots    : List[float]  per-branch mass flow [kg/s]
    diameters : List[float]  per-pass diameter [m]

    Returns
    -------
    v_min : float  minimum velocity across all pass zones [m/s]
    v_max : float  maximum velocity across all pass zones [m/s]
    pen   : float  velocity penalty in [0, 1]
    """
    m_dot_mean = float(np.mean(m_dots)) if m_dots else M_DOT_TOTAL_KG_S / max(dv.n_branches, 1)

    velocities = []
    for D in diameters:
        if D <= 0:
            continue
        A_cs = math.pi / 4.0 * D ** 2
        V    = m_dot_mean / max(fluid.rho * A_cs, 1e-12)
        velocities.append(V)

    if not velocities:
        return 0.0, 0.0, 0.0

    v_min = float(np.min(velocities))
    v_max = float(np.max(velocities))

    # --- band penalty: penalise passes outside [V_min_target, V_max_target] ---
    pens = []
    for V in velocities:
        p = 0.0
        if V < V_TARGET_MIN_M_S and V_TARGET_MIN_M_S > 0:
            p += (V_TARGET_MIN_M_S - V) / V_TARGET_MIN_M_S
        if V > V_TARGET_MAX_M_S and V_TARGET_MAX_M_S > 0:
            p += (V - V_TARGET_MAX_M_S) / V_TARGET_MAX_M_S
        pens.append(p)

    band_pen = float(np.mean(pens))

    # --- optional variation penalty for multi-zone designs ---
    if len(velocities) > 1:
        cv = float(np.std(velocities)) / max(float(np.mean(velocities)), 1e-9)
        # High CV (> 0.5) indicates large velocity jump between zones
        var_pen = min(1.0, cv / 0.5) * 0.3
    else:
        var_pen = 0.0

    pen = min(1.0, band_pen + var_pen)
    return v_min, v_max, pen


def compute_branch_heatflux_penalty(
    dv:        DesignVariables,
    Q_total_W: float,
    pitch:     float,
    L_pass:    float,
    diameters: List[float],
) -> Tuple[float, float, float, float, float]:
    """
    Change 3 (v10 corrected): Branch-count suitability metric.

    The previous formulation collapsed to q_planar because
    Q_branch / A_branch = (Q/nb) / (A/nb) = Q/A (independent of n_branches).

    This corrected version uses ABSOLUTE quantities that depend on nb:

    1.  Q_per_branch [W]  — raw heat load per branch.
        Too large → thermal overload per branch.
        Target: Q_total / nb should not exceed a threshold.

    2.  Hydraulic loading proxy — coolant temperature rise per branch:
        dT_branch = Q_branch / (m_dot_branch × cp)
        Too large a dT (>15 K) means coolant is insufficiently refreshed.
        Too small a dT (negligible flow per branch) suggests excess branching.

    3.  Branch spacing / pass pitch  [m]
        pitch = plate_height / (nb × np_)
        Very small pitch relative to channel diameter → crowded layout.
        Very large pitch → poor lateral heat spreading between channels.
        Ideal pitch ≈ 2×D_mean … 6×D_mean.

    4.  Wetted-perimeter fraction — total wetted surface per branch
        as a fraction of total plate footprint area:
        A_wet_branch = π × D_mean × L_pass × n_passes   [m²]
        A_footprint_branch = L_pass × pitch × n_passes   [m²]
        ratio = A_wet / A_footprint   (higher → better coverage)

    Penalty contributions
    ---------------------
    pen_Q     : penalise Q_per_branch outside [Q_BRANCH_MIN, Q_BRANCH_MAX]
    pen_dT    : penalise dT_branch outside [2 K, 15 K] target band
    pen_pitch : penalise pitch/D_mean outside [2, 8] ideal band
    pen_wet   : penalise low wetted-surface fraction (< 0.25 = poor coverage)

    Combined: weighted sum, clamped to [0, 1].

    Parameters
    ----------
    dv        : DesignVariables
    Q_total_W : float
    pitch     : float  pass-to-pass pitch [m]
    L_pass    : float  straight run length [m]
    diameters : List[float]  per-pass diameters [m]

    Returns
    -------
    q_planar   : float  overall planform heat flux  [W/m²]  (informational)
    Q_branch   : float  heat load per branch  [W]
    A_branch   : float  footprint area per branch  [m²]
    q_branch   : float  Q_branch / A_branch  [W/m²]  (informational)
    pen        : float  branch-count suitability penalty  [0, 1]
    """
    nb    = max(dv.n_branches, 1)
    np_   = max(dv.n_passes,   1)
    A_plate  = BATTERY_LENGTH_M * BATTERY_WIDTH_M
    q_planar = Q_total_W / max(A_plate, 1e-9)

    # --- Absolute quantities that depend on nb ---
    Q_branch = Q_total_W / nb          # W per branch
    A_branch = A_plate   / nb          # m² footprint per branch (informational)
    q_branch = Q_branch  / max(A_branch, 1e-9)   # = q_planar (informational only)

    D_mean   = float(np.mean(diameters)) if diameters else 0.004

    # ──────────────────────────────────────────────────────────────────────────
    # (1) Q_per_branch penalty
    # Target: each branch should not carry more than Q_BRANCH_TARGET_W_M2_MAX
    # nor less than Q_BRANCH_TARGET_W_M2_MIN watts (absolute, not W/m²).
    # Use the targets interpreted as absolute W values here to make them
    # branch-count-sensitive:
    #   Q_branch_max_target = Q_total / 2   (2 branches is minimum practical)
    #   Q_branch_min_target = Q_total / 8   (8 branches is maximum practical)
    Q_branch_max_abs = Q_total_W / 2.0   # W  (minimum 2 branches ↔ max Q per branch)
    Q_branch_min_abs = Q_total_W / 8.0   # W  (maximum 8 branches ↔ min Q per branch)
    if Q_branch > Q_branch_max_abs and Q_branch_max_abs > 0:
        pen_Q = min(1.0, (Q_branch - Q_branch_max_abs) / Q_branch_max_abs)
    elif Q_branch < Q_branch_min_abs and Q_branch_min_abs > 0:
        pen_Q = min(1.0, (Q_branch_min_abs - Q_branch) / Q_branch_min_abs)
    else:
        pen_Q = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # (2) Coolant temperature rise per branch
    # dT_branch = Q_branch / (m_dot_branch × cp)
    m_dot_branch = M_DOT_TOTAL_KG_S / nb
    dT_branch = Q_branch / max(m_dot_branch * CP_WATER_J_KGK, 1e-9)
    # Target band: 2 K (not worth branching further) … 15 K (coolant saturating)
    dT_min_target = 2.0    # K
    dT_max_target = 15.0   # K
    if dT_branch > dT_max_target:
        pen_dT = min(1.0, (dT_branch - dT_max_target) / dT_max_target)
    elif dT_branch < dT_min_target and dT_min_target > 0:
        pen_dT = min(1.0, (dT_min_target - dT_branch) / dT_min_target)
    else:
        pen_dT = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # (3) Pass-pitch / diameter ratio
    # Ideal: pitch = 2×D_mean … 6×D_mean  (good coverage without crowding)
    if D_mean > 0 and pitch > 0:
        pitch_to_D = pitch / D_mean
        if pitch_to_D < 2.0:
            pen_pitch = min(1.0, (2.0 - pitch_to_D) / 2.0)   # too crowded
        elif pitch_to_D > 8.0:
            pen_pitch = min(1.0, (pitch_to_D - 8.0) / 8.0)   # too sparse
        else:
            pen_pitch = 0.0
    else:
        pen_pitch = 0.5   # fallback if geometry not yet computed

    # ──────────────────────────────────────────────────────────────────────────
    # (4) Wetted-perimeter coverage fraction
    # A_wet per branch = π × D_mean × L_pass × n_passes
    # A_footprint per branch ≈ L_pass × (nb × np_ × pitch) / nb = L_pass × np_ × pitch
    A_wet_branch      = math.pi * D_mean * L_pass * np_
    A_footprint_branch = L_pass * np_ * pitch if pitch > 0 else L_pass * np_ * D_mean
    if A_footprint_branch > 0:
        wet_frac = A_wet_branch / A_footprint_branch
        # Target: > 0.25 (25 % of footprint is wetted surface)
        pen_wet = max(0.0, min(1.0, (0.25 - wet_frac) / 0.25)) if wet_frac < 0.25 else 0.0
    else:
        pen_wet = 0.5

    # ──────────────────────────────────────────────────────────────────────────
    # Combine: weighted sum
    pen = (
        0.35 * pen_Q     +   # heat load per branch (most important)
        0.35 * pen_dT    +   # coolant saturation / over-cooling
        0.20 * pen_pitch +   # channel density / lateral coverage
        0.10 * pen_wet       # wetted surface utilisation
    )
    pen = max(0.0, min(1.0, pen))

    return q_planar, Q_branch, A_branch, q_branch, pen


def evaluate_design(
    design_id: int,
    dv:        DesignVariables,
    Q_total_W: float,
    fluid:     FluidProperties,
) -> DesignResult:
    """
    Evaluate a complete cooling plate design.

    Steps
    -----
    1.  Compute pass length and pitch.
    2.  Get diameter profile.
    3.  Check geometry feasibility.
    4.  Evaluate each branch.
    5.  Compute system-level thermal and hydraulic quantities.
    6.  Compute penalties and optimization score.

    Parameters
    ----------
    design_id : int
    dv        : DesignVariables
    Q_total_W : float   total heat load [W]
    fluid     : FluidProperties

    Returns
    -------
    result : DesignResult
    """
    result = DesignResult(design_id=design_id, dv=dv)
    result.Q_total_W = Q_total_W

    # ------------------------------------------------------------------
    # Step 1: Geometry
    # ------------------------------------------------------------------
    L_pass    = compute_straight_pass_length(dv)
    pitch     = compute_branch_pitch(dv)
    diameters = get_diameter_profile(dv)

    result.straight_pass_length_m = L_pass
    result.branch_pitch_m         = pitch

    # ------------------------------------------------------------------
    # Step 2: Feasibility check
    # ------------------------------------------------------------------
    feasible, reason = check_geometry_feasibility(dv, pitch, L_pass, diameters)
    if not feasible:
        result.feasible              = False
        result.infeasibility_reason  = reason
        result.score                 = 1e9
        return result

    # ------------------------------------------------------------------
    # Step 2b: v9 — Manifold geometry + unequal branch flow allocation
    # ------------------------------------------------------------------
    # Estimate manifold geometry for this topology/manifold combination.
    # Then compute per-branch hydraulic resistances and allocate flow
    # inversely proportional to resistance.  This means near-port branches
    # (lower resistance) receive slightly more flow; far-port branches get less.
    mg = estimate_manifold_geometry(dv, pitch)

    # Per-branch R_eff values (relative, used for flow split and penalty)
    branch_R_effs = [
        estimate_branch_hydraulic_resistance(b, dv, fluid, L_pass, diameters, pitch, mg)
        for b in range(dv.n_branches)
    ]

    # Unequal flow split: m_dot_i ∝ 1 / R_eff_i
    m_dot_allocated = allocate_branch_flows(dv, fluid, L_pass, diameters, pitch, mg)

    # Manifold + connector pressure losses (added to system ΔP)
    dP_manifold_extra = evaluate_topology_specific_losses(dv, fluid, mg, m_dot_allocated)

    # ------------------------------------------------------------------
    # Step 3: Evaluate all branches (using allocated flow rates)
    # ------------------------------------------------------------------
    branches      = []
    all_Re        = []
    all_dP        = []
    all_T_out     = []
    total_channel_length = 0.0

    for b in range(dv.n_branches):
        br = evaluate_branch(
            branch_id      = b,
            dv             = dv,
            m_dot_total    = M_DOT_TOTAL_KG_S,
            Q_total_W      = Q_total_W,
            fluid          = fluid,
            L_pass         = L_pass,
            diameters      = diameters,
            best_pitch     = pitch,
            m_dot_override = m_dot_allocated[b],   # v9 unequal split
        )
        # Store v9 fields in branch result
        br.connector_length_m = mg.connector_lengths_m[b % len(mg.connector_lengths_m)]
        br.branch_resistance  = branch_R_effs[b]
        br.flow_fraction      = m_dot_allocated[b] / max(M_DOT_TOTAL_KG_S, 1e-9)

        branches.append(br)
        all_T_out.append(br.T_out_C)
        all_dP.append(br.dP_total_Pa)
        total_channel_length += br.total_length_m

        for seg in br.segments:
            all_Re.append(seg.Re)

    result.branches               = branches
    result.total_channel_length_m = total_channel_length

    # ------------------------------------------------------------------
    # Step 4: System-level thermal quantities  (v9: branch-specific model)
    # ------------------------------------------------------------------
    # v9: compute per-branch battery temperatures using each branch's
    # actual allocated mass flow (not equal split).  This captures the
    # fact that poorly fed branches run hotter.
    T_batt_branches, T_batt_mean, T_batt_max, T_out_mean_weighted = \
        aggregate_design_thermal_metrics(
            dv, fluid, L_pass, diameters, pitch, m_dot_allocated, Q_total_W,
        )

    # Store per-branch T_batt back into BranchResult objects
    for b_idx, br in enumerate(branches):
        br.T_batt_branch_C = T_batt_branches[b_idx]

    # System outlet temperature (flow-weighted mean)
    T_out_avg = T_out_mean_weighted
    result.T_out_C = T_out_avg

    # Average coolant temperature
    T_coolant_avg = (T_INLET_C + T_out_avg) / 2.0
    result.T_coolant_avg_C = T_coolant_avg

    # v9: store branch-level temperature metrics
    result.T_batt_max_C  = T_batt_max
    result.T_batt_mean_C = T_batt_mean

    # ------------------------------------------------------------------
    # Thermal resistance (parallel segment model — same as v2/v8)
    # ------------------------------------------------------------------
    all_seg_R_total = []
    for br in branches:
        for seg in br.segments:
            R_seg = seg.R_total_K_W
            if R_seg > 0:
                all_seg_R_total.append(R_seg)

    if all_seg_R_total:
        R_system = 1.0 / sum(1.0 / R for R in all_seg_R_total)
    else:
        R_system = 1e6

    result.R_total_K_W = R_system

    delta_T = Q_total_W * R_system
    result.delta_T_batt_coolant_K = delta_T

    # T_batt_est: use the MEAN branch battery temperature from v9 model
    # (more physically meaningful than T_coolant_avg + delta_T with equal split)
    T_batt_est = T_batt_mean
    result.T_batt_est_C = T_batt_est

    # ------------------------------------------------------------------
    # Step 5: System-level hydraulic quantities  (v9: includes manifold ΔP)
    # ------------------------------------------------------------------
    # Base pressure drop from branch (representative branch, branch 0)
    dP_branch = branches[0].dP_total_Pa

    # v9: add manifold + connector friction and junction losses
    result.dP_manifold_Pa = dP_manifold_extra
    result.dP_total_Pa    = dP_branch + dP_manifold_extra
    result.Re_avg         = float(np.mean(all_Re)) if all_Re else 0.0

    # ------------------------------------------------------------------
    # Step 6: Coverage ratio
    # ------------------------------------------------------------------
    result.coverage_ratio = compute_coverage_ratio(dv, pitch, L_pass)

    # ------------------------------------------------------------------
    # Step 7: Thermal uniformity penalty  (unchanged from v8)
    # ------------------------------------------------------------------
    coverage_deficit = 1.0 - result.coverage_ratio
    dT_coolant_total = T_out_avg - T_INLET_C
    coolant_rise_penalty = min(dT_coolant_total / 20.0, 1.0)

    max_D = max(diameters)
    pitch_ratio  = pitch / (3.0 * max_D) if max_D > 0 else 1.0
    pitch_penalty = max(0.0, 1.0 - pitch_ratio)

    uniformity_penalty = (
        0.4 * coverage_deficit +
        0.4 * coolant_rise_penalty +
        0.2 * pitch_penalty
    )
    result.uniformity_penalty = uniformity_penalty

    # ------------------------------------------------------------------
    # Step 8: Manufacturability penalty  (unchanged)
    # ------------------------------------------------------------------
    D_ref       = 0.002
    D_penalties = [max(0.0, (D_ref - D) / D_ref) for D in diameters]
    diam_pen    = float(np.mean(D_penalties))
    R_ref       = 0.005
    bend_pen    = max(0.0, (R_ref - dv.bend_radius_m) / R_ref)
    if dv.use_3zone:
        max_step   = max(abs(dv.D1_m - dv.D2_m), abs(dv.D2_m - dv.D3_m))
        abrupt_pen = min(max_step / D_ref, 1.0)
    else:
        abrupt_pen = 0.0
    min_D     = min(diameters)
    dense_pen = max(0.0, 1.0 - (pitch / min_D) / 1.5) if min_D > 0 else 1.0
    manuf_penalty = (
        0.35 * diam_pen  +
        0.35 * bend_pen  +
        0.15 * abrupt_pen +
        0.15 * dense_pen
    )
    result.manuf_penalty = manuf_penalty

    # ------------------------------------------------------------------
    # Step 8b: Legacy geometric manifold penalty (kept, weight now 0 in score)
    # ------------------------------------------------------------------
    manifold_dist_pen = compute_manifold_uniformity_penalty(dv, pitch)
    result.manifold_dist_penalty = manifold_dist_pen

    # ------------------------------------------------------------------
    # Step 8c: Topology complexity penalty  (v8, unchanged)
    # ------------------------------------------------------------------
    topo_pen = topology_complexity_penalty(dv.topology, dv.turn_style)
    result.topology_complexity_pen = topo_pen

    # ------------------------------------------------------------------
    # Step 8d: v9 Physical branch-flow uniformity penalty
    # ------------------------------------------------------------------
    # This REPLACES W_MANIFOLD_DIST in the score.  It is physically grounded:
    # computed from std(branch_flows), connector variation, and resistance variation.
    branch_unif_pen = compute_branch_uniformity_penalty(
        dv, m_dot_allocated, mg, branch_R_effs,
    )
    result.branch_uniformity_pen = branch_unif_pen

    # ------------------------------------------------------------------
    # Step 8e: v9 aggregate branch-flow statistics  (for CSV export)
    # ------------------------------------------------------------------
    arr_m = np.array(m_dot_allocated)
    result.mean_branch_flow_kg_s  = float(np.mean(arr_m))
    result.std_branch_flow_kg_s   = float(np.std(arr_m))
    result.min_branch_flow_kg_s   = float(np.min(arr_m))
    result.max_branch_flow_kg_s   = float(np.max(arr_m))
    arr_R = np.array(branch_R_effs)
    result.mean_branch_resistance = float(np.mean(arr_R))
    result.std_branch_resistance  = float(np.std(arr_R))
    result.manifold_length_inlet_m  = mg.inlet_manifold_length_m
    result.manifold_length_outlet_m = mg.outlet_manifold_length_m

    # ------------------------------------------------------------------
    # Step 8f-NEW: Change 2 — Velocity-variation penalty
    # ------------------------------------------------------------------
    # Compute min/max local velocity and band penalty using the mean
    # allocated branch mass flow.
    v_min, v_max, vel_pen = compute_velocity_variation_penalty(
        dv, fluid, m_dot_allocated, diameters,
    )
    result.branch_velocity_min_m_s = v_min
    result.branch_velocity_max_m_s = v_max
    result.velocity_variation_pen  = vel_pen

    # ------------------------------------------------------------------
    # Step 8g-NEW: Change 3 — Branch-count / heat-flux scaling penalty
    # ------------------------------------------------------------------
    q_planar, Q_per_branch, A_per_branch, q_branch, hf_pen = \
        compute_branch_heatflux_penalty(dv, Q_total_W, pitch, L_pass, diameters)
    result.q_planar_W_m2   = q_planar
    result.Q_per_branch_W  = Q_per_branch
    result.A_per_branch_m2 = A_per_branch
    result.q_branch_W_m2   = q_branch
    result.branch_hf_penalty = hf_pen

    # ------------------------------------------------------------------
    # Step 8f: Layout sections for plotting  (v8, unchanged)
    # ------------------------------------------------------------------
    try:
        sections = generate_layout_sections(dv, pitch, L_pass, diameters)
        result.layout_sections = sections
    except Exception:
        result.layout_sections = []

    # ------------------------------------------------------------------
    # Step 9: Aggregate optimization score  (v9d updated score terms)
    # ------------------------------------------------------------------
    # score = w_T_batt   × T_batt_mean        (mean battery temp)
    #       + w_T_max    × T_batt_max          (worst-branch battery temp)
    #       + w_dP       × dP_total            (includes manifold losses)
    #       + w_mfdP     × dP_manifold         (manifold ΔP separately weighted)
    #       + w_unif     × uniformity_penalty   (coverage + coolant rise + pitch)
    #       + w_manuf    × manuf_penalty
    #       + w_bu       × branch_unif_pen     (replaces old W_MANIFOLD_DIST)
    #       + w_topo     × topo_complexity_pen
    #       + w_vel      × velocity_variation_pen  [Change 2: velocity control]
    #       + w_hf       × branch_hf_penalty       [Change 3: heat-flux scaling]
    # Lower score = better design.
    result.score = (
        W_T_BATT              * T_batt_mean       +
        W_T_BATT_MAX          * T_batt_max         +
        W_DELTA_P             * result.dP_total_Pa +
        W_MANIFOLD_DP         * dP_manifold_extra  +
        W_UNIFORMITY          * uniformity_penalty  +
        W_MANUFACTURABILITY   * manuf_penalty       +
        W_BRANCH_UNIFORMITY   * branch_unif_pen     +
        W_TOPOLOGY_COMPLEXITY * topo_pen            +
        W_VELOCITY_PEN        * vel_pen             +   # Change 2
        W_BRANCH_HF_PEN       * hf_pen                  # Change 3
    )

    return result


# =============================================================================
# 10. PARAMETER SWEEP
# =============================================================================

def build_design_variable_list() -> List[DesignVariables]:
    """
    Generate all combinations of design variables for the brute-force sweep.

    v10 update: both CONSTANT-diameter and TAPERED (3-zone) designs are
    included in the same sweep.  The flag USE_3ZONE_DIAMETER is no longer
    used as a hard gate — instead, both modes are always evaluated so the
    optimizer can compare them on an equal footing.

    Iterates over (v8 extension + v10 dual-mode):
        topology     (from TOPOLOGY_OPTIONS)
        manifold     (from MANIFOLD_OPTIONS, filtered by compatibility)
        turn_style   (from TURN_STYLE_OPTIONS)
        n_branches   (from N_BRANCHES_OPTIONS)
        n_passes     (from N_PASSES_OPTIONS)
        bend_radius  (from BEND_RADIUS_OPTIONS_M)
        [CONSTANT MODE]  D from D_CONST_OPTIONS_M
        [TAPERED  MODE]  D1/D2/D3 from D1/D2/D3_OPTIONS_M

    A design variable named `channel_mode` is set to "CONSTANT" or "TAPERED"
    for clear labelling in the CSV and console output.

    Incompatible topology–manifold pairs are skipped using
    _TOPO_MANIFOLD_COMPAT.

    Returns
    -------
    dvs : List[DesignVariables]
    """
    dvs = []

    # Outer loop: topology × manifold × turn_style
    for topology, manifold, turn_style in itertools.product(
        TOPOLOGY_OPTIONS,
        MANIFOLD_OPTIONS,
        TURN_STYLE_OPTIONS,
    ):
        # Skip incompatible topology–manifold combos early
        if not _TOPO_MANIFOLD_COMPAT.get((topology, manifold), False):
            continue

        base_kwargs = dict(topology=topology, manifold=manifold, turn_style=turn_style)

        # ── MODE A: Constant-diameter sweep ────────────────────────────────
        for nb, np_, rb, D in itertools.product(
            N_BRANCHES_OPTIONS,
            N_PASSES_OPTIONS,
            BEND_RADIUS_OPTIONS_M,
            D_CONST_OPTIONS_M,
        ):
            dv = DesignVariables(
                n_branches    = nb,
                n_passes      = np_,
                bend_radius_m = rb,
                use_3zone     = False,
                D_const_m     = D,
                **base_kwargs,
            )
            dvs.append(dv)

        # ── MODE B: 3-zone tapered-diameter sweep ──────────────────────────
        for nb, np_, rb, D1, D2, D3 in itertools.product(
            N_BRANCHES_OPTIONS,
            N_PASSES_OPTIONS,
            BEND_RADIUS_OPTIONS_M,
            D1_OPTIONS_M,
            D2_OPTIONS_M,
            D3_OPTIONS_M,
        ):
            # Only allow monotonically tapering diameters (D1 ≥ D2 ≥ D3).
            # This avoids physically nonsensical expanding-then-contracting profiles
            # and keeps the sweep from exploding in size.
            if not (D1 >= D2 >= D3):
                continue
            # Also reject designs where all three zones are identical
            # (those are already covered by constant-diameter mode).
            if D1 == D2 == D3:
                continue
            dv = DesignVariables(
                n_branches    = nb,
                n_passes      = np_,
                bend_radius_m = rb,
                use_3zone     = True,
                D1_m          = D1,
                D2_m          = D2,
                D3_m          = D3,
                **base_kwargs,
            )
            dvs.append(dv)

    return dvs


def run_parameter_sweep(Q_total_W: float, fluid: FluidProperties) -> List[DesignResult]:
    """
    Run brute-force parameter sweep over all candidate designs.

    Parameters
    ----------
    Q_total_W : float   total heat load [W]
    fluid     : FluidProperties

    Returns
    -------
    results : List[DesignResult]
    """
    dvs      = build_design_variable_list()
    results  = []
    n_total  = len(dvs)
    n_feasible = 0

    print(f"\n  Sweeping {n_total} candidate designs …", flush=True)

    for design_id, dv in enumerate(dvs):
        res = evaluate_design(design_id, dv, Q_total_W, fluid)
        results.append(res)
        if res.feasible:
            n_feasible += 1

    print(f"  Done.  Feasible designs: {n_feasible} / {n_total}")
    return results


# =============================================================================
# 11. RANKING & RESULTS
# =============================================================================

def rank_designs(results: List[DesignResult]) -> pd.DataFrame:
    """
    Convert results list to a pandas DataFrame, filter to feasible designs,
    and sort by score (ascending = best first).

    Returns
    -------
    df : pd.DataFrame
    """
    rows = []
    for r in results:
        dv = r.dv
        row = {
            "design_id":          r.design_id,
            "feasible":           r.feasible,
            "infeasibility":      r.infeasibility_reason,
            "score":              r.score,
            # v8: topology / routing style columns
            "topology":           dv.topology,
            "manifold":           dv.manifold,
            "turn_style":         dv.turn_style,
            # Design variables
            "n_branches":         dv.n_branches,
            "n_passes":           dv.n_passes,
            "bend_radius_mm":     dv.bend_radius_m * 1e3,
            "channel_mode":       "TAPERED" if dv.use_3zone else "CONSTANT",
            "D_const_mm":         dv.D_const_m * 1e3  if not dv.use_3zone else None,
            "D1_mm":              dv.D1_m * 1e3        if dv.use_3zone     else None,
            "D2_mm":              dv.D2_m * 1e3        if dv.use_3zone     else None,
            "D3_mm":              dv.D3_m * 1e3        if dv.use_3zone     else None,
            # Geometry
            "pass_length_mm":     r.straight_pass_length_m * 1e3,
            "pitch_mm":           r.branch_pitch_m * 1e3,
            "total_channel_m":    r.total_channel_length_m,
            "coverage_ratio":     r.coverage_ratio,
            # Thermal
            "Q_total_W":          r.Q_total_W,
            "T_out_C":            r.T_out_C,
            "T_coolant_avg_C":    r.T_coolant_avg_C,
            "T_batt_est_C":       r.T_batt_est_C,
            "T_batt_mean_C":      r.T_batt_mean_C,
            "T_batt_max_C":       r.T_batt_max_C,
            "dT_batt_coolant_K":  r.delta_T_batt_coolant_K,
            "R_total_K_W":        r.R_total_K_W,
            # Hydraulic
            "dP_Pa":              r.dP_total_Pa,
            "dP_manifold_Pa":     r.dP_manifold_Pa,
            "Re_avg":             r.Re_avg,
            # v9: branch flow statistics
            "mean_branch_flow":   r.mean_branch_flow_kg_s,
            "std_branch_flow":    r.std_branch_flow_kg_s,
            "min_branch_flow":    r.min_branch_flow_kg_s,
            "max_branch_flow":    r.max_branch_flow_kg_s,
            "mean_branch_resist": r.mean_branch_resistance,
            "std_branch_resist":  r.std_branch_resistance,
            "manifold_L_inlet_mm":r.manifold_length_inlet_m * 1e3,
            "manifold_L_outlet_mm":r.manifold_length_outlet_m * 1e3,
            # Penalties
            "uniformity_pen":     r.uniformity_penalty,
            "manuf_pen":          r.manuf_penalty,
            "manifold_dist_pen":  r.manifold_dist_penalty,
            "branch_unif_pen":    r.branch_uniformity_pen,
            "topo_complexity_pen":r.topology_complexity_pen,
            # Change 2: velocity control metrics
            "velocity_min_m_s":   r.branch_velocity_min_m_s,
            "velocity_max_m_s":   r.branch_velocity_max_m_s,
            "velocity_pen":       r.velocity_variation_pen,
            # Change 3: branch heat-flux scaling
            "q_planar_W_m2":      r.q_planar_W_m2,
            "Q_per_branch_W":     r.Q_per_branch_W,
            "A_per_branch_m2":    r.A_per_branch_m2,
            "q_branch_W_m2":      r.q_branch_W_m2,
            "branch_hf_pen":      r.branch_hf_penalty,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Separate feasible and infeasible; sort feasible by score
    df_feasible   = df[df["feasible"]].sort_values("score").reset_index(drop=True)
    df_infeasible = df[~df["feasible"]].copy()
    df_all        = pd.concat([df_feasible, df_infeasible], ignore_index=True)
    return df_all


def print_top_designs(df: pd.DataFrame, n: int = TOP_N_PRINT) -> None:
    """Print the top n feasible designs to console in a readable format."""
    df_feas = df[df["feasible"]].head(n)

    print(f"\n{'='*80}")
    print(f"  TOP {n} FEASIBLE DESIGNS  (ranked by score, lower = better)")
    print(f"{'='*80}")

    col_map = {
        "design_id":       "ID",
        "score":           "Score",
        "topology":        "Topology",
        "turn_style":      "TurnStyle",
        "n_branches":      "Branches",
        "n_passes":        "Passes",
        "bend_radius_mm":  "R_bend(mm)",
        "D_const_mm":      "D(mm)",
        "T_batt_est_C":    "T_batt(C)",
        "T_out_C":         "T_out(C)",
        "dP_Pa":           "dP(Pa)",
        "Re_avg":          "Re_avg",
        "coverage_ratio":  "Coverage",
        "pitch_mm":        "Pitch(mm)",
    }

    display_cols = [c for c in col_map if c in df_feas.columns]
    df_show = df_feas[display_cols].rename(columns=col_map)

    # Format floats
    float_fmt = {
        "Score":      "{:.2f}",
        "T_batt(C)":  "{:.2f}",
        "T_out(C)":   "{:.2f}",
        "dP(Pa)":     "{:.1f}",
        "Re_avg":     "{:.0f}",
        "Coverage":   "{:.3f}",
        "Pitch(mm)":  "{:.2f}",
        "R_bend(mm)": "{:.1f}",
        "D(mm)":      "{:.2f}",
    }
    for col, fmt in float_fmt.items():
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(
                lambda x, f=fmt: f.format(x) if pd.notna(x) else "-"
            )

    print(df_show.to_string(index=False))
    print(f"{'='*80}\n")


def save_results_csv(df: pd.DataFrame, filename: str = CSV_OUTPUT_FILE) -> None:
    """Save the full results DataFrame to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"  Results saved to '{filename}'")


def _draw_bend_arc(
    ax,
    sec,
    rb_mm:   float,
    col,
    lw:      float,
    turn_style: str,
    plate_w_mm: float,
    plate_l_mm: float,
) -> None:
    """
    Draw one serpentine U-turn arc correctly.

    TURN ORIENTATION (the key fix — Change 1)
    ------------------------------------------
    A serpentine channel must alternate which side the turn bulges on:
      H_SERPENTINE (horizontal passes, vertical bends at left/right edges):
          Pass p even  → going right  → turn on RIGHT side  → arc bulges RIGHT (+x)
          Pass p odd   → going left   → turn on LEFT side   → arc bulges LEFT  (-x)
          Sign determined by: x0 position relative to plate centre.
          x0 ≥ plate_w/2  →  right turn  →  sign = +1 (arc extends further right)
          x0 <  plate_w/2  →  left turn  →  sign = -1 (arc extends further left)

      V_SERPENTINE (vertical passes, horizontal bends at top/bottom):
          Pass p even  → going up    → turn at TOP    → arc bulges UP   (+y)
          Pass p odd   → going down  → turn at BOTTOM → arc bulges DOWN (-y)
          Sign determined by: y0 position relative to plate centre.
          y0 ≥ plate_l/2  →  top turn    →  sign = +1 (arc extends further up)
          y0 <  plate_l/2  →  bottom turn →  sign = -1 (arc extends further down)

    The pass_id alone cannot be used as a sign selector because the absolute
    pass direction depends on branch index; position relative to the plate
    centre is the correct discriminant and works for all branch indices.

    Parameters
    ----------
    ax          : matplotlib Axes
    sec         : LayoutSection  (bend section)
    rb_mm       : bend radius in mm
    col         : line colour
    lw          : line width
    turn_style  : "CONNECTOR_SEMICIRCLE" | "PURE_CIRCULAR" | "SMOOTH_SPLINE"
    plate_w_mm  : plate width in mm  (PW)
    plate_l_mm  : plate length in mm (PL)
    """
    dy = abs(sec.y1 - sec.y0)
    dx = abs(sec.x1 - sec.x0)

    if turn_style == "SMOOTH_SPLINE":
        t_arr = np.linspace(0, 1, 40)
        if dy > dx:
            # Vertical bend (H_SERPENTINE style): S-curve in x
            xs = sec.x0 + (sec.x1 - sec.x0) * 0.5 * (1 - np.cos(math.pi * t_arr))
            ys = sec.y0 + (sec.y1 - sec.y0) * t_arr
        else:
            # Horizontal bend (V_SERPENTINE style): S-curve in y
            xs = sec.x0 + (sec.x1 - sec.x0) * t_arr
            ys = sec.y0 + (sec.y1 - sec.y0) * 0.5 * (1 - np.cos(math.pi * t_arr))
        ax.plot(xs, ys, color=col, lw=lw, solid_capstyle="round", zorder=3)

    elif dy > dx:
        # ── Vertical bend (H_SERPENTINE): arc in x-direction ──────────────
        # sign: +1 → arc centre is to the RIGHT of x0; -1 → to the LEFT
        sign  = 1.0 if sec.x0 >= plate_w_mm / 2.0 else -1.0
        leg_h = max(0.0, dy - 2.0 * rb_mm) / 2.0
        cy    = min(sec.y0, sec.y1) + leg_h + rb_mm
        bx    = sec.x0
        theta = np.linspace(-math.pi / 2, math.pi / 2, 60)
        if leg_h > 0.01:
            ax.plot([bx, bx], [min(sec.y0, sec.y1), min(sec.y0, sec.y1) + leg_h],
                    color=col, lw=lw, zorder=3)
        ax.plot(bx + sign * rb_mm * np.cos(theta),
                cy + rb_mm * np.sin(theta),
                color=col, lw=lw, solid_capstyle="round", zorder=3)
        if leg_h > 0.01:
            ax.plot([bx, bx], [cy + rb_mm, max(sec.y0, sec.y1)],
                    color=col, lw=lw, zorder=3)

    else:
        # ── Horizontal bend (V_SERPENTINE): arc in y-direction ────────────
        # sign: +1 → arc bulges UPWARD (top turn); -1 → bulges DOWNWARD (bottom turn)
        # The turn is at y_top when sec.y0 ≥ plate_l/2, at y_bot otherwise.
        sign  = 1.0 if sec.y0 >= plate_l_mm / 2.0 else -1.0
        leg_h = max(0.0, dx - 2.0 * rb_mm) / 2.0
        cx    = min(sec.x0, sec.x1) + leg_h + rb_mm
        by    = sec.y0
        theta = np.linspace(-math.pi / 2, math.pi / 2, 60)
        if leg_h > 0.01:
            ax.plot([min(sec.x0, sec.x1), min(sec.x0, sec.x1) + leg_h], [by, by],
                    color=col, lw=lw, zorder=3)
        ax.plot(cx + rb_mm * np.cos(theta),
                by + sign * rb_mm * np.sin(theta),
                color=col, lw=lw, solid_capstyle="round", zorder=3)
        if leg_h > 0.01:
            ax.plot([cx + rb_mm, max(sec.x0, sec.x1)], [by, by],
                    color=col, lw=lw, zorder=3)


# =============================================================================
# 12. PLOTTING
# =============================================================================

def plot_results(df: pd.DataFrame) -> None:
    """
    Generate summary plots for the parameter sweep results.

    Plots
    -----
    1. Score vs estimated battery temperature
    2. Score vs pressure drop
    3. Pressure drop vs coolant outlet temperature
    4. Score distribution histogram
    """
    df_f = df[df["feasible"]].copy()
    if df_f.empty:
        print("  No feasible designs to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Battery Cooling Plate 1-D Optimizer – Parameter Sweep Results",
                 fontsize=14, fontweight="bold")

    # Colour by number of branches for visual separation
    branches_list = sorted(df_f["n_branches"].unique())
    cmap = plt.cm.get_cmap("tab10", len(branches_list))
    color_map = {b: cmap(i) for i, b in enumerate(branches_list)}
    colors = [color_map[b] for b in df_f["n_branches"]]

    # ---- Plot 1: Score vs Battery Temperature ----
    ax = axes[0, 0]
    sc = ax.scatter(df_f["T_batt_est_C"], df_f["score"],
                    c=colors, alpha=0.6, s=30, edgecolors="none")
    ax.set_xlabel("Estimated Battery Temperature (°C)")
    ax.set_ylabel("Optimization Score (lower = better)")
    ax.set_title("Score vs. Battery Temperature")
    ax.grid(True, linestyle="--", alpha=0.4)
    # Legend for branches
    handles = [mpatches.Patch(color=color_map[b], label=f"{b} branches")
               for b in branches_list]
    ax.legend(handles=handles, fontsize=8)

    # ---- Plot 2: Score vs Pressure Drop ----
    ax = axes[0, 1]
    ax.scatter(df_f["dP_Pa"] / 1000.0, df_f["score"],
               c=colors, alpha=0.6, s=30, edgecolors="none")
    ax.set_xlabel("Pressure Drop (kPa)")
    ax.set_ylabel("Optimization Score (lower = better)")
    ax.set_title("Score vs. Pressure Drop")
    ax.grid(True, linestyle="--", alpha=0.4)

    # ---- Plot 3: Pressure Drop vs Outlet Temperature ----
    ax = axes[1, 0]
    sc3 = ax.scatter(df_f["dP_Pa"] / 1000.0, df_f["T_out_C"],
                     c=df_f["T_batt_est_C"], alpha=0.7, s=30,
                     cmap="plasma", edgecolors="none")
    plt.colorbar(sc3, ax=ax, label="T_batt_est (°C)")
    ax.set_xlabel("Pressure Drop (kPa)")
    ax.set_ylabel("Coolant Outlet Temperature (°C)")
    ax.set_title("Pressure Drop vs. Outlet Temperature")
    ax.grid(True, linestyle="--", alpha=0.4)

    # ---- Plot 4: Score histogram ----
    ax = axes[1, 1]
    ax.hist(df_f["score"], bins=40, color="steelblue", edgecolor="white",
            alpha=0.8)
    ax.set_xlabel("Optimization Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution (feasible designs)")
    ax.grid(True, linestyle="--", alpha=0.4)
    # Mark best score
    best_score = df_f["score"].min()
    ax.axvline(best_score, color="red", linestyle="--",
               label=f"Best: {best_score:.2f}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("sweep_results.png", dpi=FIG_DPI, bbox_inches="tight")
    print("  Saved 'sweep_results.png'")
    plt.show()


def plot_best_design_schematic(best: DesignResult) -> None:
    """
    Draw a 2-D top-view schematic of the best serpentine cooling plate layout.

    v5 layout: realistic single inlet port → inlet header → branches → outlet header → outlet port
    ------------------------------------------------------------------------------------------------
    Previous versions shaded the entire left/right edge as a "manifold strip".
    This was misleading: a real cold plate has one circular inlet fitting and one
    circular outlet fitting, plus short internal distribution headers.

    New layout (all in the x-y plane, top view):
    ─────────────────────────────────────────────
    Physical flow path:
        [inlet port, left edge, centred vertically]
              │
        inlet header  (horizontal pipe, runs from left edge inward ~30 mm)
              │  (vertical header pipe spans all branch y-positions)
              ├── branch 1 connector ──► serpentine branch 1
              ├── branch 2 connector ──► serpentine branch 2
              └── branch N connector ──► serpentine branch N
                                               │
                                         outlet header
                                               │  (vertical, on right side)
                                         outlet port [right edge, centred]

    Coordinate convention (unchanged from v4):
        x-axis = plate WIDTH direction  (0 → 210 mm)
        y-axis = plate LENGTH direction (0 → 420 mm)
        Serpentine runs are horizontal (along x).
        Branches are stacked along y.

    All drawing uses ax.plot() lines and plt.Circle patches.
    No Arc patches, no bbox_inches="tight", axis limits set explicitly.

    Parameters
    ----------
    best : DesignResult   the best-ranked design
    """
    dv    = best.dv
    pitch = best.branch_pitch_m           # m  centre-to-centre pass spacing
    nb    = dv.n_branches
    np_   = dv.n_passes
    rb    = dv.bend_radius_m              # m  actual optimised bend radius

    # ------------------------------------------------------------------
    # Plate dimensions in mm (all plotting is in mm)
    # ------------------------------------------------------------------
    plate_w_mm    = BATTERY_WIDTH_M  * 1e3   # 210 mm  – x axis
    plate_l_mm    = BATTERY_LENGTH_M * 1e3   # 420 mm  – y axis
    edge_mm       = EDGE_OFFSET_M    * 1e3   # 10 mm edge clearance
    bend_r_mm     = rb * 1e3                 # design bend radius in mm
    pass_pitch_mm = pitch * 1e3              # centre-to-centre pass spacing in mm

    # ------------------------------------------------------------------
    # Port and header geometry
    # ------------------------------------------------------------------
    PORT_R_MM      = 5.0    # radius of the circular port symbol (purely visual)
    HEADER_INSET_MM = 25.0  # how far the inlet/outlet header extends into the plate
    HEADER_LW      = 2.5    # line width for the header pipes
    BRANCH_CONN_LW = 1.5    # line width for branch connector stubs

    # Inlet header: runs from x=0 inward to x=HEADER_INSET_MM
    # Then a vertical distribution pipe connects all branch y-positions.
    x_inlet_header_end = HEADER_INSET_MM     # x where the vertical header sits

    # Outlet header: vertical pipe at x = (plate_w_mm - HEADER_INSET_MM)
    # then runs to the right plate edge at x = plate_w_mm.
    x_outlet_header    = plate_w_mm - HEADER_INSET_MM

    # Ports are drawn just outside the plate boundary
    x_inlet_port  = -PORT_R_MM       # inlet port centre x (just left of plate)
    x_outlet_port = plate_w_mm + PORT_R_MM   # outlet port centre x (just right)

    # Vertical centre of the plate — ports are centred here
    y_plate_centre = plate_l_mm / 2.0

    # ------------------------------------------------------------------
    # Serpentine channel x-extents
    # Straight runs go between the header region and the far turn edge.
    # The inlet branches start at x_inlet_header_end (left header).
    # The U-turns at the right happen at x_right_mm.
    # The U-turns at the left (after the first right turn) are at x_left_turn_mm.
    # For simplicity: right turns at plate_w_mm - HEADER_INSET_MM - edge_mm
    #                 left  turns at x_inlet_header_end + edge_mm   (after the header)
    # ------------------------------------------------------------------
    x_left_turn_mm  = x_inlet_header_end + edge_mm      # left turnaround x
    x_right_turn_mm = x_outlet_header    - edge_mm      # right turnaround x

    # ------------------------------------------------------------------
    # y-positions of each branch's first pass
    # Branches distributed evenly along the plate length.
    # ------------------------------------------------------------------
    def y_of_pass(b_idx: int, p_idx: int) -> float:
        """y-coordinate (mm) of pass p_idx centreline in branch b_idx."""
        return edge_mm + b_idx * np_ * pass_pitch_mm + p_idx * pass_pitch_mm

    # Vertical span of all branches (for drawing the vertical header pipe)
    y_branch_top    = y_of_pass(nb - 1, np_ - 1)  # topmost pass centreline
    y_branch_bottom = y_of_pass(0, 0)              # bottommost pass centreline

    # ------------------------------------------------------------------
    # Figure setup — explicit size, equal aspect, fixed limits
    # ------------------------------------------------------------------
    aspect = plate_w_mm / plate_l_mm   # ~0.5
    fig_h  = 9.0
    fig_w  = max(7.0, fig_h * aspect * 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Set limits first; ports sit slightly outside the plate so add margin
    margin_mm = PORT_R_MM + 4.0
    ax.set_xlim(-margin_mm - 2, plate_w_mm + margin_mm + 2)
    ax.set_ylim(-margin_mm,     plate_l_mm + margin_mm)
    ax.set_aspect("equal", adjustable="box")

    connector_leg_mm = max(0.0, pass_pitch_mm - 2.0 * bend_r_mm)
    ax.set_title(
        f"Best Design Schematic (v5)  |  "
        f"branches={nb},  passes={np_},  "
        f"D={dv.D_const_m*1e3:.1f} mm,  "
        f"R_bend={bend_r_mm:.1f} mm,  "
        f"pitch={pass_pitch_mm:.1f} mm,  "
        f"leg={connector_leg_mm:.1f} mm",
        fontsize=10, fontweight="bold",
    )

    # ------------------------------------------------------------------
    # Plate footprint rectangle
    # ------------------------------------------------------------------
    ax.add_patch(plt.Rectangle(
        (0, 0), plate_w_mm, plate_l_mm,
        linewidth=2, edgecolor="#333333", facecolor="#f8f8f8", zorder=0,
    ))

    # ------------------------------------------------------------------
    # Light shading for header zones (narrow strips, not full-height fills)
    # ------------------------------------------------------------------
    # Inlet header zone (left inset strip)
    ax.add_patch(plt.Rectangle(
        (0, y_branch_bottom - 5), x_inlet_header_end, (y_branch_top - y_branch_bottom) + 10,
        linewidth=0, facecolor="#d6eaf8", alpha=0.6, zorder=1,
    ))
    # Outlet header zone (right inset strip)
    ax.add_patch(plt.Rectangle(
        (x_outlet_header, y_branch_bottom - 5), plate_w_mm - x_outlet_header,
        (y_branch_top - y_branch_bottom) + 10,
        linewidth=0, facecolor="#fdebd0", alpha=0.6, zorder=1,
    ))

    # ------------------------------------------------------------------
    # Inlet port  (circle on left plate edge, centred vertically)
    # ------------------------------------------------------------------
    inlet_port = plt.Circle(
        (x_inlet_port, y_plate_centre),
        PORT_R_MM,
        linewidth=2.0, edgecolor="#1a5276", facecolor="#aed6f1",
        zorder=5,
    )
    ax.add_patch(inlet_port)
    ax.text(
        x_inlet_port - PORT_R_MM - 1.5, y_plate_centre,
        "INLET", fontsize=8, color="#1a5276", fontweight="bold",
        ha="right", va="center",
    )

    # ------------------------------------------------------------------
    # Outlet port  (circle on right plate edge, centred vertically)
    # ------------------------------------------------------------------
    outlet_port = plt.Circle(
        (x_outlet_port, y_plate_centre),
        PORT_R_MM,
        linewidth=2.0, edgecolor="#784212", facecolor="#f0b27a",
        zorder=5,
    )
    ax.add_patch(outlet_port)
    ax.text(
        x_outlet_port + PORT_R_MM + 1.5, y_plate_centre,
        "OUTLET", fontsize=8, color="#784212", fontweight="bold",
        ha="left", va="center",
    )

    # ------------------------------------------------------------------
    # Inlet feed line: port → plate edge → horizontal header pipe
    # ------------------------------------------------------------------
    # Short horizontal pipe from port to the plate left edge (x=0)
    ax.plot(
        [x_inlet_port + PORT_R_MM, 0.0],
        [y_plate_centre, y_plate_centre],
        color="#1a5276", linewidth=HEADER_LW, solid_capstyle="butt", zorder=4,
    )
    # Horizontal header pipe inside the plate from x=0 to x=x_inlet_header_end
    ax.plot(
        [0.0, x_inlet_header_end],
        [y_plate_centre, y_plate_centre],
        color="#1a5276", linewidth=HEADER_LW, solid_capstyle="butt", zorder=4,
    )
    # Vertical distribution header: spans all branch first-pass y-positions
    ax.plot(
        [x_inlet_header_end, x_inlet_header_end],
        [y_branch_bottom, y_branch_top],
        color="#1a5276", linewidth=HEADER_LW, solid_capstyle="round", zorder=4,
    )

    # ------------------------------------------------------------------
    # Outlet collection line: vertical header → horizontal pipe → port
    # ------------------------------------------------------------------
    # Vertical collection header
    ax.plot(
        [x_outlet_header, x_outlet_header],
        [y_branch_bottom, y_branch_top],
        color="#784212", linewidth=HEADER_LW, solid_capstyle="round", zorder=4,
    )
    # Horizontal header pipe from x=x_outlet_header to x=plate_w_mm
    ax.plot(
        [x_outlet_header, plate_w_mm],
        [y_plate_centre, y_plate_centre],
        color="#784212", linewidth=HEADER_LW, solid_capstyle="butt", zorder=4,
    )
    # Short pipe from plate edge to outlet port
    ax.plot(
        [plate_w_mm, x_outlet_port - PORT_R_MM],
        [y_plate_centre, y_plate_centre],
        color="#784212", linewidth=HEADER_LW, solid_capstyle="butt", zorder=4,
    )

    # ------------------------------------------------------------------
    # Draw each serpentine branch
    # ------------------------------------------------------------------
    cmap = matplotlib.colormaps.get_cmap("tab10")
    channel_colors = [cmap(i / max(nb - 1, 1)) for i in range(nb)]

    # Channel line width scaled to diameter for visual realism
    lw = max(1.5, dv.D_const_m * 1e3 * 0.8)

    for b_idx in range(nb):
        color = channel_colors[b_idx]

        # ---- Inlet branch connector: vertical header → first pass ----
        # Short horizontal stub from the vertical inlet header to the
        # start of the first straight pass of this branch.
        y_first = y_of_pass(b_idx, 0)
        ax.plot(
            [x_inlet_header_end, x_left_turn_mm],
            [y_first, y_first],
            color=color, linewidth=BRANCH_CONN_LW,
            linestyle="--", solid_capstyle="round", zorder=3,
        )
        # Small arrowhead at the channel entry point
        ax.plot(
            x_left_turn_mm, y_first,
            marker=">", markersize=5, color=color, zorder=4,
        )

        # ---- Serpentine passes ----
        for p_idx in range(np_):
            y_mm = y_of_pass(b_idx, p_idx)

            # Direction alternates: even passes go left→right, odd right→left
            going_right = (p_idx % 2 == 0)
            x_from = x_left_turn_mm  if going_right else x_right_turn_mm
            x_to   = x_right_turn_mm if going_right else x_left_turn_mm

            # ---- Straight run ----
            ax.plot(
                [x_from, x_to], [y_mm, y_mm],
                color=color, linewidth=lw,
                solid_capstyle="round", zorder=3,
            )

            # ---- U-turn between this pass and the next ----
            # Uses the shared _draw_bend_arc() helper so all three renderers
            # produce geometrically identical, correctly alternating turns.
            if p_idx < np_ - 1:
                y_next_mm = y_of_pass(b_idx, p_idx + 1)
                bx_turn   = x_right_turn_mm if going_right else x_left_turn_mm
                # Synthesise a LayoutSection so _draw_bend_arc receives
                # the (x0,y0) → (x1,y1) bounding box it expects.
                # For H_SERPENTINE: vertical bend, x0==x1, y varies.
                _bend_sec = LayoutSection(
                    sec_type      = "bend",
                    bend_radius_m = dv.bend_radius_m,
                    x0 = bx_turn, y0 = y_mm,
                    x1 = bx_turn, y1 = y_next_mm,
                )
                _draw_bend_arc(ax, _bend_sec, bend_r_mm, color, lw,
                               dv.turn_style, plate_w_mm, plate_l_mm)

        # ---- Outlet branch connector: last pass → outlet vertical header ----
        last_p          = np_ - 1
        y_last          = y_of_pass(b_idx, last_p)
        going_right_last = (last_p % 2 == 0)
        # Last pass ends on right if going_right_last, else on left.
        # In either case we draw a dashed stub to the outlet vertical header.
        x_last_end = x_right_turn_mm if going_right_last else x_left_turn_mm

        if going_right_last:
            # Ends on the right → already at x_right_turn_mm, stub goes to header
            ax.plot(
                [x_right_turn_mm, x_outlet_header],
                [y_last, y_last],
                color=color, linewidth=BRANCH_CONN_LW,
                linestyle="--", solid_capstyle="round", zorder=3,
            )
        else:
            # Ends on the left → draw across the serpentine zone to outlet header
            ax.plot(
                [x_left_turn_mm, x_outlet_header],
                [y_last, y_last],
                color=color, linewidth=BRANCH_CONN_LW,
                linestyle="--", solid_capstyle="round", zorder=3,
            )
        # Small arrowhead at the header connection point
        ax.plot(
            x_outlet_header, y_last,
            marker=">", markersize=5, color=color, zorder=4,
        )

        # Branch label (near the inlet side)
        ax.text(
            x_inlet_header_end + 1.5, y_first + 0.5,
            f"B{b_idx + 1}",
            fontsize=7, color=color, va="bottom",
            fontweight="bold", zorder=5,
        )

    # ------------------------------------------------------------------
    # Edge clearance guide lines (subtle dashed lines)
    # ------------------------------------------------------------------
    ax.axhline(edge_mm,              color="#aaaaaa", linewidth=0.5,
               linestyle=":", zorder=1)
    ax.axhline(plate_l_mm - edge_mm, color="#aaaaaa", linewidth=0.5,
               linestyle=":", zorder=1)

    # ------------------------------------------------------------------
    # Pitch dimension annotation (right of plate, between first two passes)
    # ------------------------------------------------------------------
    if nb > 0 and np_ > 1:
        y0 = y_of_pass(0, 0)
        y1 = y_of_pass(0, 1)
        ann_x = plate_w_mm + margin_mm * 0.4
        ax.plot([ann_x, ann_x], [y0, y1],
                color="dimgrey", linewidth=0.8, zorder=3)
        ax.plot(ann_x, y0, marker="_", markersize=6, color="dimgrey", zorder=3)
        ax.plot(ann_x, y1, marker="_", markersize=6, color="dimgrey", zorder=3)
        ax.text(ann_x + 1.5, (y0 + y1) / 2.0,
                f"pitch\n{pass_pitch_mm:.1f} mm",
                fontsize=7, color="dimgrey", va="center", ha="left")

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    legend_handles = [
        mpatches.Patch(facecolor="#aed6f1", edgecolor="#1a5276",
                       label="Inlet header"),
        mpatches.Patch(facecolor="#f0b27a", edgecolor="#784212",
                       label="Outlet header"),
    ] + [
        mpatches.Patch(color=channel_colors[i], label=f"Branch {i + 1}")
        for i in range(nb)
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=7, framealpha=0.85)

    # ------------------------------------------------------------------
    # Axis labels and grid
    # ------------------------------------------------------------------
    ax.set_xlabel("Plate width direction (mm)", fontsize=9)
    ax.set_ylabel("Plate length direction (mm)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.2, zorder=0)

    # ------------------------------------------------------------------
    # Lock axis limits AFTER all artists, then disable autoscale.
    # This prevents the port circles (which extend slightly outside the
    # plate boundary) from triggering further axis expansion.
    # ------------------------------------------------------------------
    ax.set_xlim(-margin_mm - 2, plate_w_mm + margin_mm + 2)
    ax.set_ylim(-margin_mm,     plate_l_mm + margin_mm)
    ax.autoscale(False)

    plt.tight_layout()
    plt.savefig("best_design_schematic.png", dpi=FIG_DPI)
    print("  Saved 'best_design_schematic.png'  (v5: single inlet/outlet port + headers)")
    plt.show()


# =============================================================================
# 13. PLANAR GEOMETRY SCHEMATIC  (new in v5 / engineering-drawing output)
# =============================================================================

def _build_section_table(
    best:        "DesignResult",
    diameters:   List[float],
    pitch_mm:    float,
    L_pass_mm:   float,
    rb_mm:       float,
    PORT_D_MM:   float,
    HDR_D_MM:    float,
    HEADER_INSET_MM: float,
) -> List[dict]:
    """
    Build a flat list of all hydraulic sections in flow order.
    Each entry is a dict with keys:
        section, type, branch, pass_, diameter_mm, length_mm

    Flow order:
        inlet port → inlet H-header → inlet V-header →
        [ for each branch:
              branch connector →
              [ for each pass:
                    straight run → (U-turn arc + legs if not last pass) ] ] →
        outlet V-header → outlet H-header → outlet port
    """
    dv   = best.dv
    nb   = dv.n_branches
    np_  = dv.n_passes
    rows = []

    # Inlet port
    rows.append(dict(section="Inlet port",      type_="port",    branch="-", pass_="-",
                     diameter_mm=PORT_D_MM,     length_mm=PORT_D_MM * 2.0))

    # Inlet horizontal header (port → vertical header junction)
    rows.append(dict(section="Inlet H-header",  type_="header",  branch="-", pass_="-",
                     diameter_mm=HDR_D_MM,       length_mm=HEADER_INSET_MM))

    # Inlet vertical header (spans all branch y-positions)
    total_span_mm = (nb * np_ - 1) * pitch_mm
    rows.append(dict(section="Inlet V-header",  type_="header",  branch="-", pass_="-",
                     diameter_mm=HDR_D_MM,       length_mm=total_span_mm))

    # Per-branch sections
    for b in range(nb):
        # Branch connector (stub from V-header to first pass)
        rows.append(dict(section=f"B{b+1} inlet conn", type_="connector", branch=b+1, pass_="-",
                         diameter_mm=HDR_D_MM,           length_mm=10.0))  # short stub

        for p in range(np_):
            D_mm = diameters[p] * 1e3

            # Straight pass
            rows.append(dict(section=f"B{b+1} P{p+1} straight", type_="straight",
                             branch=b+1, pass_=p+1,
                             diameter_mm=D_mm, length_mm=L_pass_mm))

            # U-turn (not after last pass)
            if p < np_ - 1:
                leg_half_mm = max(0.0, (pitch_mm - 2.0 * rb_mm) / 2.0)
                arc_len_mm  = math.pi * rb_mm
                turn_len_mm = arc_len_mm + 2.0 * leg_half_mm
                rows.append(dict(section=f"B{b+1} P{p+1}→P{p+2} U-turn", type_="bend",
                                 branch=b+1, pass_=p+1,
                                 diameter_mm=D_mm, length_mm=turn_len_mm))

        # Branch outlet connector
        rows.append(dict(section=f"B{b+1} outlet conn", type_="connector", branch=b+1, pass_="-",
                         diameter_mm=HDR_D_MM,            length_mm=10.0))

    # Outlet vertical header
    rows.append(dict(section="Outlet V-header", type_="header", branch="-", pass_="-",
                     diameter_mm=HDR_D_MM,       length_mm=total_span_mm))

    # Outlet horizontal header
    rows.append(dict(section="Outlet H-header", type_="header", branch="-", pass_="-",
                     diameter_mm=HDR_D_MM,       length_mm=HEADER_INSET_MM))

    # Outlet port
    rows.append(dict(section="Outlet port",      type_="port",   branch="-", pass_="-",
                     diameter_mm=PORT_D_MM,       length_mm=PORT_D_MM * 2.0))

    return rows


def _print_section_table(rows: List[dict]) -> None:
    """Print the section table to the console in a readable fixed-width format."""
    hdr = f"{'Section':<28} {'Type':<12} {'Branch':>6} {'Pass':>5} {'Ø (mm)':>8} {'L (mm)':>9}"
    sep = "─" * len(hdr)
    print("\n  " + sep)
    print("  " + hdr)
    print("  " + sep)
    for r in rows:
        print(f"  {r['section']:<28} {r['type_']:<12} {str(r['branch']):>6} "
              f"{str(r['pass_']):>5} {r['diameter_mm']:>8.2f} {r['length_mm']:>9.2f}")
    print("  " + sep)


def plot_planar_geometry_schematic(
    best:       "DesignResult",
    outfile:    str  = "planar_geometry_schematic.png",
    show_table: bool = True,
    label:      str  = "",
) -> None:
    """
    Produce an engineering plan-view drawing of the optimised cooling plate.

    Parameters (additions over v5)
    --------------------------------
    outfile    : filename to save the PNG to (default: planar_geometry_schematic.png)
    show_table : whether to print and save the section CSV table (default: True)
    label      : optional extra label shown in the info box (e.g. "Best Design")

    This is a CAD-driving layout map, not a conceptual flow diagram.
    It shows exact centreline routing, hydraulic diameters, and dimensions
    so that a draughter or CAD operator can reproduce the geometry without
    ambiguity.

    What is drawn
    -------------
    ┌──────────────────────────────────────────────────────────────────┐
    │  Plate outline  (420 × 210 mm, to scale)                         │
    │                                                                  │
    │  ●── H-header ──┬── branch connector ──[serpentine passes]──┬── H-header ──●  │
    │  (inlet port)   │   (per branch)                             │   (outlet port) │
    │                 V-header                               V-header               │
    └──────────────────────────────────────────────────────────────────┘

    Annotations (all dimensions in mm):
      • port diameter label   Ø8
      • header diameter label
      • per-pass channel Ø    (one label per unique diameter, e.g. in 3-zone mode)
      • pass pitch  ↕ annotation on the right margin
      • bend radius annotation inside first right-side bend
      • straight run length annotation on first branch top pass
      • info box (bottom-right) listing n_branches, n_passes, D, R_bend, pitch

    Line weights represent relative diameter:
      • port   : thickest (lw_port)
      • header : medium   (lw_hdr)
      • channel: scaled by (D/D_max) × lw_channel_max

    All drawing uses ax.plot() (no Arc patches).
    Axis limits set explicitly; autoscale disabled; no bbox_inches='tight'.

    Parameters
    ----------
    best : DesignResult
    """
    # ------------------------------------------------------------------
    # Unpack geometry
    # ------------------------------------------------------------------
    dv    = best.dv
    pitch = best.branch_pitch_m           # m
    nb    = dv.n_branches
    np_   = dv.n_passes
    rb    = dv.bend_radius_m              # m

    # Plate dims (mm — all plotting is in mm)
    PL_MM  = BATTERY_LENGTH_M * 1e3      # 420 mm  y-axis (plate length)
    PW_MM  = BATTERY_WIDTH_M  * 1e3      # 210 mm  x-axis (plate width)
    EDGE   = EDGE_OFFSET_M    * 1e3      # 10 mm

    pitch_mm  = pitch * 1e3
    rb_mm     = rb    * 1e3
    diameters = get_diameter_profile(dv)  # list[float], one per pass [m]
    D_max_mm  = max(d * 1e3 for d in diameters)

    # ------------------------------------------------------------------
    # Port and header geometry (visual / representative sizes)
    # ------------------------------------------------------------------
    PORT_D_MM    = 8.0      # nominal port inner diameter for labelling
    HDR_D_MM     = 6.0      # header pipe inner diameter for labelling
    PORT_R_MM    = 5.0      # visual radius of port circle on the drawing
    HEADER_INSET = 25.0     # how far into the plate the horizontal header runs (mm)
    STUB_MM      = 8.0      # branch connector stub length (mm)

    # x-coordinates of the two vertical distribution headers
    x_inlet_vhdr  = HEADER_INSET
    x_outlet_vhdr = PW_MM - HEADER_INSET

    # Port symbols sit just outside the plate boundary
    x_inlet_port  = -PORT_R_MM
    x_outlet_port = PW_MM + PORT_R_MM

    # Serpentine channel x extents (between the two vertical headers + small gap)
    x_left_turn  = x_inlet_vhdr  + EDGE
    x_right_turn = x_outlet_vhdr - EDGE
    L_pass_mm    = x_right_turn - x_left_turn   # straight run length (mm)

    # y-position of each pass centreline
    def y_of_pass(b: int, p: int) -> float:
        return EDGE + b * np_ * pitch_mm + p * pitch_mm

    # Vertical header spans first-pass of branch 0 → last pass of branch (nb-1)
    y_hdr_bot = y_of_pass(0,    0)
    y_hdr_top = y_of_pass(nb-1, np_-1)
    y_centre  = PL_MM / 2.0   # for port position

    # ------------------------------------------------------------------
    # Line-weight map
    # ------------------------------------------------------------------
    LW_PORT    = 4.0
    LW_HDR     = 2.5
    LW_CONN    = 1.5
    LW_CH_MAX  = 3.0
    LW_CH_MIN  = 1.0

    def ch_lw(D_m: float) -> float:
        """Channel line width proportional to diameter."""
        t = (D_m * 1e3 - D_max_mm * 0.5) / (D_max_mm * 0.5) if D_max_mm > 0 else 0.5
        t = max(0.0, min(1.0, t))
        return LW_CH_MIN + t * (LW_CH_MAX - LW_CH_MIN)

    # ------------------------------------------------------------------
    # Colour palette
    # ------------------------------------------------------------------
    INLET_COL   = "#1a5276"   # dark blue
    OUTLET_COL  = "#784212"   # dark orange
    PLATE_COL   = "#fdfefe"   # near-white plate face
    PLATE_EDGE  = "#2c3e50"   # dark border
    cmap_ch     = matplotlib.colormaps.get_cmap("tab10")
    branch_col  = [cmap_ch(i / max(nb - 1, 1)) for i in range(nb)]

    DIM_COL  = "#555555"   # dimension annotation colour
    ANN_COL  = "#1c1c1c"   # text annotation colour

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------
    fig_w = 14.0
    fig_h = 10.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#f0f0f0")
    ax.set_facecolor("#e8e8e8")

    # Generous left/right margin so port labels and annotations fit
    MARGIN = 28.0
    ax.set_xlim(-MARGIN, PW_MM + MARGIN)
    ax.set_ylim(-18.0,   PL_MM + 22.0)
    ax.set_aspect("equal", adjustable="box")

    # ------------------------------------------------------------------
    # Plate outline + hatching to indicate solid aluminium boundary
    # ------------------------------------------------------------------
    ax.add_patch(plt.Rectangle(
        (0, 0), PW_MM, PL_MM,
        linewidth=2.0, edgecolor=PLATE_EDGE, facecolor=PLATE_COL, zorder=0,
    ))
    # Corner dimension text (plate size)
    ax.text(PW_MM / 2.0, PL_MM + 3.5,
            f"Plate  {PW_MM:.0f} mm × {PL_MM:.0f} mm  (W × L)",
            fontsize=8.5, ha="center", va="bottom",
            color=ANN_COL, fontweight="bold")

    # ---- thin edge-offset guide lines ----
    for yg in [EDGE, PL_MM - EDGE]:
        ax.plot([0, PW_MM], [yg, yg],
                color="#cccccc", lw=0.5, ls=":", zorder=1)

    # ------------------------------------------------------------------
    # Inlet port circle + label
    # ------------------------------------------------------------------
    ax.add_patch(plt.Circle(
        (x_inlet_port, y_centre), PORT_R_MM,
        lw=LW_PORT, edgecolor=INLET_COL, facecolor="#aed6f1", zorder=6,
    ))
    ax.text(x_inlet_port, y_centre + PORT_R_MM + 3.5,
            f"INLET\nØ{PORT_D_MM:.0f} mm",
            ha="center", va="bottom", fontsize=7.5,
            color=INLET_COL, fontweight="bold")
    # Feed line: port → plate left edge
    ax.plot([x_inlet_port + PORT_R_MM, 0],
            [y_centre, y_centre],
            color=INLET_COL, lw=LW_HDR, solid_capstyle="butt", zorder=5)

    # ------------------------------------------------------------------
    # Outlet port circle + label
    # ------------------------------------------------------------------
    ax.add_patch(plt.Circle(
        (x_outlet_port, y_centre), PORT_R_MM,
        lw=LW_PORT, edgecolor=OUTLET_COL, facecolor="#f0b27a", zorder=6,
    ))
    ax.text(x_outlet_port, y_centre + PORT_R_MM + 3.5,
            f"OUTLET\nØ{PORT_D_MM:.0f} mm",
            ha="center", va="bottom", fontsize=7.5,
            color=OUTLET_COL, fontweight="bold")
    # Feed line: plate right edge → port
    ax.plot([PW_MM, x_outlet_port - PORT_R_MM],
            [y_centre, y_centre],
            color=OUTLET_COL, lw=LW_HDR, solid_capstyle="butt", zorder=5)

    # ------------------------------------------------------------------
    # Inlet horizontal header: plate edge (x=0) → vertical header (x=HEADER_INSET)
    # ------------------------------------------------------------------
    ax.plot([0, x_inlet_vhdr], [y_centre, y_centre],
            color=INLET_COL, lw=LW_HDR, solid_capstyle="butt", zorder=5)
    # label
    ax.text(x_inlet_vhdr / 2.0, y_centre + 2.5,
            f"Ø{HDR_D_MM:.0f}", fontsize=6.5, ha="center", va="bottom",
            color=INLET_COL)

    # ------------------------------------------------------------------
    # Outlet horizontal header: vertical header → plate right edge
    # ------------------------------------------------------------------
    ax.plot([x_outlet_vhdr, PW_MM], [y_centre, y_centre],
            color=OUTLET_COL, lw=LW_HDR, solid_capstyle="butt", zorder=5)
    ax.text((x_outlet_vhdr + PW_MM) / 2.0, y_centre + 2.5,
            f"Ø{HDR_D_MM:.0f}", fontsize=6.5, ha="center", va="bottom",
            color=OUTLET_COL)

    # ------------------------------------------------------------------
    # Inlet vertical distribution header  (Change 2: tapered visualisation)
    # ------------------------------------------------------------------
    # The inlet header is shown as a filled polygon that is wider at the
    # port end (bottom, where it connects to the horizontal feed pipe) and
    # narrows toward the top (far end).  This reflects MANIFOLD_TAPER_RATIO.
    # Width at port end  = HDR_VISUAL_W_MM
    # Width at far end   = HDR_VISUAL_W_MM * (1 - MANIFOLD_TAPER_RATIO)
    HDR_VISUAL_W_MM  = MANIFOLD_INLET_WIDTH_M * 1e3   # convert to mm for plotting
    # Taper: port is at y_hdr_bot; far end is at y_hdr_top
    w_bot = HDR_VISUAL_W_MM / 2.0                     # half-width at port end
    w_top = w_bot * (1.0 - MANIFOLD_TAPER_RATIO)      # half-width at far end
    inlet_hdr_poly_x = [
        x_inlet_vhdr - w_bot, x_inlet_vhdr + w_bot,  # bottom edge
        x_inlet_vhdr + w_top, x_inlet_vhdr - w_top,  # top edge
        x_inlet_vhdr - w_bot,                         # close
    ]
    inlet_hdr_poly_y = [y_hdr_bot, y_hdr_bot, y_hdr_top, y_hdr_top, y_hdr_bot]
    ax.fill(inlet_hdr_poly_x, inlet_hdr_poly_y,
            color="#aed6f1", alpha=0.5, zorder=4)
    ax.plot(inlet_hdr_poly_x, inlet_hdr_poly_y,
            color=INLET_COL, lw=1.0, zorder=5)
    # Taper label
    ax.text(x_inlet_vhdr - w_bot - 1.5, (y_hdr_bot + y_hdr_top) / 2.0,
            f"Inlet header\n"
            f"W_port={HDR_VISUAL_W_MM:.0f} mm\n"
            f"W_far ={HDR_VISUAL_W_MM*(1-MANIFOLD_TAPER_RATIO):.0f} mm\n"
            f"taper ={MANIFOLD_TAPER_RATIO:.0%}",
            fontsize=6.0, ha="right", va="center",
            color=INLET_COL, rotation=90, rotation_mode="anchor")

    # ------------------------------------------------------------------
    # Outlet vertical collection header  (Change 2: tapered visualisation)
    # ------------------------------------------------------------------
    # The outlet header is the mirror image: narrowest at the first collecting
    # branch (y_hdr_bot) and widens toward the port side (y_hdr_top).
    # This reflects the progressive flow accumulation as branches rejoin.
    w_out_bot = w_top    # narrowest at first collecting branch
    w_out_top = w_bot    # widest at the port side
    outlet_hdr_poly_x = [
        x_outlet_vhdr - w_out_bot, x_outlet_vhdr + w_out_bot,
        x_outlet_vhdr + w_out_top, x_outlet_vhdr - w_out_top,
        x_outlet_vhdr - w_out_bot,
    ]
    outlet_hdr_poly_y = [y_hdr_bot, y_hdr_bot, y_hdr_top, y_hdr_top, y_hdr_bot]
    ax.fill(outlet_hdr_poly_x, outlet_hdr_poly_y,
            color="#f0b27a", alpha=0.5, zorder=4)
    ax.plot(outlet_hdr_poly_x, outlet_hdr_poly_y,
            color=OUTLET_COL, lw=1.0, zorder=5)
    ax.text(x_outlet_vhdr + w_out_top + 1.5, (y_hdr_bot + y_hdr_top) / 2.0,
            f"Outlet header\n"
            f"W_bot ={HDR_VISUAL_W_MM*(1-MANIFOLD_TAPER_RATIO):.0f} mm\n"
            f"W_top ={HDR_VISUAL_W_MM:.0f} mm\n"
            f"taper ={MANIFOLD_TAPER_RATIO:.0%}",
            fontsize=6.0, ha="left", va="center",
            color=OUTLET_COL, rotation=90, rotation_mode="anchor")


    # ------------------------------------------------------------------
    # Draw branches: uses layout_sections if available (topology-aware)
    # Falls back to classic H_SERPENTINE rendering.
    # ------------------------------------------------------------------
    # Tap-off tick marks on vertical headers (H_SERP / Z_FLOW only)
    topo_s = best.dv.topology
    if topo_s in ("H_SERPENTINE", "Z_FLOW"):
        for b_idx in range(nb):
            y_tap = y_of_pass(b_idx, 0)
            ax.plot([x_inlet_vhdr - 3, x_inlet_vhdr + 3],
                    [y_tap, y_tap], color=INLET_COL, lw=1.2, zorder=5)
            y_tap_out = y_of_pass(b_idx, np_ - 1)
            ax.plot([x_outlet_vhdr - 3, x_outlet_vhdr + 3],
                    [y_tap_out, y_tap_out], color=OUTLET_COL, lw=1.2, zorder=5)

    labelled_diameters: set = set()
    use_sections = (best.layout_sections and topo_s not in ("H_SERPENTINE",))

    # ── Shared connector drawing helper ──────────────────────────────────
    def _draw_conn_schematic(x0, y0, x1, y1, col):
        """Thick solid connector + white-halo junction dots at both ends."""
        ax.plot([x0, x1], [y0, y1],
                color=col, lw=LW_CONN * 2.2,
                solid_capstyle="round", zorder=6)
        for xd, yd in [(x0, y0), (x1, y1)]:
            ax.plot(xd, yd, "o", color="white",
                    markersize=9.5, zorder=9, markeredgewidth=0)
            ax.plot(xd, yd, "o", color=col,
                    markersize=8.0, zorder=10,
                    markeredgecolor="white", markeredgewidth=1.0)

    if use_sections:
        # ── Draw straights and bends ──────────────────────────────────────
        for sec in best.layout_sections:
            if sec.sec_type not in ("straight", "bend"):
                continue
            col  = branch_col[sec.branch_id % nb] if sec.branch_id >= 0 else INLET_COL
            lw_c = ch_lw(sec.diameter_m)
            if sec.sec_type == "straight":
                ax.plot([sec.x0, sec.x1], [sec.y0, sec.y1],
                        color=col, lw=lw_c, solid_capstyle="round", zorder=3)
                D_mm = sec.diameter_m * 1e3
                diam_key = round(D_mm, 3)
                if sec.branch_id == 0 and diam_key not in labelled_diameters:
                    labelled_diameters.add(diam_key)
                    ax.text((sec.x0+sec.x1)/2, (sec.y0+sec.y1)/2+1.8,
                            f"Ø{D_mm:.1f} mm", fontsize=6.5,
                            ha="center", va="bottom", color=col, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                      edgecolor=col, linewidth=0.6, alpha=0.85), zorder=6)
            else:  # bend — use shared helper with correct turn orientation
                _draw_bend_arc(ax, sec, sec.bend_radius_m * 1e3, col, lw_c,
                               best.dv.turn_style, PW_MM, PL_MM)

        # Branch labels
        for b_idx in range(nb):
            col = branch_col[b_idx]
            bsecs_s = [s for s in best.layout_sections
                       if s.branch_id == b_idx and s.sec_type == "straight"]
            if bsecs_s:
                s0 = bsecs_s[0]
                ax.text(s0.x0 + 2.0, s0.y0 + 0.8, f"B{b_idx+1}",
                        fontsize=7, color=col, va="bottom", fontweight="bold", zorder=6)

        # ── Branch-to-manifold connectors (topology-specific coordinates) ─
        VINSET_S = HEADER_INSET   # 25 mm

        for b_idx in range(nb):
            col = branch_col[b_idx]
            bstraight = [s for s in best.layout_sections
                         if s.branch_id == b_idx and s.sec_type == "straight"]
            if not bstraight:
                continue
            s_first = bstraight[0]
            s_last  = bstraight[-1]

            if topo_s == "V_SERPENTINE":
                y_mfld_in  = PL_MM - VINSET_S    # 395 mm
                y_mfld_out = VINSET_S             # 25 mm
                x_c  = s_first.x0
                y_branch_top = max(s_first.y0, s_first.y1)
                _draw_conn_schematic(x_c, y_mfld_in, x_c, y_branch_top, col)
                x_co = s_last.x0
                y_branch_bot = min(s_last.y0, s_last.y1)
                _draw_conn_schematic(x_co, y_branch_bot, x_co, y_mfld_out, col)

            elif topo_s == "Z_FLOW":
                y_in  = s_first.y0
                y_out = s_last.y0
                x_br_in  = min(s_first.x0, s_first.x1)
                last_p_even = (s_last.pass_id % 2 == 0)
                x_br_out = max(s_last.x0, s_last.x1) if last_p_even \
                           else min(s_last.x0, s_last.x1)
                _draw_conn_schematic(x_inlet_vhdr, y_in, x_br_in, y_in, col)
                _draw_conn_schematic(x_br_out, y_out, x_outlet_vhdr, y_out, col)

            elif topo_s == "MIRRORED_U":
                y_mfld = HEADER_INSET
                x_ci = s_first.x0
                y_br_start = min(s_first.y0, s_first.y1)
                _draw_conn_schematic(x_ci, y_mfld, x_ci, y_br_start, col)
                x_co = s_last.x0
                y_br_end = min(s_last.y0, s_last.y1)
                _draw_conn_schematic(x_co, y_mfld, x_co, y_br_end, col)

            elif topo_s == "CENTRAL_INLET":
                y_mfld_in = PL_MM / 2.0
                x_ci = s_first.x0
                y_br_in = s_first.y0
                _draw_conn_schematic(x_ci, y_mfld_in, x_ci, y_br_in, col)
                x_co = s_last.x0
                y_br_out = s_last.y1
                y_out_mfld = (PL_MM - EDGE) if y_br_out > PL_MM / 2.0 else EDGE
                _draw_conn_schematic(x_co, y_br_out, x_co, y_out_mfld, col)

    else:
        # ── Classic H_SERPENTINE rendering ───────────────────────────────
        for b_idx in range(nb):
            col = branch_col[b_idx]
            y_first = y_of_pass(b_idx, 0)
            y_last  = y_of_pass(b_idx, np_ - 1)

            for p_idx in range(np_):
                D_m  = diameters[p_idx]
                D_mm = D_m * 1e3
                lw_c = ch_lw(D_m)
                y_mm = y_of_pass(b_idx, p_idx)
                going_right = (p_idx % 2 == 0)
                x_from = x_left_turn  if going_right else x_right_turn
                x_to   = x_right_turn if going_right else x_left_turn
                ax.plot([x_from, x_to], [y_mm, y_mm],
                        color=col, lw=lw_c, solid_capstyle="round", zorder=3)
                diam_key = round(D_mm, 3)
                if b_idx == 0 and diam_key not in labelled_diameters:
                    labelled_diameters.add(diam_key)
                    x_mid = (x_from + x_to) / 2.0
                    ax.text(x_mid, y_mm+1.8, f"Ø{D_mm:.1f} mm",
                            fontsize=6.5, ha="center", va="bottom",
                            color=col, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                      edgecolor=col, linewidth=0.6, alpha=0.85), zorder=6)
                if p_idx < np_ - 1:
                    y_next = y_of_pass(b_idx, p_idx + 1)
                    bx_turn = x_right_turn if going_right else x_left_turn
                    _bend_sec = LayoutSection(
                        sec_type="bend", bend_radius_m=best.dv.bend_radius_m,
                        x0=bx_turn, y0=y_mm, x1=bx_turn, y1=y_next,
                    )
                    _draw_bend_arc(ax, _bend_sec, rb_mm, col, lw_c,
                                   best.dv.turn_style, PW_MM, PL_MM)

            ax.text(x_inlet_vhdr + 2.0, y_first + 0.8, f"B{b_idx + 1}",
                    fontsize=7, color=col, va="bottom", fontweight="bold", zorder=6)

        # ── H_SERPENTINE connectors: manifold centreline → first/last pass ─
        for b_idx in range(nb):
            col     = branch_col[b_idx]
            y_first = y_of_pass(b_idx, 0)
            y_last  = y_of_pass(b_idx, np_ - 1)
            going_right_last = ((np_ - 1) % 2 == 0)
            x_last_end = x_right_turn if going_right_last else x_left_turn
            _draw_conn_schematic(x_inlet_vhdr,  y_first, x_left_turn,  y_first, col)
            _draw_conn_schematic(x_last_end,    y_last,  x_outlet_vhdr, y_last, col)

    # ------------------------------------------------------------------
    # DIMENSION ANNOTATIONS  (right margin and inside plate)
    # ------------------------------------------------------------------
    ann_x = PW_MM + 8.0   # x position for right-margin annotations

    # ── Pass pitch (between first two passes of branch 0) ──
    if np_ > 1:
        y0 = y_of_pass(0, 0)
        y1 = y_of_pass(0, 1)
        ym = (y0 + y1) / 2.0
        # Double-headed tick line
        ax.plot([ann_x, ann_x], [y0, y1], color=DIM_COL, lw=1.0, zorder=6)
        ax.plot(ann_x, y0, marker="_", markersize=7, color=DIM_COL, zorder=6)
        ax.plot(ann_x, y1, marker="_", markersize=7, color=DIM_COL, zorder=6)
        ax.text(ann_x + 1.5, ym,
                f"Pitch\n{pitch_mm:.1f} mm",
                fontsize=7, va="center", ha="left", color=DIM_COL)

    # ── Bend radius (annotate the first right-side U-turn of branch 0) ──
    if np_ > 1:
        y0 = y_of_pass(0, 0)
        leg_h   = max(0.0, (pitch_mm - 2.0 * rb_mm) / 2.0)
        arc_cy_ann = y0 + leg_h + rb_mm
        # Leader line from arc tip to annotation
        arc_tip_x = x_right_turn + rb_mm
        arc_tip_y = arc_cy_ann
        ax.annotate(
            f"R={rb_mm:.0f} mm",
            xy=(arc_tip_x, arc_tip_y),
            xytext=(arc_tip_x + 4.0, arc_tip_y - pitch_mm * 0.35),
            fontsize=7, color=DIM_COL,
            arrowprops=dict(arrowstyle="-", color=DIM_COL, lw=0.8),
            va="center", ha="left",
            zorder=7,
        )

    # ── Straight run length (annotate top of first pass) ──
    y_top = y_of_pass(nb - 1, np_ - 1)
    ax.annotate(
        "",
        xy=(x_right_turn, y_top + pitch_mm * 0.45),
        xytext=(x_left_turn,  y_top + pitch_mm * 0.45),
        arrowprops=dict(arrowstyle="<->", color=DIM_COL, lw=0.9),
        zorder=7,
    )
    ax.text((x_left_turn + x_right_turn) / 2.0, y_top + pitch_mm * 0.45 + 1.5,
            f"Straight run = {L_pass_mm:.0f} mm",
            fontsize=7, ha="center", va="bottom", color=DIM_COL)

    # ── Bottom margin: overall routing summary ──
    diams_str = ", ".join(sorted({f"{d*1e3:.1f}" for d in diameters}))
    summary = (
        f"Topology = {best.dv.topology}  |  Manifold = {best.dv.manifold}  |  "
        f"Turn = {best.dv.turn_style}  |  "
        f"Branches = {nb}    |    "
        f"Passes/branch = {np_}    |    "
        f"Channel Ø = {diams_str} mm    |    "
        f"R_bend = {rb_mm:.1f} mm    |    "
        f"Pitch = {pitch_mm:.1f} mm    |    "
        f"Plate: {PW_MM:.0f}×{PL_MM:.0f} mm"
    )
    ax.text(PW_MM / 2.0, -12.0,
            summary,
            fontsize=7.5, ha="center", va="top",
            color=ANN_COL,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="#888888",
                      linewidth=0.8, alpha=0.92),
            zorder=8)

    # ── Small info box top-left corner ──
    topo_disp = best.dv.topology.replace("_", " ")
    mani_disp = best.dv.manifold.replace("_", " ")
    ts_disp   = best.dv.turn_style.replace("_", " ")
    info_lines = [
        label if label else "PLAN VIEW — TOP FACE",
        f"Topology:   {topo_disp}",
        f"Manifold:   {mani_disp}",
        f"Turn style: {ts_disp}",
        "Scale: equal aspect",
        f"T_batt ≈ {best.T_batt_est_C:.1f} °C",
        f"ΔP  ≈ {best.dP_total_Pa:.0f} Pa",
        f"Re  ≈ {best.Re_avg:.0f}",
        f"Score = {best.score:.2f}",
        "Dashed = connector",
        "Solid  = channel",
    ]
    # Change 2: add taper mode and velocity band to info box
    if best.dv.use_3zone:
        taper_str = (f"Ø{best.dv.D1_m*1e3:.0f}→"
                     f"Ø{best.dv.D2_m*1e3:.0f}→"
                     f"Ø{best.dv.D3_m*1e3:.0f} mm")
        info_lines.append(f"Taper: {taper_str}")
    else:
        info_lines.append(f"Channel: Ø{best.dv.D_const_m*1e3:.0f} mm (const.)")
    if best.branch_velocity_min_m_s > 0:
        info_lines.append(f"V: {best.branch_velocity_min_m_s:.2f}–"
                          f"{best.branch_velocity_max_m_s:.2f} m/s")
    ax.text(1.5, PL_MM - 1.5,
            "\n".join(info_lines),
            fontsize=6.5, va="top", ha="left",
            color=ANN_COL,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="#f0f8ff", edgecolor=INLET_COL,
                      linewidth=0.8, alpha=0.92),
            zorder=8)

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    legend_handles = [
        mpatches.Patch(facecolor="#aed6f1", edgecolor=INLET_COL,  label="Inlet header"),
        mpatches.Patch(facecolor="#f0b27a", edgecolor=OUTLET_COL, label="Outlet header"),
    ] + [
        mpatches.Patch(color=branch_col[i], label=f"Branch {i+1}")
        for i in range(nb)
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=7, framealpha=0.9)

    # Axis cosmetics
    ax.set_xlabel("Plate width direction  (mm)", fontsize=9)
    ax.set_ylabel("Plate length direction  (mm)", fontsize=9)
    ax.grid(True, ls=":", lw=0.4, alpha=0.3, zorder=0)
    ax.tick_params(labelsize=8)

    # Lock limits after all artists
    ax.set_xlim(-MARGIN, PW_MM + MARGIN)
    ax.set_ylim(-18.0,   PL_MM + 22.0)
    ax.autoscale(False)

    plt.tight_layout()
    plt.savefig(outfile, dpi=FIG_DPI)
    print(f"  Saved '{outfile}'  (engineering plan-view geometry)")
    plt.show()

    # ------------------------------------------------------------------
    # Section table (only for the main/best design, not every random design)
    # ------------------------------------------------------------------
    if show_table:
        rows = _build_section_table(
            best, diameters, pitch_mm, L_pass_mm, rb_mm,
            PORT_D_MM, HDR_D_MM, HEADER_INSET,
        )
        _print_section_table(rows)

        # Save section table as CSV
        import csv as _csv
        csv_path = "planar_geometry_sections.csv"
        with open(csv_path, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved '{csv_path}'  (section table)")


# =============================================================================
# 14. MAIN ENTRY POINT
# =============================================================================

def print_header_and_assumptions() -> None:
    """Print the project header and key assumption summary."""
    print("=" * 70)
    print("  1-D SERPENTINE BATTERY COOLING PLATE OPTIMIZER  (v4 - debugged)")
    print("  University Engineering Project - First-Pass Design Screening")
    print("=" * 70)
    print()
    print("  FIXED BOUNDARY CONDITIONS:")
    print(f"    Battery size         : {BATTERY_LENGTH_M*1e3:.0f} x "
          f"{BATTERY_WIDTH_M*1e3:.0f} x {BATTERY_HEIGHT_M*1e3:.0f} mm")
    print(f"    Volumetric heat gen  : {Q_VOL_W_M3:.0f} W/m3")
    print(f"    Coolant inlet temp   : {T_INLET_C:.1f} degC")
    print(f"    Total mass flow rate : {M_DOT_TOTAL_KG_S*1e3:.1f} g/s")
    print(f"    Plate thickness      : {PLATE_THICKNESS_M*1e3:.1f} mm")
    print(f"    Max channel D (75%t) : {0.75*PLATE_THICKNESS_M*1e3:.1f} mm")
    print(f"    Plate material       : Aluminium 6061-T4  "
          f"(k = {K_ALU_W_MK:.0f} W/m.K)")
    print()
    print("  KEY ASSUMPTIONS (1-D screening model):")
    print("    [1] Equal flow split among all parallel branches assumed.")
    print("    [2] Uniform volumetric heat distribution across battery volume.")
    print("    [3] Serpentine enhancement modelled by scalar phi = "
          f"{PHI_SERPENTINE:.2f}.")
    print(f"    [4] Bend loss K_bend={K_BEND:.2f}, entry K={K_ENTRY:.2f}, exit K={K_EXIT:.2f}.")
    print("    [5] Plate conduction resistance = simple 1-D path (no spreading).")
    print("    [6] Fluid properties evaluated at fixed ~30 degC (water).")
    print("    [7] Manifold pressure losses neglected.")
    print("    [8] Steady-state model - no transient effects.")
    print()
    print("  v4–v6 FIXES APPLIED:")
    print("    [F1] Schematic: Arc patches replaced with parametric np.linspace curves.")
    print("    [F2] Schematic: ax.annotate arrows replaced with ax.plot lines.")
    print("    [F3] Schematic: bbox_inches='tight' removed from savefig.")
    print("    [F4] Schematic: axis limits set explicitly; autoscale(False) called.")
    print("    [F5] Thermal: segment heat removal weighted by wetted area.")
    print("    [F6] Hydraulic: entrance (K=0.5) and exit (K=1.0) losses added.")
    print("    [F7] Geometry: pitch < 2*R_bend rejected as infeasible (new check).")
    print("    [F8] Hydraulic: U-turn length = pi*R + max(0, pitch-2R) per bend.")
    print("    [F9] Schematic: U-turn drawn as connector legs + semicircle (geometrically correct).")
    print(f"    [F10] PLATE_THICKNESS_M updated to 10 mm; D_MAX = {D_MAX_M*1e3:.1f} mm "
          f"(= 0.75 x plate).")
    print("    [F11] Manifold distribution penalty added to score (geometric proxy).")
    print("    [F12] Tapered header visualisation in plan-view schematic.")
    print()
    print("  ⚠  MANIFOLD PENALTY IS A SCREENING PROXY — NOT A CFD MANIFOLD SOLVER.")
    print("     The hydraulic model still assumes equal flow split among branches.")
    print("     The penalty scores geometry likely to distribute flow evenly.")
    print("     Validate maldistribution with CFD before hardware build.")
    print()
    print("  THIS IS A SCREENING TOOL ONLY - VALIDATE ALL RESULTS WITH CFD.")
    print()


def print_baseline(Q_total: float, fluid: FluidProperties) -> None:
    """Print the baseline heat load and single-pass coolant calculations."""
    V_battery = BATTERY_LENGTH_M * BATTERY_WIDTH_M * BATTERY_HEIGHT_M
    dT_coolant = Q_total / (M_DOT_TOTAL_KG_S * fluid.cp)

    print("  BASELINE HEAT LOAD CALCULATIONS:")
    print(f"    Battery volume       : {V_battery*1e6:.0f} cm³  "
          f"= {V_battery:.6f} m³")
    print(f"    Total heat load Q    : {Q_total:.1f} W")
    print(f"    Coolant ΔT (full Q)  : {dT_coolant:.2f} K  "
          f"(T_out = {T_INLET_C + dT_coolant:.2f} °C)")
    print(f"    Water Pr number      : {fluid.Pr:.2f}")
    print()


def select_random_feasible_designs(
    results:      List["DesignResult"],
    df_feasible:  "pd.DataFrame",
    n:            int = 5,
    random_state: int = 42,
) -> List["DesignResult"]:
    """
    Sample up to `n` random feasible designs from the evaluated result set.

    Parameters
    ----------
    results      : full list of DesignResult objects from the parameter sweep
    df_feasible  : feasible rows of the ranked DataFrame (used for sampling)
    n            : number of random designs to pick (default 5)
    random_state : NumPy random seed for reproducibility (default 42)

    Returns
    -------
    selected : List[DesignResult]  up to n DesignResult objects, in random order
    """
    n_avail = len(df_feasible)
    k       = min(n, n_avail)
    if k == 0:
        return []

    sampled_df = df_feasible.sample(n=k, random_state=random_state)
    sampled_ids = set(sampled_df["design_id"].astype(int).tolist())
    selected    = [r for r in results if r.design_id in sampled_ids]

    # Preserve the DataFrame sample order
    id_order = sampled_df["design_id"].astype(int).tolist()
    selected.sort(key=lambda r: id_order.index(r.design_id))
    return selected


def _draw_branch_connectors_and_junctions(
    ax:              "plt.Axes",
    nb:              int,
    np_:             int,
    branch_col:      list,
    y_of_pass,                  # callable (b, p) -> y_mm
    x_hdr_in_right:  float,     # right edge of inlet manifold band (where connector starts)
    x_branch_start:  float,     # x where serpentine first pass begins (connector ends)
    x_branch_end:    float,     # x where last pass ends (outlet connector starts)
    x_hdr_out_left:  float,     # left edge of outlet manifold band (where connector ends)
    np_last_right:   bool,      # True if last pass exits on the right side
    lw_conn:         float      = 2.2,
    dot_r:           float      = 3.5,
    inlet_col:       str        = "#1a5276",
    outlet_col:      str        = "#784212",
) -> None:
    """
    Draw inlet and outlet branch connectors with junction dots for every branch.

    This function is called by both the standalone planar schematic and the
    comparison-panel subplot renderer, ensuring consistent visual treatment.

    What is drawn for each branch b:
    ─────────────────────────────────
    Inlet side:
      1. Solid connector line: (x_hdr_in_right, y_first) → (x_branch_start, y_first)
         Colour = branch colour, lw = lw_conn, solid (not dashed), zorder=6
      2. Filled dot at manifold junction (x_hdr_in_right, y_first)
         Colour = inlet_col fill + white edge, zorder=8
      3. Filled dot at branch start (x_branch_start, y_first)
         Colour = branch colour fill + white edge, zorder=8

    Outlet side:
      4. Solid connector line: (x_branch_end, y_last) → (x_hdr_out_left, y_last)
         Colour = branch colour, lw = lw_conn, solid, zorder=6
      5. Filled dot at branch end (x_branch_end, y_last)
         Colour = branch colour fill + white edge, zorder=8
      6. Filled dot at manifold junction (x_hdr_out_left, y_last)
         Colour = outlet_col fill + white edge, zorder=8

    The dots sit at zorder=8, above everything else, so they are never hidden
    by the manifold fill or the branch lines.

    Parameters
    ----------
    x_hdr_in_right  : right outer edge of the inlet manifold band [mm]
                      Connector starts here, ON the manifold boundary.
    x_branch_start  : x of the first straight-pass start [mm]
                      Connector ends here, touching the branch line exactly.
    x_branch_end    : x of the last straight-pass end [mm]
    x_hdr_out_left  : left outer edge of the outlet manifold band [mm]
    np_last_right   : whether the last pass exits on the right (True) or left (False)
    """
    for b_idx in range(nb):
        col    = branch_col[b_idx]
        y_in   = y_of_pass(b_idx, 0)
        y_out  = y_of_pass(b_idx, np_ - 1)

        # Outlet connector x depends on which side the last pass ends on
        x_out_end = x_branch_end if np_last_right else x_branch_start

        # ── Inlet connector (solid, branch colour, thick) ──
        ax.plot([x_hdr_in_right, x_branch_start], [y_in, y_in],
                color=col, lw=lw_conn, solid_capstyle="butt", zorder=6)

        # ── Outlet connector ──
        ax.plot([x_out_end, x_hdr_out_left], [y_out, y_out],
                color=col, lw=lw_conn, solid_capstyle="butt", zorder=6)

        # ── Junction dots ──  (filled circle: branch colour inside, white ring)
        for xd, yd, fc in [
            (x_hdr_in_right,  y_in,  inlet_col),   # inlet manifold tap
            (x_branch_start,  y_in,  col),          # branch start
            (x_out_end,       y_out, col),          # branch end
            (x_hdr_out_left,  y_out, outlet_col),  # outlet manifold tap
        ]:
            # White halo (drawn first, slightly larger)
            ax.plot(xd, yd, "o", color="white",
                    markersize=dot_r * 2.0 + 1.5, zorder=7,
                    markeredgewidth=0)
            # Filled coloured dot
            ax.plot(xd, yd, "o", color=fc,
                    markersize=dot_r * 2.0, zorder=8,
                    markeredgecolor="white", markeredgewidth=0.8)


def _draw_design_onto_ax(
    ax:          "plt.Axes",
    result:      "DesignResult",
    label:       str  = "",
    full_detail: bool = False,
) -> None:
    """
    Render a plan-view layout of `result` onto an existing Axes.

    Parameters
    ----------
    ax          : matplotlib Axes to draw onto
    result      : DesignResult to render
    label       : subplot title prefix
    full_detail : if True, draw ports and manifold headers with full labels
                  (used for the "★ Best Design" panel).  If False, draw a
                  compact version suitable for smaller random-design panels.
    """
    dv    = result.dv
    nb    = dv.n_branches
    np_   = dv.n_passes
    topo  = dv.topology
    mfld  = dv.manifold

    PW_mm = BATTERY_WIDTH_M  * 1e3   # 210 mm (x-axis)
    PL_mm = BATTERY_LENGTH_M * 1e3   # 420 mm (y-axis)
    EDGE_mm = EDGE_OFFSET_M * 1e3    # 10 mm

    # Generous padding so edge-mounted ports are never clipped
    PAD = 22.0

    INLET_COL  = "#1a5276"
    OUTLET_COL = "#784212"
    PLATE_EDGE = "#2c3e50"
    cmap_ch    = matplotlib.colormaps.get_cmap("tab10")
    branch_col = [cmap_ch(i / max(nb - 1, 1)) for i in range(nb)]

    diameters = get_diameter_profile(dv)
    D_max_mm  = max(d * 1e3 for d in diameters)

    # Line weights (scaled down vs standalone schematic for compact panels)
    LW_PORT   = 3.5 if full_detail else 2.5
    LW_HDR    = 2.2 if full_detail else 1.6
    LW_CONN   = 1.6 if full_detail else 1.0
    LW_CH_MAX = 2.5 if full_detail else 2.0
    LW_CH_MIN = 0.8

    def ch_lw(D_m: float) -> float:
        t = (D_m * 1e3 - D_max_mm * 0.5) / (D_max_mm * 0.5) if D_max_mm > 0 else 0.5
        return LW_CH_MIN + max(0.0, min(1.0, t)) * (LW_CH_MAX - LW_CH_MIN)

    PORT_R_MM    = 6.0 if full_detail else 4.5
    HEADER_INSET = 25.0   # mm — how far the H-header runs into the plate
    PORT_D_MM    = MANIFOLD_TUBE_DIAMETER_M * 1e3
    HDR_VIS_W    = MANIFOLD_INLET_WIDTH_M * 1e3   # visual header band width (mm)

    # ── Derived x-coordinates for H_SERPENTINE / Z_FLOW style headers ──
    x_in_vhdr  = HEADER_INSET                  # x of inlet vertical header
    x_out_vhdr = PW_mm - HEADER_INSET          # x of outlet vertical header

    pitch_mm = result.branch_pitch_m * 1e3
    rb_mm    = dv.bend_radius_m * 1e3

    def y_of_pass(b: int, p: int) -> float:
        return EDGE_mm + b * np_ * pitch_mm + p * pitch_mm

    y_hdr_bot = y_of_pass(0,    0)
    y_hdr_top = y_of_pass(nb-1, np_-1)
    y_centre  = PL_mm / 2.0

    # ── Plate rectangle ──
    ax.add_patch(plt.Rectangle(
        (0, 0), PW_mm, PL_mm,
        lw=1.5, edgecolor=PLATE_EDGE, facecolor="#fdfefe", zorder=0,
    ))
    ax.set_facecolor("#e8e8e8" if full_detail else "#eeeeee")

    # ──────────────────────────────────────────────────────────────────────
    # PORT + MANIFOLD RENDERING  (topology-aware)
    # ──────────────────────────────────────────────────────────────────────
    # Each topology draws:
    #   1. Inlet port circle + label
    #   2. Outlet port circle + label
    #   3. Inlet manifold/header (line or shaded band) + label
    #   4. Outlet manifold/header (line or shaded band) + label
    #   5. Branch connector stubs (from header to first/last straight pass)
    #
    # Port circles sit just outside the plate boundary so they are always
    # visible within the padded axis limits.

    def _port_circle(cx, cy, col, fc, label_text, label_dx=0, label_dy=0):
        """Draw a filled port circle and optional label."""
        ax.add_patch(plt.Circle(
            (cx, cy), PORT_R_MM,
            lw=LW_PORT, edgecolor=col, facecolor=fc, zorder=8,
        ))
        if full_detail and label_text:
            ax.text(cx + label_dx, cy + label_dy, label_text,
                    fontsize=6.5, ha="center", va="bottom" if label_dy >= 0 else "top",
                    color=col, fontweight="bold", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=col, linewidth=0.7, alpha=0.90))
        elif label_text:
            # Compact: just a small text tag
            ax.text(cx + label_dx, cy + label_dy, label_text[:3],
                    fontsize=5.5, ha="center", va="center",
                    color=col, fontweight="bold", zorder=9)

    def _hdr_band(x_ctr, y_bot, y_top, col, fc, half_w, label, lside=True):
        """Draw a vertical header band (shaded rectangle + outline)."""
        xs = [x_ctr - half_w, x_ctr + half_w, x_ctr + half_w,
              x_ctr - half_w, x_ctr - half_w]
        ys = [y_bot, y_bot, y_top, y_top, y_bot]
        ax.fill(xs, ys, color=fc, alpha=0.45, zorder=4)
        ax.plot(xs, ys, color=col, lw=1.2, zorder=5)
        if full_detail and label:
            lx = x_ctr - half_w - 2 if lside else x_ctr + half_w + 2
            ha = "right" if lside else "left"
            ax.text(lx, (y_bot + y_top) / 2.0, label,
                    fontsize=5.5, ha=ha, va="center", color=col,
                    rotation=90, rotation_mode="anchor", zorder=9)

    def _hdr_line(x0, y0, x1, y1, col, label="", lx=None, ly=None):
        """Draw a horizontal feed-line from port to plate edge."""
        ax.plot([x0, x1], [y0, y1], color=col, lw=LW_HDR,
                solid_capstyle="butt", zorder=5)
        if full_detail and label and lx is not None:
            ax.text(lx, ly, label,
                    fontsize=5.5, ha="center", va="bottom", color=col, zorder=9)

    # ── topology-specific port + manifold positions ──
    if topo in ("H_SERPENTINE", "Z_FLOW"):
        if mfld == "TOP_BOTTOM":
            # Ports on the LEFT (inlet) and RIGHT (outlet) mid-height edges
            px_in  = -PORT_R_MM
            px_out = PW_mm + PORT_R_MM
            py     = y_centre
            # Horizontal feed lines crossing the plate edge
            _port_circle(px_in,  py, INLET_COL,  "#aed6f1",
                         f"INLET\nØ{PORT_D_MM:.0f}", 0, PORT_R_MM + 3)
            _port_circle(px_out, py, OUTLET_COL, "#f0b27a",
                         f"OUTLET\nØ{PORT_D_MM:.0f}", 0, PORT_R_MM + 3)
            _hdr_line(px_in + PORT_R_MM, py, 0, py, INLET_COL,
                      "Inlet header", (px_in + PORT_R_MM) / 2, py + 2)
            _hdr_line(PW_mm, py, px_out - PORT_R_MM, py, OUTLET_COL,
                      "Outlet header", (PW_mm + px_out - PORT_R_MM) / 2, py + 2)
            # Vertical distribution headers (bands)
            _hdr_band(x_in_vhdr,  y_hdr_bot, y_hdr_top, INLET_COL,  "#aed6f1",
                      HDR_VIS_W / 2, "Inlet\nmanifold", lside=True)
            _hdr_band(x_out_vhdr, y_hdr_bot, y_hdr_top, OUTLET_COL, "#f0b27a",
                      HDR_VIS_W / 2, "Outlet\nmanifold", lside=False)
            _hdr_line(0, py, x_in_vhdr, py, INLET_COL)
            _hdr_line(x_out_vhdr, py, PW_mm, py, OUTLET_COL)
        else:  # LEFT_RIGHT (port at y_centre on left/right edges) — default
            px_in  = -PORT_R_MM
            px_out = PW_mm + PORT_R_MM
            py     = y_centre
            _port_circle(px_in,  py, INLET_COL,  "#aed6f1",
                         f"INLET\nØ{PORT_D_MM:.0f}", 0, PORT_R_MM + 3)
            _port_circle(px_out, py, OUTLET_COL, "#f0b27a",
                         f"OUTLET\nØ{PORT_D_MM:.0f}", 0, PORT_R_MM + 3)
            # Feed lines port → plate edge
            _hdr_line(px_in + PORT_R_MM, py, 0,      py, INLET_COL)
            _hdr_line(PW_mm,             py, px_out - PORT_R_MM, py, OUTLET_COL)
            # H-header inside plate: plate edge → vertical header
            _hdr_line(0, py, x_in_vhdr, py, INLET_COL,
                      "Inlet header", x_in_vhdr / 2, py + 2)
            _hdr_line(x_out_vhdr, py, PW_mm, py, OUTLET_COL,
                      "Outlet header", (x_out_vhdr + PW_mm) / 2, py + 2)
            # Vertical distribution headers (bands)
            _hdr_band(x_in_vhdr,  y_hdr_bot, y_hdr_top, INLET_COL,  "#aed6f1",
                      HDR_VIS_W / 2, "Inlet\nmanifold", lside=True)
            _hdr_band(x_out_vhdr, y_hdr_bot, y_hdr_top, OUTLET_COL, "#f0b27a",
                      HDR_VIS_W / 2, "Outlet\nmanifold", lside=False)

    elif topo == "V_SERPENTINE":
        if mfld == "TOP_BOTTOM":
            # Ports on TOP (inlet) and BOTTOM (outlet)
            px      = PW_mm / 2.0
            py_in   = PL_mm + PORT_R_MM
            py_out  = -PORT_R_MM
            _port_circle(px, py_in,  INLET_COL,  "#aed6f1",
                         f"INLET\nØ{PORT_D_MM:.0f}", 0, PORT_R_MM + 3)
            _port_circle(px, py_out, OUTLET_COL, "#f0b27a",
                         f"OUTLET\nØ{PORT_D_MM:.0f}", 0, -(PORT_R_MM + 9))
            # Horizontal top/bottom manifold lines
            y_top_hdr = PL_mm - HEADER_INSET
            y_bot_hdr = HEADER_INSET
            _hdr_line(0, y_top_hdr, PW_mm, y_top_hdr, INLET_COL,
                      "Inlet manifold", PW_mm / 2, y_top_hdr + 2)
            _hdr_line(0, y_bot_hdr, PW_mm, y_bot_hdr, OUTLET_COL,
                      "Outlet manifold", PW_mm / 2, y_bot_hdr - 6)
            # Feed stubs from port circle to manifold
            _hdr_line(px, py_in - PORT_R_MM, px, PL_mm, INLET_COL)
            _hdr_line(px, py_out + PORT_R_MM, px, 0, OUTLET_COL)
            _hdr_line(px, PL_mm, px, y_top_hdr, INLET_COL)
            _hdr_line(px, 0, px, y_bot_hdr, OUTLET_COL)
        else:  # LEFT_RIGHT: ports on short (left/right) sides
            py_in  = PL_mm + PORT_R_MM
            py_out = -PORT_R_MM
            # For V_SERPENTINE LEFT_RIGHT the ports still go top/bottom
            # (branches run along length, stacked across width)
            px = PW_mm / 2.0
            _port_circle(px, py_in,  INLET_COL,  "#aed6f1",
                         f"INLET\nØ{PORT_D_MM:.0f}", 0, PORT_R_MM + 3)
            _port_circle(px, py_out, OUTLET_COL, "#f0b27a",
                         f"OUTLET\nØ{PORT_D_MM:.0f}", 0, -(PORT_R_MM + 9))
            y_top_hdr = PL_mm - HEADER_INSET
            y_bot_hdr = HEADER_INSET
            _hdr_line(0, y_top_hdr, PW_mm, y_top_hdr, INLET_COL,
                      "Inlet manifold", PW_mm / 2, y_top_hdr + 2)
            _hdr_line(0, y_bot_hdr, PW_mm, y_bot_hdr, OUTLET_COL,
                      "Outlet manifold", PW_mm / 2, y_bot_hdr - 6)
            _hdr_line(px, py_in - PORT_R_MM, px, PL_mm, INLET_COL)
            _hdr_line(px, py_out + PORT_R_MM, px, 0, OUTLET_COL)

    elif topo == "MIRRORED_U":
        # Both ports on bottom edge (SAME_SIDE)
        px_in  = PW_mm * 0.30
        px_out = PW_mm * 0.70
        py     = -PORT_R_MM
        _port_circle(px_in,  py, INLET_COL,  "#aed6f1",
                     f"INLET\nØ{PORT_D_MM:.0f}", 0, -(PORT_R_MM + 9))
        _port_circle(px_out, py, OUTLET_COL, "#f0b27a",
                     f"OUTLET\nØ{PORT_D_MM:.0f}", 0, -(PORT_R_MM + 9))
        # Horizontal manifold lines along bottom
        y_hdr = HEADER_INSET
        _hdr_line(0, y_hdr, PW_mm * 0.50, y_hdr, INLET_COL,
                  "Inlet manifold", PW_mm * 0.20, y_hdr + 2)
        _hdr_line(PW_mm * 0.50, y_hdr, PW_mm, y_hdr, OUTLET_COL,
                  "Outlet manifold", PW_mm * 0.75, y_hdr + 2)
        # Feed stubs from ports to manifold line
        _hdr_line(px_in,  py + PORT_R_MM, px_in,  0,     INLET_COL)
        _hdr_line(px_in,  0,              px_in,  y_hdr, INLET_COL)
        _hdr_line(px_out, py + PORT_R_MM, px_out, 0,     OUTLET_COL)
        _hdr_line(px_out, 0,              px_out, y_hdr, OUTLET_COL)

    elif topo == "CENTRAL_INLET":
        # Central inlet at plate mid-length, outlets at top+bottom edges
        py_in  = PL_mm / 2.0
        px_in  = -PORT_R_MM
        _port_circle(px_in, py_in, INLET_COL, "#aed6f1",
                     f"INLET\nØ{PORT_D_MM:.0f}", 0, PORT_R_MM + 3)
        # Feed line into central manifold
        _hdr_line(px_in + PORT_R_MM, py_in, 0, py_in, INLET_COL)
        # Central horizontal manifold (full width at mid-length)
        _hdr_line(0, py_in, PW_mm, py_in, INLET_COL,
                  "Central inlet manifold", PW_mm / 2, py_in + 2)
        # Outlet ports at top and bottom
        py_out_top = PL_mm + PORT_R_MM
        py_out_bot = -PORT_R_MM
        px_out = PW_mm + PORT_R_MM
        _port_circle(px_out, py_in, OUTLET_COL, "#f0b27a",
                     f"OUTLET\nØ{PORT_D_MM:.0f}", 0, PORT_R_MM + 3)
        # Outlet collection lines along top/bottom
        _hdr_line(0, PL_mm - EDGE_mm, PW_mm, PL_mm - EDGE_mm, OUTLET_COL,
                  "Outlet manifold", PW_mm / 2, PL_mm - EDGE_mm + 2)
        _hdr_line(0, EDGE_mm, PW_mm, EDGE_mm, OUTLET_COL)
        _hdr_line(PW_mm, py_in, px_out - PORT_R_MM, py_in, OUTLET_COL)

    else:  # fallback: LEFT_RIGHT
        px_in  = -PORT_R_MM
        px_out = PW_mm + PORT_R_MM
        py     = y_centre
        _port_circle(px_in,  py, INLET_COL,  "#aed6f1",
                     f"INLET\nØ{PORT_D_MM:.0f}", 0, PORT_R_MM + 3)
        _port_circle(px_out, py, OUTLET_COL, "#f0b27a",
                     f"OUTLET\nØ{PORT_D_MM:.0f}", 0, PORT_R_MM + 3)
        _hdr_line(px_in + PORT_R_MM, py, 0,      py, INLET_COL, "Inlet header",
                  x_in_vhdr / 2, py + 2)
        _hdr_line(PW_mm, py, px_out - PORT_R_MM, py, OUTLET_COL, "Outlet header",
                  (x_out_vhdr + PW_mm) / 2, py + 2)
        _hdr_band(x_in_vhdr,  y_hdr_bot, y_hdr_top, INLET_COL,  "#aed6f1",
                  HDR_VIS_W / 2, "Inlet manifold", lside=True)
        _hdr_band(x_out_vhdr, y_hdr_bot, y_hdr_top, OUTLET_COL, "#f0b27a",
                  HDR_VIS_W / 2, "Outlet manifold", lside=False)

    # ──────────────────────────────────────────────────────────────────────
    # BRANCH TAP-OFF TICK MARKS on vertical headers (H_SERP / Z_FLOW)
    # ──────────────────────────────────────────────────────────────────────
    if topo in ("H_SERPENTINE", "Z_FLOW"):
        for b_idx in range(nb):
            y_in  = y_of_pass(b_idx, 0)
            y_out = y_of_pass(b_idx, np_ - 1)
            ax.plot([x_in_vhdr - 3, x_in_vhdr + 3], [y_in,  y_in],
                    color=INLET_COL,  lw=1.2, zorder=6)
            ax.plot([x_out_vhdr - 3, x_out_vhdr + 3], [y_out, y_out],
                    color=OUTLET_COL, lw=1.2, zorder=6)

    # ──────────────────────────────────────────────────────────────────────
    # SERPENTINE BRANCH ROUTING
    # ──────────────────────────────────────────────────────────────────────
    sections = result.layout_sections
    use_sections = (sections and topo not in ("H_SERPENTINE",))

    # ── Shared helper: draw one connector segment + endpoint junction dots ─
    def _draw_connector(x0, y0, x1, y1, col):
        """
        Draw a branch-to-manifold connector with filled junction dots.

        The segment runs from (x0,y0) to (x1,y1) and is rendered:
          - solid line, slightly thicker than branch channels, zorder=6
          - white-halo + filled-colour dots at both endpoints, zorder=9/10
        This makes the connection visually unambiguous against any background.
        """
        lw_c = LW_CONN * 2.0          # thicker than branch lines
        ms   = 5.5 if full_detail else 4.2   # junction dot diameter

        ax.plot([x0, x1], [y0, y1],
                color=col, lw=lw_c,
                solid_capstyle="round", zorder=6)

        for xd, yd in [(x0, y0), (x1, y1)]:
            ax.plot(xd, yd, "o", color="white",
                    markersize=ms + 2.0, zorder=9, markeredgewidth=0)
            ax.plot(xd, yd, "o", color=col,
                    markersize=ms, zorder=10,
                    markeredgecolor="white", markeredgewidth=0.9)

    if use_sections:
        # ── Draw straights and bends from LayoutSection objects ───────────
        for sec in sections:
            if sec.sec_type not in ("straight", "bend"):
                continue
            col = branch_col[sec.branch_id % nb] if sec.branch_id >= 0 else INLET_COL
            lw  = ch_lw(sec.diameter_m)
            if sec.sec_type == "straight":
                ax.plot([sec.x0, sec.x1], [sec.y0, sec.y1],
                        color=col, lw=lw, solid_capstyle="round", zorder=3)
            else:  # bend — use shared helper with correct turn orientation
                _draw_bend_arc(ax, sec, sec.bend_radius_m * 1e3, col, lw,
                               dv.turn_style, PW_mm, PL_mm)

        # Branch labels
        for b_idx in range(nb):
            bsecs = [s for s in sections
                     if s.branch_id == b_idx and s.sec_type == "straight"]
            if bsecs:
                s0 = bsecs[0]
                ax.text(s0.x0+2, s0.y0+1, f"B{b_idx+1}",
                        fontsize=6.5, color=branch_col[b_idx],
                        va="bottom", fontweight="bold", zorder=7)

        # ── Draw explicit branch-to-manifold connectors ───────────────────
        # Coordinates are derived from the same manifold line positions used
        # in the port/manifold drawing block above — guaranteeing exact joins.
        VINSET_P = HEADER_INSET   # 25 mm — matches the manifold draw calls

        for b_idx in range(nb):
            col = branch_col[b_idx]
            bstraight = [s for s in sections
                         if s.branch_id == b_idx and s.sec_type == "straight"]
            if not bstraight:
                continue
            s_first = bstraight[0]
            s_last  = bstraight[-1]

            if topo == "V_SERPENTINE":
                # Manifold lines (horizontal):
                #   inlet  at y = PL_mm - VINSET_P  (top)
                #   outlet at y = VINSET_P           (bottom)
                # First pass (p=0, even) goes UP: from y_bot=35 → y_top=385.
                #   The pass END (y=385) is near the inlet manifold (y=395).
                #   Inlet connector: vertical from manifold (395) down to pass end (385).
                # Last pass determines where outlet connector goes.
                #   The outlet is always at the y_bot (25→35) end.
                y_mfld_in  = PL_mm - VINSET_P    # 395 mm
                y_mfld_out = VINSET_P             # 25 mm
                x_c  = s_first.x0                # x of first pass (inlet connector)
                y_branch_top = max(s_first.y0, s_first.y1)   # top end of first pass
                _draw_connector(x_c, y_mfld_in, x_c, y_branch_top, col)
                x_co = s_last.x0                 # x of last pass (outlet connector)
                y_branch_bot = min(s_last.y0, s_last.y1)     # bottom end of last pass
                _draw_connector(x_co, y_branch_bot, x_co, y_mfld_out, col)

            elif topo == "Z_FLOW":
                # Manifold bands (vertical): inlet at x_in_vhdr, outlet at x_out_vhdr.
                # Branches run horizontally.  First pass starts at x_left=30;
                # inlet band right edge is at x_in_vhdr+HDR_VIS_W/2=32.5.
                # Draw connector from manifold CENTRE to branch start so it
                # emerges visibly from the band interior.
                y_in  = s_first.y0
                y_out = s_last.y0
                x_br_in  = min(s_first.x0, s_first.x1)   # left end of first pass
                last_p_even = (s_last.pass_id % 2 == 0)
                x_br_out = max(s_last.x0, s_last.x1) if last_p_even \
                           else min(s_last.x0, s_last.x1)
                _draw_connector(x_in_vhdr, y_in, x_br_in, y_in, col)
                _draw_connector(x_br_out, y_out, x_out_vhdr, y_out, col)

            elif topo == "MIRRORED_U":
                # Manifold line at y = HEADER_INSET (bottom horizontal line, y=25).
                # OUT leg starts at y = EDGE_mm (y=10, below the manifold line).
                # Inlet connector: from manifold (y=25) down to branch start (y=10).
                # Return leg ends near y = EDGE_mm or below; use the bottom y.
                y_mfld = HEADER_INSET   # 25 mm
                x_ci = s_first.x0
                y_br_start = min(s_first.y0, s_first.y1)
                _draw_connector(x_ci, y_mfld, x_ci, y_br_start, col)
                x_co = s_last.x0
                y_br_end = min(s_last.y0, s_last.y1)
                _draw_connector(x_co, y_mfld, x_co, y_br_end, col)

            elif topo == "CENTRAL_INLET":
                # Inlet manifold: y = PL_mm / 2 (horizontal centre line).
                # Outlet manifolds: y = PL_mm - EDGE_mm (top) or y = EDGE_mm (bottom).
                y_mfld_in = PL_mm / 2.0
                x_ci = s_first.x0
                y_br_in = s_first.y0    # start of first pass (adjacent to inlet)
                _draw_connector(x_ci, y_mfld_in, x_ci, y_br_in, col)
                x_co = s_last.x0
                y_br_out = s_last.y1    # end of last pass (adjacent to outlet)
                y_out_mfld = (PL_mm - EDGE_mm) if y_br_out > PL_mm / 2.0 else EDGE_mm
                _draw_connector(x_co, y_br_out, x_co, y_out_mfld, col)

    else:
        # ── H_SERPENTINE classic rendering ───────────────────────────────
        x_left_turn  = x_in_vhdr  + EDGE_mm
        x_right_turn = x_out_vhdr - EDGE_mm

        for b_idx in range(nb):
            col     = branch_col[b_idx]
            y_first = y_of_pass(b_idx, 0)
            y_last  = y_of_pass(b_idx, np_ - 1)

            for p_idx in range(np_):
                D_m  = diameters[p_idx]
                lw_c = ch_lw(D_m)
                y_mm = y_of_pass(b_idx, p_idx)
                going_right = (p_idx % 2 == 0)
                x_from = x_left_turn  if going_right else x_right_turn
                x_to   = x_right_turn if going_right else x_left_turn
                ax.plot([x_from, x_to], [y_mm, y_mm],
                        color=col, lw=lw_c, solid_capstyle="round", zorder=3)

                if p_idx < np_ - 1:
                    y_next  = y_of_pass(b_idx, p_idx + 1)
                    bx_turn = x_right_turn if going_right else x_left_turn
                    _bend_sec = LayoutSection(
                        sec_type="bend", bend_radius_m=dv.bend_radius_m,
                        x0=bx_turn, y0=y_mm, x1=bx_turn, y1=y_next,
                    )
                    _draw_bend_arc(ax, _bend_sec, rb_mm, col, lw_c,
                                   dv.turn_style, PW_mm, PL_mm)

            ax.text(x_in_vhdr + 2, y_first + 1, f"B{b_idx+1}",
                    fontsize=6.5, color=col, va="bottom",
                    fontweight="bold", zorder=7)

        # ── H_SERPENTINE branch-to-manifold connectors ───────────────────
        # Inlet manifold centreline at x = x_in_vhdr.
        # First pass starts at x = x_left_turn = x_in_vhdr + EDGE_mm.
        # Last pass ends at x_right_turn (even) or x_left_turn (odd).
        for b_idx in range(nb):
            col     = branch_col[b_idx]
            y_first = y_of_pass(b_idx, 0)
            y_last  = y_of_pass(b_idx, np_ - 1)
            going_right_last = ((np_ - 1) % 2 == 0)
            x_last_end = x_right_turn if going_right_last else x_left_turn
            _draw_connector(x_in_vhdr,  y_first, x_left_turn, y_first, col)
            _draw_connector(x_last_end, y_last,  x_out_vhdr,  y_last,  col)

    # ──────────────────────────────────────────────────────────────────────
    # AXIS LIMITS — padded so edge-mounted ports are never clipped
    # ──────────────────────────────────────────────────────────────────────
    ax.set_xlim(-PAD, PW_mm + PAD)
    ax.set_ylim(-PAD, PL_mm + PAD)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # ──────────────────────────────────────────────────────────────────────
    # SUBPLOT TITLE
    # ──────────────────────────────────────────────────────────────────────
    diams_str = "/".join(sorted({f"{d*1e3:.0f}" for d in diameters}))
    topo_short = {
        "H_SERPENTINE":  "H-Serp",
        "V_SERPENTINE":  "V-Serp",
        "MIRRORED_U":    "Mirror-U",
        "Z_FLOW":        "Z-Flow",
        "CENTRAL_INLET": "Central",
    }.get(topo, topo)
    ts_short = {
        "CONNECTOR_SEMICIRCLE": "Conn+Semi",
        "PURE_CIRCULAR":        "PureCirc",
        "SMOOTH_SPLINE":        "Spline",
    }.get(dv.turn_style, dv.turn_style)
    title_line2 = (
        f"[{topo_short} | {ts_short}] "
        f"B{nb}×P{np_}  Ø{diams_str} mm  R={dv.bend_radius_m*1e3:.0f} mm\n"
        f"T={result.T_batt_est_C:.1f}°C  ΔP={result.dP_total_Pa:.0f} Pa"
        f"  Score={result.score:.1f}"
    )
    ax.set_title(f"{label}\n{title_line2}",
                 fontsize=7.0, fontweight="bold", pad=3, linespacing=1.4)


def print_design_comparison_table(
    selected:    List["DesignResult"],
    best_result: "DesignResult",
) -> None:
    """
    Print a side-by-side console table for the random designs and best design.

    Parameters
    ----------
    selected    : list of random DesignResult objects
    best_result : the top-ranked DesignResult
    """
    header = (f"{'Label':<22} {'Score':>7} {'Topology':<16} {'TurnStyle':<20} "
              f"{'B':>3} {'P':>3} {'Ø mm':>6} {'R mm':>6} {'T°C':>7} {'ΔP Pa':>8}")
    sep = "─" * len(header)
    print("\n  " + sep)
    print("  DESIGN COMPARISON TABLE")
    print("  " + sep)
    print("  " + header)
    print("  " + sep)

    all_rows: List[Tuple[str, "DesignResult"]] = []
    for i, r in enumerate(selected, start=1):
        all_rows.append((f"Random Design {i}", r))
    all_rows.append(("★ Best Design", best_result))

    for label, r in all_rows:
        dv     = r.dv
        D_str  = "/".join(sorted({f"{d*1e3:.1f}" for d in get_diameter_profile(dv)}))
        topo   = dv.topology[:15]
        ts     = dv.turn_style[:19]
        print(f"  {label:<22} {r.score:>7.2f} {topo:<16} {ts:<20} "
              f"{dv.n_branches:>3} {dv.n_passes:>3} "
              f"{D_str:>6} {dv.bend_radius_m*1e3:>6.1f} "
              f"{r.T_batt_est_C:>7.2f} {r.dP_total_Pa:>8.1f}")
    print("  " + sep)


def plot_design_comparison_panel(
    random_results: List["DesignResult"],
    best_result:    "DesignResult",
    outfile:        str = "design_comparison_panel.png",
) -> None:
    """
    Generate a 2×3 panel figure showing 5 random feasible designs and the
    final chosen (best) design for presentation / reporting.

    Layout
    ------
    [ Random 1 ] [ Random 2 ] [ Random 3 ]
    [ Random 4 ] [ Random 5 ] [ ★ Best   ]

    Each panel is rendered by _draw_design_onto_ax() using exactly the same
    geometry logic as the full-size plan-view schematic.

    Parameters
    ----------
    random_results : list of up to 5 randomly sampled DesignResult objects
    best_result    : the top-ranked DesignResult
    outfile        : filename for the saved PNG
    """
    n_rand  = len(random_results)
    n_total = n_rand + 1          # +1 for best design
    n_cols  = 3
    n_rows  = math.ceil(n_total / n_cols)

    fig_w = n_cols * 6.0
    fig_h = n_rows * 7.5
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(fig_w, fig_h),
                             facecolor="#d8d8d8")
    fig.suptitle(
        "Cooling Plate Design Comparison\n"
        f"{n_rand} Random Feasible Designs  +  Final Chosen Design",
        fontsize=13, fontweight="bold", y=1.01,
    )

    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    # ── Draw random designs (compact mode) ──
    for i, result in enumerate(random_results):
        _draw_design_onto_ax(axes_flat[i], result,
                             label=f"Random Design {i+1}", full_detail=False)
        for spine in axes_flat[i].spines.values():
            spine.set_edgecolor("#888888")
            spine.set_linewidth(1.0)

    # ── Draw best design with full port + manifold detail ──
    best_ax = axes_flat[n_rand]
    _draw_design_onto_ax(best_ax, best_result,
                         label="★  Final Chosen Design", full_detail=True)
    # Gold border to make the best design visually distinct
    for spine in best_ax.spines.values():
        spine.set_edgecolor("#c0932e")
        spine.set_linewidth(3.0)
    best_ax.set_facecolor("#fffde8")

    # ── Hide any unused panels (if n_total < n_rows*n_cols) ──
    for j in range(n_total, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved '{outfile}'  (6-panel design comparison)")
    plt.show()


# =============================================================================
# 14.  ANALYSIS PLOTS  (v11 addition — read-only use of results DataFrame)
# =============================================================================
# All six functions below only READ the ranked DataFrame produced by
# rank_designs().  They never modify the optimizer logic, scoring, or
# any design results.  Each function saves one PNG file and returns None.
# =============================================================================

def _analysis_style() -> dict:
    """Return a shared rcParams dict applied to every analysis figure."""
    return {
        "figure.facecolor":  "#f8f9fb",
        "axes.facecolor":    "#ffffff",
        "axes.edgecolor":    "#cccccc",
        "axes.linewidth":    1.0,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        "#e5e5e5",
        "grid.linewidth":    0.8,
        "font.family":       "sans-serif",
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "legend.framealpha": 0.85,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Pareto Frontier — battery temperature vs pressure drop
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto_frontier(
    df:      pd.DataFrame,
    outfile: str = "analysis_01_pareto_frontier.png",
) -> None:
    """
    Scatter all feasible designs on T_batt_mean vs dP axes and highlight
    the Pareto-optimal front (lowest T for any given dP and vice-versa).
    The final chosen design (rank 0) is labelled with a star.

    A design is Pareto-optimal if no other design is strictly better in
    BOTH objectives simultaneously (lower T AND lower dP).
    """
    df_f = df[df["feasible"]].copy()
    if df_f.empty:
        print("  [analysis] No feasible designs — skipping Pareto plot.")
        return

    T_col = "T_batt_mean_C"
    P_col = "dP_Pa"

    # ── Pareto front computation ─────────────────────────────────────────────
    T = df_f[T_col].values
    P = df_f[P_col].values
    pareto_mask = np.ones(len(df_f), dtype=bool)
    for i in range(len(df_f)):
        for j in range(len(df_f)):
            if i == j:
                continue
            if T[j] <= T[i] and P[j] <= P[i] and (T[j] < T[i] or P[j] < P[i]):
                pareto_mask[i] = False
                break

    df_pareto = df_f[pareto_mask].sort_values(P_col)
    df_other  = df_f[~pareto_mask]
    best      = df_f.iloc[0]   # rank-0 = best score

    # ── Figure ───────────────────────────────────────────────────────────────
    with matplotlib.rc_context(_analysis_style()):
        fig, ax = plt.subplots(figsize=(9, 6))

        # Topology colour coding
        topos = df_f["topology"].unique()
        cmap_t = matplotlib.colormaps.get_cmap("Set2")
        topo_col = {t: cmap_t(i / max(len(topos) - 1, 1)) for i, t in enumerate(topos)}

        for topo in topos:
            sub = df_other[df_other["topology"] == topo]
            if not sub.empty:
                ax.scatter(
                    sub[P_col] / 1e3, sub[T_col],
                    color=topo_col[topo], alpha=0.45, s=28,
                    edgecolors="none", label=topo,
                )

        # Pareto points (topology-coloured, larger, darker outline)
        for topo in topos:
            sub = df_pareto[df_pareto["topology"] == topo]
            if not sub.empty:
                ax.scatter(
                    sub[P_col] / 1e3, sub[T_col],
                    color=topo_col[topo], alpha=0.95, s=80,
                    edgecolors="#222222", linewidths=1.2, zorder=5,
                )

        # Pareto front step line
        ax.step(
            df_pareto[P_col].values / 1e3,
            df_pareto[T_col].values,
            where="post", color="#e74c3c", linewidth=1.8,
            linestyle="--", zorder=4, label="Pareto front",
        )

        # Shade dominated region
        x_front = np.append(df_pareto[P_col].values / 1e3, ax.get_xlim()[1] if df_pareto[P_col].max() / 1e3 < 1e4 else df_pareto[P_col].max() / 1e3 + 5)
        y_front = np.append(df_pareto[T_col].values, df_pareto[T_col].values[-1])
        ax.fill_stepx = None   # placeholder — use fill_between

        # Best chosen design star
        ax.scatter(
            best[P_col] / 1e3, best[T_col],
            marker="*", s=320, color="#f39c12", edgecolors="#222222",
            linewidths=1.5, zorder=10, label="Chosen design (best score)",
        )
        ax.annotate(
            f"  Best\n  T={best[T_col]:.1f}°C\n  ΔP={best[P_col]/1e3:.1f} kPa",
            xy=(best[P_col] / 1e3, best[T_col]),
            xytext=(15, -30), textcoords="offset points",
            fontsize=8, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
        )

        ax.set_xlabel("Pressure Drop  [kPa]")
        ax.set_ylabel("Mean Battery Temperature  [°C]")
        ax.set_title("Pareto Frontier — Battery Temperature vs Pressure Drop")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper right", framealpha=0.9)

        n_pareto = pareto_mask.sum()
        ax.text(
            0.02, 0.97,
            f"Feasible designs: {len(df_f)}\nPareto-optimal: {n_pareto}",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved '{outfile}'  (Pareto frontier)")
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Sensitivity Analysis — 4-panel scatter plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensitivity_analysis(
    df:      pd.DataFrame,
    outfile: str = "analysis_02_sensitivity.png",
) -> None:
    """
    Four scatter panels showing how individual design parameters relate to
    battery temperature and pressure drop:
      (a) n_branches          vs T_batt_mean_C
      (b) n_passes            vs T_batt_mean_C
      (c) D_const_mm          vs dP_Pa
      (d) bend_radius_mm      vs dP_Pa
    Points are jittered slightly on the x-axis to reduce overplotting.
    Median lines per discrete value show the central tendency clearly.
    """
    df_f = df[df["feasible"]].copy()
    if df_f.empty:
        print("  [analysis] No feasible designs — skipping sensitivity plot.")
        return

    # For channel diameter: use D_const_mm for CONSTANT; mean of D1/D2/D3 for TAPERED
    df_f = df_f.copy()
    def _rep_diam(row):
        if row.get("channel_mode", "CONSTANT") == "TAPERED":
            d1 = row.get("D1_mm") or 0
            d2 = row.get("D2_mm") or 0
            d3 = row.get("D3_mm") or 0
            valid = [v for v in [d1, d2, d3] if v and v > 0]
            return float(np.mean(valid)) if valid else np.nan
        return row.get("D_const_mm", np.nan)

    df_f["D_rep_mm"] = df_f.apply(_rep_diam, axis=1)

    panels = [
        ("n_branches",     "T_batt_mean_C", "Branches",              "Mean Battery Temp [°C]",   "Branches vs Battery Temperature",       "#2980b9", False),
        ("n_passes",       "T_batt_mean_C", "Passes per Branch",     "Mean Battery Temp [°C]",   "Passes vs Battery Temperature",         "#27ae60", False),
        ("D_rep_mm",       "dP_Pa",         "Representative D [mm]", "Pressure Drop [Pa]",       "Channel Diameter vs Pressure Drop",     "#8e44ad", True),
        ("bend_radius_mm", "dP_Pa",         "Bend Radius [mm]",      "Pressure Drop [Pa]",       "Bend Radius vs Pressure Drop",          "#e67e22", True),
    ]

    with matplotlib.rc_context(_analysis_style()):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.ravel()

        rng = np.random.default_rng(42)

        for ax, (xcol, ycol, xlabel, ylabel, title, color, log_y) in zip(axes, panels):
            sub = df_f[[xcol, ycol]].dropna()
            if sub.empty:
                ax.set_visible(False)
                continue

            x_raw = sub[xcol].values.astype(float)
            y_raw = sub[ycol].values.astype(float)

            # Jitter on x for discrete variables
            x_unique = np.unique(x_raw)
            jitter = rng.uniform(-0.08 * (x_unique[1] - x_unique[0]) if len(x_unique) > 1 else 0,
                                  0.08 * (x_unique[1] - x_unique[0]) if len(x_unique) > 1 else 0,
                                  size=len(x_raw))
            x_jit = x_raw + jitter

            ax.scatter(x_jit, y_raw, color=color, alpha=0.4, s=22,
                       edgecolors="none", zorder=3)

            # Median per unique x value
            for xv in x_unique:
                mask = (x_raw == xv)
                med  = np.median(y_raw[mask])
                ax.plot([xv - 0.15, xv + 0.15], [med, med],
                        color="#222222", linewidth=2.5, solid_capstyle="round", zorder=5)
                ax.text(xv, med, f" {med:.0f}" if ycol == "dP_Pa" else f" {med:.1f}°",
                        va="center", fontsize=7.5, color="#333333", zorder=6)

            if log_y:
                ax.set_yscale("log")

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            if len(x_unique) <= 10:
                ax.set_xticks(x_unique)

        fig.suptitle("Sensitivity Analysis — Parameter Effect on Key Metrics",
                     fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved '{outfile}'  (sensitivity analysis)")
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Design Space Heatmap — branches × passes → avg T_batt
# ─────────────────────────────────────────────────────────────────────────────

def plot_design_space_heatmap(
    df:      pd.DataFrame,
    outfile: str = "analysis_03_design_heatmap.png",
) -> None:
    """
    Heatmap: rows = n_branches, columns = n_passes.
    Cell colour = mean T_batt_mean_C across all feasible designs with that
    (n_branches, n_passes) combination.  Cell text shows the mean value and
    the number of feasible designs in that cell.
    A second panel shows mean dP_Pa using the same grid.
    """
    df_f = df[df["feasible"]].copy()
    if df_f.empty:
        print("  [analysis] No feasible designs — skipping heatmap.")
        return

    branches = sorted(df_f["n_branches"].unique())
    passes   = sorted(df_f["n_passes"].unique())

    def _make_grid(col):
        grid = np.full((len(branches), len(passes)), np.nan)
        cnt  = np.zeros_like(grid, dtype=int)
        for i, nb in enumerate(branches):
            for j, np_ in enumerate(passes):
                sub = df_f[(df_f["n_branches"] == nb) & (df_f["n_passes"] == np_)]
                if not sub.empty:
                    grid[i, j] = sub[col].mean()
                    cnt[i, j]  = len(sub)
        return grid, cnt

    T_grid, cnt_T  = _make_grid("T_batt_mean_C")
    dP_grid, cnt_dP = _make_grid("dP_Pa")

    with matplotlib.rc_context(_analysis_style()):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for ax, grid, cnt, cmap_name, label, fmt, title in [
            (axes[0], T_grid,  cnt_T,  "YlOrRd",  "Mean Battery Temp [°C]",  "{:.1f}°C", "Mean Battery Temperature"),
            (axes[1], dP_grid, cnt_dP, "YlGnBu",  "Mean Pressure Drop [Pa]", "{:.0f} Pa","Mean Pressure Drop"),
        ]:
            im = ax.imshow(grid, aspect="auto", cmap=cmap_name,
                           origin="lower", interpolation="nearest")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(label, fontsize=9)

            ax.set_xticks(range(len(passes)))
            ax.set_xticklabels([str(p) for p in passes])
            ax.set_yticks(range(len(branches)))
            ax.set_yticklabels([str(b) for b in branches])
            ax.set_xlabel("Passes per Branch")
            ax.set_ylabel("Number of Branches")
            ax.set_title(title, fontsize=11)

            # Annotate each cell
            for i in range(len(branches)):
                for j in range(len(passes)):
                    if not np.isnan(grid[i, j]):
                        val_str = fmt.format(grid[i, j])
                        n_str   = f"n={cnt[i, j]}"
                        # Choose white or black text for contrast
                        vmin, vmax = np.nanmin(grid), np.nanmax(grid)
                        norm_val = (grid[i, j] - vmin) / max(vmax - vmin, 1e-9)
                        txt_color = "white" if norm_val > 0.6 else "#222222"
                        ax.text(j, i, f"{val_str}\n{n_str}",
                                ha="center", va="center", fontsize=8,
                                color=txt_color, fontweight="bold")

        fig.suptitle("Design Space Heatmap: Branches × Passes",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved '{outfile}'  (design space heatmap)")
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Top-10 Design Ranking — horizontal bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_design_ranking(
    df:      pd.DataFrame,
    n:       int  = 10,
    outfile: str  = "analysis_04_top_designs.png",
) -> None:
    """
    Horizontal bar chart of the top-n feasible designs ranked by score.
    Bars are coloured by topology.  Labels on the right show T_batt and dP.
    A secondary axis shows the temperature axis for easy reading.
    """
    df_f = df[df["feasible"]].head(n).copy().reset_index(drop=True)
    if df_f.empty:
        print("  [analysis] No feasible designs — skipping ranking plot.")
        return

    # Build a readable label for each design
    def _label(row):
        mode = "T" if row.get("channel_mode", "CONSTANT") == "TAPERED" else "C"
        nb   = int(row["n_branches"])
        np_  = int(row["n_passes"])
        topo = row["topology"].replace("_SERPENTINE", "SN").replace("MIRRORED_U", "MU").replace("Z_FLOW","ZF").replace("CENTRAL_INLET","CI")
        diam = row.get("D_const_mm")
        d_str = f"D={diam:.0f}" if diam and not np.isnan(float(diam)) else "TAP"
        return f"#{int(row.get('design_id',0))}  {topo}  {nb}br/{np_}p  {d_str}  [{mode}]"

    df_f["label"] = df_f.apply(_label, axis=1)

    topos    = df_f["topology"].unique()
    cmap_t   = matplotlib.colormaps.get_cmap("Set2")
    topo_col = {t: cmap_t(i / max(len(topos) - 1, 1)) for i, t in enumerate(topos)}
    colors   = [topo_col[t] for t in df_f["topology"]]

    with matplotlib.rc_context(_analysis_style()):
        fig, ax = plt.subplots(figsize=(12, max(4, n * 0.55 + 1.5)))

        bars = ax.barh(
            range(len(df_f)), df_f["score"],
            color=colors, edgecolor="#555555", linewidth=0.6,
            height=0.65,
        )

        # Rank labels on left; metric labels on right
        for i, row in df_f.iterrows():
            ax.text(-0.3, i, f"#{i+1}", ha="right", va="center",
                    fontsize=8.5, color="#444444", fontweight="bold")
            ax.text(
                row["score"] + df_f["score"].max() * 0.005, i,
                f"T={row['T_batt_mean_C']:.1f}°C  ΔP={row['dP_Pa']/1e3:.1f}kPa",
                ha="left", va="center", fontsize=8, color="#333333",
            )

        ax.set_yticks(range(len(df_f)))
        ax.set_yticklabels(df_f["label"], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("Optimization Score  (lower = better)")
        ax.set_title(f"Top {len(df_f)} Designs Ranked by Score")

        # Score range annotation
        score_range = df_f["score"].max() - df_f["score"].min()
        ax.set_xlim(df_f["score"].min() * 0.97,
                    df_f["score"].max() + df_f["score"].max() * 0.12)

        # Topology legend
        legend_patches = [
            matplotlib.patches.Patch(color=topo_col[t], label=t)
            for t in topos
        ]
        ax.legend(handles=legend_patches, loc="lower right", title="Topology",
                  fontsize=8, title_fontsize=8)

        plt.tight_layout()
        plt.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved '{outfile}'  (top-{len(df_f)} ranking)")
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Topology Performance — grouped bar / violin chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_topology_performance(
    df:      pd.DataFrame,
    outfile: str = "analysis_05_topology_performance.png",
) -> None:
    """
    Two-panel figure:
      Left:  mean optimization score per topology (bar chart, error = std).
      Right: distribution of T_batt_mean_C per topology (violin + strip plot).
    Only feasible designs are used.
    """
    df_f = df[df["feasible"]].copy()
    if df_f.empty:
        print("  [analysis] No feasible designs — skipping topology plot.")
        return

    topos  = sorted(df_f["topology"].unique())
    cmap_t = matplotlib.colormaps.get_cmap("Set2")
    colors = [cmap_t(i / max(len(topos) - 1, 1)) for i in range(len(topos))]

    topo_mean_score = [df_f[df_f["topology"] == t]["score"].mean() for t in topos]
    topo_std_score  = [df_f[df_f["topology"] == t]["score"].std()  for t in topos]
    topo_count      = [len(df_f[df_f["topology"] == t])            for t in topos]

    with matplotlib.rc_context(_analysis_style()):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        # ── Left: mean score ────────────────────────────────────────────────
        ax = axes[0]
        x  = np.arange(len(topos))
        bars = ax.bar(x, topo_mean_score, yerr=topo_std_score,
                      color=colors, edgecolor="#444444", linewidth=0.8,
                      capsize=5, error_kw={"linewidth": 1.5, "capthick": 1.5},
                      zorder=3)

        for i, (mean, std, n) in enumerate(zip(topo_mean_score, topo_std_score, topo_count)):
            ax.text(i, mean + std + topo_mean_score[0] * 0.01,
                    f"{mean:.1f}\n(n={n})",
                    ha="center", va="bottom", fontsize=8, color="#222222")

        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("_SERPENTINE", "\nSERPENTINE") for t in topos],
                            fontsize=8.5)
        ax.set_ylabel("Mean Optimization Score  (lower = better)")
        ax.set_title("Mean Score by Topology")

        # ── Right: T_batt distribution ──────────────────────────────────────
        ax2 = axes[1]
        rng = np.random.default_rng(42)

        for i, (topo, color) in enumerate(zip(topos, colors)):
            sub = df_f[df_f["topology"] == topo]["T_batt_mean_C"].dropna().values
            if len(sub) < 2:
                ax2.scatter([i] * len(sub), sub, color=color, s=40, zorder=5,
                            edgecolors="#333333", linewidths=0.7)
                continue

            # Violin body
            parts = ax2.violinplot(sub, positions=[i], widths=0.6,
                                   showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.55)
                pc.set_edgecolor("#444444")
                pc.set_linewidth(0.8)

            # Strip plot (jittered)
            jitter = rng.uniform(-0.12, 0.12, len(sub))
            ax2.scatter(i + jitter, sub, color=color, s=18, alpha=0.65,
                        edgecolors="none", zorder=4)

            # Median line
            med = np.median(sub)
            ax2.plot([i - 0.22, i + 0.22], [med, med],
                     color="#222222", linewidth=2.2, zorder=5)
            ax2.text(i + 0.27, med, f"{med:.1f}°",
                     va="center", fontsize=7.5, color="#333333")

        ax2.set_xticks(range(len(topos)))
        ax2.set_xticklabels([t.replace("_SERPENTINE", "\nSERPENTINE") for t in topos],
                             fontsize=8.5)
        ax2.set_ylabel("Mean Battery Temperature  [°C]")
        ax2.set_title("Battery Temperature Distribution by Topology")

        fig.suptitle("Topology Performance Comparison",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved '{outfile}'  (topology performance)")
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: Score Component Breakdown — stacked bar for top-N designs
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_component_breakdown(
    df:      pd.DataFrame,
    n:       int  = 15,
    outfile: str  = "analysis_06_score_breakdown.png",
) -> None:
    """
    Stacked horizontal bar chart showing how each score component contributes
    to the total score for the top-n feasible designs.

    Components are computed directly from the DataFrame columns using the
    same formula as evaluate_design():
        T_component    = W_T_BATT   × T_batt_mean_C
        Tmax_component = W_T_BATT_MAX × T_batt_max_C
        dP_component   = W_DELTA_P  × dP_Pa
        unif_component = W_UNIFORMITY × uniformity_pen
        manuf_component= W_MANUFACTURABILITY × manuf_pen
        bu_component   = W_BRANCH_UNIFORMITY × branch_unif_pen
        vel_component  = W_VELOCITY_PEN × velocity_pen
        hf_component   = W_BRANCH_HF_PEN × branch_hf_pen
        topo_component = W_TOPOLOGY_COMPLEXITY × topo_complexity_pen
    """
    df_f = df[df["feasible"]].head(n).copy().reset_index(drop=True)
    if df_f.empty:
        print("  [analysis] No feasible designs — skipping breakdown plot.")
        return

    # Build component columns (safe .get with zero fallback)
    def _gc(col, default=0.0):
        return df_f[col].fillna(default) if col in df_f.columns else pd.Series(default, index=df_f.index)

    components = {
        "Temperature\n(mean)"    : W_T_BATT            * _gc("T_batt_mean_C"),
        "Temperature\n(max)"     : W_T_BATT_MAX         * _gc("T_batt_max_C"),
        "Pressure\nDrop"         : W_DELTA_P            * _gc("dP_Pa"),
        "Uniformity"             : W_UNIFORMITY         * _gc("uniformity_pen"),
        "Manufacturability"      : W_MANUFACTURABILITY  * _gc("manuf_pen"),
        "Branch\nUniformity"     : W_BRANCH_UNIFORMITY  * _gc("branch_unif_pen"),
        "Velocity\nPenalty"      : W_VELOCITY_PEN       * _gc("velocity_pen"),
        "Heat-Flux\nPenalty"     : W_BRANCH_HF_PEN      * _gc("branch_hf_pen"),
        "Topology\nComplexity"   : W_TOPOLOGY_COMPLEXITY * _gc("topo_complexity_pen"),
    }

    comp_df = pd.DataFrame(components)

    # Colour palette
    palette = [
        "#e74c3c", "#c0392b",   # red shades — temperature terms
        "#3498db",               # blue — pressure
        "#2ecc71",               # green — uniformity
        "#f39c12",               # orange — manufacturability
        "#9b59b6",               # purple — branch uniformity
        "#1abc9c",               # teal — velocity
        "#e67e22",               # amber — heat flux
        "#95a5a6",               # grey — topology
    ]

    # Design labels
    def _short_label(row):
        topo = row["topology"].replace("_SERPENTINE", "SN").replace("MIRRORED_U","MU").replace("Z_FLOW","ZF").replace("CENTRAL_INLET","CI")
        mode = "T" if row.get("channel_mode", "CONSTANT") == "TAPERED" else "C"
        return f"#{int(row.get('design_id',0))} {topo} {int(row['n_branches'])}br/{int(row['n_passes'])}p [{mode}]"

    labels = df_f.apply(_short_label, axis=1)

    with matplotlib.rc_context(_analysis_style()):
        fig, axes = plt.subplots(1, 2, figsize=(16, max(5, n * 0.55 + 2)),
                                 gridspec_kw={"width_ratios": [3, 1]})

        # ── Left: stacked horizontal bars ───────────────────────────────────
        ax = axes[0]
        lefts = np.zeros(len(df_f))

        for (comp_name, comp_vals), color in zip(components.items(), palette):
            ax.barh(
                range(len(df_f)), comp_vals.values, left=lefts,
                color=color, edgecolor="white", linewidth=0.4,
                height=0.72, label=comp_name,
            )
            lefts += comp_vals.values

        # Total score marker
        ax.scatter(df_f["score"], range(len(df_f)),
                   color="#222222", marker="|", s=80, linewidths=1.8, zorder=5,
                   label="Total score")

        ax.set_yticks(range(len(df_f)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Score Contribution  (lower = better)")
        ax.set_title(f"Score Component Breakdown — Top {len(df_f)} Designs")
        ax.legend(loc="lower right", fontsize=7.5, ncol=2,
                  title="Component", title_fontsize=8)

        # ── Right: pie chart of average component contributions ──────────────
        ax2 = axes[1]
        comp_means = comp_df.mean()
        pct_vals   = comp_means / comp_means.sum() * 100
        # Only show slices > 1 %
        visible = pct_vals[pct_vals > 1.0]
        wedge_colors = [palette[list(components.keys()).index(k)] for k in visible.index]

        wedges, texts, autotexts = ax2.pie(
            visible.values,
            labels=[k.replace("\n", " ") for k in visible.index],
            colors=wedge_colors,
            autopct="%1.1f%%",
            pctdistance=0.78,
            startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        )
        for at in autotexts:
            at.set_fontsize(7.5)
        for t in texts:
            t.set_fontsize(7.5)
        ax2.set_title("Avg Component\nShare", fontsize=10)

        fig.suptitle("Optimization Score Component Breakdown",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved '{outfile}'  (score breakdown)")
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Master dispatcher: generate all six analysis plots
# ─────────────────────────────────────────────────────────────────────────────

def generate_analysis_plots(df: pd.DataFrame) -> None:
    """
    Generate all six analysis plots from the ranked results DataFrame.
    Each plot is saved as a separate PNG file.

    Parameters
    ----------
    df : pd.DataFrame  output of rank_designs()
    """
    print("\n  ── Generating analysis plots ──────────────────────────────────")
    plot_pareto_frontier(df)
    plot_sensitivity_analysis(df)
    plot_design_space_heatmap(df)
    plot_top_design_ranking(df)
    plot_topology_performance(df)
    plot_score_component_breakdown(df)
    print("  ── Analysis plots complete ────────────────────────────────────\n")


# =============================================================================
# 15.  MAIN
# =============================================================================

def main() -> None:
    """
    Main execution sequence:
    1.  Print header and assumptions.
    2.  Compute battery heat load.
    3.  Get fluid and material properties.
    4.  Run brute-force parameter sweep.
    5.  Rank and display results.
    6.  Save CSV.
    7.  Generate plots.
    """
    print_header_and_assumptions()

    # ---- Step 1: Heat load ----
    Q_total = compute_battery_heat_load()
    fluid   = get_fluid_properties(T_C=30.0)
    print_baseline(Q_total, fluid)

    # ---- Step 2: Sweep ----
    results = run_parameter_sweep(Q_total, fluid)

    # ---- Step 3: Rank ----
    df = rank_designs(results)

    # ---- Step 4: Print top designs ----
    print_top_designs(df, n=TOP_N_PRINT)

    # ---- Step 5: Summary statistics ----
    df_f = df[df["feasible"]]
    n_infeasible = (~df["feasible"]).sum()
    print(f"  Total candidates   : {len(df)}")
    print(f"  Feasible designs   : {len(df_f)}")
    print(f"  Infeasible designs : {n_infeasible}")
    if not df_f.empty:
        best_row = df_f.iloc[0]
        print(f"\n  ★ BEST DESIGN:")
        print(f"      Topology       = {best_row.get('topology', 'H_SERPENTINE')}")
        print(f"      Manifold       = {best_row.get('manifold', 'LEFT_RIGHT')}")
        print(f"      Turn style     = {best_row.get('turn_style', 'CONNECTOR_SEMICIRCLE')}")
        print(f"      n_branches     = {int(best_row['n_branches'])}")
        print(f"      n_passes       = {int(best_row['n_passes'])}")
        print(f"      Channel mode   = {best_row.get('channel_mode', 'CONSTANT')}")
        print(f"      D_channel      = {best_row['D_const_mm']:.1f} mm")
        print(f"      Bend radius    = {best_row['bend_radius_mm']:.1f} mm")
        print(f"      T_batt_mean    = {best_row.get('T_batt_mean_C', best_row['T_batt_est_C']):.2f} °C")
        print(f"      T_batt_max     = {best_row.get('T_batt_max_C', best_row['T_batt_est_C']):.2f} °C")
        print(f"      T_out          = {best_row['T_out_C']:.2f} °C")
        print(f"      Pressure drop  = {best_row['dP_Pa']:.1f} Pa  (manifold: {best_row.get('dP_manifold_Pa',0):.1f} Pa)")
        print(f"      Re_avg         = {best_row['Re_avg']:.0f}")
        print(f"      Coverage ratio = {best_row['coverage_ratio']:.3f}")
        print(f"      Branch flows   = mean {best_row.get('mean_branch_flow',0)*1e3:.2f} g/s, "
              f"std {best_row.get('std_branch_flow',0)*1e3:.2f} g/s")
        print(f"      Branch unif pen= {best_row.get('branch_unif_pen',0):.4f}")
        print(f"      Score          = {best_row['score']:.4f}")

        # Change 2: velocity band summary
        v_min = best_row.get("velocity_min_m_s", 0.0)
        v_max = best_row.get("velocity_max_m_s", 0.0)
        v_pen = best_row.get("velocity_pen", 0.0)
        in_band = "✓ in band" if v_pen < 0.05 else "⚠ outside band"
        print(f"      Velocity range = {v_min:.2f}–{v_max:.2f} m/s  "
              f"(target {V_TARGET_MIN_M_S}–{V_TARGET_MAX_M_S} m/s)  {in_band}")

        # Change 3: branch-count / heat-flux recommendation
        nb_best  = int(best_row["n_branches"])
        q_branch = best_row.get("q_branch_W_m2", 0.0)
        hf_pen   = best_row.get("branch_hf_pen", 0.0)
        print(f"\n  ── Branch-Count Heat-Flux Assessment (Change 3) ──")
        print(f"      Planform heat flux  : {best_row.get('q_planar_W_m2', 0):.0f} W/m²")
        print(f"      Heat per branch     : {best_row.get('Q_per_branch_W', 0):.1f} W  "
              f"({nb_best} branches)")
        print(f"      Area per branch     : {best_row.get('A_per_branch_m2', 0)*1e4:.1f} cm²")
        print(f"      Branch heat flux    : {q_branch:.0f} W/m²  "
              f"(target {Q_BRANCH_TARGET_W_M2_MIN:.0f}–"
              f"{Q_BRANCH_TARGET_W_M2_MAX:.0f} W/m²)")
        # Recommendation logic (Change 3 optional feature)
        Q_total_for_reco = compute_battery_heat_load()
        A_plate = BATTERY_LENGTH_M * BATTERY_WIDTH_M
        print(f"\n  ── Branch-Count Recommendation ──")
        for nb_test in N_BRANCHES_OPTIONS:
            Q_b_test   = Q_total_for_reco / nb_test
            m_dot_b    = M_DOT_TOTAL_KG_S / nb_test
            dT_b       = Q_b_test / max(m_dot_b * CP_WATER_J_KGK, 1e-9)
            flag = ""
            if dT_b > 15.0:
                flag = "  ⚠ coolant rise too high — may need more branches"
            elif dT_b < 2.0:
                flag = "  ⚠ very low ΔT per branch — possible maldistribution risk"
            else:
                flag = "  ✓ coolant temperature rise in target range"
            marker = " ← best" if nb_test == nb_best else ""
            print(f"      {nb_test} branches  →  Q={Q_b_test:.1f} W, ΔT={dT_b:.1f} K{flag}{marker}")

    # ---- Step 6: Save CSV ----
    print()
    save_results_csv(df)

    # ---- Step 7: Plots ----
    print("\n  Generating plots …")
    plot_results(df)

    # ---- Step 8: Best design schematic ----
    if not df_f.empty:
        best_design_id = int(df_f.iloc[0]["design_id"])
        best_result    = next(r for r in results if r.design_id == best_design_id)
        plot_best_design_schematic(best_result)

    # ---- Step 9: Engineering plan-view geometry schematic ----
    if not df_f.empty:
        print("\n  Generating engineering plan-view geometry schematic …")
        plot_planar_geometry_schematic(best_result,
                                       outfile="planar_geometry_schematic.png",
                                       show_table=True,
                                       label="★ Best Design — Plan View")

    # ---- Step 10: Select 5 random feasible designs ----
    #
    #   After the full parameter sweep, randomly sample up to 5 feasible
    #   designs using a fixed seed (random_state=42) so the selection is
    #   reproducible across runs.  These are used purely for visualisation
    #   and reporting — the optimization result is unchanged.
    #
    N_RANDOM     = 5
    RANDOM_SEED  = 42
    random_results: List[DesignResult] = []
    if not df_f.empty:
        random_results = select_random_feasible_designs(
            results, df_f, n=N_RANDOM, random_state=RANDOM_SEED,
        )
        print(f"\n  Selected {len(random_results)} random feasible designs "
              f"(seed={RANDOM_SEED}) for visualisation.")

    # ---- Step 11: Individual plan-view schematics for each random design ----
    for i, rr in enumerate(random_results, start=1):
        fname = f"random_design_{i}.png"
        print(f"  Plotting random design {i} (ID={rr.design_id}) …")
        plot_planar_geometry_schematic(
            rr,
            outfile=fname,
            show_table=False,
            label=f"Random Design {i}  (ID={rr.design_id})",
        )

    # ---- Step 12: 6-panel comparison figure ----
    if not df_f.empty and random_results:
        print_design_comparison_table(random_results, best_result)
        plot_design_comparison_panel(
            random_results, best_result,
            outfile="design_comparison_panel.png",
        )

    # ---- Step 13: Analysis plots (v11) ----
    print("\n  Generating analysis plots …")
    generate_analysis_plots(df)

    print("\n  Optimization complete.")
    print("=" * 70)
    print()
    print("  HOW TO EDIT THIS CODE")
    print("  =====================")
    print("  ► Adjust fixed BCs: lines 57–68  (battery size, Q_vol, m_dot, …)")
    print("  ► Change sweep ranges: lines 96–101  (N_BRANCHES_OPTIONS, …)")
    print("  ► Change diameter options: lines 86–93  (D_CONST_OPTIONS_M, …)")
    print("  ► Tune objective weights: lines 103–107 (W_T_BATT, W_DELTA_P, …)")
    print("  ► Switch to 3-zone diameters: set USE_3ZONE_DIAMETER = True  (line 84)")
    print("  ► Change phi or K_bend: lines 73–74")
    print()
    print("  SUGGESTED FUTURE IMPROVEMENTS")
    print("  ==============================")
    print("  1. Unequal flow split: minimise branch pressure drop residuals")
    print("     using scipy.optimize to solve for flow distribution.")
    print("  2. Better Nusselt correlations: Gnielinski (transitional),")
    print("     Dean-number correlation for curved channels.")
    print("  3. 2-D spreading resistance (Lee & Vafai, 1999) to capture")
    print("     lateral conduction between channels.")
    print("  4. Genetic algorithm (DEAP or scipy.differential_evolution)")
    print("     for faster convergence on larger parameter spaces.")
    print("  5. CFD validation: export best design geometry as a CSV of")
    print("     channel centreline coordinates for CAD import.")
    print("  6. Transient model: add thermal capacitance of battery and plate")
    print("     to simulate charge/discharge cycles.")
    print("  7. Variable pitch: allow non-uniform pass spacing to target")
    print("     hot zones identified from cell-level thermal maps.")
    print("=" * 70)


if __name__ == "__main__":
    main()
