import numpy as np
import pandas as pd

# -----------------------------
# Heat rates (MMBtu/MWh)
# -----------------------------
HR_BY_FUEL = {
    "NG":  7.75,
    "DFO": 11.2,
    "KER": 13.2,
    "JF":  13.2,
    "RFO": 14.2,
    "BIT": 10.8,
    "LFG": 12.0,
    "MSW": 15.0,
    "WDS": 14.0,
    "WAT": 0.0,
    "PS":  0.0,
    "BAT": 0.0,
    "NUC": 0.0,
    "WON": 0.0,
    "WOF": 0.0,
    "SUN": 0.0,
    "OBG": 0.0,
    "HSB": 0.0,
}

# -----------------------------
# Non-input fuel prices ($/MMBtu)
# -----------------------------
FUEL_PRICE_DEFAULT = {
    "DFO": 22.0,
    "KER": 23.0,
    "JF":  25.0,
    "RFO": 15.0,
    "BIT": 5.2,
    "LFG": 2.0,
    "MSW": 0.0,
    "WDS": 0.0,
    "WAT": 0.0,
    "PS":  0.0,
    "BAT": 0.0,
    "NUC": 0.0,
    "WON": 0.0,
    "WOF": 0.0,
    "SUN": 0.0,
    "HSB": 0.0,
    "OBG": 0.0,
}

# -----------------------------
# Availability by fuel (internal generators only)
# -----------------------------
AVAIL_BY_FUEL = {
    "NG":  0.80,
    "DFO": 0.80,
    "KER": 0.80,
    "BIT": 0.80,
    "NUC": 0.80,
    "RFO": 0.80,
    "WAT": 0.50,
    "LFG": 0.80,
    "MSW": 0.80,
    "WDS": 0.80,
    "PS":  0.80,

}

# RGGI Adder

RGGI_PRICE = 23.0  # $/short ton CO2

CO2_TON_PER_MMBTU = {
    "NG": 0.053,   # approx 117 lb/MMBtu
    "DFO": 0.074,
    "KER": 0.074,
    "JF":  0.074,
    "RFO": 0.078,
    "BIT": 0.095,
}



def fuel_price_per_mmbtu(fuel: str, gas_price: float) -> float:
    if str(fuel).upper() == "NG":
        return float(gas_price)
    return float(FUEL_PRICE_DEFAULT.get(str(fuel).upper(), 0.0))

# -----------------------------
# Load generator stack
# -----------------------------
gen = pd.read_csv("Generator capacities v2.csv").rename(columns={"generator": "gen_name"})

# normalize column names a bit (optional safety)
gen.columns = [c.strip() for c in gen.columns]

gen["capacity_mw"] = pd.to_numeric(gen["capacity_mw"], errors="coerce")
gen["heat_rate"] = pd.to_numeric(gen.get("heat_rate", np.nan), errors="coerce")

# fill missing heat rates from HR_BY_FUEL
mask_missing_hr = gen["heat_rate"].isna()
gen.loc[mask_missing_hr, "heat_rate"] = gen.loc[mask_missing_hr, "fuel_type"].map(HR_BY_FUEL)

unknown = gen[gen["heat_rate"].isna()]
if not unknown.empty:
    print("WARNING: some units still have no heat rate after fallback:")
    print(unknown[["gen_name", "fuel_type", "capacity_mw"]].head(30))

gen_stack = gen[["gen_name", "fuel_type", "capacity_mw", "heat_rate"]].copy()

# apply availability derates (internal generators only)
gen_stack["avail_factor"] = gen_stack["fuel_type"].map(AVAIL_BY_FUEL).fillna(1.0)
gen_stack["capacity_mw"] = gen_stack["capacity_mw"] * gen_stack["avail_factor"]

# -----------------------------
# Economic stack clearing
# -----------------------------
def clear_economic_stack(load_mw: float, gas_price: float):

    load_mw = float(load_mw)
    if load_mw <= 0:
        return (0.0, "NONE", "NONE", {}, pd.DataFrame())

    df = gen_stack.copy()
    df["fuel_price"] = df["fuel_type"].apply(lambda f: fuel_price_per_mmbtu(f, gas_price))
    df["mc"] = df["fuel_price"] * df["heat_rate"]

    df["co2_ton_per_mwh"] = df["heat_rate"] * df["fuel_type"].map(CO2_TON_PER_MMBTU).fillna(0.0)
    df["rggi_adder"] = df["co2_ton_per_mwh"] * float(RGGI_PRICE)
    df["mc"] = df["mc"] + df["rggi_adder"]

    df = df[["gen_name", "fuel_type", "capacity_mw", "mc"]].copy()
    df = df.sort_values("mc").reset_index(drop=True)

    df["cum_mw"] = df["capacity_mw"].cumsum()
    df["cum_mw_prev"] = df["cum_mw"] - df["capacity_mw"]

    idx = df["cum_mw"].searchsorted(load_mw, side="left")
    if idx >= len(df):
        idx = len(df) - 1

    marginal_row = df.iloc[idx]

    flows = {}
    for _, r in df.iterrows():
        used = max(0.0, min(float(r["capacity_mw"]), float(load_mw) - float(r["cum_mw_prev"])))
        if used > 0:
            flows[str(r["gen_name"])] = used

    return (
        float(marginal_row["mc"]),
        str(marginal_row["gen_name"]),
        str(marginal_row["fuel_type"]),
        flows,
        df,
    )

# -----------------------------
# Interface inputs + shapes
# -----------------------------
inputs = pd.read_csv("Interface_inputs.csv")
shapes = pd.read_csv("Interface shapes.csv")

# strip column whitespace
inputs.columns = [c.strip() for c in inputs.columns]
shapes.columns = [c.strip() for c in shapes.columns]

# coerce hour
inputs["hour"] = pd.to_numeric(inputs["hour"], errors="coerce")
shapes["hour"] = pd.to_numeric(shapes["hour"], errors="coerce")

inputs = inputs.dropna(subset=["hour"]).copy()
shapes = shapes.dropna(subset=["hour"]).copy()

inputs["hour"] = inputs["hour"].astype(int)
shapes["hour"] = shapes["hour"].astype(int)

# numeric coercion for everything except hour
for df in (inputs, shapes):
    for c in df.columns:
        if c != "hour":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

# merge
df = inputs.merge(shapes, on="hour", how="inner", suffixes=("_cap", "_shape"))

if df.empty:
    raise ValueError("After merging Interface_inputs and Interface_shapes on hour, dataframe is empty. Check hour values (0–23).")

# -----------------------------
# FIX: shapes must be 0..1
# If any *_shape column has max > 1, normalize by dividing by max
# (this directly fixes your hq_highgate_imp = 225 issue)
# -----------------------------
shape_cols = [c for c in df.columns if c.endswith("_shape") and c != "hour_shape"]

bad_shapes = []
for c in shape_cols:
    mx = float(df[c].max())
    if mx > 1.0 + 1e-9:
        bad_shapes.append((c, mx))
        if mx > 0:
            df[c] = df[c] / mx  # normalize to 0..1

if bad_shapes:
    print("WARNING: Detected shape columns with values > 1. Normalized by dividing by column max:")
    for c, mx in bad_shapes:
        print(f"  - {c}: max was {mx}")

# -----------------------------
# Compute net load & interchange
# -----------------------------
required_cols = {"fcst_load", "fcst_solar", "fcst_wind", "gas_price"}
missing = required_cols - set(inputs.columns)
if missing:
    raise ValueError(f"Interface_inputs.csv missing required columns: {sorted(missing)}")

df["net_load"] = df["fcst_load"] - df["fcst_solar"] - df["fcst_wind"]

# find all interface base names from *_cap columns
cap_cols = [c for c in df.columns if c.endswith("_cap") and (c[:-4].endswith("_imp") or c[:-4].endswith("_exp"))]

# compute per-interface MW = cap * shape
mw_cols = []
for cap_col in cap_cols:
    base = cap_col[:-4]              # e.g. "ny_ac_imp"
    shape_col = base + "_shape"      # e.g. "ny_ac_imp_shape"
    if shape_col not in df.columns:
        continue
    mw_col = base + "_mw"
    df[mw_col] = df[cap_col] * df[shape_col]
    mw_cols.append(mw_col)

# total imports/exports
imp_mw_cols = [c for c in mw_cols if c.endswith("_imp_mw")]
exp_mw_cols = [c for c in mw_cols if c.endswith("_exp_mw")]

df["total_import_mw"] = df[imp_mw_cols].sum(axis=1) if imp_mw_cols else 0.0
df["total_export_mw"] = df[exp_mw_cols].sum(axis=1) if exp_mw_cols else 0.0

# load after interchange:
# imports reduce load; exports add to load
df["load_after_interchange"] = df["net_load"] + df["total_export_mw"] - df["total_import_mw"]

# -----------------------------
# Run stack pricing hour by hour
# -----------------------------
prices = []
marginal_gens = []
marginal_fuels = []
flows_list = []

for _, row in df.iterrows():
    price, mg, mf, flows, _stack = clear_economic_stack(row["load_after_interchange"], row["gas_price"])
    prices.append(price)
    marginal_gens.append(mg)
    marginal_fuels.append(mf)
    flows_list.append(flows)

df["stack_price"] = prices
df["marginal_gen"] = marginal_gens
df["marginal_fuel"] = marginal_fuels
df["flows"] = flows_list

# Output

df = df.sort_values("hour").reset_index(drop=True)

peak_mask = (df["hour"] >= 7) & (df["hour"] <= 22)
peak_price = df.loc[peak_mask, "stack_price"].mean()
offp_price = df.loc[~peak_mask, "stack_price"].mean()

print("=== Day Ahead Stack Pricer ===")
print(f"\n\033[1mPeak strip price:     {peak_price:0.2f} $/MWh\033[0m")
print(f"\033[1mOff-peak strip price: {offp_price:0.2f} $/MWh\033[0m")

# show a compact table for debugging interchange
show_cols = [
    "hour",
    "fcst_load", "fcst_solar", "fcst_wind",
    "net_load",
    "load_after_interchange",
    "gas_price",
    "stack_price",
    "marginal_gen",
]
# include a few interface MW columns if present
for extra in ["necec_imp_mw", "hq_p2_imp_mw", "hq_highgate_imp_mw", "ny_ac_imp_mw", "ny_nnc_imp_mw", "ny_csc_imp_mw"]:
    if extra in df.columns:
        show_cols.append(extra)

print("\nHourly detail:\n")
print(df[show_cols].to_string(index=False))
