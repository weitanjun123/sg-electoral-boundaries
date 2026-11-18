"""Election results (GE2006-GE2025) + MRT test.

Inputs (all in data/raw/):
  results/SGParliamentaryGeneralElectionResultsbyCandidate.csv  ELD results by candidate (data.gov.sg)
  ElectoralBoundary{year}.geojson                                ELD boundaries
  hdb/hdb_property_info.json, hdb/hdb_block_coords.json          HDB blocks: units, year completed, lat/lon
  mrt/stations.json                                              MRT station points (LTA via sgraildata)
Output: docs/results.json (page data) and docs/analysis.txt (full regression output).
"""
import csv, json, math, re, sys, datetime as dt
from collections import defaultdict
import numpy as np
from matplotlib.path import Path
from scipy import stats

RAW, OUT = sys.argv[1], sys.argv[2]
YEARS = [2006, 2011, 2015, 2020, 2025]
POLL = {2006: dt.date(2006, 5, 6), 2011: dt.date(2011, 5, 7), 2015: dt.date(2015, 9, 11),
        2020: dt.date(2020, 7, 10), 2025: dt.date(2025, 5, 3)}
WALKOVER_2025 = {"MARINE PARADE-BRADDELL HEIGHTS"}  # not in the results file: uncontested
OPP_COLOURLESS = True

# ---------- MRT opening dates (station code -> date). Sources: LTA / sgtrains.com network figures.
D = dt.date
OPEN = {}
def stage(codes, date):
    for c in codes: OPEN[c] = date
# everything opened before GE2006 is treated as 'always open' (date 2000-01-01)
OLD = D(2000, 1, 1)
stage([f"NS{i}" for i in range(1, 28) if i not in (6, 12)], OLD)
stage([f"EW{i}" for i in range(1, 28)], OLD); stage(["CG1", "CG2"], OLD)
stage([f"NE{i}" for i in range(1, 18) if i not in (2, 11, 15)], OLD)
stage(["NE15"], D(2006, 1, 15)); stage(["NE11"], D(2011, 6, 20)); stage(["NE18"], D(2024, 12, 10))
stage(["NS28"], D(2014, 11, 23)); stage(["NS12"], D(2019, 11, 2))
stage(["EW28", "EW29"], D(2009, 2, 28)); stage(["EW30", "EW31", "EW32", "EW33"], D(2017, 6, 18))
stage([f"CC{i}" for i in range(12, 17)], D(2009, 5, 28))
stage([f"CC{i}" for i in range(1, 12)], D(2010, 4, 17))
stage([f"CC{i}" for i in range(17, 30) if i != 18], D(2011, 10, 8))
stage(["CE1", "CE2"], D(2012, 1, 14))
stage([f"DT{i}" for i in range(14, 20)], D(2013, 12, 22))
stage([f"DT{i}" for i in range(1, 14) if i != 4], D(2015, 12, 27))
stage([f"DT{i}" for i in range(20, 36)], D(2017, 10, 21))
stage(["DT4"], D(2025, 2, 28))
stage(["TE1", "TE2", "TE3"], D(2020, 1, 31))
stage([f"TE{i}" for i in range(4, 10)], D(2021, 8, 28))
stage([f"TE{i}" for i in range(11, 23) if i != 21] + ["TE21"], D(2022, 11, 13))
stage([f"TE{i}" for i in range(23, 30)], D(2024, 6, 23))
# CC30-CC34 (CCL6, Jul 2026), TE30/31, DT36/37 open after GE2025 -> not in table -> ignored

LON0, LAT0 = 103.60, 1.475
KX = 111320 * math.cos(math.radians(1.35)); KY = 110574
def proj(lon, lat):  # metres
    return np.array([(lon - LON0) * KX, (LAT0 - lat) * KY])

def norm(n):
    n = re.sub(r"\s*-\s*", "-", n.strip().upper())
    return re.sub(r"\s+(GRC|SMC)$", "", n)

# ---------- boundaries
FILES = {y: f"{RAW}/ElectoralBoundary{y}.geojson" for y in YEARS}
def ed_name(p):
    if "ED_DESC" in p: return norm(p["ED_DESC"])
    t = dict(re.findall(r"<th>(\w+)</th>\s*<td>([^<]*)</td>", p["Description"]))
    return norm(t["ED_DESC"])
polys = {}
for y in YEARS:
    gj = json.load(open(FILES[y])); m = defaultdict(list)
    for f in gj["features"]:
        g = f["geometry"]; P = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in P:
            for ring in poly:
                m[ed_name(f["properties"])].append(np.array([proj(c[0], c[1]) for c in ring]))
    polys[y] = m

# ---------- HDB blocks
info = json.load(open(f"{RAW}/hdb/hdb_property_info.json"))["records"]
coords = json.load(open(f"{RAW}/hdb/hdb_block_coords.json"))
blocks = []
for r in info:
    if r["residential"] != "Y": continue
    k = f'{r["blk_no"]}|{r["street"]}'
    if k not in coords: continue
    lat, lon = coords[k][0], coords[k][1]
    u = int(r["total_dwelling_units"] or 0)
    if u <= 0: continue
    blocks.append((proj(lon, lat), u, int(r["year_completed"])))
BX = np.array([b[0] for b in blocks]); BU = np.array([b[1] for b in blocks], float); BY = np.array([b[2] for b in blocks])
n_info_res = sum(1 for r in info if r["residential"] == "Y")
print(f"HDB residential blocks: {n_info_res}, with coordinates: {len(blocks)} ({BU.sum():,.0f} units)")

# block -> constituency per year (even-odd for holes)
bcon = {}
for y in YEARS:
    a = np.full(len(blocks), "", object)
    for n, rings in polys[y].items():
        inside = np.zeros(len(blocks), bool)
        for r in rings: inside ^= Path(r).contains_points(BX)
        a[inside] = n
    bcon[y] = a
    print(y, "blocks unassigned:", (a == "").sum())

# ---------- stations
st = json.load(open(f"{RAW}/mrt/stations.json"))["stations"]
SX, SOPEN, SLINES, SNAME, SCODES = [], [], [], [], []
for s in st:
    codes = [c["code"] for c in s["codes"] if c["code"] in OPEN]
    if not codes: continue
    SX.append(proj(s["lon"], s["lat"])); SOPEN.append(min(OPEN[c] for c in codes))
    SLINES.append({re.match(r"[A-Z]+", c).group(0) for c in codes}); SNAME.append(s["name"]); SCODES.append(codes)
SX = np.array(SX)
print("MRT stations with dates:", len(SX))
DIST = np.sqrt(((BX[:, None, :] - SX[None, :, :]) ** 2).sum(-1))  # blocks x stations, metres

def open_mask(date): return np.array([d <= date for d in SOPEN])
def access(date):
    m = open_mask(date); return (DIST[:, m] <= 800).any(1)
def nearest(date):
    m = np.where(open_mask(date))[0]; j = m[np.argmin(DIST[:, m], 1)]; return j

# ---------- results
rows = [r for r in csv.DictReader(open(f"{RAW}/results/SGParliamentaryGeneralElectionResultsbyCandidate.csv"))
        if int(r["year"]) in YEARS]
res = {y: {} for y in YEARS}
for r in rows:
    y = int(r["year"]); n = norm(r["constituency"])
    e = res[y].setdefault(n, dict(parties=[], walkover=False, kind=r["constituency_type"]))
    try: v = int(float(r["vote_count"]))
    except ValueError: e["walkover"] = True; e["parties"].append([r["party"], None]); continue
    e["parties"].append([r["party"], v])
for n in WALKOVER_2025: res[2025][n] = dict(parties=[["PAP", None]], walkover=True, kind="GRC")
for y in YEARS:
    bnames = set(polys[y]); rnames = set(res[y])
    assert bnames == rnames, (y, bnames ^ rnames)
    for n, e in res[y].items():
        if e["walkover"]: e["pap"] = None; e["total"] = 0; continue
        e["total"] = sum(v for _, v in e["parties"])
        e["pap"] = sum(v for p, v in e["parties"] if p == "PAP") / e["total"]
        e["parties"].sort(key=lambda x: -x[1])

national = []
for y in YEARS:
    tot = defaultdict(int)
    for e in res[y].values():
        for p, v in e["parties"]:
            if v is not None: tot[p] += v
    T = sum(tot.values())
    wo = sorted(n for n, e in res[y].items() if e["walkover"])
    national.append(dict(year=y, valid=T, parties=sorted(([p, v] for p, v in tot.items()), key=lambda x: -x[1]),
                         pap=tot["PAP"] / T, walkovers=wo,
                         contested=sum(1 for e in res[y].values() if not e["walkover"])))
    print(y, f"PAP {100*tot['PAP']/T:.2f}% of {T:,} valid votes; walkovers {wo}")

# ---------- per constituency-election features
obs = []
for i, y in enumerate(YEARS):
    date = POLL[y]
    exist = BY < date.year  # blocks completed before the election year
    acc_now = access(date)
    acc_prev = access(POLL[YEARS[i - 1]]) if i else None
    near = nearest(date)
    nsew = np.array([bool(SLINES[j] & {"NS", "EW", "CG"}) for j in near])
    west9 = {"EW19", "EW20", "EW21", "EW22", "EW23", "EW24", "EW25", "EW26", "EW27"}  # Queenstown..Boon Lay
    ewwest = np.array([bool(set(SCODES[j]) & west9) for j in near])
    for n in polys[y]:
        m = exist & (bcon[y] == n); u = BU[m].sum()
        f = dict(year=y, name=n, units=float(u), grc=res[y][n]["kind"] == "GRC" or len([1 for _ in []]) > 0,
                 pap=res[y][n]["pap"])
        f["grc"] = res[y][n]["kind"] == "GRC"
        if u > 0:
            f["access"] = float(BU[m & acc_now].sum() / u)
            f["nsew"] = float(BU[m & nsew].sum() / u)
            f["ewwest"] = float(BU[m & ewwest].sum() / u)
            f["new_homes"] = float(BU[m & (BY >= date.year - 5)].sum() / u)  # completed in the 5 yrs before
        else:
            f["access"] = f["nsew"] = f["ewwest"] = f["new_homes"] = None
        if i:
            py = YEARS[i - 1]
            f["d_access"] = float(BU[m & acc_now].sum() / u - BU[m & acc_prev].sum() / u) if u else None
            # notional previous PAP share: units-weighted over blocks' previous constituency
            pw, ps, wo = 0.0, 0.0, 0.0
            for pn in set(bcon[py][m]):
                if not pn: continue
                w = BU[m & (bcon[py] == pn)].sum()
                if res[py][pn]["walkover"]: wo += w
                else: pw += w; ps += w * res[py][pn]["pap"]
            f["prev_pap"] = ps / pw if pw else None
            f["prev_walkover_share"] = wo / (wo + pw) if (wo + pw) else None
            # stations newly opened since the previous election, inside or serving this constituency
            newst = [SNAME[j] for j in range(len(SX)) if POLL[py] < SOPEN[j] <= date and (DIST[m, j] <= 800).any()]
            f["new_stations"] = sorted(set(newst))
        obs.append(f)

# ---------- regression helpers (OLS + HC1 robust SE)
def ols(y, X, names):
    X = np.column_stack([np.ones(len(y))] + X); n, k = X.shape
    b, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X); V = XtXi @ (X.T * e ** 2) @ X @ XtXi * n / (n - k)
    se = np.sqrt(np.diag(V)); t = b / se; p = 2 * stats.t.sf(np.abs(t), n - k)
    r2 = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(n=int(n), r2=float(r2), coef={nm: dict(b=float(b[i]), se=float(se[i]), p=float(p[i]))
                                               for i, nm in enumerate(["const"] + names)})

def perm_p(y, x, groups, controls, n=5000, seed=1):
    """Permutation p-value for x's coefficient, shuffling x within groups (elections)."""
    rng = np.random.default_rng(seed)
    def coef(xx):
        X = np.column_stack([np.ones(len(y)), xx] + controls)
        return np.linalg.lstsq(X, y, rcond=None)[0][1]
    b0 = coef(x); cnt = 0
    for _ in range(n):
        xx = x.copy()
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]; xx[idx] = rng.permutation(xx[idx])
        cnt += abs(coef(xx)) >= abs(b0)
    return (cnt + 1) / (n + 1)

lines = []
def log(s=""): print(s); lines.append(s)

swing = [o for o in obs if o["year"] > 2006 and o["pap"] is not None and o.get("prev_pap") is not None
         and (o["prev_walkover_share"] or 0) <= 0.25 and o["units"] > 0]
log(f"Swing sample: {len(swing)} contested constituency-elections (2011-2025) with a notional previous result")
yv = np.array([100 * (o["pap"] - o["prev_pap"]) for o in swing])
yr = np.array([o["year"] for o in swing])
fe = [(yr == y).astype(float) for y in YEARS[2:]]  # 2011 = base
prev = np.array([100 * o["prev_pap"] for o in swing]); grc = np.array([float(o["grc"]) for o in swing])
dacc = np.array([100 * o["d_access"] for o in swing]); newd = np.array([float(bool(o["new_stations"])) for o in swing])
acc = np.array([100 * o["access"] for o in swing])

tests = {}
m1 = ols(yv, [dacc, prev, grc] + fe, ["d_access_pp", "prev_pap", "grc", "y2015", "y2020", "y2025"])
m1["perm_p"] = perm_p(yv, dacc, yr, [prev, grc] + fe); tests["access_change"] = m1
m2 = ols(yv, [newd, prev, grc] + fe, ["new_station", "prev_pap", "grc", "y2015", "y2020", "y2025"])
m2["perm_p"] = perm_p(yv, newd, yr, [prev, grc] + fe); tests["new_station"] = m2
m3 = ols(yv, [acc, prev, grc] + fe, ["access_pp", "prev_pap", "grc", "y2015", "y2020", "y2025"])
m3["perm_p"] = perm_p(yv, acc, yr, [prev, grc] + fe); tests["access_level_swing"] = m3

lev = [o for o in obs if o["pap"] is not None and o["units"] > 0]
ly = np.array([100 * o["pap"] for o in lev]); lyr = np.array([o["year"] for o in lev])
lfe = [(lyr == y).astype(float) for y in YEARS[1:]]
m4 = ols(ly, [np.array([100 * o["access"] for o in lev]), np.array([float(o["grc"]) for o in lev])] + lfe,
         ["access_pp", "grc", "y2011", "y2015", "y2020", "y2025"])
m4["perm_p"] = perm_p(ly, np.array([100 * o["access"] for o in lev]), lyr, [np.array([float(o["grc"]) for o in lev])] + lfe)
tests["access_level_share"] = m4

def event(year, key, label):
    s = [o for o in swing if o["year"] == year]
    y_ = np.array([100 * (o["pap"] - o["prev_pap"]) for o in s]); x = np.array([100 * o[key] for o in s])
    c = [np.array([100 * o["prev_pap"] for o in s]), np.array([float(o["grc"]) for o in s])]
    m = ols(y_, [x] + c, [key + "_pp", "prev_pap", "grc"]); m["perm_p"] = perm_p(y_, x, np.zeros(len(s)), c)
    m["label"] = label; return m
tests["nsewl_2015"] = event(2015, "nsew", "GE2015, two months after the 7 Jul 2015 NSL+EWL breakdown")
tests["ewwest_2025"] = event(2025, "ewwest", "GE2025, seven months after the Sep 2024 Boon Lay-Queenstown EWL disruption")

# robustness: add share of brand-new homes and whether the main challenger was WP
nh = np.array([100 * o["new_homes"] for o in swing])
wp = np.array([float(res[o["year"]][o["name"]]["parties"][1][0] == "WP" if len(res[o["year"]][o["name"]]["parties"]) > 1 else 0) for o in swing])
for key, x, nm in [("new_station_robust", newd, "new_station"), ("access_level_swing_robust", acc, "access_pp"),
                   ("access_change_robust", dacc, "d_access_pp")]:
    m = ols(yv, [x, prev, grc, nh, wp] + fe, [nm, "prev_pap", "grc", "new_homes_pp", "wp_challenger", "y2015", "y2020", "y2025"])
    m["perm_p"] = perm_p(yv, x, yr, [prev, grc, nh, wp] + fe); tests[key] = m

for k, m in tests.items():
    log(f"\n== {k}  n={m['n']}  R2={m['r2']:.2f}  permutation p (key var)={m['perm_p']:.3f}")
    for nm, c in m["coef"].items(): log(f"   {nm:14s} b={c['b']:+.3f}  se={c['se']:.3f}  p={c['p']:.3f}")

# ---------- write page data
out = dict(national=national, tests=tests,
           eds={str(y): {n: dict(parties=res[y][n]["parties"], walkover=res[y][n]["walkover"], total=res[y][n]["total"],
                                 pap=res[y][n]["pap"]) for n in res[y]} for y in YEARS},
           features=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()} for o in obs],
           swing_points=[dict(year=o["year"], name=o["name"], swing=round(100 * (o["pap"] - o["prev_pap"]), 2),
                              d_access=round(100 * o["d_access"], 1), access=round(100 * o["access"], 1),
                              new=o["new_stations"]) for o in swing])
json.dump(out, open(OUT, "w"), separators=(",", ":"))
open(OUT.replace(".json", ".txt"), "w").write("\n".join(lines) + "\n")
print("wrote", OUT)
