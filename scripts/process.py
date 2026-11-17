"""Process ELD electoral boundary GeoJSONs (2006-2025) into compact SVG paths,
a rasterised change analysis, and stats, written to out/data.json."""
import json, re, math, sys
import numpy as np
from matplotlib.path import Path
from scipy import ndimage

DATA = sys.argv[1]
OUT = sys.argv[2]

FILES = {
    2006: "ElectoralBoundary2006.geojson", 2011: "ElectoralBoundary2011.geojson", 2015: "ElectoralBoundary2015.geojson",
    2020: "ElectoralBoundary2020.geojson", 2025: "ElectoralBoundary2025.geojson",
}
POLL = {2006: "6 May 2006", 2011: "7 May 2011", 2015: "11 Sep 2015", 2020: "10 Jul 2020", 2025: "3 May 2025"}

# Seats per constituency (Wikipedia lists of electoral divisions per term); SMC = 1
GRC = {
 2006: {"ALJUNIED":5,"ANG MO KIO":6,"BISHAN-TOA PAYOH":5,"EAST COAST":5,"HOLLAND-BUKIT TIMAH":5,"HONG KAH":5,
        "JALAN BESAR":5,"JURONG":5,"MARINE PARADE":6,"PASIR RIS-PUNGGOL":6,"SEMBAWANG":6,"TAMPINES":5,
        "TANJONG PAGAR":6,"WEST COAST":5},
 2011: {"ALJUNIED":5,"ANG MO KIO":6,"BISHAN-TOA PAYOH":5,"CHUA CHU KANG":5,"EAST COAST":5,"HOLLAND-BUKIT TIMAH":4,
        "JURONG":5,"MARINE PARADE":5,"MOULMEIN-KALLANG":4,"NEE SOON":5,"PASIR RIS-PUNGGOL":6,"SEMBAWANG":5,
        "TAMPINES":5,"TANJONG PAGAR":5,"WEST COAST":5},
 2015: {"ALJUNIED":5,"ANG MO KIO":6,"BISHAN-TOA PAYOH":5,"CHUA CHU KANG":4,"EAST COAST":4,"HOLLAND-BUKIT TIMAH":4,
        "JALAN BESAR":4,"JURONG":5,"MARINE PARADE":5,"MARSILING-YEW TEE":4,"NEE SOON":5,"PASIR RIS-PUNGGOL":6,
        "SEMBAWANG":5,"TAMPINES":5,"TANJONG PAGAR":5,"WEST COAST":4},
 2020: {"ALJUNIED":5,"ANG MO KIO":5,"BISHAN-TOA PAYOH":4,"CHUA CHU KANG":4,"EAST COAST":5,"HOLLAND-BUKIT TIMAH":4,
        "JALAN BESAR":4,"JURONG":5,"MARINE PARADE":5,"MARSILING-YEW TEE":4,"NEE SOON":5,"PASIR RIS-PUNGGOL":5,
        "SEMBAWANG":5,"SENGKANG":4,"TAMPINES":5,"TANJONG PAGAR":5,"WEST COAST":5},
 2025: {"ALJUNIED":5,"ANG MO KIO":5,"BISHAN-TOA PAYOH":4,"CHUA CHU KANG":4,"EAST COAST":5,"HOLLAND-BUKIT TIMAH":4,
        "JALAN BESAR":4,"JURONG EAST-BUKIT BATOK":5,"MARINE PARADE-BRADDELL HEIGHTS":5,"MARSILING-YEW TEE":4,
        "NEE SOON":5,"PASIR RIS-CHANGI":4,"PUNGGOL":4,"SEMBAWANG":5,"SENGKANG":4,"TAMPINES":5,"TANJONG PAGAR":5,
        "WEST COAST-JURONG WEST":5},
}
TOTAL = {2006: (84, 14, 9), 2011: (87, 15, 12), 2015: (89, 16, 13), 2020: (93, 17, 14), 2025: (97, 18, 15)}

LON0, LAT0 = 103.60, 1.475
KX = 111320 * math.cos(math.radians(1.35))
KY = 110574
UNIT = 10.0  # svg unit = 10 m


def proj(lon, lat):
    return ((lon - LON0) * KX / UNIT, (LAT0 - lat) * KY / UNIT)


def norm(n):
    n = re.sub(r"\s*-\s*", "-", n.strip().upper())
    return re.sub(r"\s+(GRC|SMC)$", "", n)


def name_of(year, props):
    if "ED_DESC" in props:
        return norm(props["ED_DESC"])
    t = dict(re.findall(r"<th>(\w+)</th>\s*<td>([^<]*)</td>", props["Description"]))
    return norm(t["ED_DESC"])


def polys_of(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"]]
    return geom["coordinates"]


def dp(pts, tol):
    """Douglas-Peucker on an (n,2) array."""
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, q = pts[a], pts[b]
        seg = pts[a + 1:b]
        d = q - p
        L = math.hypot(*d)
        if L == 0:
            dist = np.hypot(*(seg - p).T)
        else:
            dist = np.abs(d[0] * (seg[:, 1] - p[1]) - d[1] * (seg[:, 0] - p[0])) / L
        i = int(np.argmax(dist))
        if dist[i] > tol:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return pts[keep]


def ring_area(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# --- load
years = sorted(FILES)
eds = {}  # year -> list of dict(name, rings[list of arrays], type, seats)
for y in years:
    gj = json.load(open(f"{DATA}/{FILES[y]}"))
    merged = {}
    for f in gj["features"]:
        n = name_of(y, f["properties"])
        rings = merged.setdefault(n, [])
        for poly in polys_of(f["geometry"]):
            for ri, ring in enumerate(poly):
                arr = np.array([proj(c[0], c[1]) for c in ring])
                rings.append(arr)
    lst = []
    for n, rings in merged.items():
        seats = GRC[y].get(n, 1)
        lst.append(dict(name=n, rings=rings, kind="GRC" if n in GRC[y] else "SMC", seats=seats))
    assert all(g in merged for g in GRC[y]), (y, set(GRC[y]) - set(merged))
    s = sum(e["seats"] for e in lst)
    ng = sum(e["kind"] == "GRC" for e in lst)
    assert (s, ng, len(lst) - ng) == TOTAL[y], (y, s, ng, len(lst) - ng)
    eds[y] = lst
    print(y, len(lst), "constituencies,", s, "seats OK")

# --- raster grid (100 m cells)
allpts = np.vstack([r for y in years for e in eds[y] for r in e["rings"]])
xmin, ymin = allpts.min(0)
xmax, ymax = allpts.max(0)
CELL = 10.0  # svg units = 100 m
W = int(math.ceil((xmax) / CELL)) + 1
H = int(math.ceil((ymax) / CELL)) + 1
cx = (np.arange(W) + 0.5) * CELL
cy = (np.arange(H) + 0.5) * CELL
GX, GY = np.meshgrid(cx, cy)
pts = np.column_stack([GX.ravel(), GY.ravel()])

lab = {}
for y in years:
    grid = np.full(H * W, -1, int)
    for i, e in enumerate(eds[y]):
        inside = np.zeros(H * W, bool)
        for r in e["rings"]:
            inside ^= Path(r).contains_points(pts)  # even-odd handles holes
        grid[inside] = i
    lab[y] = grid.reshape(H, W)

# land mask = 2025 coverage (2025 file is land-only; older files include territorial sea)
land = lab[2025] >= 0
print("grid", W, H, "land km2", land.sum() * 0.01)
for y in years:
    print(y, "coverage km2", (lab[y] >= 0).sum() * 0.01)

# --- change detection between consecutive elections
# A cell is 'redrawn' if the constituency it lands in is mainly built (by area) from a
# different predecessor than the one it came from. Pure renames/expansions keep identity.
trans = []
churn = np.zeros((H, W), int)
for a, b in zip(years, years[1:]):
    A, B = lab[a], lab[b]
    valid = land & (A >= 0) & (B >= 0)
    pred = {}
    for j in range(len(eds[b])):
        m = valid & (B == j)
        vals, cnt = np.unique(A[m], return_counts=True)
        pred[j] = int(vals[np.argmax(cnt)]) if len(vals) else -1
    succ = {}
    for i in range(len(eds[a])):
        m = valid & (A == i)
        vals, cnt = np.unique(B[m], return_counts=True)
        succ[i] = int(vals[np.argmax(cnt)]) if len(vals) else -1
    predmap = np.vectorize(lambda j: pred.get(j, -1))(np.where(B >= 0, B, 0))
    succmap = np.vectorize(lambda i: succ.get(i, -1))(np.where(A >= 0, A, 0))
    # identity kept only if A's main successor is B AND B's main predecessor is A
    changed = valid & ~((predmap == A) & (succmap == B))
    churn += changed
    # per-new-constituency lineage summary
    lineage = []
    for j, e in enumerate(eds[b]):
        m = valid & (B == j)
        tot = m.sum()
        vals, cnt = np.unique(A[m], return_counts=True)
        order = np.argsort(-cnt)
        parts = [(eds[a][int(vals[k])]["name"], round(float(cnt[k] / tot), 3)) for k in order if cnt[k] / tot >= 0.03]
        lineage.append(dict(name=e["name"], from_=parts,
                            new=(len(parts) == 0) or eds[a][pred[j]]["name"] != e["name"]))
    oldnames = {e["name"] for e in eds[a]}
    newnames = {e["name"] for e in eds[b]}
    trans.append(dict(frm=a, to=b, changed=changed, share=float(changed.sum() / valid.sum()),
                      km2=float(changed.sum() * 0.01), lineage=lineage,
                      gone=sorted(oldnames - newnames), born=sorted(newnames - oldnames)))
    print(f"{a}->{b}: {changed.sum()*0.01:.0f} km2 redrawn ({100*changed.sum()/valid.sum():.1f}% of land)",
          "gone", sorted(oldnames - newnames), "born", sorted(newnames - oldnames))


def mask_to_path(mask):
    """Run-length rectangles -> compact SVG path (merging identical row runs vertically)."""
    rects = []
    open_runs = {}
    for r in range(H):
        row = mask[r]
        runs = []
        c = 0
        while c < W:
            if row[c]:
                s = c
                while c < W and row[c]:
                    c += 1
                runs.append((s, c))
            else:
                c += 1
        nxt = {}
        for run in runs:
            if run in open_runs:
                nxt[run] = open_runs.pop(run)
            else:
                nxt[run] = r
        for run, r0 in open_runs.items():
            rects.append((run[0], r0, run[1], r))
        open_runs = nxt
    for run, r0 in open_runs.items():
        rects.append((run[0], r0, run[1], H))
    C = int(CELL)
    return "".join(f"M{x0*C} {y0*C}h{(x1-x0)*C}v{(y1-y0)*C}h{-(x1-x0)*C}z" for x0, y0, x1, y1 in rects)


# --- label anchors: point deepest inside each constituency (on land)
def anchor(grid, i):
    m = (grid == i) & land
    if not m.any():
        m = grid == i
    d = ndimage.distance_transform_edt(m)
    r, c = np.unravel_index(np.argmax(d), d.shape)
    return [round(float(cx[c]), 1), round(float(cy[r]), 1), round(float(d[r, c] * CELL), 1)]


def ring_path(r, tol):
    s = dp(r, tol)
    if len(s) < 4 or abs(ring_area(s)) < 30:  # drop specks < ~3000 m2
        return ""
    s = np.round(s).astype(int)
    out = f"M{s[0][0]} {s[0][1]}"
    prev = s[0]
    for p in s[1:]:
        d = p - prev
        if d[0] or d[1]:
            out += f"l{d[0]} {d[1]}"
        prev = p
    return out + "z"


out = dict(meta=dict(W=W * CELL, H=H * CELL, unit_m=UNIT), years=[])
for y in years:
    lst = []
    for i, e in enumerate(eds[y]):
        path = "".join(ring_path(r, 1.5) for r in e["rings"])
        km2 = float(((lab[y] == i) & land).sum() * 0.01)
        lst.append(dict(name=e["name"], kind=e["kind"], seats=e["seats"], d=path, at=anchor(lab[y], i), km2=round(km2, 1)))
    lst.sort(key=lambda e: e["name"])
    n_grc = sum(e["kind"] == "GRC" for e in lst)
    out["years"].append(dict(year=y, poll=POLL[y], seats=TOTAL[y][0], grc=n_grc, smc=len(lst) - n_grc, eds=lst))

out["transitions"] = [dict(frm=t["frm"], to=t["to"], share=round(t["share"], 4), km2=round(t["km2"], 1),
                           gone=t["gone"], born=t["born"], d=mask_to_path(t["changed"]),
                           lineage=[dict(name=l["name"], parts=l["from_"], new=bool(l["new"])) for l in t["lineage"]])
                      for t in trans]
out["churn"] = {str(k): mask_to_path(churn == k) for k in range(1, 5)}
out["churn_share"] = {str(k): round(float(((churn == k) & land).sum() / land.sum()), 4) for k in range(0, 5)}
out["land"] = mask_to_path(land)
json.dump(out, open(OUT, "w"), separators=(",", ":"))
import os
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB", "churn", out["churn_share"])
