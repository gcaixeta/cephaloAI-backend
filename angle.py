import math
import numpy as np

# Landmark indices — verify against dataset documentation
SELLA = 0        # S
NASION = 1       # N
ANS = 2          # Anterior Nasal Spine
PNS = 3          # Posterior Nasal Spine
A_POINT = 4      # A (Subspinale)
B_POINT = 5      # B (Supramentale)
POGONION = 6     # Pg
MENTON = 7       # Me
GONION_ANT = 8   # Gonion (mandibular plane anterior reference)
GONION = 9       # Go
U1_TIP = 10      # Upper incisor tip (MW)
L1_TIP = 11      # Lower incisor tip (MW)
# indices 12–15 not used in current measurements
FH_ANT = 16      # Frankfort Horizontal anterior reference
FH_POST = 17     # Frankfort Horizontal posterior reference

# Classification thresholds (clinical reference ranges; normal = class "1")
THRESHOLDS = {
    "ANB":  {"low": 3.2,  "high": 5.7,  "below": "3", "above": "2"},
    "SNB":  {"low": 74.6, "high": 78.7, "below": "2", "above": "3"},
    "SNA":  {"low": 79.4, "high": 83.2, "below": "3", "above": "2"},
    "ODI":  {"low": 68.4, "high": 80.5, "below": "3", "above": "2"},
    "APDI": {"low": 77.6, "high": 85.2, "below": "2", "above": "3"},
    "FHI":  {"low": 0.65, "high": 0.75, "below": "3", "above": "2"},
    "FMA":  {"low": 26.8, "high": 31.4, "below": "3", "above": "2"},
}


def _classify(value, key):
    t = THRESHOLDS[key]
    if value < t["low"]:
        return t["below"]
    elif value > t["high"]:
        return t["above"]
    return "1"


class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return f"{self.x},{self.y}"


class Vector:
    def __init__(self, pa: Point, pb: Point):
        self.x = float(pb.x) - float(pa.x)
        self.y = float(pb.y) - float(pa.y)

    def __str__(self):
        return f"{self.x},{self.y}"

    def norm(self):
        return math.hypot(self.x, self.y)


class Angle:
    def __init__(self, va: Vector, vb: Vector):
        self.va = va
        self.vb = vb

    def theta(self):
        norm_a = self.va.norm()
        norm_b = self.vb.norm()

        if norm_a == 0 or norm_b == 0:
            return 0.0

        dot = self.va.x * self.vb.x + self.va.y * self.vb.y
        cos_theta = dot / (norm_a * norm_b)
        cos_theta = max(-1.0, min(1.0, cos_theta))

        return math.degrees(math.acos(cos_theta))


class Distance:
    def __init__(self, pa: Point, pb: Point):
        dx = float(pb.x) - float(pa.x)
        dy = float(pb.y) - float(pa.y)
        self.value = math.hypot(dx, dy)

    def dist(self):
        return self.value


def getCross(va, vb):
    return va.x * vb.y - va.y * vb.x


def getODI(pa, pb, pc, pd, pe, pf, pg, ph):
    va = Vector(pa, pb)
    vb = Vector(pc, pd)
    vc = Vector(pe, pf)
    vd = Vector(pg, ph)

    aa = Angle(va, vb).theta()
    ab = Angle(vc, vd).theta()
    cb = getCross(vc, vd)

    if cb < 0:
        ab = -ab

    return aa + ab


def getAPDI(pa, pb, pc, pd, pe, pf, pg, ph, pi, pj):
    va = Vector(pa, pb)
    vb = Vector(pc, pd)
    vc = Vector(pe, pf)
    vd = Vector(pg, ph)
    ve = Vector(pi, pj)

    aa = Angle(va, vb).theta()
    ab = Angle(vb, vc).theta()
    ac = Angle(vd, ve).theta()

    cb = getCross(vb, vc)
    cc = getCross(vd, ve)

    if cb > 0:
        ab = -ab
    if cc < 0:
        ac = -ac

    return aa + ab + ac


def classification(points):
    results = {}

    # --- ANB ---
    va = Vector(points[NASION], points[SELLA])
    vb = Vector(points[NASION], points[B_POINT])
    vc = Vector(points[NASION], points[SELLA])
    vd = Vector(points[NASION], points[A_POINT])

    ANB = Angle(vc, vd).theta() - Angle(va, vb).theta()
    results["ANB"] = {"value": ANB, "class": _classify(ANB, "ANB")}

    # --- SNB ---
    va = Vector(points[NASION], points[SELLA])
    vb = Vector(points[NASION], points[B_POINT])
    SNB = Angle(va, vb).theta()
    results["SNB"] = {"value": SNB, "class": _classify(SNB, "SNB")}

    # --- SNA ---
    va = Vector(points[NASION], points[SELLA])
    vb = Vector(points[NASION], points[A_POINT])
    SNA = Angle(va, vb).theta()
    results["SNA"] = {"value": SNA, "class": _classify(SNA, "SNA")}

    # --- ODI ---
    ODI = getODI(
        points[MENTON],
        points[GONION],
        points[B_POINT],
        points[A_POINT],
        points[PNS],
        points[ANS],
        points[FH_ANT],
        points[FH_POST],
    )
    results["ODI"] = {"value": ODI, "class": _classify(ODI, "ODI")}

    # --- APDI ---
    APDI = getAPDI(
        points[ANS],
        points[PNS],
        points[NASION],
        points[POGONION],
        points[A_POINT],
        points[B_POINT],
        points[PNS],
        points[ANS],
        points[FH_ANT],
        points[FH_POST],
    )
    results["APDI"] = {"value": APDI, "class": _classify(APDI, "APDI")}

    # --- FHI (Face Height Index = PFH / AFH) ---
    pfh = Distance(points[SELLA], points[GONION]).dist()
    afh = Distance(points[NASION], points[MENTON]).dist()
    ratio = pfh / afh if afh != 0 else 0
    results["FHI"] = {"value": ratio, "class": _classify(ratio, "FHI")}

    # --- FMA ---
    va = Vector(points[SELLA], points[NASION])
    vb = Vector(points[GONION], points[GONION_ANT])
    FMA = Angle(va, vb).theta()
    results["FMA"] = {"value": FMA, "class": _classify(FMA, "FMA")}

    # --- MW (Maxillary Width) ---
    mw = Distance(points[U1_TIP], points[L1_TIP]).dist() / 10
    if points[L1_TIP].x < points[U1_TIP].x:
        mw = -mw
    if mw >= 2:
        if mw <= 4.5:
            mwtype = "1"
        else:
            mwtype = "4"
    elif mw == 0:
        mwtype = "2"
    else:
        mwtype = "3"
    results["MW"] = {"value": mw, "class": mwtype}

    return results
