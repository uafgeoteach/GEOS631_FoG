"""Make the synthetic station data set for Lab 7b (figure of the Earth from surface gravity).

g_obs = gamma_GRS80(phi) [Somigliana] - 0.3086 h [free air] + dg_FA(EGM96, deg 2-360, at sea level) + noise
Stations: random, uniform in area; land stations get a random height, ocean stations are at sea level.
Run: python make_data.py   (needs pyshtools, and egm96_to360.ascii, ne_coast.geojson, gravlib.py from 2026_lectures/L2_beamer/common in this folder)
"""
import json
import numpy as np
import pyshtools as pysh
from matplotlib.path import Path
from gravlib import read_egm96, remove_reference, A_E, G0

rng = np.random.default_rng(631)
N = 800

# GRS80 Somigliana
GE, K, E2 = 978032.67715, 0.001931851353, 0.00669438002290  # mGal
def gamma(phi):
    s2 = np.sin(np.radians(phi))**2
    return GE * (1 + K * s2) / np.sqrt(1 - E2 * s2)

# random points uniform on the sphere
lat = np.degrees(np.arcsin(rng.uniform(-1, 1, N)))
lon = rng.uniform(-180, 180, N)

# land mask: closed coastline rings, plus the three big land masses stitched from open segments
gj = json.load(open("ne_coast.geojson"))
rings, segs = [], []
for f in gj["features"]:
    g = f["geometry"]
    for p in (g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]):
        p = np.array(p)
        (rings if np.allclose(p[0], p[-1]) else segs).append(p)
rings = [Path(r) for r in rings if len(r) > 20]
def seg(start, end):
    for p in segs:
        if np.allclose(p[0], start, atol=0.1) and np.allclose(p[-1], end, atol=0.1):
            return p
    raise ValueError((start, end))
americas = np.vstack([seg((-77.9, 7.2), (-82.5, 9.6)), seg((-82.5, 9.6), (-77.4, 8.7)),
                      seg((-77.4, 8.7), (-74.9, -52.3)), seg((-77.9, 7.2), (-74.9, -52.3))[::-1]])
afroeurasia = np.vstack([seg((-16.3, 19.1), (180, 69.0)), seg((180, 65.0), (-16.3, 19.1))])
antarctica = np.vstack([seg((-180, -84.7), (180, -84.7)), [[180, -90], [-180, -90]]])
rings += [Path(americas), Path(afroeurasia), Path(antarctica)]
pts = np.column_stack([lon, lat])
land = np.zeros(N, bool)
for r in rings:
    land |= r.contains_points(pts)

# heights: land stations from a gamma distribution, ocean stations at sea level
h = np.where(land, np.clip(rng.gamma(1.5, 350, N), 0, 4500), 0.0)

# EGM96 free-air anomaly at sea level, degrees 2-360 relative to the WGS84/GRS80 normal field
C, S = read_egm96()
C = remove_reference(C)
C[1, :] = 0; S[1, :] = 0
n = np.arange(361)[:, None]
arr = np.zeros((2, 361, 361)); arr[0], arr[1] = C, S
arr = arr * (n - 1) * G0 * 1e5   # mGal
arr[:, :2, :] = 0
cl = pysh.SHCoeffs.from_array(arr, normalization="4pi", csphase=1)
dg = np.array([cl.expand(lat=la, lon=lo) for la, lo in zip(lat, lon)])

noise = rng.normal(0, 0.05, N)
g_obs = gamma(lat) - 0.3086 * h + dg + noise

np.savez("gravity_stations.npz", lat=np.round(lat, 4), lon=np.round(lon, 4), h=np.round(h, 1),
         g_obs=np.round(g_obs, 3), land=land)
print(N, "stations,", land.sum(), "on land; dg rms %.1f mGal, range %.0f to %.0f" % (dg.std(), dg.min(), dg.max()))
