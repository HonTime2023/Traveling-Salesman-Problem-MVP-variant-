# Traveling Sales Man Problem (MVP — Multi‑Vehicle Routing Problem)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-%20-%23150458?logo=pandas&logoColor=white)
![OR-Tools](https://img.shields.io/badge/OR--Tools-https-orange)
![GeoPandas](https://img.shields.io/badge/GeoPandas-optional-blue)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-black)

Overview
--------
This repository captures the original solution and artifacts for the Traveling Sales Man Problem (MVP) as implemented during the analysis phase. In this context "MVP" is used to denote the multi‑vehicle routing problem variant (multi‑agent routing), where the objective is to assign routes to multiple agents so that:

- All locations are visited exactly once.
- Each agent receives a route (sequence of locations) and returns to its depot if required.
- The total distance is minimized while keeping per‑agent workloads balanced.

Quick links
-----------
- Notebook: `TSP_Problem.ipynb`
- Original data: `TSP1.xlsm` (includes a sheet listing the 41 agents)
- Summary: `Solution Approach Summary.pdf` / `Solution Approach Summary.docx`
- Final routes map (embedded):

![Map of Final Routes](Map_of_final_routes.png)

Problem & data
--------------
The primary dataset contains location coordinates (latitude, longitude). A separate sheet lists 41 agents. The assignment goal is to allocate locations to the 41 agents and produce ordered routes per agent.

Solution summary
----------------
Two complementary strategies were explored in the notebook:

1) Balanced partition + per‑cluster TSP

- Partition the entire location set into 41 spatially coherent clusters (balanced by node count or estimated workload).
- Solve an independent TSP within each cluster (local improvement: 2‑opt, 3‑opt; optionally LKH for higher quality).
- Pros: parallel solves and fast. Cons: cluster boundaries may cause suboptimal global routing.

2) Single‑stage Vehicle Routing Problem (VRP)

- Model the problem as a multi‑vehicle VRP with 41 vehicles using Google OR‑Tools RoutingModel.
- Use a geodesic distance callback (Haversine or GeographicLib) for accurate earth‑curved distances.
- Add soft constraints or objective penalties to balance route lengths (minimize variance or minimize the maximum route length).
- Pros: captures global interactions and supports constraints (capacities, time windows). Cons: higher compute and solver tuning.

Distance computation (why Haversine)
----------------------------------
Euclidean straight lines on latitude/longitude are incorrect for geographic distances. For reasonably sized geographic extents the Haversine great‑circle formula gives a simple, efficient, and accurate distance model. The Haversine distance between two points with latitudes $\varphi_1,\varphi_2$ and longitudes $\lambda_1,\lambda_2$ (in radians) using Earth radius $R$ is:

$$
\Delta\varphi = \varphi_2 - \varphi_1, \quad \Delta\lambda = \lambda_2 - \lambda_1
$$

$$
a = \sin^2\left(\frac{\Delta\varphi}{2}\right) + \cos(\varphi_1)\cos(\varphi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)
$$

$$
d = 2R\,\arcsin(\sqrt{a})
$$

This calculation is compact, numerically stable for small distances, and widely used in routing prototypes. For maximum geodetic accuracy (ellipsoidal Earth), use `pyproj.Geod` or `GeographicLib`.

Why not straight lines?
- Lat/lon are angular coordinates; straight Euclidean distance in lat/lon space distorts distances (especially at higher latitudes).

Tech stack (high level)
-----------------------
- Python 3.10+ — core implementation language.
- pandas, numpy — data handling and numeric work.
- geopandas, shapely, pyproj — geospatial helpers and geometry processing.
- geopy / geographiclib — geodesic distance tools (Haversine or more accurate options).
- OR‑Tools — routing solver (VRP) and local search.
- matplotlib / plotly — static and interactive plotting.
- Mapbox GL JS or Leaflet (for web mapping visualizations).
- Tesseract (`pytesseract`) and/or Google Cloud Vision — OCR tools for data ingestion when needed.

Reproducibility & setup
-----------------------
Suggested Python environment commands:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Minimal `requirements.txt` (examples)

```
pandas
numpy
geopandas
pyproj
geopy
ortools
matplotlib
plotly
pytesseract
jupyterlab
```

Notes on OCR
-----------
- Use `pytesseract` (local) for scanned images and `google-cloud-vision` for higher accuracy or large‑scale OCR jobs.
- Keep raw OCR outputs and a processing log (CSV) to support traceability.

Visualization & exports
-----------------------
- The notebook exports per‑agent route summaries (CSV) and a GeoJSON of the final routes.
- Maps are rendered using geodesic polylines; where the map provider supports it, use native great‑circle arcs or interpolate intermediate points between lat/lon pairs to create smooth curved lines.

Evaluation metrics
------------------
- Total distance (sum of all agent routes).
- Per‑agent distance and node count.
- Fairness: max route length, mean and standard deviation, and max/min ratio.
- Runtime and solver status.

Files in this folder
--------------------
- `TSP_Problem.ipynb` — full analysis and experiments.
- `TSP1.xlsm` — original Excel dataset (locations + agents).
- `Solution Approach Summary.pdf` / `.docx` — design notes.
- `Map_of_final_routes.png` — exported map image used above.

License & contact
-----------------
Add a license file (e.g., `LICENSE` with MIT) if you intend to publish. For help pushing this repository to GitHub or scaffolding the web app, open an issue or contact the author.

Acknowledgements
----------------
Workflows and recommendations in this README are based on the analysis in `TSP_Problem.ipynb` and the `Solution Approach Summary` documents found in this folder.
