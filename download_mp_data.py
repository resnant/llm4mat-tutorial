import datetime
import os
import pickle
from pathlib import Path

from mp_api.client import MPRester

data_path = Path(".")
download_path = data_path / "mp_download"
download_path.mkdir(parents=True, exist_ok=True)

print(f"download_path: {download_path}")

API_KEY_path = Path("api_keys/MP_API_KEY.txt")


print(f"API_KEY_path: {API_KEY_path}")

if not API_KEY_path.exists():
    print("Please put your API_KEY in the file API_KEY.txt")
    print("See: https://next-gen.materialsproject.org/api")
    exit(1)

with open(API_KEY_path, "r") as f:
    API_KEY = f.read().strip()

print(f"API_KEY: {API_KEY}")
mpr = MPRester(API_KEY, use_document_model=False)

entries = mpr.materials.summary._search(fields=["material_id"], nsites_min=-1, nsites_max=50)
# entries = mpr.materials.summary._search(fields=["material_id"], nsites_min=-1, nsites_max=1) # for debugging

# for debugging with small data
# entries = entries[0:200]

mpids = [e["material_id"] for e in entries]

all_properties = [
    "builder_meta",
    "nsites",
    "elements",
    "nelements",
    "composition",
    "composition_reduced",
    "formula_pretty",
    "formula_anonymous",
    "chemsys",
    "volume",
    "density",
    "density_atomic",
    "symmetry",
    "property_name",
    "material_id",
    "deprecated",
    "deprecation_reasons",
    "last_updated",
    "origins",
    "warnings",
    "structure",
    "task_ids",
    "uncorrected_energy_per_atom",
    "energy_per_atom",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "equilibrium_reaction_energy_per_atom",
    "decomposes_to",
    "xas",
    "grain_boundaries",
    "band_gap",
    "cbm",
    "vbm",
    "efermi",
    "is_gap_direct",
    "is_metal",
    "es_source_calc_id",
    "bandstructure",
    "dos",
    "dos_energy_up",
    "dos_energy_down",
    "is_magnetic",
    "ordering",
    "total_magnetization",
    "total_magnetization_normalized_vol",
    "total_magnetization_normalized_formula_units",
    "num_magnetic_sites",
    "num_unique_magnetic_sites",
    "types_of_magnetic_species",
    "k_voigt",
    "k_reuss",
    "k_vrh",
    "g_voigt",
    "g_reuss",
    "g_vrh",
    "universal_anisotropy",
    "homogeneous_poisson",
    "e_total",
    "e_ionic",
    "e_electronic",
    "n",
    "e_ij_max",
    "weighted_surface_energy_EV_PER_ANG2",
    "weighted_surface_energy",
    "weighted_work_function",
    "surface_anisotropy",
    "shape_factor",
    "has_reconstructed",
    "possible_species",
    "has_props",
    "theoretical",
    "database_IDs",
]

today = datetime.date.today()
today = today.strftime("%Y%m%d")
print(f"dataset date: {today}")

num_chunks = len(mpids) // 10000 + 1

for i in range(num_chunks):
    print(f"chunk {i+1}/{num_chunks}")
    this_mpids = mpids[i * 10000 : (i + 1) * 10000]
    this_mpids = ",".join(this_mpids)
    entries = mpr.materials.summary._search(
        material_ids=this_mpids, all_fields=True, chunk_size=100
    )
    # print(len(entries))
    with open(download_path / f"{today}_{i:04}.pkl", "wb") as f:
        pickle.dump(entries, f)

print("done")