from atomic_configuration_analysis import AtomicConfigurationAnalysis as ACA
import pprint
import numpy as np

poscar = "/jet/home/mohammes/projects/atomic_site_analysis/POSCAR_structures/POSCAR-R1"

analysis = ACA()
analysis.parse_poscar(poscar)
analysis.generate_cells()
analysis.bond_lengths_for_type('Te')
analysis.bond_angles_for_type('Te')
analysis.analyze_neighbor_ratios('Te')

_2Ag4SbAdj = '2Ag-4Sb_Adjacent'
_2Ag4SbOpp = '2Ag-4Sb_Opposite'
_6Ag0Sb = '6Ag-0Sb'
_0Ag6Sb = '0Ag-6Sb'
_1Ag5Sb = '1Ag-5Sb'
_5Ag1Sb = '5Ag-1Sb'
_4Ag2SbAdj = '4Ag-2Sb_Adjacent'
_4Ag2SbOpp = '4Ag-2Sb_Opposite'
_3Ag3SbClu = '3Ag-3Sb_Cluster'
_3Ag3SbLin = '3Ag-3Sb_Line'

classifications = {
    "2Ag4SbAdj": _2Ag4SbAdj,
    "2Ag4SbOpp": _2Ag4SbOpp,
    "6Ag0Sb": _6Ag0Sb,
    "0Ag6Sb": _0Ag6Sb,
    "1Ag5Sb": _1Ag5Sb,
    "5Ag1Sb": _5Ag1Sb,
    "4Ag2SbAdj": _4Ag2SbAdj,
    "4Ag2SbOpp": _4Ag2SbOpp,
    "3Ag3SbClu": _3Ag3SbClu,
    "3Ag3SbLin": _3Ag3SbLin,
}

for name, value in classifications.items():
    globals()[f"atoms_{name}"] = [
        atom_key
        for atom_key, classification in analysis.environment_class_analysis.items()
        if classification == value
    ]

print(f"\nSummary of classifications:")
print(f"    2Ag4Sb-Adjacent: {len(atoms_2Ag4SbAdj)}")
print(f"    1Ag5Sb: {len(atoms_1Ag5Sb)}")
print(f"    3Ag3Sb-Line: {len(atoms_3Ag3SbLin)}")
print(f"    2Ag4Sb-Opposite: {len(atoms_2Ag4SbOpp)}")
print(f"    3Ag3Sb-Cluster: {len(atoms_3Ag3SbClu)}")
print(f"    6Ag: {len(atoms_6Ag0Sb)}")
print(f"    4Ag2Sb-Adjacent:{len(atoms_4Ag2SbAdj)}")
print(f"    5Ag1Sb: {len(atoms_5Ag1Sb)}")
print(f"    4Ag2Sb-Opposite: {len(atoms_4Ag2SbOpp)}")
print(f"    6Sb: {len(atoms_0Ag6Sb)}")

te_count = analysis.atom_counts_dict['Te']
print(f"\nTotal number of Te-centered octahedra analyzed: '{te_count}'")

for name, value in classifications.items():
    globals()[f"bl_{name}_Ag"] = [
        neighbor['bond_length']
        for atom_key in globals()[f"atoms_{name}"]
        for neighbor in analysis.bond_length_analysis[atom_key]
        if neighbor['type'] == 'Ag'
    ]
    globals()[f"bl_{name}_Sb"] = [
        neighbor['bond_length']
        for atom_key in globals()[f"atoms_{name}"]
        for neighbor in analysis.bond_length_analysis[atom_key]
        if neighbor['type'] == 'Sb'
    ]
    globals()[f"bl_{name}_Ag_mean"] = np.mean(globals()[f"bl_{name}_Ag"]) if globals()[f"bl_{name}_Ag"] else None
    globals()[f"bl_{name}_Sb_mean"] = np.mean(globals()[f"bl_{name}_Sb"]) if globals()[f"bl_{name}_Sb"] else None
    globals()[f"bl_{name}_Ag_std"] = np.std(globals()[f"bl_{name}_Ag"]) if globals()[f"bl_{name}_Ag"] else None
    globals()[f"bl_{name}_Sb_std"] = np.std(globals()[f"bl_{name}_Sb"]) if globals()[f"bl_{name}_Sb"] else None
    
print(f"\nSummary of classifications:")
print(f"    1Ag5Sb - Ag: mean = '{bl_1Ag5Sb_Ag_mean:.4f}'A, std = '{bl_1Ag5Sb_Ag_std:.4f}'A")
print(f"    1Ag5Sb - Sb: mean = '{bl_1Ag5Sb_Sb_mean:.4f}'A, std = '{bl_1Ag5Sb_Sb_std:.4f}'A")
print(f"    2Ag4Sb-Adjacent - Ag: mean = '{bl_2Ag4SbAdj_Ag_mean:.4f}'A, std = '{bl_2Ag4SbAdj_Ag_std:.4f}'A")
print(f"    2Ag4Sb-Adjacent - Sb: mean = '{bl_2Ag4SbAdj_Sb_mean:.4f}'A, std = '{bl_2Ag4SbAdj_Sb_std:.4f}'A")
print(f"    2Ag4Sb-Opposite - Ag: mean = '{bl_2Ag4SbOpp_Ag_mean:.4f}'A, std = '{bl_2Ag4SbOpp_Ag_std:.4f}'A")
print(f"    2Ag4Sb-Opposite - Sb: mean = '{bl_2Ag4SbOpp_Sb_mean:.4f}'A, std = '{bl_2Ag4SbOpp_Sb_std:.4f}'A")
print(f"    3Ag3Sb-Cluster - Ag: mean = '{bl_3Ag3SbClu_Ag_mean:.4f}'A, std = '{bl_3Ag3SbClu_Ag_std:.4f}'A")
print(f"    3Ag3Sb-Cluster - Sb: mean = '{bl_3Ag3SbClu_Sb_mean:.4f}'A, std = '{bl_3Ag3SbClu_Sb_std:.4f}'A")
print(f"    3Ag3Sb-Line - Ag: mean = '{bl_3Ag3SbLin_Ag_mean:.4f}'A, std = '{bl_3Ag3SbLin_Ag_std:.4f}'A")
print(f"    3Ag3Sb-Line - Sb: mean = '{bl_3Ag3SbLin_Sb_mean:.4f}'A, std = '{bl_3Ag3SbLin_Sb_std:.4f}'A")
print(f"    4Ag2Sb-Adjacent - Ag: mean = '{bl_4Ag2SbAdj_Ag_mean:.4f}'A, std = '{bl_4Ag2SbAdj_Ag_std:.4f}'A")
print(f"    4Ag2Sb-Adjacent - Sb: mean = '{bl_4Ag2SbAdj_Sb_mean:.4f}'A, std = '{bl_4Ag2SbAdj_Sb_std:.4f}'A")
print(f"    4Ag2Sb-Opposite - Ag: mean = '{bl_4Ag2SbOpp_Ag_mean:.4f}'A, std = '{bl_4Ag2SbOpp_Ag_std:.4f}'A")
print(f"    4Ag2Sb-Opposite - Sb: mean = '{bl_4Ag2SbOpp_Sb_mean:.4f}'A, std = '{bl_4Ag2SbOpp_Sb_std:.4f}'A")
print(f"    5Ag1Sb - Ag: mean = '{bl_5Ag1Sb_Ag_mean:.4f}'A, std = '{bl_5Ag1Sb_Ag_std:.4f}'A")
print(f"    5Ag1Sb - Sb: mean = '{bl_5Ag1Sb_Sb_mean:.4f}'A, std = '{bl_5Ag1Sb_Sb_std:.4f}'A")
print(f"    6Ag - Ag: mean = '{bl_6Ag0Sb_Ag_mean:.4f}'A, std = '{bl_6Ag0Sb_Ag_std:.4f}'A")
print(f"    6Sb - Sb: mean = '{bl_0Ag6Sb_Sb_mean:.4f}'A, std = '{bl_0Ag6Sb_Sb_std:.4f}'A")

Ag_bonds = [
    neighbor['bond_length']
    for neighbor_list in analysis.bond_length_analysis.values()
    for neighbor in neighbor_list
    if neighbor['type'] == 'Ag'
]
Sb_bonds = [
    neighbor['bond_length']
    for neighbor_list in analysis.bond_length_analysis.values()
    for neighbor in neighbor_list
    if neighbor['type'] == 'Sb'
]
Ag_bonds_mean = np.mean(Ag_bonds)
Sb_bonds_mean = np.mean(Sb_bonds)
Ag_bonds_std = np.std(Ag_bonds)
Sb_bonds_std = np.std(Sb_bonds)

print(f"\nTotal bond length statistics:")
print(f"    Ag - : mean = '{Ag_bonds_mean:4f}'A, std = '{Ag_bonds_std:4f}'A")
print(f"    Sb - : mean = '{Sb_bonds_mean:4f}'A, std = '{Sb_bonds_std:4f}'A")
