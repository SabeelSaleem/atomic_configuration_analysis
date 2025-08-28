import numpy as np
import itertools
from collections import Counter

class AtomicConfigurationAnalysis:
    """
    An atomic structure analysis tool for VASP POSCAR files, designed to analyze
    bond lengths, bond angles, and classify local atomic environments.
    """
    def __init__(self, filepath=None):
        """
        Initializes an empty object to store structural and analysis data.
        """
        self.filepath = filepath
        self.title = None
        self.scaling_factor = None
        self.lattice_matrix = None
        self.atom_types = None
        self.atom_counts = None
        self.atom_counts_dict = None
        self.atomic_frac_positions = None
        self.cells = {}
        self.number_of_neighbors = None
        self.analysis_type = None
        self.bond_length_analysis = {}
        self.bond_angle_analysis = {}
        self.all_atoms_list = []
        self.neighbor_ratio_analysis = {}
        self.environment_class_analysis = {}
        self.octahedral_classifications = {
            "6Ag0Sb": '6Ag-0Sb',
            "5Ag1Sb": '5Ag-1Sb',
            "4Ag2SbAdj": '4Ag-2Sb_Adjacent',
            "4Ag2SbOpp": '4Ag-2Sb_Opposite',
            "3Ag3SbClu": '3Ag-3Sb_Cluster',
            "3Ag3SbLin": '3Ag-3Sb_Line',
            "2Ag4SbAdj": '2Ag-4Sb_Adjacent',
            "2Ag4SbOpp": '2Ag-4Sb_Opposite',
            "1Ag5Sb": '1Ag-5Sb',
            "0Ag6Sb": '0Ag-6Sb',
        }
        self.classifications = {}

    @staticmethod
    def atom(atom_type, index, cell='cell_0_0_0'):
        """A static helper method to create a standardized atom identifier dictionary."""
        return {'type': atom_type, 'index': index, 'cell': cell}

    def parse_poscar(self, filepath):
        """
        Parses a POSCAR file and populates the object's structural attributes.
        """
        self.filepath = filepath
        print(f"Parsing data from {self.filepath}...")
        with open(self.filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        self.title = lines[0]
        self.scaling_factor = float(lines[1])
        a_vec = np.array([float(x) for x in lines[2].split()])
        b_vec = np.array([float(x) for x in lines[3].split()])
        c_vec = np.array([float(x) for x in lines[4].split()])
        self.lattice_matrix = np.array([a_vec, b_vec, c_vec]) * self.scaling_factor
        self.atom_types = lines[5].split()
        self.atom_counts = list(map(int, lines[6].split()))
        self.atom_counts_dict = dict(zip(self.atom_types, self.atom_counts))
        
        coord_start_line = 7
        if lines[7].lower().startswith('s'):
            coord_start_line += 1
        positions_start_line = coord_start_line + 1
        total_atoms = sum(self.atom_counts)
        frac_positions_raw = lines[positions_start_line : positions_start_line + total_atoms]
        
        frac_positions_dict = {}
        index = 0
        for atom_type, count in zip(self.atom_types, self.atom_counts):
            temp_positions_list = []
            for _ in range(count):
                coords = list(map(float, frac_positions_raw[index].split()[:3]))
                temp_positions_list.append(coords)
                index += 1
            frac_positions_dict[atom_type] = np.array(temp_positions_list)
        self.atomic_frac_positions = frac_positions_dict
        print("Parsing complete.")

    def generate_cells(self):
        """
        Generates a 3x3x3 supercell and a master list of all atoms for efficient analysis.
        """
        print("Generating coordinates for 27 cells...")
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    cell_vector = np.array([i, j, k])
                    cell = f"cell_{i}_{j}_{k}"
                    current_cell_coords = {}
                    for atom_type, frac_coords in self.atomic_frac_positions.items():
                        translated_frac_coords = frac_coords + cell_vector
                        cart_coords = np.dot(translated_frac_coords, self.lattice_matrix)
                        current_cell_coords[atom_type] = cart_coords
                    self.cells[cell] = current_cell_coords
        
        print("Building master atom list for analysis...")
        temp_list = []
        for cell_key, cell_data in self.cells.items():
            for atom_type, coords_array in cell_data.items():
                for j, coords in enumerate(coords_array):
                    temp_list.append({"type": atom_type, "index": j, "cell": cell_key, "coords": coords})
        self.all_atoms_list = temp_list

        print("\n--- Analysis Setup Complete ---")
        print(f"Cells Generated Successfully from POSCAR: {self.title}")

    def bond_lengths_for_type(self, atom_type):
        """
        Calculates and stores the 6 nearest neighbors for all atoms of a given type.
        """
        if atom_type not in self.atom_counts_dict:
            print(f"Error: Atom type '{atom_type}' not found in structure.")
            return
        
        count = self.atom_counts_dict[atom_type]
        print(f"\n--- Running bond length analysis for all {count} '{atom_type}' atoms ---")
        for i in range(count):
            try:
                target_coords = self.cells['cell_0_0_0'][atom_type][i]
            except (KeyError, IndexError):
                print(f"Error: Cannot find {atom_type} with index {i} in the central cell. Skipping.")
                continue

            distances = []
            for atom in self.all_atoms_list:
                if atom['type'] == atom_type:
                    continue
                dist = np.linalg.norm(target_coords - atom['coords'])
                distances.append({"type": atom['type'], "index": atom['index'], "cell": atom['cell'], "bond_length": float(dist), "coords": atom['coords']})
            
            distances.sort(key=lambda x: x['bond_length'])
            analysis_key = f"{atom_type}_{i}"
            self.bond_length_analysis[analysis_key] = distances[:self.number_of_neighbors]
        print(f"Bond length analysis for {atom_type} has been stored.")

    def bond_angles_for_type(self, atom_type):
        """
        Calculates and stores all bond angles for the nearest neighbors of a given atom type.
        """
        if atom_type not in self.atom_counts_dict:
            print(f"Error: Atom type '{atom_type}' not found in structure.")
            return
            
        count = self.atom_counts_dict[atom_type]
        print(f"\n--- Running bond angle analysis for all {count} '{atom_type}' atoms ---")
        for i in range(count):
            analysis_key = f"{atom_type}_{i}"
            if analysis_key not in self.bond_length_analysis:
                print(f"Error: Bond lengths for {analysis_key} must be calculated first. Skipping.")
                continue

            central_atom_coords = self.cells['cell_0_0_0'][atom_type][i]
            closest_neighbors = self.bond_length_analysis[analysis_key]
            
            bond_angles = []
            for neighbor1, neighbor2 in itertools.combinations(closest_neighbors, 2):
                vector1 = neighbor1['coords'] - central_atom_coords
                vector2 = neighbor2['coords'] - central_atom_coords
                dot_product = np.dot(vector1, vector2)
                mag1 = neighbor1['bond_length']
                mag2 = neighbor2['bond_length']
                
                cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)
                
                bond_angles.append({
                    'neighbor_1': f"{neighbor1['type']}{neighbor1['index']}",
                    'central_atom': f"{atom_type}{i}",
                    'neighbor_2': f"{neighbor2['type']}{neighbor2['index']}",
                    'angle_degrees': float(angle_deg)
                })
            
            self.bond_angle_analysis[analysis_key] = bond_angles
        print(f"Bond angle analysis for {atom_type} has been stored.")

    def analyze_neighbor_ratios(self, atom_type):
        """
        Analyzes neighbor composition. If number_of_neighbors is 6, it also
        performs a detailed octahedral geometric classification for Ag/Sb environments.
        """
        if atom_type not in self.atom_counts_dict:
            print(f"Error: Atom type '{atom_type}' not found in structure.")
            return
            
        count = self.atom_counts_dict[atom_type]
        print(f"\n--- Running neighbor analysis for all {count} '{atom_type}' atoms ---")
        
        if self.analysis_type == 'octahedral':
            self.classifications = self.octahedral_classifications.copy()
        
        for i in range(count):
            analysis_key = f"{atom_type}_{i}"
            if analysis_key not in self.bond_length_analysis:
                print(f"Error: Bond lengths for {analysis_key} must be calculated first. Skipping.")
                continue

            central_coords = self.cells['cell_0_0_0'][atom_type][i]
            neighbors = self.bond_length_analysis[analysis_key]

            neighbor_types = [n['type'] for n in neighbors]
            self.neighbor_ratio_analysis[analysis_key] = dict(Counter(neighbor_types))
            
            def get_angle(v1, v2):
                mag1 = np.linalg.norm(v1)
                mag2 = np.linalg.norm(v2)
                if mag1 == 0 or mag2 == 0: return 0.0
                cos_angle = np.clip(np.dot(v1, v2) / (mag1 * mag2), -1.0, 1.0)
                return np.degrees(np.arccos(cos_angle))
            
            if self.analysis_type == 'octahedral':
                env_class = self._classify_octahedral_ag_sb(neighbors, central_coords)
                self.environment_class_analysis[analysis_key] = env_class
                
        print(f"Neighbor analysis for {atom_type} has been stored.")

    def get_bond_length(self, atom1, atom2):
        """
        Calculates the bond length between two specific atoms defined by identifier dictionaries.
        """
        try:
            coords1 = self.cells[atom1['cell']][atom1['type']][atom1['index']]
            coords2 = self.cells[atom2['cell']][atom2['type']][atom2['index']]
        except (KeyError, IndexError) as e:
            print(f"Error: Could not find one of the specified atoms. Details: {e}")
            return None
            
        distance = np.linalg.norm(coords1 - coords2)
        return float(distance)
    
    def get_bond_angle(self, central_atom, atom1, atom2):
        """
        Calculates a single bond angle between three specific atoms defined by identifier dictionaries.
        """
        try:
            central_coords = self.cells[central_atom['cell']][central_atom['type']][central_atom['index']]
            neighbor1_coords = self.cells[atom1['cell']][atom1['type']][atom1['index']]
            neighbor2_coords = self.cells[atom2['cell']][atom2['type']][atom2['index']]
        except (KeyError, IndexError) as e:
            print(f"Error: Could not find one of the specified atoms. Details: {e}")
            return None

        vector1 = neighbor1_coords - central_coords
        vector2 = neighbor2_coords - central_coords
        dot_product = np.dot(vector1, vector2)
        mag1 = np.linalg.norm(vector1)
        mag2 = np.linalg.norm(vector2)
        
        if mag1 == 0 or mag2 == 0:
            return 0.0

        cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        
        return float(np.degrees(angle_rad))

    """
    Internal Classification Code for Various Structures (Work In Progress)
    """
    def _classify_octahedral_ag_sb(self, neighbors, central_coords):
        """
        Helper function to classify an octahedral environment with Ag and Sb neighbors.
        """
        ag_neighbors = [n for n in neighbors if n['type'] == 'Ag']
        sb_neighbors = [n for n in neighbors if n['type'] == 'Sb']
        num_ag = len(ag_neighbors)
        if num_ag == 6: return '6Ag-0Sb'
        if num_ag == 5: return '5Ag-1Sb'
        if num_ag == 1: return '1Ag-5Sb'
        if num_ag == 0: return '0Ag-6Sb'
        def get_angle(v1, v2):
            mag1 = np.linalg.norm(v1)
            mag2 = np.linalg.norm(v2)
            if mag1 == 0 or mag2 == 0: return 0.0
            cos_angle = np.clip(np.dot(v1, v2) / (mag1 * mag2), -1.0, 1.0)
            return np.degrees(np.arccos(cos_angle))
        if num_ag == 2:
            v1 = ag_neighbors[0]['coords'] - central_coords
            v2 = ag_neighbors[1]['coords'] - central_coords
            angle = get_angle(v1, v2)
            return "2Ag-4Sb_Opposite" if angle > 150 else "2Ag-4Sb_Adjacent"
        if num_ag == 4:
            v1 = sb_neighbors[0]['coords'] - central_coords
            v2 = sb_neighbors[1]['coords'] - central_coords
            angle = get_angle(v1, v2)
            return "4Ag-2Sb_Opposite" if angle > 150 else "4Ag-2Sb_Adjacent"
        if num_ag == 3:
            coords = [n['coords'] for n in ag_neighbors]
            vectors = [c - central_coords for c in coords]
            angles = [get_angle(vectors[0], vectors[1]), get_angle(vectors[0], vectors[2]), get_angle(vectors[1], vectors[2])]
            return "3Ag-3Sb_Line" if any(a > 150 for a in angles) else "3Ag-3Sb_Cluster"
        return "Unknown"
    


if __name__ == "__main__":
    
    ## Configuration Settings ##
    poscar = "home/FILEPATH/POSCAR"
    CORE_ATOM = 'Te'
    NUM_NEIGH = 6
    STRUC_TYPE = 'octahedral'
    
    analysis = AtomicConfigurationAnalysis()
    analysis.number_of_neighbors = NUM_NEIGH
    analysis.analysis_type = STRUC_TYPE
    analysis.parse_poscar(poscar)
    analysis.generate_cells()
    analysis.bond_lengths_for_type(CORE_ATOM)
    analysis.bond_angles_for_type(CORE_ATOM)
    analysis.analyze_neighbor_ratios(CORE_ATOM) 

    class_summary = Counter(analysis.environment_class_analysis.values())

    statistics_by_class = {}
    for full_name in analysis.classifications.values():
        atom_keys_in_class = [
            key for key, value in analysis.environment_class_analysis.items()
            if value == full_name
        ]
        bl_Ag = [
            n['bond_length'] for key in atom_keys_in_class
            for n in analysis.bond_length_analysis[key] if n['type'] == 'Ag'
        ]
        bl_Sb = [
            n['bond_length'] for key in atom_keys_in_class
            for n in analysis.bond_length_analysis[key] if n['type'] == 'Sb'
        ]
        statistics_by_class[full_name] = {
            'Ag_mean': np.mean(bl_Ag) if bl_Ag else None,
            'Ag_std': np.std(bl_Ag) if bl_Ag else None,
            'Sb_mean': np.mean(bl_Sb) if bl_Sb else None,
            'Sb_std': np.std(bl_Sb) if bl_Sb else None,
        }

    print("\n--- Summary of Classifications ---")
    for full_name in analysis.classifications.values():
        count = class_summary.get(full_name, 0)
        print(f"  {full_name:<18}: {count}")

    te_count = analysis.atom_counts_dict.get('Te', 0)
    print(f"\n--- Total number of Te-centered octahedra analyzed: {te_count} ---")

    print("\n--- Bond Length Statistics ---")
    for full_name, stats in statistics_by_class.items():
        if stats['Ag_mean'] is not None:
            print(f"  {full_name:<18} - Ag: mean = {stats['Ag_mean']:.4f} A, std = {stats['Ag_std']:.4f} A")
        if stats['Sb_mean'] is not None:
            print(f"  {full_name:<18} - Sb: mean = {stats['Sb_mean']:.4f} A, std = {stats['Sb_std']:.4f} A")

    Ag_bonds = [
        neighbor['bond_length'] for neighbor_list in analysis.bond_length_analysis.values()
        for neighbor in neighbor_list if neighbor['type'] == 'Ag'
    ]
    Sb_bonds = [
        neighbor['bond_length'] for neighbor_list in analysis.bond_length_analysis.values()
        for neighbor in neighbor_list if neighbor['type'] == 'Sb'
    ]

    print("\n--- Total Bond Length Statistics ---")
    if Ag_bonds:
        Ag_bonds_mean = np.mean(Ag_bonds)
        Ag_bonds_std = np.std(Ag_bonds)
        print(f"  Overall Ag: mean = {Ag_bonds_mean:.4f} A, std = {Ag_bonds_std:.4f} A")

    if Sb_bonds:
        Sb_bonds_mean = np.mean(Sb_bonds)
        Sb_bonds_std = np.std(Sb_bonds)

        print(f"  Overall Sb: mean = {Sb_bonds_mean:.4f} A, std = {Sb_bonds_std:.4f} A")
