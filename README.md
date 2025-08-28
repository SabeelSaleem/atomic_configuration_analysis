# Atomic Configuration Analysis
Atomic Configuration Analysis is a Python script that will parse a VASP input POSCAR file and statistically analyze local atomic properties such as bond lengths, angles, neighboring atom ratios, etc. 

  -- POSCAR Path must be defined and Core atom must be chosen within structure to search for neighbors. Defined at Configuration Settings in main. 
  
  -- Analyzes set number of closest neighboring atoms which can be set to any number of neighbors defined in init as self.number_of_neighbors = # 
      *Note: Will need to set classifications manually if changing number of neighboring atoms to analyze. 
