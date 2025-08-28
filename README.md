# Atomic Configuration Analysis
Atomic Configuration Analysis is a Python script that will parse a VASP input POSCAR file and statistically analyze local atomic properties such as bond lengths, angles, neighboring atom ratios, etc. 

**Configure Settings in main before running code**

  -- poscar = "/jet/home/mohammes/projects/atomic_site_analysis/POSCAR_structures/POSCAR-R1"
  
  -- CORE_ATOM = 'Te'
  
  -- NUM_NEIGH = 6
    
  -- STRUC_TYPE = 'octahedral'
      
      *Note: Will need to set classifications manually if changing number of neighboring atoms to analyze. 

Only aca.py needed to run analysis tool. Other files are for archival purposes only. 
