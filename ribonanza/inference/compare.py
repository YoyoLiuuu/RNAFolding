from calculate_score import get_scores 
ground_truth = "data/4OJI.pdb"
predicted = "data/4OJI_Twister.pdb"

from prody import *
from prody import *

ref = parsePDB(ground_truth)
mob = parsePDB(predicted)

ref_ca = ref.select('name CA')
mob_ca = mob.select('name CA')

# Superimpose structures
tmscore, transformation = superpose(mob_ca, ref_ca)

# Compute RMSD
rmsd = calcRMSD(mob_ca, ref_ca)
print("Superposition RMSD:", rmsd)