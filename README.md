# Atomic_coordinates-energy_NNPs
WORK IN PROGRESS
* Requires the ANI dataset (20 Million molecules)

## Project impulse
This project is to study the application of Graph Learning methods (GCN - Kipf and Welling, GraphSage - Leskovec et al, MPNN - Gilmer et al, etc.) used in combination with various Neural Network Potential methods (NNPs - Behler et al, ANI-1 - Roitberg et al, SchNet - Schutt et al). Preprocessing code attempts to extract atomic coordinates (features) and energies (targets) to be used to train a single fully connected NN, with each unique atom learning a specific set of model weights that are swapped (transfer learning) as needed. The predicted energies of the fully connected NNs are then used as features in downstream Graph Learning tasks (molecular property prediction, generation, etc.)
