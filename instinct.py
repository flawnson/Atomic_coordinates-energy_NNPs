# TODO: Figure out how to delete the last list of each overflowed energy list using the index list

import math
import numpy as np
import itertools

from pprint import pprint
from itertools import islice
from ANI1_dataset_master.readers import pyanitools as pya

# Set the HDF5 file containing the data
hdf5file = r'C:\Users\Flawnson\env\ANI1_dataset_master\benchmark\ani_gdb_s01.h5'

"DATA LOADER INITIALIZATION"
# Construct the data loader class and initialize lists
adl = pya.anidataloader(hdf5file)
# heavy_atom = "O"
species_list = []
species_positions = []
coordinates_list = []
energy_list = []

assert len(coordinates_list) == len(species_list), "Number of species and number of coordinate sets must be the same"

# Retrieve species coordinates
for datapoint in adl:
    P = datapoint['path']
    X = datapoint['coordinates']
    E = datapoint['energies']
    S = datapoint['species']
    sm = datapoint['smiles']

    species_list.append(S)
    coordinates_list.append(X)
    energy_list.append(E)

species_len_list = [len(species) for species in species_list]

"ATOM COORDINATE INITIALIZATION"
def get_unique_atoms(species_list):
    flattened_species = list(itertools.chain.from_iterable(species_list))
    unique_species = np.unique(flattened_species)
    
    return unique_species

unique_atoms = get_unique_atoms(species_list)

def coordinates_divider(species_list, coordinates_list, unique_atoms):
    """
    Returns a list of lists (number of unique atom) of lists (number of molecules)
    of the index posiion of each unique atom in each molecule
    """
    divided_idx = []

    # Picking out the atom coordinates of a specific type of atom in each molecule of the dataset
    for heavy_atom in unique_atoms:
        for species in species_list:
            position_coordinates = [idx for idx, atom in enumerate(species) if atom == heavy_atom]
            species_positions.append(position_coordinates)

    for i in range(0, len(species_positions), len(species_list)):
        divided_idx.append((species_positions[i:i + len(species_list)]))
    
    return divided_idx

divided_idx = coordinates_divider(species_list, coordinates_list, (unique_atoms))
print(divided_idx, "\n")

divided_idx_dict = dict(zip(unique_atoms, divided_idx))
print(divided_idx_dict)

# for idx_list in divided_idx:
#     per_mol_idx = list(zip((idx_list)))
#     print(per_mol_idx)

for idx_list in divided_idx:
    per_mol_idx = list(map(list, zip(*divided_idx)))
print(per_mol_idx, "\n")

for mol_idx_list in per_mol_idx:
    mol_idx_dict = dict(zip(unique_atoms, mol_idx_list))
    print(mol_idx_dict)
#     for idx in mol_idx_dict.values():
#         print(idx)
        # for num in idx:
        #     if num == 0:
        #         print(mol_idx_dict.keys())

# for coordinates, mol_idx in zip(coordinates_list, per_mol_idx):

def get_coordinate_portions(coordinates_list, species_list):
    # Retrieve relative atomic energies per atom in each molecule
    package = zip(coordinates_list, species_list)
    portions = []

    for coordinates, species in package:
        portion = math.floor(len(coordinates) / len(species))
        portions.append(portion)

    return portions

coordinate_portion_list = zip(coordinates_list, get_coordinate_portions(coordinates_list, species_list))

# print(list(coordinate_portion_list))

def get_coordinates(coordinate_portions, species_len_list):
    # Create two new lists of lengths (to determine the overflows) and one for modification after applying the chunker function
    divided_coordinates_len = []
    divided_coordinates = []
    trim_idx = []

    # Divide each atomic energy vector by the respective value
    for coordinates, portions in coordinate_portions:
        chunks = [coordinates[i:i + portions].tolist() for i in range(0, len(coordinates), portions)]
        divided_coordinates_len.append(len(chunks))
        divided_coordinates.append(chunks)

    # Get rid of overflowed coordinates
    trim_check = (np.array(species_len_list) == np.array(divided_coordinates_len))
    trim_idx = [idx for idx, item in enumerate(trim_check) if item == False]

    for index in trim_idx:
        divided_coordinates[index].pop(-1)

    return divided_coordinates

divided_coordinates = get_coordinates(coordinate_portion_list, species_len_list)
# pprint(divided_coordinates)

packed = list(zip(species_list, divided_coordinates))
for atoms_in_species, divided_coordinate in packed:
    subpack = zip(atoms_in_species, divided_coordinate)
    pprint(list(subpack)[2])
print(species_list)
# for ping in divided_coordinates:
#         print("\n")
#         test = zip(species_list, ping)
#         pprint(list(test))
#         print(len(ping))
        # for ing in ping:
        #         print(len(ing))
                # test = zip(species_list, ing)
                # print(list(test))

# "ATOM ENERGY INITIALIZATION"
# class EnergiesDivider:
#     def __init__(self, energy_list, species_list, species_len_list):
#         self.energy_list = energy_list
#         self.species_list = species_list
#         self.species_len_list = species_len_list

#     def get_energy_portions(energy_list, species_list):
#         # Retrieve relative atomic energies per atom in each molecule
#         package = zip(energy_list, species_list)
#         portions = []

#         for energy, species in package:
#             portion = math.floor(len(energy) / len(species))
#             portions.append(portion)

#         return portions

#     energy_portion_list = zip(energy_list, get_energy_portions(energy_list, species_list))

#     def get_energies(energy_portions, species_len_list):
#         # Create two new lists of lengths (to determine the overflows) and one for modification after applying the chunker function
#         divided_energies_len = []
#         divided_energies = []

#         # Divide each atomic energy vector by the respective value
#         for energies, portions in energy_portions:
#             chunks = [energies[i:i + portions].tolist() for i in range(0, len(energies), portions)]
#             divided_energies_len.append(len(chunks))
#             divided_energies.append(chunks)

#         # Get rid of overflowed energies
#         trim_check = (np.array(species_len_list) == np.array(divided_energies_len))
#         trim_idx = [idx for idx, item in enumerate(trim_check) if item == False]

#         for index in trim_idx:
#             divided_energies[index].pop(-1)

#         return divided_energies

# # print(EnergiesDivider.get_energy_portions(energy_list, species_list))
# # print(list(EnergiesDivider.energy_portion_list))

# energies = EnergiesDivider.get_energies(EnergiesDivider.energy_portion_list, species_len_list)