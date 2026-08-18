# QMugs: Quantum Mechanical Properties of Drug-like Molecules 

The QMugs data collection comprises quantum mechanical properties of more than 665k biologically and pharmacologically relevant molecules extracted from the ChEMBL27 database (http://www.ebi.ac.uk/chembl), totaling ~2M conformers. QMugs contains (i) their optimized geometries and thermodynamic properties (including vibrational frequencies) obtained via the semi-empirical method GFN2-xTB, (ii) atomic and molecular properties (e.g., partial charges, bond orders, energies, and dipoles) on both the GFN2-xTB and on the DFT (ωB97X-D/def2-SVP) levels of theory, and (iii) quantum mechanical wavefunction as local basises of atomic orbitals (DFT density and orbital matrices), totaling over 7 terabytes of uncompressed data. This dataset is intended to facilitate the development of machine learning models that learn from molecular data on different levels of theory while also providing insight into the corresponding relationships between molecular structure and biological activity.


## Downloading the dataset
While you can download the individual files through the website's GUI, we recommend a programmatic access for larger files. For that, download the "download_links.txt" file manually, then run the following command: 
```
cat download_links.txt | while read line || [[ -n $line ]]; do linearr=($line); link=${linearr[1]}; wget -c ${linearr[1]} -O ${linearr[0]}; done
```


## Content of QMugs and file structures:
* summary.csv: Overview of basic structural properties of molecules (e.g., number of heavy atoms, number of rings, molecular weight) and calculated properties (e.g., energies, dipoles, rotational constants). No atomic or bond level information is contained in this file (see individual structure-data files (SDFs) for those). The column "nonunique_smiles" denotes structures which, as described in the paper, share their SMILES representation with at least one other ChEMBL-ID. Users may wish to use those examples together in e.g., only the training set to avoid information leakage.
* tarball_assignment.csv: Assignment of CHEMBL-IDs to the corresponding wavefunction tarballs. 
* structures.tar.gz: Each SDF contains the optimized 3D geometry of the indvidual conformer and all calculated properties. See paper for a detailed description and units. Each CHEMBL ID has its own subdirectory.
* wfns: Tarballs for wavefunction files of all structures, grouped in 100 files, each containing wavefunctions for all conformers of approx. 6650 molecules (see tarball_assignment.csv).
* vibspectra.tar.gz: Vibrational spectra (as output by GFN2-xTB). 


## Reading properties from SDFs
A dictionary of 44 keys including 42 quantum properties can be loaded from the SDFs using e.g., RDKit. See paper for a detailed description. Example of loading properties:
```python
from rdkit import Chem
mol = next(Chem.SDMolSupplier(<PATH_TO_SDF>, removeHs=False))
props = mol.GetPropsAsDict()
dft_urt = props["DFT:FORMATION_ENERGY"]
```
Atomic and bond properties have the same ordering as the atoms and bonds in the SDF and are separated by vertical lines ("|").
```python
dft_charges = props["DFT:MULLIKEN_CHARGES"].split("|")
dft_charges = [float(charge) for charge in dft_charges]
```

Example for bond properties: 
```python
# Load properties from dictionary
dft_wbo = props["DFT:WIBERG_LOWDIN_BOND_ORDER"].split("|")
dft_wbo = [float(x) for x in dft_wbo]
gfn2_wbo = props["GFN2:WIBERG_BOND_ORDER"].split("|")
gfn2_wbo = [float(x) for x in gfn2_wbo]

# Loop over bonds
for idx, bond in enumerate(mol.GetBonds()):
    atomid1 = bond.GetBeginAtom().GetSymbol() # atom type of the first atom of the bond
    atomid2 = bond.GetEndAtom().GetSymbol() # atom type of the second atom of the bond
    atomidx1 = bond.GetBeginAtomIdx() # idx of the first atom of the bond
    atomidx2 = bond.GetEndAtomIdx() # idx of the second atom of the bond
    dft_order = float(dft_wbo[idx])
    gfn2_order = float(gfn2_wbo[idx]) 
```

Example for atomic properties: 
```python
# Load quantum properties from dictionary
gfn2_charges = props["GFN2:MULLIKEN_CHARGES"].split("|")
gfn2_charges = [float(x) for x in gfn2_charges]
dft_charges = props["DFT:MULLIKEN_CHARGES"].split("|")
dft_charges = [float(x) for x in dft_charges]

# Loop over atoms
for idx, atom in enumerate(mol.GetAtoms()):
    atomid = atom[idx].GetSymbol() 
    gfn2_charge = gfn2_charges[idx] 
    dft_charge = dft_charges[idx] 
```

## Reading properties from the wavefunctions
A variety of properties can also be loaded directly from the wavefunctions. See main paper for more details. Properties include DFT matrices:
```python
import numpy as np
import psi4
wfn = np.load(<PATH_TO_WFN>, allow_pickle=True).tolist()
density_matrix_a = wfn["matrix"]["Ca"]
density_matrix_b = wfn["matrix"]["Cb"]
orbital_matrix_a = wfn["matrix"]["Db"]
orbital_matrix_b = wfn["matrix"]["Db"]
aotoso_matrix = wfn["matrix"]["aotoso"]
```
and bond orders for covalent and non-covalent interactions: 
```python
wfn = psi4.core.Wavefunction.from_file(<PATH_TO_WFN>)
psi4.oeprop(wfn, "MAYER_INDICES")
psi4.oeprop(wfn, "WIBERG_LOWDIN_INDICES")
meyer_bos = wfn.array_variables()["MAYER_INDICES"]
lodwin_bos = wfn.array_variables()["WIBERG_LOWDIN_INDICES"]
```
In order to reduce DFT matrix filesize, the original double precision (float64) provided by Psi4 has been reduced to single precision (float32). Fock matrices for alpha and beta orbitals have also been removed. 


## Calculating electron densities from the wavefunctions
Psi4 allows the calculation of .cube files from the .wfn files for properties such as electron densities and electrostatic potential. See <http://www.psicode.org/psi4manual/master/cubeprop.html> for additional details

## Converting .wfn files into .fchk files
Psi4 can convert .wfn files directly into .fchk files:
```python
import psi4
wfn = psi4.core.Wavefunction.from_file(<PATH_TO_WFN>)
psi4.fchk(wfn, "output.fchk")
```

## ChEMBL information
The data content in ChEMBL is licensed under a highly permissive Creative
Commons license - specifically the "CC Attribution-ShareAlike 3.0 Unported
license", see LICENSE file. The required attribution should contain the url
of the ChEMBL resource, and also the release version, e.g.:

ChEMBL data is from <http://www.ebi.ac.uk/chembl> - the version of ChEMBL is
chembl_27.

## Bioactivity and pharmacological information
Each molecule included in QMugs has an annotated bioactivity to a macromolecular target with binding affinities (Ki, EC50, IC50 etc.) in the range of 1 mM (10e-3 M) to 1 pM (10e-12 M). This bioactivity related information can be retrieved from the ChEMBL database.

## Changelog
- 30.07.21: Using unpaired electrons in atomic energy calculation. 

