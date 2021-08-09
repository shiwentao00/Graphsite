# Copyright: Wentao Shi, 2021
import numpy as np
import pandas as pd
from biopandas.mol2 import PandasMol2
from scipy.spatial import distance
import statistics

class Graphsite:
    """
    A callable class that reads a pocket and compute its graph representation.
    
    The graph representation: Each atom represents a node. If the distance 
    between two atoms are less than or equal to a threshold (default is 4.5 Angstrom), 
    an undirected edge is formed between these two atoms.
    
    Parameters:
        mol_path (string): The path of the input .mol2 file.
        profile_path (string): The path of the input .profile file.
        pop_path (string): The path of the input .popsa file.
        hydrophobicity: A dictionary that maps residue to hydrophobicity as node feature.
        binding_probability: A dictionary that maps residue to binding_probability as node feature.
        features_to_use: A list of node features to select.
        threshold (float): The threshold for forming an edge between two atoms.

    Returns:
        node_feature: Node feature matrix with shape [num_nodes, num_node_features].
        
        edge_index: Graph connectivity list in Coordinate list (COO) format with shape 
        [2, num_edges * 2]. Example: 
        
                edge_index = [[0, 1, 1, 2],
                              [1, 0, 2, 1]]

        Graphsite always generates undirected graph, so the edge_index always has pairs
        of edges. In this example, we have node 0 connected to node 1, node 1 connected
        to node 2.
    
        edge_attr: Edge feature matrix with shape [num_edges, 1]. There is only 1 edge
        feature, which is the number of chemical bonds.
    """
    def __init__(self):
        """
        Initialize default parameters.
        """
        # hard coded hydrophobicity node feature
        self.hydrophobicity = {
            'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5,
            'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4,
            'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9,
            'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8,
            'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
        }

        # hard coded binding probability node feature
        self.binding_probability = {
            'ALA': 0.701, 'ARG': 0.916, 'ASN': 0.811, 'ASP': 1.015,
            'CYS': 1.650, 'GLN': 0.669, 'GLU': 0.956, 'GLY': 0.788,
            'HIS': 2.286, 'ILE': 1.006, 'LEU': 1.045, 'LYS': 0.468,
            'MET': 1.894, 'PHE': 1.952, 'PRO': 0.212, 'SER': 0.883,
            'THR': 0.730, 'TRP': 3.084, 'TYR': 1.672, 'VAL': 0.884
        }

        # all available node features
        self.features_to_use = [
            'x', 'y', 'z', 
            'r', 'theta', 'phi', 
            'sasa', 'charge', 
            'hydrophobicity', 
            'binding_probability', 
            'sequence_entropy'
        ]  

        self.threshold = 4.5

    def __call__(self, mol_path, profile_path, pop_path,
                    hydrophobicity=None, binding_probability=None,
                    features_to_use=None, threshold=None):
        if not hydrophobicity:
            hydrophobicity = self.hydrophobicity
        
        if not binding_probability:
            binding_probability = self.binding_probability

        if not features_to_use:
            features_to_use = self.features_to_use

        if not threshold:
            threshold = self.threshold

        # read basic info from input file
        atoms = PandasMol2().read_mol2(mol_path)
        atoms = atoms.df[['atom_id', 'subst_name',
                          'atom_type', 'atom_name',
                          'x', 'y', 'z', 'charge']]
        atoms['residue'] = atoms['subst_name'].apply(lambda x: x[0:3])

        # compute hydrophobicity of atoms/residues
        atoms['hydrophobicity'] = atoms['residue'].apply(
            lambda x: hydrophobicity[x])

        # compute binding_probability of atoms/residues
        atoms['binding_probability'] = atoms['residue'].apply(
            lambda x: binding_probability[x])

        # compute spherical coordinates of atoms
        r, theta, phi = compute_spherical_coord(atoms[['x', 'y', 'z']].to_numpy())
        if 'r' in features_to_use:
            atoms['r'] = r
        if 'theta' in features_to_use:
            atoms['theta'] = theta
        if 'phi' in features_to_use:
            atoms['phi'] = phi

        # a list of residues
        siteresidue_list = atoms['subst_name'].tolist()

        # compute solvent-accessible surface area (SASA) node feature
        if 'sasa' in features_to_use:
            qsasa_data = extract_sasa_data(siteresidue_list, pop_path)
            atoms['sasa'] = qsasa_data

        # compute sequence_entropy node feature
        if 'sequence_entropy' in features_to_use:
            # sequence entropy data with subst_name as keys
            seq_entropy_data = extract_seq_entropy_data(
                siteresidue_list, profile_path)
            atoms['sequence_entropy'] = atoms['subst_name'].apply(
                lambda x: seq_entropy_data[x])

        if atoms.isnull().values.any():
            print('invalid input data (containing nan):')
            print(mol_path)

        # parse chemical bonds from the input file
        bonds = bond_parser(mol_path)

        # compute the graph representation from pandas dataframe
        node_feature, edge_index, edge_attr = form_graph(
            atoms, bonds, features_to_use, threshold)

        return node_feature, edge_index, edge_attr


def form_graph(atoms, bonds, features_to_use, threshold, coordinate_normalize_factor=300):
    """
    Form a graph representation in numpy matrices according to the input data frame.

    Parameters:
    atoms: Dataframe containing the 3-d coordinates of atoms.
    bonds: Dataframe of bond info.
    threshold: Distance threshold to form the edge.
    """
    A = atoms.loc[:, 'x':'z']
    A_dist = distance.cdist(A, A, 'euclidean')  # the distance matrix

    # set the element whose value is larger than threshold to 0
    threshold_condition = A_dist > threshold

    # set the element whose value is larger than threshold to 0
    A_dist[threshold_condition] = 0

    edge_index = np.where(A_dist > 0)
    edge_index = np.vstack((edge_index[0], edge_index[1]))
    edge_attr = compute_edge_attr(edge_index, bonds)

    # normalize large features
    atoms['x'] = atoms['x'] / coordinate_normalize_factor
    atoms['y'] = atoms['y'] / coordinate_normalize_factor
    atoms['z'] = atoms['z'] / coordinate_normalize_factor

    # select node feautures
    node_feature = atoms[features_to_use].to_numpy()

    return node_feature, edge_index, edge_attr


def bond_parser(pocket_path):
    f = open(pocket_path, 'r')
    f_text = f.read()
    f.close()
    bond_start = f_text.find('@<TRIPOS>BOND')
    bond_end = -1
    df_bonds = f_text[bond_start:bond_end].replace('@<TRIPOS>BOND\n', '')
    df_bonds = df_bonds.replace('am', '1')  # amide
    df_bonds = df_bonds.replace('ar', '1.5')  # aromatic
    df_bonds = df_bonds.replace('du', '1')  # dummy
    df_bonds = df_bonds.replace('un', '1')  # unknown
    df_bonds = df_bonds.replace('nc', '0')  # not connected
    df_bonds = df_bonds.replace('\n', ' ')

    # convert the the elements to integer
    df_bonds = np.array([np.float(x) for x in df_bonds.split()]).reshape(
        (-1, 4))

    df_bonds = pd.DataFrame(
        df_bonds, columns=['bond_id', 'atom1', 'atom2', 'bond_type'])

    df_bonds.set_index(['bond_id'], inplace=True)

    return df_bonds


def compute_edge_attr(edge_index, bonds):
    """Compute the edge attributes according to the chemical bonds."""
    sources = edge_index[0, :]
    targets = edge_index[1, :]
    edge_attr = np.zeros((edge_index.shape[1], 1))
    for index, row in bonds.iterrows():
        # find source == row[1], target == row[0]
        # minus one because in new setting atom id starts with 0
        source_locations = set(list(np.where(sources == (row[1] - 1))[0]))
        target_locations = set(list(np.where(targets == (row[0] - 1))[0]))
        edge_location = list(
            source_locations.intersection(target_locations))[0]
        edge_attr[edge_location] = row[2]

        # find source == row[0], target == row[1]
        source_locations = set(list(np.where(sources == (row[0] - 1))[0]))
        target_locations = set(list(np.where(targets == (row[1] - 1))[0]))
        edge_location = list(
            source_locations.intersection(target_locations))[0]
        edge_attr[edge_location] = row[2]

    return edge_attr


def compute_spherical_coord(data):
    """
    Shift the geometric center of the pocket to origin,
    then compute its spherical coordinates.
    """
    # center the data around origin
    center = np.mean(data, axis=0)
    shifted_data = data - center

    r, theta, phi = cartesian_to_spherical(shifted_data)

    return r, theta, phi


def cartesian_to_spherical(data):
    """
    Convert cartesian coordinates to spherical coordinates.

    Parameters:
    data: Numpy array with shape (n, 3) which is the cartesian 
    coordinates (x, y, z) of n points.

    Returns:
    numpy array with shape (n, 3) which is the spherical
    coordinates (r, theta, phi) of n points.
    """
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # distances to origin
    r = np.sqrt(x**2 + y**2 + z**2)

    # angle between x-y plane and z
    theta = np.arccos(z/r)/np.pi

    # angle on x-y plane
    phi = np.arctan2(y, x)/np.pi

    # spherical_coord = np.vstack([r, theta, phi])
    # spherical_coord = np.transpose(spherical_coord)
    return r, theta, phi


def extract_sasa_data(siteresidue_list, pop):
    """
    Extracts accessible surface area data from .out file 
    generated by POPSlegacy, then matches the data in the 
    .out file to the binding site in the mol2 file.
    
    Used POPSlegacy https://github.com/Fraternalilab/POPSlegacy.
    """
    # extracting sasa data from .out file
    residue_list = []
    qsasa_list = []

    # opening .out file
    with open(pop) as popsa:
        for line in popsa:
            line_list = line.split()

            # extracting relevant information
            if len(line_list) == 12:
                residue_type = line_list[2] + line_list[4]
                if residue_type in siteresidue_list:
                    qsasa = line_list[7]
                    residue_list.append(residue_type)
                    qsasa_list.append(qsasa)

    qsasa_list = [float(x) for x in qsasa_list]
    median = statistics.median(qsasa_list)
    qsasa_new = [median if x == '-nan' else x for x in qsasa_list]

    # matching amino acids from .mol2 and .out files and creating dictionary
    qsasa_data = []
    fullprotein_data = list(zip(residue_list, qsasa_new))
    for i in range(len(fullprotein_data)):
        if fullprotein_data[i][0] in siteresidue_list:
            qsasa_data.append(float(fullprotein_data[i][1]))

    return qsasa_data


def extract_seq_entropy_data(siteresidue_list, profile):
    """Extracts sequence entropy data from .profile."""
    # opening and formatting lists of the probabilities and residues
    with open(profile) as profile:
        ressingle_list = []
        probdata_list = []

        # extracting relevant information
        for line in profile:
            line_list = line.split()
            residue_type = line_list[0]
            prob_data = line_list[1:]
            prob_data = list(map(float, prob_data))
            ressingle_list.append(residue_type)
            probdata_list.append(prob_data)
    ressingle_list = ressingle_list[1:]
    probdata_list = probdata_list[1:]

    # changing single letter amino acid to triple letter
    # with its corresponding number
    count = 0
    restriple_list = []
    for res in ressingle_list:
        newres = res.replace(res, amino_single_to_triple(res))
        count += 1
        restriple_list.append(newres + str(count))

    # Calculating information entropy
    # suppress warning
    with np.errstate(divide='ignore'):
        prob_array = np.asarray(probdata_list)
        log_array = np.log2(prob_array)

        # change all infinite values to 0
        log_array[~np.isfinite(log_array)] = 0
        entropy_array = log_array * prob_array
        entropydata_array = np.sum(a=entropy_array, axis=1) * -1
        entropydata_list = entropydata_array.tolist()

    # matching amino acids from .mol2 and .profile files and creating dictionary
    fullprotein_data = dict(zip(restriple_list, entropydata_list))
    seq_entropy_data = {residue: float(
        fullprotein_data[residue]) for residue in siteresidue_list if residue in fullprotein_data}

    return seq_entropy_data


def amino_single_to_triple(single):
    """
    Converts the single letter amino acid abbreviation to 
    the triple letter abbreviation.
    """

    single_to_triple_dict = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'G': 'GLY', 'Q': 'GLN', 'E': 'GLU', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
    }

    for i in single_to_triple_dict.keys():
        if i == single:
            triple = single_to_triple_dict[i]

    return triple
