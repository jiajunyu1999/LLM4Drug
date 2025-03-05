from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

# Define the properties and their corresponding descriptor functions
props = ["MolLogP", "qed", "TPSA", "NumHAcceptors", "NumHDonors", "MolWt"]
prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]

# Create a dictionary to map property names to their functions
prop2func = {prop: func for prop, func in prop_pred}

def get_property(smiles):
    """
    Get the properties of a molecule from its SMILES string.

    Parameters:
        smiles (str): The SMILES representation of the molecule.

    Returns:
        str: A formatted string containing the scaffold, functional groups, and properties.
    """
    res = '{}&{}&{}&{}&{}&{}&{}&{}'
    try:
        mol = Chem.MolFromSmiles(smiles)
        scaffold = Chem.MurckoDecompose(mol)
        functional_groups = Chem.ReplaceCore(mol, scaffold)
        scaffold = Chem.MolToSmiles(scaffold)
        functional_groups = Chem.MolToSmiles(functional_groups)
    except:
        return res.format('none', 'none', -999, -999, -999, -999, -999, -999)
    
    props_value = []
    for p in props:
        try:
            props_value.append(prop2func[p](mol))
        except:
            props_value.append(-999)

    return res.format(scaffold, functional_groups, *props_value)

def generate_mol_property(df, head = 'mol'):
    """
    Calculate the properties for each molecule in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame with a 'gene_mol' column containing SMILES strings.

    Returns:
        pd.DataFrame: The DataFrame with new columns for each property.
    """

    df['property'] = df[head].apply(get_property)
    head = head.replace('mol','')
    property_cols = ['scaffold', 'functional', 'MolLogP', 'qed', 'TPSA', 'NumHAcceptors', 'NumHDonors', 'MolWt']
    for i, col in enumerate(property_cols):
        df[f'{head}{col}'] = df['property'].apply(lambda x: float(x.split('&')[i]) if i > 1 else x.split('&')[i])

    df = df.drop(columns=['property'])
    return df

def similarity_mols(smiles1, smiles2):
    """
    Calculate the Tanimoto similarity between two molecules.

    Parameters:
        smiles1 (str): The SMILES representation of the first molecule.
        smiles2 (str): The SMILES representation of the second molecule.

    Returns:
        float: The Tanimoto similarity between the two molecules.
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=1024)
        return TanimotoSimilarity(fp1, fp2)
    except:
        return 0

def same_scaffold(smiles, scaffold_smiles):
    """
    Check if a molecule has the same scaffold as a given scaffold.

    Parameters:
        smiles (str): The SMILES representation of the molecule.
        scaffold_smiles (str): The SMILES representation of the scaffold.

    Returns:
        int: 1 if the molecule has the same scaffold, otherwise 0.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        scaffold = Chem.MolFromSmiles(scaffold_smiles)
        return 1 if Chem.ReplaceCore(mol, scaffold) is not None else 0
    except:
        return 0

def get_scaffold(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            scaffold = Chem.MurckoDecompose(mol)
            functional_groups = Chem.ReplaceCore(mol, scaffold)
            return Chem.MolToSmiles(scaffold), Chem.MolToSmiles(functional_groups)
        except:
            return '', ''