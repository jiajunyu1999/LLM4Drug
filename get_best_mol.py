from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
import json
weight = json.load(open('./weight.wt', 'r'))

# 1. Molecular Weight (MW)
def calculate_molecular_weight(mol):

    if mol is None:
        return "Invalid SMILES string"
    return Descriptors.MolWt(mol)

# 2. Topological Polar Surface Area (TPSA)
def calculate_tpsa(mol):

    if mol is None:
        return "Invalid SMILES string"
    return rdMolDescriptors.CalcTPSA(mol)

# 3. Hydrogen Bond Donors
def calculate_h_bond_donors(mol):

    if mol is None:
        return "Invalid SMILES string"
    return Descriptors.NumHDonors(mol)

# 4. Hydrogen Bond Acceptors
def calculate_h_bond_acceptors(mol):

    if mol is None:
        return "Invalid SMILES string"
    return Descriptors.NumHAcceptors(mol)

# 5. Polarizability (Approximated by MolMR)
def calculate_polarizability(mol):

    if mol is None:
        return "Invalid SMILES string"
    return Descriptors.MolMR(mol)

# 6. LogP (Partition Coefficient)
def calculate_logp(mol):

    if mol is None:
        return "Invalid SMILES string"
    return Crippen.MolLogP(mol)

# 7. Number of Rotatable Bonds
def calculate_num_rotatable_bonds(mol):

    if mol is None:
        return "Invalid SMILES string"
    return Descriptors.NumRotatableBonds(mol)

# 8. Formal Charge
def calculate_charge(mol):

    if mol is None:
        return "Invalid SMILES string"
    return Chem.GetFormalCharge(mol)

# 9. Stereochemistry (Chiral Centers)
def calculate_num_chiral_centers(mol):

    if mol is None:
        return "Invalid SMILES string"
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    return len(chiral_centers)

# 10. Solubility (Approximated by LogS using Crippen LogP)
def calculate_solubility(mol):

    if mol is None:
        return "Invalid SMILES string"
    logp = Crippen.MolLogP(mol)
    logS = 0.16 - 0.63 * logp  # Approximate formula for LogS
    return logS

# 11. Synthetic Accessibility (SA)
def calculate_synthetic_accessibility(mol):

    if mol is None:
        return "Invalid SMILES string"
    mw = Descriptors.MolWt(mol)
    num_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
    return 10 - (0.01 * mw + 0.5 * num_rings)  # Simple SA approximation

# 12. Hydroxyl (-OH) Group
def calculate_num_hydroxyl_groups(mol):

    if mol is None:
        return "Invalid SMILES string"
    hydroxyl_pattern = Chem.MolFromSmarts('[OX2H]')
    return len(mol.GetSubstructMatches(hydroxyl_pattern))

# 13. Carboxyl (-COOH) Group
def calculate_num_carboxyl_groups(mol):

    if mol is None:
        return "Invalid SMILES string"
    carboxyl_pattern = Chem.MolFromSmarts('C(=O)[OH]')
    return len(mol.GetSubstructMatches(carboxyl_pattern))

# 14. Amino (-NH2) Group
def calculate_num_amino_groups(mol):

    if mol is None:
        return "Invalid SMILES string"
    amino_pattern = Chem.MolFromSmarts('[NX3H2]')
    return len(mol.GetSubstructMatches(amino_pattern))

# 15. Ether (R-O-R') Group
def calculate_num_ether_groups(mol):

    if mol is None:
        return "Invalid SMILES string"
    ether_pattern = Chem.MolFromSmarts('O')
    return len(mol.GetSubstructMatches(ether_pattern))

# 16. Phosphate (PO4) Group
def calculate_num_phosphate_groups(mol):

    if mol is None:
        return "Invalid SMILES string"
    phosphate_pattern = Chem.MolFromSmarts('P(=O)(O)(O)O')
    return len(mol.GetSubstructMatches(phosphate_pattern))

# 17. Amide (C(O)NH2) Group
def calculate_num_amide_groups(mol):

    if mol is None:
        return "Invalid SMILES string"
    amide_pattern = Chem.MolFromSmarts('C(=O)N')
    return len(mol.GetSubstructMatches(amide_pattern))

# 18. Nitrile (-C≡N) Group
def calculate_num_nitrile_groups(mol):

    if mol is None:
        return "Invalid SMILES string"
    nitrile_pattern = Chem.MolFromSmarts('C#N')
    return len(mol.GetSubstructMatches(nitrile_pattern))

# 19. Sulfonyl (SO2) Group
def calculate_num_sulfonyl_groups(mol):

    if mol is None:
        return "Invalid SMILES string"
    sulfonyl_pattern = Chem.MolFromSmarts('S(=O)(=O)')
    return len(mol.GetSubstructMatches(sulfonyl_pattern))

# 20. Nitro (-NO2) Group
def calculate_num_nitro_groups(mol):

    if mol is None:
        return "Invalid SMILES string"
    nitro_pattern = Chem.MolFromSmarts('N(=O)(=O)')
    return len(mol.GetSubstructMatches(nitro_pattern))

# 21. Thioether (R-S-R') Group
def calculate_num_thioether_groups(mol):

    if mol is None:
        return "Invalid SMILES string"
    thioether_pattern = Chem.MolFromSmarts('S')
    return len(mol.GetSubstructMatches(thioether_pattern))

# 22. Imidazole Ring (e.g., in Histidine)
def calculate_num_imidazole_rings(mol):

    if mol is None:
        return "Invalid SMILES string"
    imidazole_pattern = Chem.MolFromSmarts('n1cncc1')
    return len(mol.GetSubstructMatches(imidazole_pattern))

# 23. Carboxylate Ions (R-COO-)
def calculate_num_carboxylate_ions(mol):

    if mol is None:
        return "Invalid SMILES string"
    carboxylate_pattern = Chem.MolFromSmarts('C(=O)[O-]')
    return len(mol.GetSubstructMatches(carboxylate_pattern))


# 24. Number of Aromatic Rings
def calculate_num_aromatic_rings(mol):

    if mol is None:
        return "Invalid SMILES string"
    num_aromatic_rings = Chem.rdMolDescriptors.CalcNumAromaticRings(mol)
    return num_aromatic_rings

# 25. Presence of Halogens (F, Cl, Br, I)
def calculate_num_halogens(mol):

    if mol is None:
        return "Invalid SMILES string"
    halogen_smarts = Chem.MolFromSmarts('[F,Cl,Br,I]')
    num_halogens = len(mol.GetSubstructMatches(halogen_smarts))
    return num_halogens

# 26 Function to calculate the number of Carbonyl (C=O) groups
def calculate_num_carbonyl_groups(mol):
    if mol is None:
        return "Invalid SMILES string"
    
    # SMARTS pattern for Carbonyl (C=O) group
    carbonyl_pattern = Chem.MolFromSmarts('C=O')
    
    # Find and count matches for Carbonyl group
    return len(mol.GetSubstructMatches(carbonyl_pattern))


# 27 Function to calculate the number of Sulfonamide (-SO2NH2) groups
def calculate_num_sulfonamide_groups(mol):

    if mol is None:
        return "Invalid SMILES string"
    
    # SMARTS pattern for Sulfonamide (-SO2NH2) group
    sulfonamide_pattern = Chem.MolFromSmarts('S(=O)(=O)N')
    
    # Find and count matches for Sulfonamide group
    return len(mol.GetSubstructMatches(sulfonamide_pattern))


# 28 Function to calculate the number of Thiol (-SH) groups
def calculate_num_thiol_groups(mol):

    if mol is None:
        return "Invalid SMILES string"
    
    # SMARTS pattern for Thiol (-SH) group
    thiol_pattern = Chem.MolFromSmarts('[SH]')
    
    # Find and count matches for Thiol group
    return len(mol.GetSubstructMatches(thiol_pattern))

# 29 Function to calculate the number of Aromatic Nitro (-NO2) groups
def calculate_num_aromatic_nitro_groups(mol):

    if mol is None:
        return "Invalid SMILES string"
    
    # SMARTS pattern for Aromatic Nitro (-NO2) group
    aromatic_nitro_pattern = Chem.MolFromSmarts('[$([NX3](=O)=O)]c')
    
    # Find and count matches for Aromatic Nitro group
    return len(mol.GetSubstructMatches(aromatic_nitro_pattern))



# Updated function to compute a 'logP score' based on more rules
def logP_score(mol):
    # input: mol
    # output: int   
    # 分子量、TPSA、氢键供体、氢键受体、极化率、羟基、羧基、氨基、芳香环、卤素、醚基
    mw = calculate_molecular_weight(mol)
    tpsa = calculate_tpsa(mol)
    h_bond_donors = calculate_h_bond_donors(mol)
    h_bond_acceptors = calculate_h_bond_acceptors(mol)
    polarizability = calculate_polarizability(mol)
    hydroxyl_groups = calculate_num_hydroxyl_groups(mol)
    carboxyl_groups = calculate_num_carboxyl_groups(mol)
    amino_groups = calculate_num_amino_groups(mol)
    aromatic_rings = calculate_num_aromatic_rings(mol)
    halogens = calculate_num_halogens(mol)  
    ether_groups = calculate_num_ether_groups(mol)

    score = weight['logP'][0] * mw + weight['logP'][1] * tpsa + weight['logP'][2] * h_bond_donors + weight['logP'][3] * h_bond_acceptors + weight['logP'][4] * polarizability + weight['logP'][5] * hydroxyl_groups + weight['logP'][6] * carboxyl_groups + weight['logP'][7] * amino_groups + weight['logP'][8] * aromatic_rings + weight['logP'][9] * halogens + weight['logP'][10] * ether_groups
    
    return score

def qed_score(mol):

    mw = calculate_molecular_weight(mol)
    logp = calculate_logp(mol)
    tpsa = calculate_tpsa(mol)
    h_bond_donors = calculate_h_bond_donors(mol)
    h_bond_acceptors = calculate_h_bond_acceptors(mol)
    aromatic_rings = calculate_num_aromatic_rings(mol)
    rotatable_bonds = calculate_num_rotatable_bonds(mol)
    charge = calculate_charge(mol)
    stereochemistry = calculate_num_chiral_centers(mol)
    solubility = calculate_solubility(mol)
    sa = calculate_synthetic_accessibility(mol)


    score = weight['qed'][0] * mw + weight['qed'][1] * logp + weight['qed'][2] * tpsa + weight['qed'][3] * h_bond_donors + weight['qed'][4] * h_bond_acceptors + weight['qed'][5] * aromatic_rings + weight['qed'][6] * rotatable_bonds + weight['qed'][7] * charge + weight['qed'][8] * stereochemistry + weight['qed'][9] * solubility + weight['qed'][10] * sa
    return score


def TPSA_score(mol):
    hydroxyl_groups = calculate_num_hydroxyl_groups(mol)
    carboxyl_groups = calculate_num_carboxyl_groups(mol)
    amino_groups = calculate_num_amino_groups(mol)
    carbonyl_groups = calculate_num_carbonyl_groups(mol)
    sulfonamide_groups = calculate_num_sulfonamide_groups(mol)
    ether_groups = calculate_num_ether_groups(mol)
    thiol_groups = calculate_num_thiol_groups(mol)
    phosphate_groups = calculate_num_phosphate_groups(mol)
    amide_groups = calculate_num_amide_groups(mol)
    nitro_groups = calculate_num_nitro_groups(mol)  
    score = weight['TPSA'][0] * hydroxyl_groups + weight['TPSA'][1] * carboxyl_groups + weight['TPSA'][2] * amino_groups + weight['TPSA'][3] * carbonyl_groups + weight['TPSA'][4] * sulfonamide_groups + weight['TPSA'][5] * ether_groups + weight['TPSA'][6] * thiol_groups + weight['TPSA'][7] * phosphate_groups + weight['TPSA'][8] * amide_groups + weight['TPSA'][9] * nitro_groups
    return score

def HBA_score(mol):
    # input: mol
    # output: int
    hydroxyl_groups = calculate_num_hydroxyl_groups(mol)
    carbonyl_groups = calculate_num_carbonyl_groups(mol)
    amino_groups = calculate_num_amino_groups(mol)
    ether_groups = calculate_num_ether_groups(mol)
    phosphate_groups = calculate_num_phosphate_groups(mol)
    carboxylate_groups = calculate_num_carboxylate_ions(mol)
    nitrile_groups = calculate_num_nitrile_groups(mol)
    amide_groups = calculate_num_amide_groups(mol)
    thioether_groups = calculate_num_thioether_groups(mol)
    sulfonyl_groups = calculate_num_sulfonyl_groups(mol)    
    score = weight['HBA'][0] * hydroxyl_groups + weight['HBA'][1] * carbonyl_groups + weight['HBA'][2] * amino_groups + weight['HBA'][3] * ether_groups + weight['HBA'][4] * phosphate_groups + weight['HBA'][5] * carboxylate_groups + weight['HBA'][6] * nitrile_groups + weight['HBA'][7] * amide_groups + weight['HBA'][8] * thioether_groups + weight['HBA'][9] * sulfonyl_groups
    return score

def HBD_score(mol):
    carbonyl_groups = calculate_num_carbonyl_groups(mol)    
    nitro_groups = calculate_num_nitro_groups(mol)
    ether_groups = calculate_num_ether_groups(mol)
    amide_groups = calculate_num_amide_groups(mol)
    aromatic_nitro_compounds = calculate_num_aromatic_nitro_groups(mol)
    sulfonyl_groups = calculate_num_sulfonyl_groups(mol)
    phosphoryl_groups = calculate_num_phosphate_groups(mol)
    hydroxyl_groups = calculate_num_hydroxyl_groups(mol)
    imidazole_rings = calculate_num_imidazole_rings(mol)
    carboxylate_ions = calculate_num_carboxylate_ions(mol)      
    score = weight['HBD'][0] * carbonyl_groups + weight['HBD'][1] * nitro_groups + weight['HBD'][2] * ether_groups + weight['HBD'][3] * amide_groups + weight['HBD'][4] * aromatic_nitro_compounds + weight['HBD'][5] * sulfonyl_groups + weight['HBD'][6] * phosphoryl_groups + weight['HBD'][7] * hydroxyl_groups + weight['HBD'][8] * imidazole_rings + weight['HBD'][9] * carboxylate_ions
    return score



def mix_score(mol, task_id):
    if task_id in ['201', '202']:
        if task_id == '201':
            score = -logP_score(mol) + HBA_score(mol)
        else:
            score = logP_score(mol) + HBA_score(mol)
    elif task_id in ['203', '204']:
        if task_id == '203':
            score = -logP_score(mol) + HBD_score(mol)
        else:
            score = logP_score(mol) + HBD_score(mol) 
    elif task_id in ['205', '206']:
        if task_id == '205':
            score = - TPSA_score(mol) + logP_score(mol)
        else:
            score =  TPSA_score(mol) - logP_score(mol) 
    else:
        raise ValueError(f"未知的任务ID: {task_id}")
    return score


# 计算指定任务的分数
def calculate_scores(smiles_list, task_id):
    scores_dict = {}
    
    # 遍历每个SMILES字符串，计算分子并检查合法性
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # 跳过无效的分子
            continue
        # 根据任务ID选择合适的评分函数
        if task_id in ['101', '102']:
            score = logP_score(mol)  
        elif task_id in ['103', '104']:
            score = qed_score(mol)
        elif task_id in ['105', '106']:
            score = TPSA_score(mol)
        elif task_id == '107':
            score = HBA_score(mol)
        elif task_id == '108':
            score = HBD_score(mol)
        elif task_id in ['201', '202', '203', '204', '205', '206']:
            score = mix_score(mol, task_id)
        elif task_id == '301':
            score = qed_score(mol)
        elif task_id in ['109', '110']:
            score = TPSA_score(mol)
        else:
            raise ValueError(f"未知的任务ID: {task_id}")
        
        scores_dict[smiles] = score

    return scores_dict

def is_valid_molecule(smiles):
    props = ["MolLogP", "qed", "TPSA", "NumHAcceptors", "NumHDonors"]
    prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
    prop2func = {prop: func for prop, func in prop_pred}
    
    
    if smiles.count('CCCCCCCC') != 0:
        return False

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False  # 无效的 SMILES

    try:
        logP_test = prop2func['MolLogP'](mol)
    except:
        return False

    # 获取分子中的原子s
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

    # 如果分子全部由C或O组成，则过滤掉
    if all(atom == 'C' for atom in atoms) or all(atom == 'O' for atom in atoms):
        return False
    
    return True



# 筛选合法分子并计算分数的函数
def find_best_molecule(smiles_list, task_id):
    task_id = task_id.strip('x').strip('r').strip('z')
    if not smiles_list:
        return None  # 如果没有分子列表，直接返回None

    # 过滤合法且不是全由C或O组成的分子
    valid_molecules = [smiles for smiles in smiles_list if is_valid_molecule(smiles) ]
    if not valid_molecules:
        return None  # 如果没有合法分子，返回None
    

    # find the most frequency scaffold in the smiles_list:
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in valid_molecules]
    scaffold_list = [Chem.MurckoDecompose(mol) for mol in mol_list]
    # Count the occurrences of each scaffold
    from collections import Counter
    scaffold_counts = Counter(Chem.MolToSmiles(scaffold) for scaffold in scaffold_list)
    most_common_scaffold = scaffold_counts.most_common(1)[0][0]
    valid_molecules = [smiles for smiles, mol in zip(valid_molecules, mol_list) 
                       if Chem.MolToSmiles(Chem.MurckoDecompose(mol)) == most_common_scaffold]

    # 计算所有合法分子的评分
    scores = calculate_scores(valid_molecules, task_id)

    if not scores:
        return None  # 如果没有有效的分数，返回None
    
    # 确定每个任务对应的模式（最低分或最高分）
    mode_dict = {
        '101': 'lower', '102': 'higher', '103': 'higher', '104': 'lower',
        '105': 'lower', '106': 'higher', '107': 'higher', '108': 'higher',
        '201': 'higher', '202': 'higher', '203': 'higher', '204': 'higher',
        '205': 'higher', '206': 'higher', '301': 'higher', '109': 'higher',
        '110': 'lower'
    }
    
    mode = mode_dict.get(task_id, None)
    if mode is None:
        raise ValueError(f"未知的任务ID: {task_id}")
    
    # 根据模式找到最佳分子
    if mode == 'lower':
        best_smiles = min(scores, key=scores.get)
    else:  # mode == 'higher'
        best_smiles = max(scores, key=scores.get)

    return best_smiles