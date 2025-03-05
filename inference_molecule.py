import copy
import json
import argparse
import random
import warnings
import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
from rdkit import Chem
from rdkit.Chem import Descriptors

from para import *
from utils_new import *
from get_best_mol import *

warnings.filterwarnings("ignore")

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def load_model_pipeline(backbone, lora_path):
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    # )
    if args.tune_model.count('full') != 0:
        model = AutoModelForCausalLM.from_pretrained( 
        lora_path, 
        # quantization_config = bnb_config,
        device_map = 'auto'
    )
        
    else:
        model = AutoModelForCausalLM.from_pretrained( 
            backbone_path[backbone], 
            # quantization_config = bnb_config,
            device_map = 'auto'
        )
        if lora_path:
            try:
                lora_model = PeftModel.from_pretrained(model, lora_path)
            except:
                model.resize_token_embeddings(50001)
                lora_model = PeftModel.from_pretrained(model, lora_path)
            model = lora_model.merge_and_unload()
    
    tokenizer = AutoTokenizer.from_pretrained(backbone_path[backbone], trust_remote_code=False, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=args.max_length)

def calculate_molecule_properties(smiles_list)->dict:
    '''
        input: 
            the list of lots of smiles: ['C','COC']
        output: 
            the dictionary of the molecule and its property:
                {
                    'mol': 'C', 'scaffold': 'C', 'MolLogP': xx.
                }
    '''

    def get_property_function_map(properties):
        descriptors = Descriptors.descList
        return {n: func for n, func in descriptors if any(prop in n for prop in properties)}

    def get_property_value(smiles, prop, prop_func_map):
        try:
            return prop_func_map[prop](Chem.MolFromSmiles(smiles))
        except:
            return -999

    def get_scaffold(smiles):
        mol = Chem.MolFromSmiles(smiles)
        try:
            scaffold = Chem.MurckoDecompose(mol)
            functional_groups = Chem.ReplaceCore(mol, scaffold)
            return Chem.MolToSmiles(scaffold), Chem.MolToSmiles(functional_groups)
        except:
            return '', ''
        
    properties = ["MolLogP", "qed", "TPSA", "NumHAcceptors", "NumHDonors"]
    prop_func_map = get_property_function_map(properties)
    results = []

    for smiles in smiles_list:
        mol_properties = {prop: get_property_value(smiles, prop, prop_func_map) for prop in properties}
        mol_properties["mol"] = smiles
        mol_properties["scaffold"], mol_properties["functional_groups"] = get_scaffold(smiles)
        if mol_properties['MolLogP'] != -999:
            results.append(mol_properties)
        
    return results

def get_alternative_smiles(smiles):
        """
        Convert a SMILES string to a different but equivalent SMILES representation
        
        Args:
            smiles (str): Input SMILES string
            
        Returns:
            str: Alternative SMILES representation, or original if conversion fails
        """
        try:
            # Convert SMILES to mol object
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
                
            # Generate a new SMILES string with different atom ordering
            new_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
            
            return new_smiles
        except:
            return smiles

def generate_molecules(df, pipeline, prompt_template, task_id, prop_ranges, prop_trends):
    '''
        input: 
            the dataframe format data: columns: ['mol', 'MolLogP' ... ]
            the prompt_template of inference: <s> Masked molecule [START_SMILES]{}[END_SMILES], property: {}. Completed molecule: [START_SMILES]
            the task id: 101...
            the prop_ranges: {  "MolLogP": [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  }
            the prop_trend: {  "101": "0", "102": "1", "103": "1", "104": "0"  }
        
        output:
            the best molecule that is filtered by rules.
    '''
    def generate_property_prompts(task_id, prop_values, prop_ranges, prop_trends, num_prompts):
        prompts = []
        task_props = taskid_prop[task_id]

        for i in range(num_prompts):
            temp = []

            for j, prop in enumerate(task_props):
                round_num = 4
                if prop in ['NumHDonors', 'NumHAcceptors']:
                    round_num = 0
                if i>len(prop_ranges[prop])-1:
                    prop_ranges_value = prop_ranges[prop][0] + random.uniform(0,0.1)
                else:
                    prop_ranges_value = prop_ranges[prop][i]
                if prop_trends[task_id][j] == "1":
                    desired_value = prop_values[prop] + prop_ranges_value + random.uniform(0,0.1)
                    if prop == 'qed' : desired_value = random.uniform(0.85, 1.99)
                    temp.append(f'|{prop}|{round(desired_value, round_num)}')
                else:
                    desired_value = prop_values[prop] - prop_ranges_value - random.uniform(0,0.1)
                    if desired_value <= 0:
                        if prop in ['NumHAcceptors', 'NumHDonors']: desired_value = 1.0 
                        elif prop in ['qed']: desired_value = 0.1 + random.uniform(0,0.1)
                        elif prop in ['TPSA']: desired_value = 10.0 + random.uniform(0,0.1)
                    temp.append(f'|{prop}|{round(desired_value, round_num)}')
            prompt = ', '.join(temp)
            prompt = prompt
            prompts.append(prompt)
        
        return prompts
    
    def clean_generated_text(text):
        return text.replace('\n', '').replace('\t', '')

    def extract_smiles(text):
        if args.tune_model.count('general') != 0:
            res = text.split('The recover molecule is [START_SMILES]')[-1].split(',')[0].split('.')[0].strip(" ,.:\"").strip('[END_SMILES]')
        else:
            res = text.split('[START_SMILES]')[-1].split(',')[0].split(' ')[0].split('.')[0].strip(" ,.:\"").strip('[END_SMILES]')
        return res
    
    generated_molecules = []
    df['drd2'] = 0.5
    for idx in tqdm(range(df.shape[0])):
        prop_values = {prop: df[prop][idx] for prop in ["MolLogP", "qed", "TPSA", "NumHAcceptors", "NumHDonors", "drd2"]}

        scaffold, mol = df['scaffold'][idx], df['mol'][idx]
        if scaffold in ['c1cscn1','c1ccccc1']:
            scaffold = df['mol'][idx]
        source_mol = df['mol'][idx]
        prop_prompts = generate_property_prompts(task_id, prop_values, prop_ranges, prop_trends, args.k)

        gene_mol_set = set()
        ans_set = set()

        mol = source_mol
        for i,prop_prompt in enumerate(prop_prompts):
            fix_seed(i)

            if args.task_id[0] != '3':
                question = prompt_template.format(scaffold, prop_prompt)
            else:
                question = prompt_template.format(scaffold)    
            # if args.tune_model.count('full')!=0:
            #     question = 'User: {}\nAssistant: '.format(question)
            answer = pipeline(question)[0]['generated_text']
            answer = clean_generated_text(answer)
            gene_molecule = extract_smiles(answer)
            scaffold_source = df['scaffold'][idx]
            scaffold_generate = get_scaffold(gene_molecule)[0]
            print('qs:', question,  'ans:', answer.replace(question,''), 'gene:', gene_molecule, 'scaffold', scaffold_generate, 'same',same_scaffold(gene_molecule, scaffold_source))
            ans_set.add(answer)
            if args.replace == 'iter':
                if scaffold_generate != '':
                    scaffold = scaffold_generate
                else:
                    scaffold = scaffold_source
            elif args.replace == 'iter2':
                if abs(len(gene_molecule) - len(source_mol)) <= 10:
                    scaffold = gene_molecule
                else:
                    scaffold = scaffold_source
            elif args.replace == 'alter':
                scaffold = get_alternative_smiles(scaffold_source)
                mol = get_alternative_smiles(mol)
            elif args.replace == 'scaffold':
                pass
            # if Chem.MolFromSmiles(gene_molecule) is not None and similarity_mols(gene_molecule, mol)!=1:
            gene_mol_set.add(gene_molecule)
            if len(list(gene_mol_set)) >= 30:
                break
        tune_model_name = args.tune_model.split('/')[-1]
        if args.task_id != '301':
            with open(f'./generated_new/{dataset_type}_{task_id}_{args.backbone}_{args.output}_{args.replace}.csv','a+')as f:
                f.write(f'{str(list(gene_mol_set))}\n')
        else:
            with open(f'./generated_new/drd2_{dataset_type}_{task_id}_{args.backbone}_{args.output}_{args.replace}.csv','a+')as f:
                f.write(f'{str(list(gene_mol_set))}\n')
        best_mol = find_best_molecule(list(gene_mol_set), task_id)
        generated_molecules.append(best_mol)
    return generated_molecules

def evaluate_molecule_predictions(df, task_id, prop_trends, prop_thresholds):
    '''
        input: 
            the dataframe format, ['mol', 'MolLogP', 'scaffold', 'gene_mol', 'gene_MolLogP', 'gene_scaffold']
            the task_id: which should be removed the suffix 'x'
        output:
            the 
    '''
    task_id = task_id.strip('x').strip('r').strip('z')

    def evaluate_hits(row, task_id, prop_trends, prop_thresholds):
        loose_hit = all((row[f'delta_{prop}'] < 0 if trend == '0' else row[f'delta_{prop}'] > 0) for prop, trend in zip(taskid_prop[task_id], prop_trends[task_id]))
        strict_hit = all((row[f'delta_{prop}'] < -prop_thresholds[prop] if trend == '0' else row[f'delta_{prop}'] > prop_thresholds[prop]) for prop, trend in zip(taskid_prop[task_id], prop_trends[task_id]))
        return loose_hit, strict_hit
    source_lens = df.shape[0]
    df = df[df['gene_MolLogP'] != -999].reset_index(drop=True)
    df = df[df['scaffold'] != ''].reset_index(drop=True)
    
    valid_ratio = (df.shape[0] / source_lens) * 100
    
    for prop in ["MolLogP", "qed", "TPSA", "NumHAcceptors", "NumHDonors"]:
        df[f'delta_{prop}'] = df[f'gene_{prop}'] - df[prop]
    
    hits = df.apply(lambda row: evaluate_hits(row, task_id, prop_trends, prop_thresholds), axis=1)
    
    df['loose_hit'], df['strict_hit'] = zip(*hits)
    df['same_scaffold'] = df.apply(lambda row: same_scaffold(row['gene_mol'], row['scaffold']), axis=1)
    df['similarity'] = df.apply(lambda row: similarity_mols(row['mol'], row['gene_mol']), axis=1)

    loose_hit_rate = df['loose_hit'].mean() * 100
    strict_hit_rate = df['strict_hit'].mean() * 100
    same_scaffold_ratio = df['same_scaffold'].mean() * 100
    average_similarity = df[df['similarity'] != 0]['similarity'].mean()

    print(f"\nLoose Accuracy: {loose_hit_rate:.2f}%")
    print(f"Strict Accuracy: {strict_hit_rate:.2f}%")
    print(f"Valid Ratio: {valid_ratio:.2f}%")
    print(f"Same Scaffold Ratio: {same_scaffold_ratio:.2f}%")
    print(f"Similarity: {average_similarity:.2f}")
    
    return loose_hit_rate, strict_hit_rate, valid_ratio, same_scaffold_ratio, average_similarity

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=str, default='101', help='Task ID')
    parser.add_argument('--backbone', type=str, default='galactica6.7b', help='Backbone model')
    parser.add_argument('--tune_model', type=str, default='base', help='Tune model path')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--k', type=int, default=16, help='Number of generated molecules')
    parser.add_argument('--file', type=str, default='103', help='File path')
    parser.add_argument('--test_data', type=str, default='./base_data/test_chatdrug.csv', help='Test file path')
    parser.add_argument('--config_path', type=str, default='./config.json')
    parser.add_argument('--replace', type=str, default='scaffold')
    parser.add_argument('--max_length', type=int, default=120)
    parser.add_argument('--output', type=str, default='6.7b_100')
    
    
    args = parser.parse_args()
    return args


from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
def similarity_mols(smiles1, smiles2):
    """
    Calculate the Tanimoto similarity between two molecules.

    Parameters:
        smiles1 (str): The SMILES representation of the first molecule.
        smiles2 (str): The SMILES representation of the second molecule.

    Returns:
        float: The Tanimoto similarity between the two molecules.
    """
    # try:
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048, useChirality=False)
    # fp1 = AllChem.GetMorganFingerprint(mol1, 2, useChirality=False)
    # fp2 = AllChem.GetMorganFingerprint(mol2, 2, useChirality=False)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def get_file_line(file):
    with open(file, "r") as file:
        lines = sum(1 for line in file)
    return lines

if __name__ == "__main__":
    
    args = get_args()
    fix_seed(args.seed)

    ## loading the config
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    task_id = args.task_id
    backbone_path = config['backbone_path']
    taskid_prop = config['taskid_prop']
    prop_range = config['prop_range']
    prop_trend = config['prop_trend']
    prop_threshold = config['prop_threshold']
    if args.task_id[0] == '1' or args.task_id[0] == '2':
        if args.tune_model.count('full') == 0:
            prompt_template = config['inference_template_molecule']
        else:
            prompt_template = config['inference_template_full']
    elif args.task_id == '301':
        prompt_template = config['inference_template_301']
    
    
    if args.test_data.count('chatdrug') != 0:
        dataset_type = 'base'
    elif args.test_data.count('drd2') != 0:
        dataset_type = 'drd2'
    else:
        dataset_type = 'new'

    ## loading the pipeline
    df = pd.read_csv(args.test_data)
    df = generate_mol_property(df, head = 'mol')
    if args.tune_model.count('full') == 0:
        lora_path = os.path.join('./tune_model', f'{args.backbone}_{args.tune_model}')
        # lora_path = args.tune_model
    else:
        lora_path = args.tune_model
    # lora_path = None
    pipeline = load_model_pipeline(args.backbone, lora_path)

    generated_molecules = generate_molecules(df, pipeline, prompt_template, task_id, prop_range, prop_trend)
    df['gene_mol'] = generated_molecules
    df = generate_mol_property(df, 'gene_mol')

    df.to_csv(f'./output/{dataset_type}_{args.backbone}_{args.output}_{task_id}_{args.k}_{args.replace}.csv', index=False)
    
    ## evaluate
    loose_hit, strict_hit, valid_ratio, same_scaffold_ratio, similarity = evaluate_molecule_predictions(df, task_id, prop_trend, prop_threshold)
    with open('/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/yujiajun-240108120114/llm4drug/output/log/base_submit.log', 'a+') as log_file:
        log_file.write(f"{args.replace},{args.test_data},{args.backbone},{args.tune_model},{task_id},{args.k},{loose_hit:.4f},{strict_hit:.4f},{valid_ratio:.4f},{same_scaffold_ratio:.4f},{similarity:.4f}\n")
