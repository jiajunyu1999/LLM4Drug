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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from rdkit import Chem
from rdkit.Chem import Descriptors

from para import *
from utils_new import *

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def load_model_pipeline(backbone, lora_path):
    model = AutoModelForCausalLM.from_pretrained(backbone_path[backbone], 
                                                 low_cpu_mem_usage=True, 
                                                 return_dict=True, 
                                                 torch_dtype=torch.float16, 
                                                 device_map='auto')
    if lora_path != '':
        lora_model = PeftModel.from_pretrained(model, lora_path)
        model = lora_model.merge_and_unload()
    
    tokenizer = AutoTokenizer.from_pretrained(backbone_path[backbone], trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=max_seq_length)

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
    def generate_property_prompts(task_id, prop_values):
        task_props = taskid_prop[task_id]

        temp = []
        for j, prop in enumerate(task_props):
            temp.append(f'|{prop}|{prop_values[prop]}')
        prompt = ', '.join(temp)

        return prompt
    
    def clean_generated_text(text):
        return text.replace('\n', '').replace('\t', '')

    def extract_smiles(text):
        return text.split('[START_SMILES]')[-1].split(',')[0].split('.')[0].strip(" ,.:\"")
    
    generated_molecules = []
    for idx in tqdm(range(df.shape[0])):
        prop_values = {prop: df[prop][idx] for prop in ["MolLogP", "qed", "TPSA", "NumHAcceptors", "NumHDonors"]}
        scaffold = df['scaffold'][idx]

        prop_prompt = generate_property_prompts(task_id, prop_values)

        question = prompt_template.format(scaffold, prop_prompt)
        answer = pipeline(question)[0]['generated_text']
        answer = clean_generated_text(answer)
        gene_molecule = extract_smiles(answer)

        generated_molecules.append(gene_molecule)
    print(generated_molecules)
    return generated_molecules

def evaluate_regression(df, task_id):
    task_id = task_id.strip('x')

    source_lens = df.shape[0]
    df = df[df['gene_MolLogP'] != -999].reset_index(drop=True)
    df = df[df['scaffold'] != ''].reset_index(drop=True)    
    valid_ratio = (df.shape[0] / source_lens) * 100

    prop = taskid_prop[task_id][0]
    df['mae'] = df.apply(lambda x: abs(x[f'gene_{prop}'] - x[f'{prop}']), axis = 1)
    df['mse'] = df.apply(lambda x: pow(x[f'gene_{prop}'] - x[f'{prop}'],2), axis = 1)
    df['same_scaffold'] = df.apply(lambda row: same_scaffold(row['gene_mol'], row['scaffold']), axis=1)
    df['similarity'] = df.apply(lambda row: similarity_mols(row['mol'], row['gene_mol']), axis=1)


    mae, mse = df['mae'].mean(), df['mse'].mean()
    same_mol_ratio = (df['gene_mol'] == df['mol']).mean()
    same_scaffold_ratio = (df['scaffold'] == df['gene_scaffold']).mean() * 100
    average_similarity = df[df['similarity'] != 0]['similarity'].mean()

    return valid_ratio, mae, mse, same_mol_ratio, same_scaffold_ratio, average_similarity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=str, default='101', help='Task ID')
    parser.add_argument('--backbone', type=str, default='galactica', help='Backbone model')
    parser.add_argument('--tune_model', type=str, default='combined_final', help='Tune model path')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--test_data', type=str, default='./base_data/test_chatdrug.csv', help='Test file path')
    parser.add_argument('--config_path', type=str, default='./config.json')
    parser.add_argument('--seq', type=str, default='seq')
    

    args = parser.parse_args()

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
    if args.seq == 'seq':
        prompt_template = config['inference_template_seq']
    else:
        prompt_template = config['inference_template_name']


    ## loading the pipeline
    df = pd.read_csv(args.test_data)
    if args.tune_model != '':
        lora_path = os.path.join('./new_model', f'{args.backbone}_{args.tune_model}')
    else:
        lora_path = ''
    pipeline = load_model_pipeline(args.backbone, lora_path)

    ## generate the molecule
    generated_molecules = generate_molecules(df, pipeline, prompt_template, task_id, prop_range, prop_trend)
    df['gene_mol'] = generated_molecules
    df = generate_mol_property(df)
    df.to_csv(f'./output/regression_{task_id}_{args.seq}.csv', index=False)

    ## evaluate
    valid_ratio, mae, mse, same_mol_ratio, same_scaffold_ratio, avg_similarity = evaluate_regression(df, task_id)
    with open('./submit_regression.log', 'a+') as log_file:
        log_file.write(f"{task_id},{args.tune_model},{valid_ratio:.4f},{mae:.4f},{mse:.4f},{same_mol_ratio:.4f},{same_scaffold_ratio:.4f},{avg_similarity:.4f}\n")

