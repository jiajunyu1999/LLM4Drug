import os 
from tqdm import *
import pandas as pd
from utils_new import *
import json
import argparse
import random
import copy
import numpy as np

def generate_property_prompts(prop_values):
    # prop_values = {prop: prop_values[prop] for prop in list(set(["MolLogP", "qed", "TPSA", "NumHAcceptors", "NumHDonors", "drd2"]) & set(prop_values.keys())) }
    prop_values = {prop: prop_values[prop] for prop in ["MolLogP", "qed", "TPSA", "NumHAcceptors", "NumHDonors"]}
    
    # Generate specific multi-property combinations
    if args.task.count('301') == 0:
        single_prompts = [f'|{prop}|{prop_values[prop]}' for prop in prop_values.keys()]
        multi_prompts = [
            f'|MolLogP|{prop_values["MolLogP"]}, |NumHAcceptors|{prop_values["NumHAcceptors"]}',
            f'|MolLogP|{prop_values["MolLogP"]}, |NumHDonors|{prop_values["NumHDonors"]}', 
            f'|MolLogP|{prop_values["MolLogP"]}, |TPSA|{prop_values["TPSA"]}'
        ]
    elif args.task.count('301') != 0:
        single_prompts = [f'|{prop}|{prop_values[prop]}' for prop in ['qed', 'drd2']]
        multi_prompts = [
            f'|qed|{prop_values["qed"]}, |drd2|{prop_values["drd2"]}'
        ]
    prompts = single_prompts + multi_prompts
    prompts = [multi_prompts[2]]
    return prompts

import json
import pandas as pd

def convert_to_jsonl(df, save_path):
    jsonl_data = []
    for idx, row in df.iterrows():
        # Split the string to get input SMILES and output SMILES
        text = row[0]  # Assuming the text is in the first column
        parts = text.split('Completed SMILES:')
        
        question = parts[0] 
        answer = 'Completed SMILES:'+parts[1]
        
        # Create conversation format
        conversation_dict = {
            "conversation_id": str(idx + 1),
            "category": "chatgpt",
            "conversation": [
                {
                    "human": question,
                    "assistant": answer
                }
            ]
        }
        jsonl_data.append(conversation_dict)
    
    # Write to jsonl file
    save_path = save_path.replace('.csv', '')
    with open(f'{save_path}.jsonl', 'a+', encoding='utf-8') as f:
        for item in jsonl_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return jsonl_data



def get_save_path():
    file_version = 0
    while os.path.exists(f'./tune_data/{args.task}_v{file_version}.csv'):
        file_version += 1
    save_path = f'./tune_data/{args.task}_v{file_version}.csv'
    return save_path

def get_df_data(args):
    print(args)
    ## data processing:
    df = pd.read_csv(f'{args.input}')
    if 'Ligand' in df.columns:
        df['mol'] = df['Ligand']

    if 'Protein' in df.columns:
        df['amino_seq'] = df['Protein']
    if 'regression_label' in df.columns:
        df['binding'] = df['regression_label']
    if 'Protein_Name' in df.columns:
        df['amino_name'] = df['Protein_Name']

    # if 'qed' not in df.columns:
    #     df = generate_mol_property(df, head = 'mol')
    #     df.to_csv(args.input)
    
    ## optimize the molecue
    # df = df[df['scaffold'] != 'c1ccccc1']
    # df['drop'] = df.apply(lambda x: 1 if x['scaffold'] == x['mol'] else 0, axis = 1)
    # df = df[df['drop'] != 1].reset_index(drop=True)
    # df = df.drop('drop', axis=1)
    
    # Sort by SMILES length and sample evenly across the range
    # df['mol_length'] = df['mol'].str.len()
    # df = df.sort_values('mol_length')
    
    # Calculate indices for even sampling
    n_samples = args.samples
    total_rows = len(df)
    indices = np.linspace(0, total_rows-1, n_samples, dtype=int)
    
    # Sample rows evenly
    df = df.iloc[indices].reset_index(drop=True)
    # df = df.drop('mol_length', axis=1)
    # df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def generate_conversation_str(idx, source_smiles, target_smiles, property_prompt):
    conversation = {
        "conversation_id": f"{idx}",
        "category": "chatgpt", 
        "conversation": [
            {
                "human": f"Please recover this masked molecule [START_SMILES]{source_smiles}[END_SMILES] the desired properties is {property_prompt}",
                "assistant": f"The recover molecule is [START_SMILES]{target_smiles}[END_SMILES]."
            }
        ]
    }
    return conversation


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='base')
    parser.add_argument('--config_path', type=str, default='./config.json')
    parser.add_argument('--input', type=str, default='./test.csv')
    parser.add_argument('--samples', type=int, default=1000000)
    
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = get_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    save_path = get_save_path()
    df = get_df_data(args)

    if args.task == 'base':
        ## building base property editing task
        template = config['tune_template_molecule']
        instruct_prompts = []
        for i in tqdm(range(df.shape[0])):
            item = df.iloc[i]
            prop_prompts = generate_property_prompts(item)
            prop_prompts = [template.format(item['scaffold'], prop_prompt, item['mol']) for prop_prompt in prop_prompts]
            instruct_prompts.extend(prop_prompts)
        random.shuffle(instruct_prompts)
        instruct_prompts = pd.DataFrame({'prompt': instruct_prompts})
        file_version = 0 
        # instruct_prompts = instruct_prompts.sample(frac=1, random_state=42).reset_index(drop=True)
        instruct_prompts.to_csv(save_path)
        instruct_prompts_jsonl = convert_to_jsonl(instruct_prompts, save_path)

    elif args.task == 'base_301':
        ## building base property editing task
        template = config['tune_template_molecule']
        instruct_prompts = []
        for i in tqdm(range(df.shape[0])):
            item = df.iloc[i]
            prop_prompts = generate_property_prompts(item)
            prop_prompts = [template.format(item['scaffold'], prop_prompt, item['mol']) for prop_prompt in prop_prompts]
            instruct_prompts.extend(prop_prompts)
        random.shuffle(instruct_prompts)
        instruct_prompts = pd.DataFrame({'prompt': instruct_prompts})
        file_version = 0 
        instruct_prompts.to_csv(save_path)
        # instruct_prompts_jsonl = convert_to_jsonl(instruct_prompts, save_path)



    elif args.task == 'base_general':
        ## building base property editing task
        template = config['tune_template_molecule_general']
        instruct_prompts = []
        for i in tqdm(range(df.shape[0])):
            item = df.iloc[i]
            prop_prompts = generate_property_prompts(item)
            prop_prompts = [template.format(item['scaffold'], prop_prompt, item['mol']) for prop_prompt in prop_prompts]
            instruct_prompts.extend(prop_prompts)
        # random.shuffle(instruct_prompts)
        instruct_prompts = pd.DataFrame({'prompt': instruct_prompts})
        file_version = 0 
        instruct_prompts.to_csv(save_path)

    elif args.task == 'amino_seq':
        ## building the amino property editing task
        template = config['tune_template_seq']
        instruct_prompts = []
        for i in tqdm(range(df.shape[0])):
            item = df.iloc[i]
            prop_prompts = generate_property_prompts(item)
            prop_prompts = [template.format(item['scaffold'], prop_prompt, item['amino_seq'],item['binding'], item['mol']) for prop_prompt in prop_prompts]
            instruct_prompts.extend(prop_prompts)
        random.shuffle(instruct_prompts)
        instruct_prompts = pd.DataFrame({'prompt': instruct_prompts})
        instruct_prompts.to_csv(save_path)

    elif args.task == '301':
        ## building the amino property editing task
        template = config['tune_template_molecule_301']
        instruct_prompts = []
        for i in tqdm(range(df.shape[0])):
            item = df.iloc[i]
            instruct_prompts.append(template.format(item['src_smi'], item['tgt_smi']))
        random.shuffle(instruct_prompts)
        instruct_prompts = pd.DataFrame({'prompt': instruct_prompts})
        instruct_prompts.to_csv(save_path)
        instruct_prompts_jsonl = convert_to_jsonl(instruct_prompts, save_path)
        
        
    elif args.task == 'amino_seq_only':
        ## building the amino property editing task
        template = config['tune_template_seqonly']
        instruct_prompts = []
        for i in tqdm(range(df.shape[0])):
            item = df.iloc[i]
            instruct_prompts.append(template.format(item['scaffold'], item['amino_seq'],item['binding'], item['mol']))
        random.shuffle(instruct_prompts)
        instruct_prompts = pd.DataFrame({'prompt': instruct_prompts})
        instruct_prompts.to_csv(save_path)
    
    elif args.task == 'amino_name':

        ## building the amino property editing task
        template = config['tune_template_name']
        instruct_prompts = []
        for i in tqdm(range(df.shape[0])):
            item = df.iloc[i]
            prop_prompts = generate_property_prompts(item)
            prop_prompts = [template.format(item['scaffold'], prop_prompt, item['amino_name'],item['binding'], item['mol']) for prop_prompt in prop_prompts]
            instruct_prompts.extend(prop_prompts)
        random.shuffle(instruct_prompts)
        instruct_prompts = pd.DataFrame({'prompt': instruct_prompts})
        instruct_prompts.to_csv(save_path)

    elif args.task == 'amino_name_only':

        ## building the amino property editing task
        template = config['tune_template_nameonly']
        instruct_prompts = []
        for i in tqdm(range(df.shape[0])):
            item = df.iloc[i]
            instruct_prompts.append(template.format(item['scaffold'], item['amino_name'],item['binding'], item['mol']))
        random.shuffle(instruct_prompts)
        instruct_prompts = pd.DataFrame({'prompt': instruct_prompts})
        instruct_prompts.to_csv(save_path)

    elif args.task == 'molecule_firefly':
        
        s = 0
        instruct_prompts = []
        for i in tqdm(range(df.shape[0])):
            item = df.iloc[i]
            prop_prompts = generate_property_prompts(item)
            for j, prop_prompt in enumerate(prop_prompts):
                prompt = generate_conversation_str(s, item['scaffold'], item['mol'], prop_prompt)
                s+=1
                instruct_prompts.append(prompt)
        random.shuffle(instruct_prompts)
        with open(save_path.replace('csv', 'jsonl'), 'a+') as f:
            for i in instruct_prompts:
                json.dump(i, f)
                f.write('\n')



