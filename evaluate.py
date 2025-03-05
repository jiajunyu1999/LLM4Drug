import os
from tqdm import *
import json
import argparse
import random
import pandas as pd
import numpy as np
import torch
from utils_new import *

def evaluate_molecule_predictions(df, task_id, prop_trends, prop_thresholds):
    '''
        input: 
            the dataframe format, ['mol', 'MolLogP', 'scaffold', 'gene_mol', 'gene_MolLogP', 'gene_scaffold']
            the task_id: which should be removed the suffix 'x'
        output:
            the 
    '''
    task_id = task_id.strip('x').strip('r')

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

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=str, default='101', help='Task ID')
parser.add_argument('--backbone', type=str, default='galactica', help='Backbone model')
parser.add_argument('--tune_model', type=str, default='combined_final', help='Tune model path')
parser.add_argument('--seed', type=int, default=2024, help='Random seed')
parser.add_argument('--k', type=int, default=10, help='Number of generated molecules')
parser.add_argument('--file', type=str, default='103', help='File path')
parser.add_argument('--test_data', type=str, default='./base_data/test_chatdrug.csv', help='Test file path')
parser.add_argument('--config_path', type=str, default='./config.json')

args = parser.parse_args()

fix_seed(args.seed)

## loading the config
with open('./config.json', 'r') as f:
    config = json.load(f)
task_id = args.task_id
backbone_path = config['backbone_path']
taskid_prop = config['taskid_prop']
prop_range = config['prop_range']
prop_trend = config['prop_trend']
prop_threshold = config['prop_threshold']
prompt_template = config['inference_template_molecule']
root_path = './output'
# for file in tqdm(sorted(os.listdir(root_path))):
#     # if file.count('base') == 1 :
#         path = os.path.join(root_path, file)
#         # if file[3] == 'r': model_name = 'random'
#         # elif file[3] == 'x': model_name = 'select'
#         # elif file[3] == '_': model_name = 'base'
task_id = '101'
df = pd.read_csv('/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/yujiajun-240108120114/llm4drug/test_chatdrug.csv')
df['gene_mol'] = df['best']
if 'gene_MolLogP' not in df.columns:
    df = generate_mol_property(df, 'gene_mol')
loose_hr, strict_hr, vr, ssr, sim = evaluate_molecule_predictions(df, task_id, prop_trend, prop_threshold)
print(loose_hr, strict_hr, vr, ssr, sim )
# with open('./res_single.csv', 'a+')as f:
#     f.write('{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n'.format(file.split('_')[1], task_id, loose_hr, strict_hr, vr, ssr, sim))
