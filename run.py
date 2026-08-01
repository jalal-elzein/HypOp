from src.run_exp import exp_centralized,  exp_centralized_for
from src.solver import QUBO_solver
import argparse
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/Partitioning_new.json',
                     help='Path to the JSON config file')
args = parser.parse_args()

with open(args.config) as f:
   params = json.load(f)

##### optional env-var overrides for batch evaluation sweeps (see run_evaluate.sh) #####
##### lets one base template config be reused across many benchmark instance      #####
##### folders instead of hand-authoring one JSON file per folder. No-ops (params   #####
##### come only from --config) unless these are explicitly set.                   #####
if os.environ.get('HYPOP_FOLDER_PATH'):
    params['folder_path'] = os.environ['HYPOP_FOLDER_PATH']
if os.environ.get('HYPOP_K'):
    params['K'] = int(os.environ['HYPOP_K'])
if os.environ.get('HYPOP_EPOCH'):
    params['epoch'] = int(os.environ['HYPOP_EPOCH'])
if os.environ.get('HYPOP_DIFFICULTY_PARAM'):
    params['difficulty_param'] = os.environ['HYPOP_DIFFICULTY_PARAM']
if os.environ.get('HYPOP_WANDB_ENABLED'):
    params['wandb_enabled'] = os.environ['HYPOP_WANDB_ENABLED'].lower() in ('1', 'true', 'yes')

exp_centralized(params)




