from src.run_exp import exp_centralized,  exp_centralized_for
from src.solver import QUBO_solver
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/Partitioning_new.json',
                     help='Path to the JSON config file')
args = parser.parse_args()

with open(args.config) as f:
   params = json.load(f)
exp_centralized(params)




