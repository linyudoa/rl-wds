from epynet import Network
import numpy as np
import pandas as pd
from scipy.optimize import minimize as neldermead
from deap import base
from deap import creator
from functools import reduce
from deap import tools
from wdsEnv import wds
import wntr 
import time
# -*- coding: utf-8 -*-
import argparse
import os
import yaml
import numpy as np
import pandas as pd
from scipy.optimize import minimize as neldermead
from deap import base
from deap import creator
from deap import tools
from wdsEnv import wds

# pathToWds = "water_networks\QDWDM_1114a.inp"

# headMaskKeys = {"HUAYI", "J40379", "J40543",
#                     "JIAHUA", "JIXI", "J49895",
#                     "J59970", "J110064", "J107956",
#                     "J79998", "HUAYU", "J111568",
#                     "J13356", "J82123", "XULE",
#                     "J95051", "JINSHUI", "J56841",
#                     "PANZHONG", "J77098", "HUAQING",
#                     "ZHUGUANG", "J54945", "J101196"}

# headDict = {}
# n_scenes = 1440
# wn = wntr.network.WaterNetworkModel(pathToWds) 
# wn.options.hydraulic.demand_model = 'DD'
# wn.options.time.hydraulic_timestep = 300
# wn.options.time.duration = n_scenes * 300
# sim = wntr.sim.EpanetSimulator(wn)

# time_start = time.time()
# results = sim.run_sim()
# pressure = results.node['demand']
# # for key in headMaskKeys:
# #     pressure_at_scene = pressure.loc[0, key]
# #     print("Head for key ", key , " is ", pressure_at_scene)
# time_end = time.time()
# print("time cost: ", time_end - time_start, "s")
# print("avg time: ", (time_end - time_start) / n_scenes, " for each scene")
parser  = argparse.ArgumentParser()
parser.add_argument('--params', default='QDMaster', type=str, help="Name of the YAML file.")
parser.add_argument('--nscenes', default=100, type=int, help="Number of the scenes to generate.")
parser.add_argument('--seed', default=None, type=int, help="Random seed for the optimization methods.")
parser.add_argument('--dbname', default=None, type=str, help="Name of the generated database.")
parser.add_argument('--nproc', default=1, type=int, help="Number of processes to raise.")
args    = parser.parse_args()
pathToRoot      = os.path.dirname(os.path.realpath(__file__))
pathToParams    = os.path.join(
                    pathToRoot,
                    'experiments',
                    'hyperparameters',
                    args.params+'.yaml')
with open(pathToParams, 'r') as fin:
    hparams = yaml.load(fin, Loader=yaml.Loader)

reset_orig_demands  = hparams['env']['resetOrigDemands']
test_db_name        = hparams['evaluation']['dbName']
wds_name            = hparams['env']['waterNet']+'_master'
if args.dbname:
    db_name = args.dbname
else:
    db_name     = test_db_name+'_db'
n_scenes    = args.nscenes
seed        = args.seed
n_proc      = args.nproc

env = wds(
        wds_name        = hparams['env']['waterNet']+'_master',
        speed_increment = hparams['env']['speedIncrement'],
        episode_len     = hparams['env']['episodeLen'],
        pump_group     = hparams['env']['pumpGroups'],
        total_demand_lo = hparams['env']['totalDemandLo'],
        total_demand_hi = hparams['env']['totalDemandHi'],
        seed            = args.seed
)

for scene_id in range(n_scenes):
    env.apply_scene(scene_id)
    env.wds.solve()
    print(env.get_junction_heads())
    print(env.get_state_value())