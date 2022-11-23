from epynet import Network
import numpy as np
import pandas as pd
from scipy.optimize import minimize as neldermead
from deap import base
from deap import creator
from functools import reduce
from deap import tools
from wdsEnv import wds
# -*- coding: utf-8 -*-
import argparse
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np
import pandas as pd
from scipy.optimize import minimize as neldermead
from deap import base
from deap import creator
from deap import tools
from wdsEnv import wds

parser  = argparse.ArgumentParser()
parser.add_argument('--params', default='QDMaster', type=str, help="Name of the YAML file.")
parser.add_argument('--nscenes', default=1440, type=int, help="Number of the scenes to generate.")
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

def plot1Dline(points : list):
    fig = plt.figure()
    ax = fig.add_subplot()
    points1D = np.array(points)
    ax.plot(points1D, 'bo-', mfc = 'red', mec = 'red', ms = 2, linewidth = 1, label='line')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')

    plt.savefig("ctr_head.png")
    plt.show()

points = []

count = 1

for scene_id in range(0, 10):
    env.apply_scene(scene_id)
    env.wds.solve()
    # print(env.wds.solved)
    # print(env.pump_speeds)
    print("ctr head: ", env.get_point_head(env.controlPoint))
    print("highest demand: ", max(env.wds.junctions.basedemand))
    print("lowest demand: ", min(env.wds.junctions.basedemand))
    # points.append(env.get_state_value())
    count += 1
    # print(count / n_scenes, " ", points[-1])

# plot1Dline(headPoints)