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

def plot2Dline(points : list):
    fig = plt.figure()
    ax = fig.add_subplot()
    points2d = np.array(points)
    x = points2d[:, 0]
    y = points2d[:, 1]
    # ax.plot(x, y, 'bo-', mfc = 'red', mec = 'red', ms = 5, linewidth = 3, label='line')
    ax.scatter(x, y, 
        marker = 'o', 
        linewidths = 1, c = "green")
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')

    plt.savefig("ctr_head.png")
    plt.show()
    
points = []
count = 1

for scene_id in range(0, 100):
    env.apply_scene(scene_id)
    env.wds.solve()
    print(env.pump_speeds)
    # # print(env.wds.solved)
    print(env.wds.reservoirs.head)
    print(env.wds.tanks.head)
    # print("2 bef head: ", env.get_point_head("1100325-A"))
    # print("2 aft head: ", env.get_point_head("1100325-B"))
    # print("4 bef head: ", env.get_point_head("1100323-A"))
    # print("4 aft head: ", env.get_point_head("1100323-B"))
    # print("9 bef head: ", env.get_point_head("1100778-A"))
    # print("9 aft head: ", env.get_point_head("1100778-B"))
    print("HX bef head: ", env.get_point_head("HX-node7"))
    print("HX aft head: ", env.get_point_head("HX-node13"))   
    # print("head of neg dmd point: ", env.get_point_head("J-HCXZ01Z_P"))
    # print("highest demand: ", max(env.wds.junctions.basedemand))
    # print("lowest demand: ", min(env.wds.junctions.basedemand))
    line = []
    line.append(sum(map(lambda x : x if x > 0 else 0, env.wds.junctions.basedemand)))
    print("tot demand: ", line[-1])
    # print("net demand: ", sum(map(lambda x : x if x > 0 else 0, env.wds.junctions.basedemand)))
    # print("net inflow: ", abs(sum(map(lambda x : x if x < 0 else 0, env.wds.junctions.basedemand))))
    line.append(env.get_point_head("XFX-OP"))
    print("XFX head: ", line[-1])
    points.append(line)
    count += 1
    # print(count / n_scenes, " ", points[-1])

plot2Dline(points)