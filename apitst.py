from epynet import Network
import numpy as np
import pandas as pd
from scipy.optimize import minimize as neldermead
from deap import base
from deap import creator
from functools import reduce
from deap import tools
import math
from wdsEnv import wds
# -*- coding: utf-8 -*-
import argparse
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize as neldermead
from deap import base
from deap import creator
from deap import tools
from wdsEnv import wds

parser  = argparse.ArgumentParser()
parser.add_argument('--params', default='QDMaster', type=str, help="Name of the YAML file.")
parser.add_argument('--idscenes_start', default=0, type=int, help="Number of the scenes to generate.")
parser.add_argument('--idscenes_end', default=2, type=int, help="Number of the scenes to generate.")
parser.add_argument('--nscenes', default=192, type=int, help="Number of the scenes to generate.")
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
n_scenes    = args.idscenes_end - args.idscenes_start
# n_scenes    = args.nscenes

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

    plt.savefig("demand-head.png")
    
def savePoints(points : list, outputpath : str):
    with open(outputpath + ".txt", "w") as file:
        for item in points:
                line = str(item).strip('[').strip(']').replace(',', ' ')
                file.write(line)
    file.close()
    
points = []
count = 0
dissatcount = 0
rewards = []

logger = logging.getLogger("logwds")
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("results/loghistory.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info(str.format("Start print log"))
for scene_id in range(args.idscenes_start, args.idscenes_end):
    print(str.format("scene_id {0}", scene_id))
    logger.info(str.format("scene_id {0}", scene_id))
    env.apply_scene(scene_id)
    print(env.get_pump_speeds())
    env.wds.solve(scene_id)
    logger.info(env.wds.reservoirs.head)
    
    # print("2 bef head: ", env.get_point_head("1100325-A"))
    # print("2 aft head: ", env.get_point_head("1100325-B"))
    # print("4 bef head: ", env.get_point_head("1100323-A"))
    # print("4 aft head: ", env.get_point_head("1100323-B"))
    # print("9 bef head: ", env.get_point_head("1100778-A"))
    # print("9 aft head: ", env.get_point_head("1100778-B"))
    # print("XFX head: ", env.get_point_pressure("XFX-OP"))
    # logger.info(str.format("XFX-OP head: {0}", env.get_point_pressure("XFX-OP")))
    # logger.info(str.format("XFX-P3 leverage: {0}", env.get_point_pressure("XFX-OP") - env.get_point_pressure("XFX-node9")))
    # logger.info(str.format("XFX-P5 leverage: {0}", env.get_point_pressure("XFX-OP") + 5.65 - env.wds.reservoirs["R00008"].head))
    # print("HX head: ", env.get_point_pressure("HX-node12"))   
    # logger.info(str.format("HX-node12 head: {0}", env.get_point_pressure("HX-node12")))
    # logger.info(str.format("HX-P3 leverage: {0}", env.get_point_pressure("HX-node14") - env.get_point_pressure("HX-node8")))
    # logger.info(str.format("HX-P5 leverage: {0}", env.get_point_pressure("HX-node12") + 3.75 - env.wds.reservoirs["R00009"].head))
    logger.info(str.format("P1 Flow: {0}", env.wds.pumps["XJ-P2"].flow))   
    logger.info(str.format("P1 Flow: {0}", env.wds.pumps["XJ-P4"].flow))   
    logger.info(str.format("P1 Flow: {0}", env.wds.pumps["XJ-P9"].flow))   
    logger.info(str.format("R1-P leverage: {0}", env.get_point_pressure("XJ-node43") + 3.75 - env.wds.reservoirs["XJ-R1"].head))
    logger.info(str.format("R2-P leverage: {0}", env.get_point_pressure("XJ-node43") + 3.75 - env.wds.reservoirs["XJ-R2"].head))
    # print("head of neg dmd point: ", env.get_point_head("J-HCXZ01Z_P"))
    # print("highest demand: ", max(env.wds.junctions.basedemand))
    # print("lowest demand: ", min(env.wds.junctions.basedemand))
    line = []
    line.append(env.totDmd)
    print("tot demand: ", line[-1])
    logger.info(str.format("tot demand: {0}", line[-1]))
    # print("net demand: ", sum(map(lambda x : x if x > 0 else 0, env.wds.junctions.basedemand)))
    # print("net inflow: ", abs(sum(map(lambda x : x if x < 0 else 0, env.wds.junctions.basedemand))))
    # line.append(env.get_point_head("XFX-OP"))
    line.append(env.get_point_pressure(env.controlPoint) if env.get_point_pressure(env.controlPoint) > -50 else -50)
    rewards.append(env.get_state_value())
    print("\033[40m", str.format("Reward is {0}", rewards[-1]), "\033[0m")
    logger.info(str.format("Reward is {0}", rewards[-1]))
    if (line[-1] < 16): 
        print("\033[31m", str.format("Ctr head: {0}", line[-1]), "\033[0m")
        logger.error(str.format("Ctr head: {0}", line[-1]))
        dissatcount += 1
    else: 
        print("\033[40m", str.format("Ctr head: {0}", line[-1]), "\033[0m")
        logger.info(str.format("Ctr head: {0}", line[-1]))
    points.append(line)
    count += 1
    print(str.format("----------{0} DONE----------", count / n_scenes))
print(str.format("Generation done, total scneces: {0}; dissatsfied count: {1}", n_scenes, dissatcount))
logger.info(str.format("Generation done, total scneces: {0}; dissatsfied count: {1}", n_scenes, dissatcount))
print(str.format("avg score: {0} max score: {1} min score: {2}", sum(rewards) / n_scenes, max(rewards), min(rewards)))
logger.info(str.format("avg score: {0} max score: {1} min score: {2}", sum(rewards) / n_scenes, max(rewards), min(rewards)))
logger.info("Finish")
savePoints(points, "results/ctrhead-demand")
savePoints(rewards, "results/rewards")
plot2Dline(points)
plot1Dline(list(np.array(points)[:, 1]))

