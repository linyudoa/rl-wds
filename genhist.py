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
import pump_efficiency_cal
from scipy.optimize import minimize as neldermead
from deap import base
from deap import creator
from deap import tools
from wdsEnv import wds

parser  = argparse.ArgumentParser()
parser.add_argument('--params', default='QDMaster', type=str, help="Name of the YAML file.")
parser.add_argument('--idscenes_start', default=1152, type=int, help="Number of the scenes to generate.")
parser.add_argument('--idscenes_end', default=1440, type=int, help="Number of the scenes to generate.")
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
                line = str(item).strip('[').strip(']') + "\r"
                file.write(line)
    file.close()
    
def extractSpeed():
    inpLines = []
    pathToInp = "results\logspeed.txt"
    fileHandler = open(pathToInp, "r", encoding='latin1')
    inpLines = fileHandler.readlines()
    mp = {}
    index = 0
    for line in inpLines:
        line = line.strip()
        lineItems = line.strip().split()
        vals = lineItems[3:]
        for val in vals:
            if (index in mp.keys()):
                mp[index].append(val)
            else:
                mp[index] = [val]
        index += 1
    return mp

speedMp = extractSpeed()

flows = []
count = 0
dissatcount = 0
rewards = []
powers = []
outPressure = []
ctrPressure = []
freq1 = []
freq2 = []
freq3 = []

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
    env.apply_pumpStatusSnapshot_v1(speedMp[scene_id], scene_id)
    print(env.get_pump_speeds())
    freq1.append(env.get_pump_speeds()[0])
    freq2.append(env.get_pump_speeds()[1])
    freq3.append(env.get_pump_speeds()[2])
    # env.wds.solve(scene_id)
    logger.info(env.wds.reservoirs.head)
    logger.info(str.format("{0}, {1}, {2}", env.wds.pumps["XJ-P2"].flow, env.wds.pumps["XJ-P4"].flow, env.wds.pumps["XJ-P9"].flow)) 
    logger.info(str.format("R1-P leverage: {0}", env.get_point_pressure("XJ-node43") + 3.75 - env.wds.reservoirs["XJ-R1"].head))
    logger.info(str.format("R2-P leverage: {0}", env.get_point_pressure("XJ-node43") + 3.75 - env.wds.reservoirs["XJ-R2"].head))
    line = []
    line.append(env.totDmd)
    rewards.append(env.get_state_value())
    ctrPressure.append(env.get_point_pressure(env.controlPoint))
    logger.info(str.format("tot demand: {0}", line[-1]))
    line.append(env.get_point_pressure(env.controlPoint) if env.get_point_pressure(env.controlPoint) > -50 else -50)
    powers.append(pump_efficiency_cal.pump_efficiency_cal("xj2", env.wds.pumps["XJ-P2"].flow, 
        env.get_point_pressure("XJ-node43") + 3.75 - env.wds.reservoirs["XJ-R1"].head)[2] + 
        pump_efficiency_cal.pump_efficiency_cal("xj4", env.wds.pumps["XJ-P4"].flow, 
        env.get_point_pressure("XJ-node43") + 3.75 - env.wds.reservoirs["XJ-R1"].head)[2]+
        pump_efficiency_cal.pump_efficiency_cal("xj9", env.wds.pumps["XJ-P9"].flow, 
        env.get_point_pressure("XJ-node43") + 3.75 - env.wds.reservoirs["XJ-R2"].head)[2])
    outPressure.append(env.get_point_pressure("XJ-node43") + 3.75)
    flows.append(env.wds.pumps["XJ-P2"].flow+env.wds.pumps["XJ-P4"].flow+env.wds.pumps["XJ-P9"].flow)
    count += 1
# savePoints(powers, "histlog/powers")
# savePoints(flows, "histlog/flows")
# savePoints(rewards, "histlog/rewards")
# savePoints(outPressure, "histlog/outPressure")
# savePoints(ctrPressure, "histlog/ctrPressure")
# savePoints(powers, "optilog/powers")
# savePoints(flows, "optilog/flows")
# savePoints(rewards, "optilog/rewards")
savePoints(freq1, "optilog/freq1")
savePoints(freq2, "optilog/freq2")
savePoints(freq3, "optilog/freq3")
# savePoints(outPressure, "optilog/outPressure")
# savePoints(ctrPressure, "optilog/ctrPressure")

