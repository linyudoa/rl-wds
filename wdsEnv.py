# -* coding: utf-8 -*-
import os
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import gym.spaces
import random
from scipy.interpolate import interp1d
from MyParser import MyParser
from functools import reduce
from pipe import select, where
from epynet import Network
from opti_algorithms import nm, rs
import math

class wds():
    """Gym-like environment for water distribution systems."""
    def __init__(self,
            wds_name        = 'anytown_master',
            speed_increment = .05,
            episode_len     = 10,
            pump_group     = [['78', '79']],
            total_demand_lo = .3,
            total_demand_hi = 1.1,
            reset_orig_pump_speeds  = False,
            reset_orig_demands      = False,
            seed    = None):

        self.seedNum   = seed
        if self.seedNum:
            np.random.seed(self.seedNum)
        else:
            np.random.seed()
    ## setting path to .inp of waternetwork
        self.pathToRoot  = os.path.dirname(os.path.realpath(__file__))
        self.wds_name = wds_name
        self.pathToWDS   = os.path.join(self.pathToRoot, 'water_networks', wds_name + '.inp')
        self.pathToTankLevel   = os.path.join(self.pathToRoot, 'water_networks', wds_name+'_tank_level.txt')
        self.wds        = Network(self.pathToWDS)
        self.demandDict = self.build_demand_dict()
        self.pumpGroup = pump_group
        self.pump_speeds = np.ones(shape=(len(self.pumpGroup)), dtype=np.float32)
        self.pumpEffs   = np.empty(shape=(len(self.pumpGroup)), dtype=np.float32)
        self.tmpfile_name = ""
        self.pathToTmpWds = ""
        self.totDmd = 0

        self.parser = MyParser(self.pathToWDS, self.pathToTankLevel)
        
        self.headMaskKeys = {"HUAYI", "QINGDONG", "J40543",
                            "JIAHUA", "JIXI", "J49895",
                            "J59970", "J110064", "J107956",
                            "J79998", "HUAYU", "J111568",
                            "J13356", "J82123", "XULE",
                            "J95051", "JINSHUI", "J56841",
                            "PANZHONG", "J77098", "HUAQING",
                            "ZHUGUANG", "J54945", "J101196"} # should fill observable junctionIDs here, to filter needed junctions
        self.headDict = {}
        self.tankKeys = ["R00008", "R00009"]
        self.reserviorKeys = ["XJ-R1", "XJ-R2", "R00003", "R00004", "R00005", "R00006", "R00006"]
        self.controlPoint = "QINGDONG"
        self.demandLimitLo = 0
        self.demandLimitHi = 5000
        self.scene_id = -1
        self.apply_scene(0) # using demand at timestamp 0 as original demand

        if (len(self.headMaskKeys) != 0):
            for i, key in enumerate(self.headMaskKeys):
                self.headDict[i] = key
        else:
            for i, junction in enumerate(self.wds.junctions):
                self.headDict[i] = junction.uid

        nomHCurvePtsDict, nomECurvePtsDict = self.get_performance_curve_points()
        nomHCurvePoliDict       = self.fit_polinomials(
                                    nomHCurvePtsDict,
                                    degree=2,
                                    encapsulated=True)
        self.nomECurvePoliDict  = self.fit_polinomials(
                                    nomECurvePtsDict,
                                    degree=4,
                                    encapsulated=True)
        self.sumOfDemands       = sum(
                            [demand for demand in self.wds.junctions.basedemand])
        self.demandRandomizer   = self.build_truncnorm_randomizer(
                                    lo=.7, hi=1.3, mu=1.0, sigma=1.0)
        
        # Theoretical bounds of {head, efficiency}
        # peak_heads   = []
        # for key in nomHCurvePoliDict.keys():
        #     max_q       = np.max(nomHCurvePtsDict[key][:,0])
        #     opti_result = minimize(
        #         -nomHCurvePoliDict[key], x0=1, bounds=[(0, max_q)])
        #     peak_heads.append(nomHCurvePoliDict[key](opti_result.x[0]))
        # peak_effs  = []
        # for key in nomHCurvePoliDict.keys():
        #     max_q       = np.max(nomHCurvePtsDict[key][:,0])
        #     q_list      = np.linspace(0, max_q, 10)
        #     head_poli   = nomHCurvePoliDict[key]
        #     eff_poli    = self.nomECurvePoliDict[key]
        #     opti_result = minimize(-eff_poli, x0=1, bounds=[(0, max_q)])
        #     peak_effs.append(eff_poli(opti_result.x[0]))

        # Reward control
        self.dimensions     = len(self.pumpGroup)
        self.episodeLength  = episode_len
        self.headLimitLo    = 16.5
        self.headLimitHi    = 60
        self.rewScale       = [4, 4, 2] # mut factors of valid head, head ratio, pump eff
        # 4,4,2   3.5,3.5,3     3,3,4     2.5,2.5,5
        self.baseReward     = +1
        self.bumpPenalty    = -1
        self.distanceRange  = .5
        self.wrongMovePenalty   = -1
        self.lazinessPenalty    = -1
        # ----- ----- ----- ----- -----
        # Tweaking reward
        # ----- ----- ----- ----- -----
        # maxReward   = 5
        # ----- ----- ----- ----- -----
        self.maxReward   = +1
        self.minReward   = -1

        # Inner variables
        self.spec           = None
        self.metadata       = None
        self.totalDemandLo  = total_demand_lo
        self.totalDemandHi  = total_demand_hi
        self.speedIncrement = speed_increment # increment of pump speed, can adjust
        self.speedLimitLo   = .75
        self.speedLimitHi   = 1.0
        self.validSpeeds   = np.arange(
                                self.speedLimitLo,
                                self.speedLimitHi+.001,
                                self.speedIncrement,
                                dtype=np.float32) # the speeds that can choose
        self.resetOrigPumpSpeeds= reset_orig_pump_speeds
        self.resetOrigDemands   = reset_orig_demands
        self.optimized_speeds   = np.empty(shape=(len(self.pumpGroup)),
                                    dtype=np.float32)
        self.optimized_speeds.fill(np.nan)
        self.optimized_value    = np.nan
        self.previous_distance  = np.nan
        self.pump_heads = [] # pump leverage to calculate
        # initialization of {observation, steps, done}
        self.action_space   = gym.spaces.Discrete(2*self.dimensions+1)
        self.observation_space  = gym.spaces.Box(
                                    low     = -1,
                                    high    = +1,
                                    shape   = (len(self.headDict) + len(self.pumpGroup),),
                                    dtype   = np.float32)

        # for one-shot tests
        self.one_shot   = rs.rs(
            target      = self.reward_to_deap,
            dims        = self.dimensions,
            limit_lo    = self.speedLimitLo,
            limit_hi    = self.speedLimitHi,
            step_size   = self.speedIncrement,
            maxfev      = 1)

    def step(self, action, training=True):
        """ Compute reward from the Euclidean distance between the speed of the pumps
            and the optimized speeds."""
        self.steps  += 1
        self.done   = (self.steps == self.episodeLength)
        group_id    = action // 2
        command     = action % 2
        if training:
            if group_id != self.dimensions:
                self.n_siesta       = 0
                first_pump_in_grp   = self.wds.pumps[self.pumpGroup[group_id][0]]
                if command == 0:
                    if first_pump_in_grp.speed < self.speedLimitHi:
                        for pump in self.pumpGroup[group_id]:
                            self.wds.pumps[pump].speed  += self.speedIncrement
                        self.update_pump_speeds(self.scene_id)
                        distance    = np.linalg.norm(self.optimized_speeds-self.pump_speeds)
                        if distance < self.previous_distance:
        # ----- ----- ----- ----- -----
        # Tweaking reward
        # ----- ----- ----- ----- -----
                            #reward  = distance * self.baseReward / self.distanceRange
                            reward  = distance * self.baseReward / self.distanceRange / self.maxReward
        # ----- ----- ----- ----- -----
                        else:
                            reward  = self.wrongMovePenalty
                        self.previous_distance  = distance
                    else:
                        self.n_bump += 1
                        reward  = self.bumpPenalty
                else:
                    if first_pump_in_grp.speed > self.speedLimitLo:
                        for pump in self.pumpGroup[group_id]:
                            self.wds.pumps[pump].speed  -= self.speedIncrement
                        self.update_pump_speeds(self.scene_id)
                        distance    = np.linalg.norm(self.optimized_speeds-self.pump_speeds)
                        if distance < self.previous_distance: 
        # ----- ----- ----- ----- -----
        # Tweaking reward
        # ----- ----- ----- ----- -----
                            #reward  = distance * self.baseReward / self.distanceRange
                            reward  = distance * self.baseReward / self.distanceRange /self.maxReward
        # ----- ----- ----- ----- -----
                        else:
                            reward  = self.wrongMovePenalty
                        self.previous_distance  = distance
                    else:
                        self.n_bump += 1
                        reward  = self.bumpPenalty
            else:
                self.n_siesta   += 1
                self.wds.solve(self.scene_id)
                value   = self.get_state_value()
                if self.n_siesta == 3:
                    self.done   = True
                    if value/self.optimized_value > 0.98:
        # ----- ----- ----- ----- -----
        # Tweaking reward
        # ----- ----- ----- ----- -----
                        #reward  = 5
                        reward  = 5 /self.maxReward
        # ----- ----- ----- ----- -----
                    else:
                        reward = self.lazinessPenalty
                else:
                    if value/self.optimized_value > 0.98:
                        reward  = self.n_siesta * self.baseReward
                    else:
                        reward = self.lazinessPenalty
            self.wds.solve(self.scene_id)
        else:
            if group_id != self.dimensions:
                self.n_siesta       = 0
                first_pump_in_grp   = self.wds.pumps[self.pumpGroup[group_id][0]]
                if command == 0:
                    if first_pump_in_grp.speed < self.speedLimitHi:
                        for pump in self.pumpGroup[group_id]:
                            self.wds.pumps[pump].speed  += self.speedIncrement
                    else:
                        self.n_bump += 1
                else:
                    if first_pump_in_grp.speed > self.speedLimitLo:
                        for pump in self.pumpGroup[group_id]:
                            self.wds.pumps[pump].speed  -= self.speedIncrement
                    else:
                        self.n_bump += 1
            else:
                self.n_siesta   += 1
                if self.n_siesta == 3:
                    self.done   = True
            self.update_pump_speeds(self.scene_id)
            self.wds.solve(self.scene_id)
            reward  = self.get_state_value()
        observation = self.get_observation()
        return observation, reward, self.done, {}

    def reset(self, training=True):
        """Reset to Original pump speeds and demand, original means historic"""
        if training:
            scene_id = random.randint(0, 864) # gen a snapshot from 1440 scenes
            self.scene_id = scene_id
            self.apply_scene(self.scene_id)
            self.optimize_state(self.scene_id)
        self.wds.solve(self.scene_id)
        observation = self.get_observation()
        self.done   = False
        self.steps  = 0
        self.n_bump = 0
        self.n_siesta   = 0
        return observation

    def seed(self, seed=None):
        """Collecting seeds."""
        return [seed]

    def optimize_state(self, scene_id):
        """Optimize pump states with nm"""
        speeds, target_val, _   = nm.minimize(
            self.reward_to_scipy, scene_id, self.dimensions)
        self.optimized_speeds   = speeds
        self.optimized_value    = -target_val

    def optimize_state_with_one_shot(self):
        speeds, target_val, _   = self.one_shot.maximize()
        self.optimized_speeds   = speeds
        self.optimized_value    = target_val

    def fit_polinomials(self, pts_dict, degree, encapsulated=False):
        """Fitting polinomials to points stored in dict."""
        polinomials = dict()
        if encapsulated:
            for curve in pts_dict:
                # polinomials[curve]  = np.poly1d(np.polyfit(
                #     pts_dict[curve][:,0], pts_dict[curve][:,1], degree))
                polinomials[curve] = interp1d(pts_dict[curve][:,0], pts_dict[curve][:,1], bounds_error = False, fill_value=(pts_dict[curve][:,1][0], pts_dict[curve][:,1][-1]), kind="slinear")
        else:
            for curve in pts_dict:
                polinomials[curve]  = np.polyfit(
                    pts_dict[curve][:,0], pts_dict[curve][:,1], degree)
        return polinomials
# read the curves to dict, original models
    # def get_performance_curve_points(self):
    #     """Reader for H(Q) and P(Q) curves."""
    #     head_curves = dict()
    #     eff_curves  = dict()

    #     # Loading data to dictionary
    #     for curve in self.wds.curves:
    #         if curve.uid[0] == 'H': # starting with 'H' means this is an H(Q) curve
    #             head_curves[curve.uid[1:]]  = np.empty([len(curve.values), 2], dtype=np.float32)
    #             for i, op_pnt in enumerate(curve.values):
    #                 head_curves[curve.uid[1:]][i, 0]    = op_pnt[0]
    #                 head_curves[curve.uid[1:]][i, 1]    = op_pnt[1]
    #         elif curve.uid[0] == 'E': # starting with 'E' means this is an E(Q) curve
    #             eff_curves[curve.uid[1:]]   = np.empty([len(curve.values), 2], dtype=np.float32)
    #             for i, op_pnt in enumerate(curve.values):
    #                 eff_curves[curve.uid[1:]][i, 0] = op_pnt[0]
    #                 eff_curves[curve.uid[1:]][i, 1] = op_pnt[1]
    #         else:
    #             print("Error, curve is either head nor efficiency") 
    #     # Checking consistency
    #     # Mistake here, should compare one by one
    #     for head_key in head_curves.keys():
    #         if all(head_key != eff_key for eff_key in eff_curves.keys()):
    #             print('\nInconsistency in H(Q) and P(Q) curves.\n')
    #             raise IndexError
    #     return head_curves, eff_curves
    
    def get_performance_curve_points(self):
        """Reader for H(Q) and P(Q) curves. New models"""
        head_curves = dict()
        eff_curves  = dict()

        # Loading data to dictionary
        for curve in self.wds.curves:
            if curve.uid[-1] != 'E': # not end with 'E' means this is an H(Q) curve
                head_curves[curve.uid[:]]  = np.empty([len(curve.values), 2], dtype=np.float32)
                for i, op_pnt in enumerate(curve.values):
                    head_curves[curve.uid[:]][i, 0]    = op_pnt[0]
                    head_curves[curve.uid[:]][i, 1]    = op_pnt[1]
        for curve in self.wds.curves:
            if curve.uid[-1] == 'E': # this is an E(Q) curve
                eff_curves[curve.uid[:-1]]   = np.empty([len(curve.values), 2], dtype=np.float32)
                for i, op_pnt in enumerate(curve.values):
                    eff_curves[curve.uid[:-1]][i, 0] = op_pnt[0]
                    eff_curves[curve.uid[:-1]][i, 1] = op_pnt[1]
        return head_curves, eff_curves

    def get_junction_heads(self):
        return self._get_junction_heads()

    def get_point_pressure(self, junc_uid):
        '''Get a single point head'''
        return self.wds.junctions[str(junc_uid)].pressure

    def get_pump_state(self, pump_id):
        '''Get a single point head'''
        return self.wds.pumps[str(pump_id)].flow, self.wds.pumps[str(pump_id)].speed

    def _get_junction_heads(self):
        """fill an array of junction head dict"""
        junc_heads = np.empty(
                        shape   = (len(self.headDict),),
                        dtype   = np.float32)
        for junc_id in range(len(self.headDict)):
            junc_heads[junc_id] = self.wds.junctions[str(self.headDict[junc_id])].head
        return junc_heads

    def get_observation(self):
        heads   = (2*self.get_junction_heads() / 30) - 1
        self.update_pump_speeds()
        speeds  = self.pump_speeds / self.speedLimitHi
        return np.concatenate([heads, speeds])

    def restore_original_demands(self, i):
        mp = self.parser.demandSnapshot(i)
        for junc in self.wds.junctions:
            junc.basedemand = mp[junc.uid] if (junc.uid in mp.keys()) else junc.basedemand

    def build_truncnorm_randomizer(self, lo, hi, mu, sigma):
        randomizer = stats.truncnorm(
                        (lo-mu)/sigma, (hi-mu)/sigma, loc=mu, scale=sigma)
        return randomizer

    def randomize_demands(self):
        target_sum_of_demands   = self.sumOfDemands * (self.totalDemandLo +
                np.random.rand()*(self.totalDemandHi-self.totalDemandLo))
        sum_of_random_demands   = 0
        if self.seedNum:
            for junction in self.wds.junctions:
                junction.basedemand     = (self.demandDict[junction.uid] *
                    self.demandRandomizer.rvs(random_state=self.seedNum *
                        int(np.abs(np.floor(junction.coordinates[0])))))
                sum_of_random_demands   += junction.basedemand
        else:
            for junction in self.wds.junctions:
                junction.basedemand     = (self.demandDict[junction.uid] *
                    self.demandRandomizer.rvs())
                sum_of_random_demands   += junction.basedemand
        for junction in self.wds.junctions:
            junction.basedemand *= target_sum_of_demands / sum_of_random_demands

    def apply_scene(self, i):
        self.scene_id = i
        self.calc_totdemand(i)
        self.apply_pumpStatusSnapshot(i)

    def calc_totdemand(self, i):
        """Generate demand from pattern with step index i"""
        demandMap = self.parser.demandSnapshot(i)
        self.totDmd = sum(demandMap.values())

    def randomize_demand(self, lo, hi):
        totDemand = random.randint(lo, hi)
        num = len(self.wds.junctions)
        for junction in self.wds.junctions:
            junction.basedemand = totDemand / num

    def apply_pumpStatusSnapshot(self, i):
        """Generate pump speeds from pattern with timestamp i"""
        pumpSpeedMap = self.parser.pumpSpeedSnapshot(i)
        for pump in self.wds.pumps:
            if (pump.uid in pumpSpeedMap.keys()):
                pump.speed = pumpSpeedMap[pump.uid] if pumpSpeedMap[pump.uid] > 0.65 else 0
                pump.initstatus = 1 if (pump.speed > 0.001) else 0
            else: print("\033[0;31;40m", pump.uid, pumpSpeedMap[pump.uid], "\033[0m")
        self.update_pump_speeds(i)

    def apply_pumpStatusSnapshot_v1(self, pump_speeds, i):
        """Generate pump speeds from pattern with timestamp i"""
        pumpSpeedMap = self.parser.pumpSpeedSnapshot(i)
        self.wds.pumps["XJ-P2"].speed = float(pump_speeds[0])
        self.wds.pumps["XJ-P4"].speed = float(pump_speeds[1])
        self.wds.pumps["XJ-P9"].speed = float(pump_speeds[2])
        self.update_pump_speeds(i)

        # # for orig
#     def calculate_pump_efficiencies(self):
#         """calculate efficiencies from speeds"""
#         for i, group in enumerate(self.pumpGroup):
#             pump        = self.wds.pumps[group[0]]
#             curve_id    = pump.curve.uid[1:]
#             self.pump_heads.append(pump.downstream_node.head - pump.upstream_node.head)
#             eff_poli    = self.nomECurvePoliDict[curve_id]
#             self.pumpEffs[i]   = eff_poli(pump.flow / pump.speed)

    def calculate_pump_efficiencies(self):
        """calculate efficiencies from speeds"""
        for i, group in enumerate(self.pumpGroup):
            pump        = self.wds.pumps[group[0]]
            curve_id    = pump.curve.uid[:]
            if (curve_id[-1] == 'E'): curve_id    = curve_id[:-1]
            pump_head   = pump.downstream_node.head - pump.upstream_node.head
            eff_poli    = self.nomECurvePoliDict[curve_id]
            self.pumpEffs[i]   = eff_poli(pump.flow / pump.speed)

    # mapping junction uid->basedemand
    def build_demand_dict(self):
        demand_dict = dict()
        for junction in self.wds.junctions:
            demand_dict[junction.uid] = junction.basedemand
        return demand_dict

    def get_state_value_separated(self):
        self.calculate_pump_efficiencies()
        pump_ok = (self.pumpEffs < 1).all() and (self.pumpEffs > 0).all()
        if pump_ok:
            heads   = np.array([head for head in self.wds.junctions.head])
            invalid_heads_count = (np.count_nonzero(heads < self.headLimitLo) +
                np.count_nonzero(heads > self.headLimitHi))
            valid_heads_ratio   = 1 - (invalid_heads_count / len(heads))

            total_demand    = sum(
                [junction.basedemand for junction in self.wds.junctions])
            total_tank_flow = sum(
                [tank.inflow+tank.outflow for tank in self.wds.tanks])
            demand_to_total = total_demand / (total_demand+total_tank_flow)

            total_efficiency    = np.prod(self.pumpEffs)
            eff_ratio           = total_efficiency / self.peakTotEff
        else:
            eff_ratio           = 0
            valid_heads_ratio   = 0
            demand_to_total     = 0
        return eff_ratio, valid_heads_ratio, demand_to_total

    def get_state_value(self):
        self.calculate_pump_efficiencies()
        result = self.calc_reward()
        return result

    def calc_reward(self):
        # pump_ok = (self.pumpEffs < 1).all() and (self.pumpEffs > 0).all()
        pump_ok = (self.pumpEffs > 0).all()
        if pump_ok:
            heads   = self.get_junction_heads()
            invalid_heads_count = (np.count_nonzero(heads < self.headLimitLo) +
                np.count_nonzero(heads > self.headLimitHi))
            valid_heads_ratio   = 1 - (invalid_heads_count / len(heads)) # calc valid head ratio

            # total_efficiency    = np.prod(self.pumpEffs)
            control_head = self.get_point_pressure(self.controlPoint)

            peakEffs = [80.14655, 79.52216, 76.04018]
            
            control_head_ratio = math.exp(-(abs(control_head - self.headLimitLo)) / 10)
            energy_eff_ratio = np.prod(self.pumpEffs / peakEffs)
            result  = ( self.rewScale[0] * valid_heads_ratio + 
                        self.rewScale[1] * control_head_ratio + 
                        self.rewScale[2] * energy_eff_ratio) / sum(self.rewScale)
            # print(self.pump_speeds)
            # print(self.pumpEffs)
            # print("valid_heads_score: ", valid_heads_ratio)
            # print("control_head_score: ", control_head_ratio)
            # print("energy_eff_score: ", energy_eff_ratio)
        else:
            result = 0
        return result

# restrict pump speed to limits and call .get_state_value()
    def get_state_value_to_opti(self, pump_speeds, scene_id = -1):
        np.clip(a   = pump_speeds,
            a_min   = self.speedLimitLo,
            a_max   = self.speedLimitHi,
            out     = pump_speeds)
        for group_id, pump_group in enumerate(self.pumpGroup):
            for pump in pump_group:
                self.wds.pumps[pump].speed  = pump_speeds[group_id]
        if (scene_id < 0 and self.scene_id > 0):
            scene_id = self.scene_id
        self.update_pump_speeds(scene_id)
        self.wds.solve(scene_id)
        return self.get_state_value()
        
    def reward_to_scipy(self, pump_speeds, scene_id):
        """Only minimization allowed."""
        return -self.get_state_value_to_opti(pump_speeds, scene_id)

    def reward_to_deap(self, pump_speeds):
        """Return should be tuple."""
        return self.get_state_value_to_opti(np.asarray(pump_speeds)),

    def update_pump_speeds(self, scene_id = -1):
        """Update pump group speeds by epynet wrapped pump objects. Enables consistency among epynet wrapper & pump_speeds[] & epanet model"""
        for i, pump_group in enumerate(self.pumpGroup):
            self.pump_speeds[i] = self.wds.pumps[pump_group[0]].speed
        if (scene_id != -1):
            self.wds.ep.ENsetpatternvalue(index = self.wds.ep.ENgetpatternindex("XJ-P2"), period = scene_id + 1, value = self.pump_speeds[0])
            self.wds.ep.ENsetpatternvalue(index = self.wds.ep.ENgetpatternindex("XJ-P4"), period = scene_id + 1, value = self.pump_speeds[1])
            self.wds.ep.ENsetpatternvalue(index = self.wds.ep.ENgetpatternindex("XJ-P9"), period = scene_id + 1, value = self.pump_speeds[2])
        return self.pump_speeds

    # def update_pump_speeds_v1(self, pump_speeds, scene_id = -1):
    #     """Update pump group speeds by epynet wrapped pump objects. Enables consistency among epynet wrapper & pump_speeds[] & epanet model"""
    #     for i, pump_group in enumerate(self.pumpGroup):
    #         self.pump_speeds[i] = self.wds.pumps[pump_group[0]].speed
    #     if (scene_id != -1):
    #         self.wds.ep.ENsetpatternvalue(index = self.wds.ep.ENgetpatternindex("XJ-P2"), period = scene_id + 1, value = float(pump_speeds[0]))
    #         self.wds.ep.ENsetpatternvalue(index = self.wds.ep.ENgetpatternindex("XJ-P4"), period = scene_id + 1, value = float(pump_speeds[1]))
    #         self.wds.ep.ENsetpatternvalue(index = self.wds.ep.ENgetpatternindex("XJ-P9"), period = scene_id + 1, value = float(pump_speeds[2]))
    #     return self.pump_speeds

    def get_pump_speeds(self):
        self.update_pump_speeds()
        return self.pump_speeds
    
    def printState(self):
        id = self.wds.ep.ENgetlinkindex("XJ-P2")
        print("XJ-P2 Speed:", self.wds.ep.ENgetlinkvalue(index = id, paramcode = 5))
        id = self.wds.ep.ENgetlinkindex("XJ-P4")
        print("XJ-P4 Speed:", self.wds.ep.ENgetlinkvalue(index = id, paramcode = 5))
        id = self.wds.ep.ENgetlinkindex("XJ-P9")
        print("XJ-P9 Speed:", self.wds.ep.ENgetlinkvalue(index = id, paramcode = 5))
        print("XFX head: ", self.get_point_pressure("XFX-OP"))
        print("HX head: ", self.get_point_pressure("HX-node12"))   
        print(str.format("XJ head: {0}", self.get_point_pressure("XJ-node43")))   
        print(str.format("R1-P leverage: {0}", self.get_point_pressure("XJ-node43") + 3.75 - self.wds.reservoirs["XJ-R1"].head))
        print(str.format("R2-P leverage: {0}", self.get_point_pressure("XJ-node43") + 3.75 - self.wds.reservoirs["XJ-R2"].head))
        print("\033[40m", str.format("Reward is {0}", self.get_state_value()), "\033[0m")
        print("\033[40m", str.format("Ctr head: {0}", self.get_point_pressure(self.controlPoint)), "\033[0m")