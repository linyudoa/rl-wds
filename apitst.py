from epynet import Network
import wdsEnv

pathToNetwork = "./water_networks/QDMaster1031_master.inp"

wds = Network(pathToNetwork)

# wds.solve()
node = wds.junctions["J90212_B"]

# print(sum([demand for demand in wds.junctions.basedemand]))

# for demand in wds.junctions.basedemand:
#     print(demand)