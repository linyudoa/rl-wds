from epynet import Network
import wdsEnv

pathToNetwork = "./water_networks/QDMaster1031_master.inp"

wds = Network(pathToNetwork)

wds.run()

wds.solve()

# node = wds.junctions["J90212_B"]
# pat = node.pattern

# print(sum([demand for demand in wds.junctions.basedemand]))

# for demand in wds.junctions.basedemand:
#     print(demand)