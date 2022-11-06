from epynet import Network
import wdsEnv

pathToNetwork = "water_networks/QDMaster1031_master.inp"

wds = Network(pathToNetwork)

node = wds.junctions["J119449_B"]
linecount = 0
for node in wds.junctions:
    try:
         if (node.pattern != ""): 
            print(node, node.pattern)
            linecount += 1
    except:
         1
    if (linecount == 100): break

# print(sum([demand for demand in wds.junctions.basedemand]))

# for demand in wds.junctions.basedemand:
#     print(demand)