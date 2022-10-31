from epynet import Network
import wdsEnv

pathToNetwork = "./water_networks/QDMaster1031.inp"

pathToNetwork1 = "./water_networks/anytown_master.inp"

wds = Network(pathToNetwork)

wds.solve()