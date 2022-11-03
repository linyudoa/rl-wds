from epynet import Network
import wdsEnv

pathToNetwork = "./water_networks/QDMaster.inp"

pathToNetwork1 = "./water_networks/anytown_master.inp"

wds = Network(pathToNetwork1)

wds.solve()