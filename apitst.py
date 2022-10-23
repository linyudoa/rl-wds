import wdsEnv
import random

env = wdsEnv.wds(
        wds_name        = 'anytown_master',
        speed_increment = .05,
        episode_len     = 10,
        pump_group     = [['78', '79']],
        total_demand_lo = .3,
        total_demand_hi = 1.1,
        reset_orig_pump_speeds  = False,
        reset_orig_demands      = False,
        seed    = None)

# for pipe in env.wds.pipes:
#     print("pipe index: ", pipe.index)
#     print("uid: ", pipe.uid)
#     print("pipe status: ", pipe.status)
#     print("pipe diameter: ", pipe.diameter)

key = "4"
prop_code = 2
print("original roughness of pipe", key, " = ", env.wds.pipes[key].get_property(prop_code))
noise = random.gauss(0, 10)
print("---------------------- trying to add noise ----------------------")
env.wds.pipes[key].set_static_property(2, env.wds.pipes[key].get_property(prop_code) + noise)
print("new roughness", " = ", env.wds.pipes[key].get_property(prop_code))
env.wds.solve()
env.wds.solve()
print("new roughness", " = ", env.wds.pipes[key].get_property(prop_code))

# for pump in env.wds.pumps:
#     print("pump index: ", pump.index)
#     print("uid: ", pump.uid)
#     print("pump props: ", pump.properties)