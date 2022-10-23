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

prop_code = 2
mu = 0
sigma = 10
print("score", env.get_state_value())
print("---------------------- trying to add noise ----------------------")
env.randomize_wds_roughness(mu, sigma)
print("score", env.get_state_value())

# for pump in env.wds.pumps:
#     print("pump index: ", pump.index)
#     print("uid: ", pump.uid)
#     print("pump props: ", pump.properties)