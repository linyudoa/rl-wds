import wdsEnv

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

key = "40"
print("---------------------- trying to close pipe uid = ", key, " ----------------------")
env.wds.pipes[key].set_static_property(11, 0.0)
print("pipe status of uid", key, " = ", env.wds.pipes[key].status)
env.wds.solve()
print("pipe status of uid", key, " = ", env.wds.pipes[key].status)

# for pump in env.wds.pumps:
#     print("pump index: ", pump.index)
#     print("uid: ", pump.uid)
#     print("pump props: ", pump.properties)