

from scipy.interpolate import interp1d




def pump_efficiency_cal(pump, flow, liftHead):

    if pump == 'xj1':
        x=[1560, 1920, 2710, 3050, 3430]
        y=[61,66.2,77.9,78.7,79.6]
        f = interp1d(x, y, kind="slinear")
        if flow > x[-1]:
            efficiency = y[-1]
        elif flow < x[0]:
            efficiency = y[0]
        else:
            efficiency = f(flow)
        # efficiency单位%
    elif pump == 'xj2':
        x=[1560,1920,2710,3050,3430]
        y=[61,66.2,77.9,78.7,79.6]
        f = interp1d(x, y, kind="slinear")
        if flow > x[-1]:
            efficiency = y[-1]
        elif flow < x[0]:
            efficiency = y[0]
        else:
            efficiency = f(flow)
        # efficiency单位%
    elif pump == 'xj3':
        x=[2030,2510,3024,3547,3760]
        y=[72.1,76.8,80.3,80.2,78.9]
        f = interp1d(x, y, kind="slinear")
        if flow > x[-1]:
            efficiency = y[-1]
        elif flow < x[0]:
            efficiency = y[0]
        else:
            efficiency = f(flow)
        # efficiency单位%
    elif pump == 'xj4':
        x=[1932,2410,2802,3380,3661]
        y=[67.8,73.4,75.5,79.5,77.4]
        f = interp1d(x, y, kind="slinear")
        if flow > x[-1]:
            efficiency = y[-1]
        elif flow < x[0]:
            efficiency = y[0]
        else:
            efficiency = f(flow)
        # efficiency单位%
    elif pump == 'xj5':
        x=[1364.5833,2135.4167,2531.25,3187.5,3562.5]
        y=[56.2,75.7,76.5,82.8,81]
        f = interp1d(x, y, kind="slinear")
        if flow > x[-1]:
            efficiency = y[-1]
        elif flow < x[0]:
            efficiency = y[0]
        else:
            efficiency = f(flow)
        # efficiency单位%
    elif pump == 'xj6':
        x=[1633,2040,2850,3420,3680]
        y=[65.1,72.6,78.7,82.8,80.2]
        f = interp1d(x, y, kind="slinear")
        if flow > x[-1]:
            efficiency = y[-1]
        elif flow < x[0]:
            efficiency = y[0]
        else:
            efficiency = f(flow)
        # efficiency单位%
    elif pump == 'xj7':
        x=[558.3333,841.6667,1181.25,1394.7917,1511.4583]
        y=[67.5,86.8,91.2,78.1,61]
        f = interp1d(x, y, kind="slinear")
        if flow > x[-1]:
            efficiency = y[-1]
        elif flow < x[0]:
            efficiency = y[0]
        else:
            efficiency = f(flow)
        # efficiency单位%
    elif pump == 'xj8':
        x=[583.3333,955.2083,1232.2917,1328.125,1534.375]
        y=[60.6,72.7,74.8,72.4,61.3]
        f = interp1d(x, y, kind="slinear")
        if flow > x[-1]:
            efficiency = y[-1]
        elif flow < x[0]:
            efficiency = y[0]
        else:
            efficiency = f(flow)
        # efficiency单位%
    elif pump == 'xj9':
        x=[471.875,1030.2083,1313.5417,1461.4583,1588.5417]
        y=[52.9,74.7,76,70.5,62.9]
        f = interp1d(x, y, kind="slinear")
        if flow > x[-1]:
            efficiency = y[-1]
        elif flow < x[0]:
            efficiency = y[0]
        else:
            efficiency = f(flow)
        # efficiency单位%
    elif pump == 'xfx1':
        x=[592,968,1390,1641,1949]
        y=[54,68,76.9,80.7,51]
        f = interp1d(x, y, kind="slinear")
        if flow > x[-1]:
            efficiency = y[-1]
        elif flow < x[0]:
            efficiency = y[0]
        else:
            efficiency = f(flow)
        # efficiency单位%
    elif pump == 'xfx2':
        x=[331,1025,1486,1774,1996]
        y=[31.5,69.6,72.6,63.5,60.9]
        f = interp1d(x, y, kind="slinear")
        if flow > x[-1]:
            efficiency = y[-1]
        elif flow < x[0]:
            efficiency = y[0]
        else:
            efficiency = f(flow)
        # efficiency单位%
    elif pump == 'xfx3':
        x=[422,919,1575,1763,1879]
        y=[40.6,73.7,91.9,66.4,51.1]
        f = interp1d(x, y, kind="slinear")
        if flow > x[-1]:
            efficiency = y[-1]
        elif flow < x[0]:
            efficiency = y[0]
        else:
            efficiency = f(flow)
        # efficiency单位%
    elif pump == 'xfx4':
        x=[251,457,664,934,1037]
        y=[38.7,58.6,71.6,77.6,73.6]
        f = interp1d(x, y, kind="slinear")
        if flow > x[-1]:
            efficiency = y[-1]
        elif flow < x[0]:
            efficiency = y[0]
        else:
            efficiency = f(flow)
        # efficiency单位%
    elif pump == 'xfx5':
        x=[251,457,664,934,1037]
        y=[38.7,58.6,71.6,77.6,73.6]
        f = interp1d(x, y, kind="slinear")
        if flow > x[-1]:
            efficiency = y[-1]
        elif flow < x[0]:
            efficiency = y[0]
        else:
            efficiency = f(flow)
        # efficiency单位%


    elif pump == 'hx1':
        x=[1775,2675,3220,3630,3884]
        y=[57.4,74,73.8,69.2,66]
        f = interp1d(x, y, kind="slinear")
        if flow > x[-1]:
            efficiency = y[-1]
        elif flow < x[0]:
            efficiency = y[0]
        else:
            efficiency = f(flow)
        # efficiency单位%

    elif pump == 'hx2':
        x=[2169,2713,3245,3526,3825]
        y=[66.9,73.5,74.5,70.6,64.4]
        f = interp1d(x, y, kind="slinear")
        if flow > x[-1]:
            efficiency = y[-1]
        elif flow < x[0]:
            efficiency = y[0]
        else:
            efficiency = f(flow)
        # efficiency单位%

    elif pump == 'hx3':
        x=[1735,2612,3146,3510,3819]
        y=[58.1,70.8,73.2,68.7,60]
        f = interp1d(x, y, kind="slinear")
        if flow > x[-1]:
            efficiency = y[-1]
        elif flow < x[0]:
            efficiency = y[0]
        else:
            efficiency = f(flow)
        # efficiency单位%

    elif pump == 'hx4':
        x=[1230,1470,1707,1845,1960]
        y=[65,64.4,61.7,55.5,48.2]
        f = interp1d(x, y, kind="slinear")
        if flow > x[-1]:
            efficiency = y[-1]
        elif flow < x[0]:
            efficiency = y[0]
        else:
            efficiency = f(flow)
        # efficiency单位%

    elif pump == 'hx5':
        x=[739,1350,1750,1960,2100]
        y=[46.3,69.5,60.1,57.2,50.4]
        f = interp1d(x, y, kind="slinear")
        if flow>x[-1]:
            efficiency = y[-1]
        elif flow<x[0]:
            efficiency = y[0]
        else:
            efficiency = f(flow)
        # efficiency单位%


    power1 = 1000 * liftHead / 367.2 / efficiency * 100
    # power1单位，kWh每1000m3水
    power2 = power1 * flow * 5 / 60 /1000
    # print("%.2f" % efficiency, "%.2f" % power1,"%.2f" % power2)
    # power2单位，kWh



    return (efficiency, power1, power2)



if __name__ == '__main__':
    pump_efficiency_cal('hx3',2000, 13.43)
    print(pump_efficiency_cal('xj2', 2500, 20))

# 说明：输入参数依次为（泵名（小写加数字），流量（CMH，可从SCADA读取），扬程（m，可从SCADA之差得到））
# 可以先试试和实际数据差多少
# 输入示例pump_efficiency_cal('hx1', 3000,15)
# 输入示例pump_efficiency_cal('xj3', 2500, 20)
# 输入示例pump_efficiency_cal('xfx4', 1000, 15)
# pump_efficiency_cal('hx2', 3621.65, 13.43)