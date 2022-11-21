import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from MyDotCloud import *

def plot3DCloud(inputPath : str, outputPath : str):
    dotCloud = MyDotCloud(inputPath)
    points = dotCloud.toArr()
    points = np.array(points)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

# Set dimensions and plotting format
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    ax.scatter(xs, ys, zs, 
           marker = 'o', 
           linewidths = 1, c = "orange")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig(outputPath)
    
def plot3DCloudAndLine(cloudPath : str, linePath : str, outputPath : str):
    dotCloud = MyDotCloud(cloudPath)
    cloudPoints = dotCloud.toArr()
    cloudPoints = np.array(cloudPoints)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = cloudPoints[:, 0]
    ys = cloudPoints[:, 1]
    zs = cloudPoints[:, 2]
    ax.scatter(xs, ys, zs, 
           marker = 'o', alpha = 0.03,
           linewidths = 0.5, c = "orange")
    
    lineDots = MyDotCloud(linePath)
    points3d = lineDots.toArr()
    points3d = np.array(points3d)
    x = points3d[:, 0]
    y = points3d[:, 1]
    z = points3d[:, 2]
    ax.plot(x, y, z, 'o-', mfc = 'b', mec = 'r', ms = 5, linewidth = 3, label = 'line')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')

    plt.savefig(outputPath)
    
def plot2DCloudAndLine(cloudPath : str, dim_ignored : int, linePath : str, outputPath : str):
    dotCloud = MyDotCloud(cloudPath)
    cloudPoints = dotCloud.to2DArr(dim_ignored)
    cloudPoints = np.array(cloudPoints)
    fig = plt.figure()
    ax = fig.add_subplot()
    d1 = cloudPoints[:, 0]
    d2 = cloudPoints[:, 1]
    ax.scatter(d1, d2, 
           marker = 'o', 
           linewidths = 1, c = "green")
    
    lineDots = MyDotCloud(linePath)
    points2d = lineDots.toArr()
    points2d = np.array(points2d)
    x = points2d[:, 0]
    y = points2d[:, 1]
    ax.plot(x, y, 'bo-', mfc = 'red', mec = 'red', ms = 5, linewidth = 3, label='line')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')

    plt.savefig(outputPath)

inputFile = "Points.txt"
linePath2d = "exp_2D.txt"
linePath3d = "exp_3D.txt"

plot3DCloud(inputFile, "dotCloud.png")
plot2DCloudAndLine(inputFile, 2, linePath2d, "dots2d.png")
plot3DCloudAndLine(inputFile, linePath3d, "dots3d.png")
