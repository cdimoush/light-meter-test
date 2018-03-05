import matplotlib
import matplotlib.colors as colors
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import style
import math

fig = plt.figure()
axis = fig.gca(projection='3d')
df = pd.read_csv('sc_data.csv')
df_maximum = df.max().max()
df_mean = df.mean().mean()
reg = False

cmap = plt.get_cmap("gnuplot")

# Brightness is the actual value
# for a&b row number is distance from vertex and column number is height on wall
# for c row is distance from vertex along right wall and column number is distance from vertex on front wall
out_high = df.mean().mean() + (df.stack().std())
out_low = df.mean().mean() - (df.stack().std())
print(out_high)
print(out_low)
for i in range(0, df.shape[0]):
        for j in range(0, df.shape[1]):
            if df.iloc[i][j] > out_high:
                df.iloc[i][j] = out_high
            if df.iloc[i][j] < out_low:
                df.iloc[i][j] = out_low

#df = df / df_maximum
#df = (df - df.mean().mean()) / df.stack().std()
df = (df - df.min().min())/(df.max().max() - df.min().min())
normalize = colors.Normalize(vmin=df.min().min(), vmax=df.max().max())
a0 = df[df.columns[0:5]]  # front
b0 = df[df.columns[5:10]]  # right
c0 = df[df.columns[10:15]]  # floor
ab0 = df[df.columns[15]]
a1 = df[df.columns[16:21]]  # front
b1 = df[df.columns[21:26]]  # right
c1 = df[df.columns[26:31]]  # floor
ab1 = df[df.columns[31]]
a2 = df[df.columns[32:37]]  # front
b2 = df[df.columns[37:42]]  # right
c2 = df[df.columns[42:47]]  # floor
ab2 = df[df.columns[47]]

ax, az = np.meshgrid(range(1, 6), range(5))
ay = np.zeros((5, 5))

by, bz = np.meshgrid(range(1, 6), range(5))
bx = np.zeros((5, 5))

cx, cy = np.meshgrid(range(1,6), range(1,6))
cz = np.zeros((5, 5))


def regression(axis, jet, ac, bc, cc):
    skip = 0
    axr, azr = np.meshgrid(np.arange(.5, 5.5, 1), range(5))
    ayr = np.zeros((5, 5))
    acr = []
    for i in range(0, len(ac)):
        skip += 1
        if skip != 6:
            acr.append((ac[i] + ac[i+1])/2)
        if skip == 6:
            skip = 0

    skip = 0
    byr, bzr = np.meshgrid(np.arange(.5, 5.5, 1), range(5))
    bxr = np.zeros((5, 5))
    bcr = []
    for i in range(0, len(bc)):
        skip += 1
        if skip == 1: # figure out the spaces between a and b
            bcr.append(())
        elif skip != 5:
            bcr.append((bc[i] + bc[i+1])/2)
        elif skip == 5:
            skip = 0

    axis.scatter(axr, ayr, azr, c=acr, marker='o', s=5, cmap=jet, depthshade=False)
    axis.scatter(bxr, byr, bzr, c=bcr, marker='o', s=5, cmap=jet, depthshade=False)


    by, bz = np.meshgrid(range(1, 6), range(5))
    bx = np.zeros((5, 5))
    bc = []

    cx, cy = np.meshgrid(range(1,6), range(1,6))
    cz = np.zeros((5, 5))
    cc = []


def make_plot(a, b, c):
    ac = []
    bc = []
    cc = []
    for i in range(0, 5):
        for j in range(0, 5):
            ac.append(a.iloc[i][j])
    for i in range(0, 5):
        for j in range(0, 5):
            bc.append(b.iloc[i][j])
    for i in range(0, 5):
        for j in range(0, 5):
            cc.append(c.iloc[i][j])
    return ac, bc, cc


if True:
    plot_num = input()
    if plot_num == '1':
        ac, bc, cc = make_plot(a0, b0, c0)
        p = axis.scatter(ax, ay, az, c=ac, marker='o', s=25, cmap=cmap, norm=normalize,  depthshade=False)
        p = axis.scatter(bx, by, bz, c=bc, marker='o', s=25, cmap=cmap, norm=normalize, depthshade=False)
        p = axis.scatter(cx, cy, cz, c=cc, marker='o', s=25, cmap=cmap, norm=normalize, depthshade=False)
        y1 = 5
        y2 = y1 + 2
        bot_pos_x = np.array([[2, 3], [2, 3], [2, 3], [2, 3], [2, 3]])
        bot_pos_y = np.array([[y1, y1], [y2, y2], [y2, y2], [y1, y1], [y1, y1]])
        bot_pos_z = np.array([[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]])
        axis.plot_wireframe(bot_pos_x, bot_pos_y, bot_pos_z)
        fig.colorbar(p, shrink=0.5, aspect=5)
        axis.set_title('Plot 1')

    if plot_num == '2':
        ac, bc, cc = make_plot(a1, b1, c1)
        p = axis.scatter(ax, ay, az, c=ac, marker='o', s=25, cmap=cmap, norm=normalize,  depthshade=False)
        p = axis.scatter(bx, by, bz, c=bc, marker='o', s=25, cmap=cmap, norm=normalize, depthshade=False)
        p = axis.scatter(cx, cy, cz, c=cc, marker='o', s=25, cmap=cmap, norm=normalize, depthshade=False)
        y1 = 3
        y2 = y1 + 2
        bot_pos_x = np.array([[2, 3], [2, 3], [2, 3], [2, 3], [2, 3]])
        bot_pos_y = np.array([[y1, y1], [y2, y2], [y2, y2], [y1, y1], [y1, y1]])
        bot_pos_z = np.array([[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]])
        axis.plot_wireframe(bot_pos_x, bot_pos_y, bot_pos_z)
        fig.colorbar(p, shrink=0.5, aspect=5)
        axis.set_title('Plot 2')
    if plot_num == '3':
        ac, bc, cc = make_plot(a2, b2, c2)
        p = axis.scatter(ax, ay, az, c=ac, marker='o', s=25, cmap=cmap, norm=normalize,  depthshade=False)
        p = axis.scatter(bx, by, bz, c=bc, marker='o', s=25, cmap=cmap, norm=normalize, depthshade=False)
        p = axis.scatter(cx, cy, cz, c=cc, marker='o', s=25, cmap=cmap, norm=normalize, depthshade=False)
        y1 = 1
        y2 = y1 + 2
        bot_pos_x = np.array([[2, 3], [2, 3], [2, 3], [2, 3], [2, 3]])
        bot_pos_y = np.array([[y1, y1], [y2, y2], [y2, y2], [y1, y1], [y1, y1]])
        bot_pos_z = np.array([[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]])
        axis.plot_wireframe(bot_pos_x, bot_pos_y, bot_pos_z)
        fig.colorbar(p, shrink=0.5, aspect=5)
        axis.set_title('Plot 3')

    axis.set_xlabel('$X$')
    axis.set_ylabel('$Y$')
    axis.set_zlabel('$Z$')

    plt.show()

