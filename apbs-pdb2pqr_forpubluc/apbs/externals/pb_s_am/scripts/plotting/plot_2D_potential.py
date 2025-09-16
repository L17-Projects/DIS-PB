import sys
import numpy as np
import matplotlib.pyplot as plt

'''
Program to plot a 2D version of the ESP from PB-[S]AM
'''
fileName = sys.argv[1]
outFile  = sys.argv[2]

#-----------------------------------------------------------------------
def FileOpen(fileName):
    """Gets data from 2D plot output of PB-SAM"""
    lines = open(fileName).readlines()

    grid,org,dl = np.zeros(2), np.zeros(2),np.zeros(2)
    axval, ax, units, mn, mx = 0.0, 'x', 'jmol', 0.0, 0.0
    pot = np.zeros((100,100))
    ct = 0

    for line in lines:
        temp = line.split()
        if 'units' in line[0:10]:
            units = temp[1]
        elif 'grid' in line[0:10]:
            grid[0], grid[1] = int(temp[1]), int(temp[2])
            pot = np.zeros((int(grid[0]), int(grid[1])))
        elif 'axis' in line[0:10]:
            ax, axval = temp[1], float(temp[2])
        elif 'origin' in line[0:10]:
            org[0], org[1] = float(temp[1]), float(temp[2])
        elif 'delta' in line[0:10]:
            dl[0], dl[1] = float(temp[1]), float(temp[2])
        elif 'maxmin' in line[0:10]:
             mx, mn = float(temp[1]), float(temp[2])
        elif '#' not in line:
            temp = [float(x) for x in line.split()]

            for i in range(int(grid[0])):
                pot[ct][i] = temp[i]
            ct += 1

    return(pot, org, dl, ax, axval, units, mx, mn)

def dispPlot( org, bn, count, potential,
                mx = 0.1, mn = -0.1, title = '',
                xlab = r'$X \AA$', ylab = r'$Y \, (\AA)$',
                lege = '', outFile = None ):
    """Plots the colormap of potential plot, 2D"""
    fig = plt.figure(1, figsize = (3.5, 3.))
    ax = fig.add_subplot(1,1,1)

    nbins = len(potential[0])

    X = np.arange(org[0], org[0]+ nbins*bn[0], bn[0])
    Y = np.arange(org[1], org[1]+ nbins*bn[1], bn[1])
    big = max( abs(mn)-abs(0.1*mn), abs(mx)+abs(mx)*0.1)
    plt.pcolor(X, Y, potential, cmap = 'seismic_r',
                    vmin=-big, vmax=big)
    plt.colorbar()

    ax.set_xlim([X[0], X[-1]])
    ax.set_ylim([Y[0], Y[-1]])

    ax.xaxis.labelpad = -1.4
    ax.yaxis.labelpad = -1.4

    plt.title(title, fontsize = 13);
    ax.set_ylabel(ylab, fontsize = 12);
    ax.set_xlabel(xlab, fontsize = 12)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    if outFile != None:
        plt.savefig(outFile,bbox_inches='tight', dpi = 300)
    plt.close()

#--------------------------------------------------------------------------------
# main

esp, org, dl, ax, axval, units, mx, mn = FileOpen(fileName)
boxl = len(esp[0])

for i in range(len(esp)):
    for j in range(len(esp[i])):
        if np.isnan(esp[i][j]):
            esp[i][j] = float('Inf')

titl = 'Cross section at ' + ax + ' = ' + str(axval)
titl += ' in ' + units
xla = r'Y ($\AA$)'; yla = r'Z ($\AA$)'
if ax == 'y':
    xla = r'X ($\AA$)'; yla = r'Z ($\AA$)'
elif ax == 'z':
    xla = r'X ($\AA$)'; yla = r'Y ($\AA$)'

dispPlot( org, dl, len(esp[0]), np.transpose(esp),
                mx, mn, titl,
                xlab = xla, ylab = yla,
                outFile=outFile)
