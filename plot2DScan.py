#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import uproot
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('main', help='Main input file for the scan',) #default='higgsCombinectW_ctZ.MultiDimFit.mH120.root') # for testing
#parser.add_argument('--y-cut', type=float, default=7., help='Remove points with y > y-cut')
#parser.add_argument('--y-max', type=float, default=8., help='y-axis maximum')
#parser.add_argument('--output', '-o', help='output name without file extension', default='scan')
parser.add_argument('--POIs', nargs='+', help='use this parameter of interest', default=['ctW','ctZ']) # for testing
#parser.add_argument('--translate', default=None, help='json file with POI name translation')
#parser.add_argument('--main-label', default='Observed', type=str, help='legend label for the main scan')
#parser.add_argument('--main-color', default=1, type=int, help='line and marker color for main scan')
#parser.add_argument('--others', nargs='*', help='add secondary scans processed as main: FILE:LABEL:COLOR')
#parser.add_argument('--breakdown', help='do quadratic error subtraction using --others')
#parser.add_argument('--logo', default='CMS')
#parser.add_argument('--logo-sub', default='Internal')
args = parser.parse_args()

if len (args.POIs) != 2:
    raise ValueError("You must pass two POIs to --POIs")

lim = uproot.open(args.main)['limit']

nll  = 2*lim.array('deltaNLL')
poi1 = lim.array(args.POIs[0])
poi2 = lim.array(args.POIs[1])

fig, ax = plt.subplots()
ts = ax.tricontour(poi1,poi2, nll, levels=[1,4,10], colors=['gold','blue','green'])
ax.clabel(ts, fmt={1:r'$1\sigma$',4:r'$2\sigma$',10:r'$3\sigma$'}, inline=1, fontsize=10)
tslabels = [r'$1\sigma$',r'$2\sigma$',r'$3\sigma$']
for i in range(len(tslabels)):
    ts.collections[i].set_label(tslabels[i])
#help(ax.set_xlabel)
ax.set_xlabel(r'${0}_{{\mathrm{{{1}}}}}$'.format(args.POIs[0][0],args.POIs[0][1:]),
              position=(0.95, 1), fontsize=16)
ax.set_ylabel(r'${0}_{{\mathrm{{{1}}}}}$'.format(args.POIs[1][0],args.POIs[1][1:]),
              position=(1, 0.95), fontsize=16)

# draw SM point at 0,0
sm = ax.scatter([0],[0], c='r', marker='d', label='SM value', zorder=5)

#
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.grid()
ax.legend(ncol=len(tslabels)+1, scatterpoints=1, fontsize='small', bbox_to_anchor=(-0.012, .985), loc='lower left')
#
fig.text(0.70,0.94, r"$\bf{CMS}$ $Preliminary$", fontsize = 12)
fig.text(0.70,0.91, '137'+r' fb$^{-1}$ (13 TeV)',  fontsize = 12)

plt.savefig('run2_scan2d_{}.pdf'.format('_'.join(args.POIs)))

