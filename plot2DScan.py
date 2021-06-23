#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import AutoMinorLocator
import uproot
import argparse
import numpy as np

plt.rc("font", size=10, family="sans-serif", **{"sans-serif" : [u'TeX Gyre Heros', u'Helvetica', u'Arial']})
#plt.rc("xaxis", labellocation='right')
#plt.rc("yaxis", labellocation='top')
plt.rc("legend", fontsize=10, scatterpoints=1, numpoints=1, borderpad=0.15, labelspacing=0.2,
                 handlelength=0.7, handletextpad=0.25, handleheight=0.7, columnspacing=0.6,
                 fancybox=False, edgecolor='none', borderaxespad=0.15)
plt.rc("savefig", dpi=200)
plt.rc("figure", figsize=(3.375, 3.375*(6./8.)), dpi=200)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble='\n'.join([r"\usepackage{amsmath}",
                                         r"\usepackage{helvet}",
                                         r"\usepackage{sansmath}",
                                         r"\sansmath"]))

parser = argparse.ArgumentParser()

parser.add_argument('main', help='Main input file for the scan',) #default='higgsCombinectW_ctZ.MultiDimFit.mH120.root') # for testing
#parser.add_argument('--y-cut', type=float, default=7., help='Remove points with y > y-cut')
#parser.add_argument('--y-max', type=float, default=8., help='y-axis maximum')
#parser.add_argument('--output', '-o', help='output name without file extension', default='scan')
parser.add_argument('--POIs', nargs='+', help='use this parameter of interest', default=['ctW','ctZ']) # for testing
parser.add_argument('--nCLs', default=3, type=int, help='Number of Confidence levels')
parser.add_argument('--isSM', action='store_true', help='Center SM marker at 1,1')
#parser.add_argument('--translate', default=None, help='json file with POI name translation')
#parser.add_argument('--main-label', default='Observed', type=str, help='legend label for the main scan')

#parser.add_argument('--others', nargs='*', help='add secondary scans processed as main: FILE:LABEL:COLOR')
#parser.add_argument('--breakdown', help='do quadratic error subtraction using --others')
#parser.add_argument('--logo', default='CMS')
#parser.add_argument('--logo-sub', default='Internal')
args = parser.parse_args()

if len (args.POIs) != 2:
    raise ValueError("You must pass two POIs to --POIs")

l_dict = {
    'r_ttZ' : r'${\mu}_{\mathrm{t\bar{t}Z}}$',
    'r_ttZ0' : r'${\mu}_{\mathrm{t\bar{t}Z}}0$',
    'r_ttZ2' : r'${\mu}_{\mathrm{t\bar{t}Z}}2$',
    'r_ttH0' : r'${\mu}_{\mathrm{t\bar{t}H}}0$',
    'r_ttH1' : r'${\mu}_{\mathrm{t\bar{t}H}}1$',
    'r_ttH2' : r'${\mu}_{\mathrm{t\bar{t}H}}2$',
    'r_ttH' : r'${\mu}_{\mathrm{t\bar{t}H}}$',
}

lim_dict = {'r_ttH': [-2,4], 'r_ttZ': [-2,4]}

def main():
    # get results from input root file
    lim = uproot.open(args.main)['limit']
    nll  = 2*lim.array('deltaNLL')
    poi1 = lim.array(args.POIs[0])
    poi2 = lim.array(args.POIs[1])
    # make contour plots
    fig, ax = plt.subplots()
    fig.subplots_adjust(
        top=0.85,
        bottom=0.14,
        left=0.14,
        right=0.85,
        wspace=0.0,
        hspace=0.0
    )
    levels = [2.29574893,  6.18007431, 11.82915808][:args.nCLs]
    tslabels = ['68\% CL','95\% CL','99.7\% CL'][:args.nCLs]
    ts = ax.tricontour(poi1,poi2, nll, levels=levels, colors=['gold','blue','green'])
    ax.clabel(ts, fmt={l:ls for l,ls in zip(levels, tslabels)}, inline=1, fontsize=10)

    for i in range(len(tslabels)):
        ts.collections[i].set_label(tslabels[i])
    #help(ax.set_xlabel)
    ax.set_xlabel(l_dict[args.POIs[0]],
                  fontsize=10)
    ax.xaxis.set_label_coords(0.95, -0.1)
    ax.set_ylabel(l_dict[args.POIs[1]],
                  fontsize=10)
    ax.yaxis.set_label_coords(-0.07, 0.95)
    
    # draw SM point at 0,0
    if args.isSM:
        sm = ax.scatter([1],[1], c='r', marker='d', label='SM value', zorder=5)
    else:
        sm = ax.scatter([0],[0], c='r', marker='d', label='SM value', zorder=5)
    
    #
    #if args.POIs[0] in lim_dict and args.POIs[1] in lim_dict:
    #    ax.set_xlim(lim_dict[args.POIs[0]])
    #    ax.set_ylim(lim_dict[args.POIs[1]])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid()
    ax.legend(ncol=len(tslabels)+1, scatterpoints=1, fontsize=10, bbox_to_anchor=(0.05, .985), loc='upper left')
    #ax.legend(scatterpoints=1, fontsize=10, bbox_to_anchor=(1.01, .97), loc='upper left')
    #
    CMSlabel(fig, ax)
    #fig.text(0.70,0.94, r"$\bf{CMS}$ $Preliminary$", fontsize = 12)
    #fig.text(0.70,0.91, '137'+r' fb$^{-1}$ (13 TeV)',  fontsize = 12)
    #plt.tight_layout()
    #plt.show()
    plt.gca().set_aspect('equal','datalim')
    plt.savefig('run2_scan2d_{}.pdf'.format('_'.join(args.POIs)))



def upperlefttext(s):
    trans = gca().transAxes + matplotlib.transforms.ScaledTranslation(3/72, -3/72, gcf().dpi_scale_trans)
    return text(0, 1, s, transform=trans, ha='left', va='top')

def CMSlabel(fig=None, ax=None):
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    if type(ax) is np.ndarray:
        ax0, ax1 = ax[0,0], ax[0,1]
    #
    else:
        ax0, ax1 = ax, ax
    trans = ax0.transAxes + matplotlib.transforms.ScaledTranslation(0/72, 3/72, fig.dpi_scale_trans)
    #ax0.text(1, 1.12, r"\textbf{CMS} {\footnotesize \textit{Preliminary}}",
    #        transform=trans, ha='right', va='baseline')
    ax0.text(0, 1.02, r"\textbf{CMS} {\footnotesize \textit{Preliminary}}",
            transform=trans, ha='left', va='baseline')
    trans = ax1.transAxes + matplotlib.transforms.ScaledTranslation(0/72, 3/72, fig.dpi_scale_trans)
    ax1.text(1, 1.02, r"$137\,\mathrm{fb}^{\text{-1}}$ (13 TeV)",
            transform=trans, ha='right', va='baseline')


if __name__ == '__main__':
    main()
