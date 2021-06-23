import uproot
import re
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


#r16 = uproot.open("Higgs-Combine-Tool/datacard_partblind_2016.root")
#r17 = uproot.open("Higgs-Combine-Tool/datacard_partblind_2017.root")
#r18 = uproot.open("Higgs-Combine-Tool/datacard_partblind_2018.root")
r16 = uproot.open("Higgs-Combine-Tool/datacard_2016.root")
r17 = uproot.open("Higgs-Combine-Tool/datacard_2017.root")
r18 = uproot.open("Higgs-Combine-Tool/datacard_2018.root")


#proc = 'data_obs'
#proc = 'TTBar'
proc = 'ttZ3'

ddict = {k.decode(): {'2016':r16[k].values, '2017':r17[k].values, '2018':r18[k].values} for k in r16.keys() if re.match(r"Zhpt\d_{};1".format(proc), k.decode()) is not None}
ddict_err = {k.decode(): {'2016':r16[k].variances, '2017':r17[k].variances, '2018':r18[k].variances} for k in r16.keys() if re.match(r"Zhpt\d_{};1".format(proc), k.decode()) is not None}
print(ddict_err)

def plot_channel(ch):
    bins = np.arange(0,len(ddict[ch]['2016']))+.5
    for i,y in enumerate(ddict[ch]):
        plt.errorbar(
            x=bins+i*.15,
            y=ddict[ch][y]/sum(ddict[ch][y]),
            yerr= np.sqrt(ddict_err[ch][y])/sum(ddict[ch][y]),
            fmt='.',
            label=y
        )
    plt.yscale('log')
    plt.title(ch)
    plt.legend()
    plt.savefig(ch.rstrip(';1')+'.pdf')
    plt.close()
    #plt.show()

for ch in ddict:
    plot_channel(ch)

command = "pdfunite Zhpt*{0}*.pdf Zhpt_{0}_norm.pdf ; rm Zhpt*{0}.pdf ; mv Zhpt_{0}_norm.pdf WC_pdf/".format(proc)

#os.system("pdfunite Zhpt*data_obs*.pdf Zhpt_data_norm.pdf")
#os.system("rm Zhpt*data_obs*.pdf")
#os.system("mv Zhpt_data_norm.pdf WC_pdf/")
os.system(command)
