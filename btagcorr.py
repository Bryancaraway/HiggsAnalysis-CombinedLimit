import ROOT 


input_file = 'fitdiag_roots/fitDiagnostics_partblind_run2.root'

roo =  ROOT.TFile.Open(input_file, "READ") 
fit_s = roo.Get("fit_s")
jes_list = [jec+y for jec in ['jesRelativeSample','jesHF' , 'jesAbsolute', 'jesEC2', 'jesBBEC1'] for y in ['2016','2017','2018']] + ['jesHF' , 'jesAbsolute', 'jesEC2', 'jesBBEC1', 'jesRelativeBal', 'jesFlavorQCD','jesHEMIssue']

print("Post-fit correlation between jes source and btagjes")
sort_dict = {}
for jes in jes_list:
    sort_dict[fit_s.correlation("btgjes",jes)] = jes
    #print(jes," : ",fit_s.correlation("btgjes",jes))
sort_corr_list = sorted(sort_dict.keys(),key=(lambda _ : -abs(_)))
for corr in sort_corr_list:
    print(sort_dict[corr]," : ",corr)
