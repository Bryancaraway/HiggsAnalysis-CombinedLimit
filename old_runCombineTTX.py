#                      #  
##                    ##
########################                               
### TTZ/H, Z/H to bb ###
### python script    ###                               
### to run combine   ###
### for multiSignal  ###                               
########################                               
### written by:      ###                               
### Bryan Caraway    ###                               
########################                               
##                    ##                                 
#                      #

import sys
import subprocess
import os
'''
Use cases --- 

To combine cards in Higgs-Combine-Tool dir:
=== python runCombineTTX.py C ===

To build work space to generate .root workspace combine input:
=== python runCombineTTX.py [input data card .txt file] ===

To run combine with input .root file to generate limits, expected significance:
=== python runCombineTTX.py [input workspace .root file] M [csv of POI to scan over]===
Keys: 
M  --- multidimensional fit
MS --- multidimensional fit with fast scan
'''
points = 400
# Channel list
ch_list = [
    'HZpt_200to300',
    #'HZpt_ge300'   ,
    'HZpt_300to400',
    'HZpt_ge400'
    #'HZpt_300to450',
    #'HZpt_ge450'
]
# POI list
poi_list = [
    'ttZbin1',
    'ttHbin1',
    'ttZbin2',
    'ttHbin2',
    'ttZbin3',
    'ttHbin3',
    'ttZbin4',
    'ttHbin4'


    #'ttHgenlopt' ,
    #'ttZgenlopt' 
    #'ttHgenhipt' ,
    #'ttZgenhipt' 
    #'ttHelse'    ,
    #'ttZelse'
]

#
if len(sys.argv) == 2:
    if sys.argv[1] == 'C':
        command = 'cd Higgs-Combine-Tool; combineCards.py '#HZpt_200to300=HZpt_200to300.txt HZpt_ge300=HZpt_ge300.txt > HZpt.txt; cd -'
        for ch in ch_list:
            command += ch+'='+ch+'.txt '
        command += '> HZpt_'+ch_list[-1].split('ge')[-1]+'inc.txt; cd -'
        print(command)
        os.system(command)
        exit()
    #
    input_dc= sys.argv[1]
    if '/' in input_dc:
        outputWSP = sys.argv[1].split('/')[1].split('.')[0]+'.root'
    else:
        outputWSP = input_dc.split('.')[0]+'.root'
    command = 'text2workspace.py -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel  --PO verbose '
    for poi in poi_list:
        command += '--PO \'map=.*/'+poi+':r_'+poi+'[1,-3,3]\' '
    command += input_dc+' -o '+outputWSP
    #
elif len(sys.argv) == 4 and '.root' in sys.argv[1]:
    poi_list = sys.argv[3].split(',')
    if sys.argv[2] == 'M':
        command = 'combine -M MultiDimFit '+sys.argv[1]+' --floatOtherPOIs=1 --algo=grid --points='+str(points)+' -n '+sys.argv[3]+' -t -1'#--fastScan '  
    if sys.argv[2] == 'MS':
        command = 'combine -M MultiDimFit '+sys.argv[1]+' --floatOtherPOIs=0 --algo=grid --points='+str(points)+' --fastScan '
    p_str = ' '
    r_str = ' --setParameterRanges '
    sP_str= ' --setParameters '
    for poi in poi_list:
        p_str += '-P r_'+poi+' '
        r_str += 'r_'+poi+'=-3.0,5.0'
        sP_str+= 'r_'+poi+'=1.0'
        if poi != poi_list[-1]:
            r_str += ':'
            sP_str+= ','
    command += p_str+r_str+sP_str
    #
else : 
    print(30*'=-=')
    print('Please refer to file for intructions on how to run this code!!!')
    print(30*'=-=')
    exit()
#
print(command)
os.system(command)


# combine -M MultiDimFit HZpt.root -P r_ttHbb_lowpt -P r_ttZbb_lowpt --floatOtherPOIs=0 --algo=grid --points=900 --setParameterRanges r_ttHbb_lowpt=-3,3:r_ttZbb_lowpt=-3,3 --fastScan -t -1
