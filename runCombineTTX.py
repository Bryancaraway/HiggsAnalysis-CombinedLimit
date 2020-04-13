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

input_dc= sys.argv[1]
outputWP= sys.argv[1].split('/')[1].split('.')[0]+'.root'
# POI list
poi_list = [
    'ttHbb_lowpt' ,
    'ttZbb_lowpt' ,
    'ttHbb_highpt',
    'ttZbb_highpt'
]
#
command = 'text2workspace.py -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel  --PO verbose '
for poi in poi_list:
    command += '--PO \'map=.*/'+poi+':r_'+poi+'[1,0,10]\' '
command += input_dc+' -o '+outputWP
print(command)
os.system(command)


# combine -M MultiDimFit HZpt.root -P r_ttHbb_lowpt -P r_ttZbb_lowpt --floatOtherPOIs=0 --algo=grid --points=900 --setParameterRanges r_ttHbb_lowpt=-3,3:r_ttZbb_lowpt=-3,3 --fastScan -t -1
