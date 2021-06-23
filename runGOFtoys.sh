#!/bin/bash

# very simple job submission script 

cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=1
source /home/$USER/.bashrc
eval 'source ~cmssoft/shrc; cd /home/bcaraway/CombineTool/CMSSW_10_2_9/src/HiggsAnalysis/CombinedLimit'
eval `scramv1 runtime -sh`
#args="--qc $tag --nogoft0"
#if [ ! -z "$jec" ]; then
#    args="$args -j $jec"
#fi
#echo "$args"
#eval 'python runCombineTTX.py $args'
eval 'combine -M GoodnessOfFit Higgs-Combine-Tool/datacard_${tag}.txt -n _${tag}_toy --fixedSignalStrength=1 --algo=saturated -t 150 -s ${seed}'
