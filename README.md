HiggsAnalysis-CombinedLimit
===========================

### Official documentation

[Manual to run combine](http://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/)


### TTX workspace guide
```
cmsrel CMS_10_2_9
cd CMS_10_2_9/src
cmsenv
git clone https://github.com/Bryancaraway/TTX/HiggsAnalysis-CombinedLimit.git
git checkout TTX
scramv1 b clean; scramv1 b
```
### Run combine for TTX analysis
```
python runCombineTTX.py
combine -M MultiDimFit HZpt.root -P r_ttHbb_lowpt -P r_ttZbb_lowpt --floatOtherPOIs=0 --algo=grid --points=900 --setParameterRanges r_ttHbb_lowpt=-3,3:r_ttZbb_lowpt=-3,3 --fastScan -t -1   
```
for example...
