#!/usr/bin/env python
import argparse
import os
import re
from glob import glob
#
years = ['2016','2017','2018']

#
parser = argparse.ArgumentParser(description='Run Combine scripts and tools on analysis')
parser.add_argument('-c','--combine', dest='combine', action='store_true', required=False, 
                    help='run combineCards on datacards', default=False)
parser.add_argument('--tag', dest='tag', type=str, required=False, 
                    help='tag for input datacards', default='')
parser.add_argument('-t','--t2w', dest='t2w', type=str, required=False, 
                    help='run text2workspace with given .txt file', default=None)
parser.add_argument('--t2weft', dest='t2weft', type=str, required=False, 
                    help='run text2workspace eft-fit with given .txt file', default=None)
parser.add_argument('-m','-M', dest='multi', action='store_true', required=False, 
                    help='run multidimfit', default=False)
parser.add_argument('--wcs', dest='wcs', nargs='+', required=False, 
                    help='run wc scan', default=None)
parser.add_argument('--wcs2d', dest='wcs2d', nargs='+', required=False, 
                    help='run 2d wc countour scan', default=None)
parser.add_argument('--crab', dest='crab', action='store_true', required=False, 
                    help='run 2d multidimfit using grid', default=False)
parser.add_argument('--floatpoi', dest='floatPOI', action='store_true', required=False, 
                    help='run multidimfit while floating other POIs', default=False)
parser.add_argument('--fastscan', dest='fastscan', action='store_true', required=False, 
                    help='run multidimfit with fast scan', default=False)
parser.add_argument('-f','-F', dest='fitd', action='store_true', required=False, 
                    help='run fit diagnostic', default=False)
parser.add_argument('-y', dest='year', type=str, choices=years+['all'],required=False, 
                    help='year or all years', default='all')
parser.add_argument('--gbins', type=int, required=False,
                    help='number of gen bins for signal', default=4)
parser.add_argument('--npoi', type=int, required=False,
                    help='number of pois per signal', default=4)
parser.add_argument('--wp', type=str, required=False,
                    help='input workspace', default='datacard_run2.wp.root')
parser.add_argument('--initial', required=False,
                    action='store_true', help='to only run initial fit', default=False)
parser.add_argument('--json', type=str, required=False,
                    help='json file to generate impact plots from', default=None)
parser.add_argument('--wcjson', type=str, required=False,
                    help='wc json file to generate impact plots from', default=None)
parser.add_argument('--qc', type=str, required=False,
                    help='one variable fits for validation, uses FitDiagnostic, and diffn', default=None)
parser.add_argument('--qcnn', type=bool, required=False,
                    help='run multiprocessing for multiple fitDiagnostics for NN inputs, NNcuts?', default=None)
#
parser.add_argument('--isnotblind', dest='isblind', action='store_false', required=False, 
                    help='run with Asimov (calling this argument stores false)', default=True)
#
args = parser.parse_args()

class runCombine():
    
    combine_dir = '/home/bcaraway/CombineTool/CMSSW_10_2_9/src/HiggsAnalysis/CombinedLimit'
    #
    dc_dir = 'Higgs-Combine-Tool/'
    dc_namef = 'datacard_tag_year.txt'
    roo_namef = 'datacard_tag_year.root'
    input_eft = 'EFT_Parameterization_v3.npy'
    crab_dir  = 'crab_projects/'
    crab_store = '/cms/data/store/user/bcaraway/crab_test/Combine'
    #
    n_gen_bins = args.gbins
    wp = args.wp
    #pois = ['ttZbb','ttHbb'] 
    pois = ['ttZ','ttH'] 
    wc_list  = ['cQei','cQl3i','cQlMi','cbW','cpQ3','cpQM','cpt','cptb','ctG','ctW','ctZ','ctei','ctlSi','ctlTi','ctli','ctp']
    relwc_list = ['cbW','cpQ3','cpQM','cpt','cptb','ctG','ctW','ctZ','ctp']
    wc_range = {'cQei' :[-200,200],
                'cQl3i':[-200,200],
                'cQlMi':[-200,200],
                'cbW'  :[-12,12],
                'cpQ3' :[-10,10],
                'cpQM' :[-30,30],
                'cpt'  :[-30,25],
                'cptb' :[-25,25],
                'ctG'  :[-1.0,1.0],
                'ctW'  :[-4,4],
                'ctZ'  :[-4,4],
                'ctei' :[-200,200],
                'ctlSi':[-200,200],
                'ctlTi':[-200,200],
                'ctli' :[-200,200],
                'ctp'  :[-20,50]}
    #poi_list = [p+str(i) for i in range(args.npoi) for p in pois]
    poi_list = [p+str(i) for i in range(args.npoi) for p in pois]

    def __init__(self):
        #
        self.years = [args.year] if args.year != 'all' else years
        self.datacards = [self.dc_namef.replace('year', y).replace('tag',args.tag).replace('__','_') for y in self.years] # which datacards to combine
        # run selectred script
        if args.combine:   
            self.combine_datacards()
        elif args.t2w is not None:     
            self.run_t2w(args.t2w)
        elif args.t2weft is not None:
            self.run_t2weft(args.t2weft)
        elif args.multi:   
            self.run_multi(args.fastscan)
        elif args.wcs is not None:
            wc_scan = self.run_WCscan
            if args.wcs[0] == 'all':
                #for wc in self.relwc_list:
                for wc in self.wc_list:
                    wc_scan([wc], floatPOI=args.floatPOI, isfast=args.fastscan, makepdf=True)
            else:
                wc_scan(args.wcs, args.fastscan)

        elif args.wcs2d is not None:
            wc_scan = self.run_WCscan
            if args.wcs2d[0] == 'all':
                #for wc in self.relwc_list
                wcs2ds = []
                for i in range(len(self.relwc_list)): # only look at relavent WC
                    for j in range(i+1, len(self.relwc_list)):
                        wcs2ds.append([self.relwc_list[i],self.relwc_list[j]]) # store all permutations
                #
                #if args.crab: self.issue_command('rm -rf '+self.crab_store) # need to remove Combine outputs first
                for wcs in wcs2ds:
                    break
                    wc_scan( wcs, floatPOI=args.floatPOI, isfast=args.fastscan, makepdf=True)
                if args.crab: # run the postpreocessing script 
                    for wcs in wcs2ds:
                        self.process_crab_output(wcs)
            else : 
                print("non-'all' value unsupported")
                exit()

        elif args.fitd:
            self.run_fitd()
        elif args.json is not None:
            self.run_impactplots()
        elif args.wcjson is not None:
            self.run_wcimpactplots()
        elif args.qcnn is not None:
            self.run_qcfitnn()
        elif args.qc is not None:
            self.run_qcfit()
        #
        else: print("Please select script mode from options: python runCombineTTX.py --help")

    def combine_datacards(self):
        out_datacard = 'datacard_tag_run2.txt'.replace('tag',args.tag).replace('__','_')# when combining datacards
        #
        command = 'cd {}; combineCards.py '.format(self.dc_dir)
        for dc in self.datacards:
            #command += re.search(r'201\d',dc).group()+'='+dc+' '
            command += 'y{}='.format(re.search(r'201\d',dc).group()) + dc+' '
        command += '> {}; cd -'.format(out_datacard)
        self.issue_command(command)

    def run_t2w(self, dc):
        if '/' in dc:
            out_wsp = dc.split('/')[1].split('.')[0]+'.wp.root'
        else:
            out_wsp = dc.split('.')[0]+'.wp.root'
        #
        command = 'text2workspace.py -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --channel-masks  --PO verbose '
        for poi in self.poi_list:
            r_param = poi
            if self.n_gen_bins-1 < int(re.search(r'\d$',poi).group()): r_param = re.sub(r'\d$',str(self.n_gen_bins-1),poi)
            command += '--PO \'map=.*/'+poi+':r_'+r_param+'[1,-10,10]\' '
        #command += '--PO \'map=.*/tt_b*:r_tt_bb_norm[1,-10,10]\' '
        command += dc+' -o '+out_wsp #.replace('eft','').replace('__','')
        self.issue_command(command)

    def run_t2weft(self, dc):
        if '/' in dc:
            out_wsp = dc.split('/')[1].split('.')[0]+'.wc.wp.root'
        else:
            out_wsp = dc.split('.')[0]+'.wp.root'
        command = 'text2workspace.py -P HiggsAnalysis.CombinedLimit.EFTModel:eftmodel  '
        command += '--PO fits=Higgs-Combine-Tool/{} '.format(self.input_eft)
        command += dc+' -o '+out_wsp
        self.issue_command(command)
    
    def run_multi(self, isfast=False):
        for poi in self.poi_list:
            command = 'combine -M MultiDimFit {} --floatOtherPOIs=1 --algo=grid --points=400 -n {} -t -1'.format(self.wp,poi)#--fastScan '
            if isfast: 
                command += ' --fastScan '
            p_str = ' '
            r_str = ' --setParameterRanges '
            sP_str= ' --setParameters '
            p_str += '-P r_'+poi+' '
            r_str += 'r_'+poi+'=-3.0,5.0'
            sP_str+= 'r_'+poi+'=1.0'
            command += p_str+r_str+sP_str 
            self.issue_command(command)
            os.system('sleep 5')

    ## need WC scanner
    def run_WCscan(self, wcs, floatPOI=False, isfast=False, dofreeze=True, makepdf=False):
        #wc_list = set(self.wc_list)
        #command = 'combine -M MultiDimFit {} --floatOtherPOIs={} --algo=grid -v 0 --points=50 -n {} -t -1'.format(self.wp,int(floatPOI),'_'.join(wcs))#--fastScan ' 
        command = 'combine -M MultiDimFit -d {} --floatOtherPOIs={} -v 0 -n {} -t -1 '.format(self.wp,int(floatPOI),'_'.join(wcs))#--fastScan ' 
        if args.wcs is not None:
            command += ' --algo=grid --points=50 '
        elif args.wcs2d is not None:
            command += ' --algo=grid --points=2000 '
        if isfast: 
            command += ' --fastScan '

        p_str =  ' '.join([' -P '+wc for wc in wcs]) 
        r_str =  ' --setParameterRanges ' + ':'.join([wc+'={0},{1}'.format(*self.wc_range[wc]) for wc in wcs])
        sP_str = ' --setParameters ' +      ','.join([wc+'=0.0'.format(*self.wc_range[wc]) for wc in wcs])
        command += p_str+r_str+sP_str

        if makepdf and len(wcs) == 1: # only do this for 1d scans
            command += '; plot1DScan.py higgsCombine{}.MultiDimFit.mH120.root -o run2_{} --POI {} --main-label Expected'.format(wcs[0],wcs[0],wcs[0])
        #print(command)
        if args.crab: # run on grid
            command = 'rm -rf {0}crab_{1}'.format(self.crab_dir, '_'.join(wcs))+' ; '+command # remove any existing crab directory first
            command += ' --job-mode crab3 --split-points 200 --task-name {} --custom-crab custom_crab.py '.format('_'.join(wcs))
            command = command.replace('combine', ' combineTool.py')
        self.issue_command(command)
        #os.system('sleep 5')

    def process_crab_output(self, wcs):
        # check to see if all files are there, should be 1000/100 = 10
        crab_o_dir = '{0}/{1}'.format(self.crab_store, '_'.join(wcs))
        while len(glob(crab_o_dir+'/*/*/*.tar')) < 10: # hardcoded 10
            os.system('sleep 60')
        # all files should now be here
        local_crab = '{0}crab_{1}/results'.format(self.crab_dir, '_'.join(wcs))
        #
        cp_command  = 'cp '+crab_o_dir+'/*/*/*.tar'+' {}/.'.format(local_crab)
        cd_untar_rm = 'cd {0}; for f in *.tar; do tar xf $f; done ; rm *.tar'.format(local_crab)
        hadd        = 'hadd higgsCombine{0}.MultiDimFit.mH120.root higgsCombine{0}.POINTS.*; rm higgsCombine{0}.POINTS.*; cp higgsCombine{0}.MultiDimFit.mH120.root {1} ; cd -'.format(
            '_'.join(wcs), self.combine_dir)
        self.issue_command(cp_command+';'+cd_untar_rm+';'+hadd) # cp to local results file and upack, cd, 
        # make contour plots
        plot_contour = './plot2DScan.py higgsCombine{0}_{1}.MultiDimFit.mH120.root --POIs {0} {1}'.format(wcs[0],wcs[1])
        self.issue_command(plot_contour)
        

    def run_fitd(self):
        if args.isblind:
            asimov_str = '-t -1 --expectSignal 1 '
        else:
            asimov_str = ''
        command = 'combineTool.py -M Impacts -d {0} -m 125 --doInitialFit --robustFit 1 {1}'.format(self.wp, asimov_str) # why do i have to input stupid mass
        self.issue_command(command)
        #
        if args.initial: return
        command = 'combineTool.py -M Impacts -d {0} -m 125 --doFits       --robustFit 1 {1} --parallel 30'.format(self.wp, asimov_str)
        self.issue_command(command)
        #
        command = 'combineTool.py -M Impacts -d {0} -m 125 -o impacts_{1}.json'.format(self.wp, self.wp.replace('datacard_','').replace('.wp.root','').rstrip('_'))
        self.issue_command(command)
        
    def run_impactplots(self):
        for i in range(args.npoi):
            for j in ['Z','H']:
                command = 'plotImpacts.py -i {0} -o {1} --POI r_tt{2}{3} --max-pages 1'.format(args.json, args.json.replace('.json','_tt{0}{1}'.format(j,i)), j,i)
                self.issue_command(command)
        #command = 'for i in 0 1 2 3; do for j in Z H; do plotImpacts.py -i nomcstats_asi_scrun2_2018_impacts.json -o nomcstats_asi_scrun2_2018_tt$j\bb$i --POI r_tt$j\bb$i;  done; done'

    def run_wcimpactplots(self):
        for wc in self.relwc_list:
            command = 'plotImpacts.py -i {0} -o {1} --POI {2} --max-pages 1'.format(args.wcjson, args.wcjson.replace('wc.json',wc), wc)
            self.issue_command(command)
                                                                   
    # worker for pool
    @staticmethod
    def worker(tag) :
        if args.qcnn:
            tag += '_NNcuts'
        command = './new_runCombineTTX.py --qc {}'.format(tag)
        #self.issue_command('./new_runCombineTTX.py --qc {}'.format(tag))
        os.system(command)
    #

    def run_qcfitnn(self):
        dnn_ZH_vars = [
            'max_lb_dr','max_lb_invM', 'n_Zh_btag_sj', 'n_ak4jets', 'Zh_score', 'best_rt_score',
            'n_q_outZh', 'n_b_outZh', 'Zh_l_dr', 'n_Zh_sj', 'n_b_inZh', 'Zh_bestb_sj', 'Zh_worstb_sj',
            'Zh_eta','Zh_deepB','b1_outZh_score', 'best_Zh_b_invM_sd', 'Zh_b1_invM_sd', 'Zh_b2_invM_sd','Zh_l_invM_sd',
            'Zh_Wscore', 'Zh_Tscore', 'n_ak8_Zhbb', 'n_ak8jets', 
            'nonZhbb_b1_dr', 'nonZhbb_b2_dr', 
            'Zh_bbscore_sj', 
            'b1_over_Zhpt', 'bb_over_Zhpt',
            'spher','aplan','n_q_inZh']
        
        #import multiprocessing
        #import pathos.multiprocesssing as multiprocessing
        #pool = multiprocessing.Pool()
        #_ = pool.map(self.worker, dnn_ZH_vars)
        for v in dnn_ZH_vars:
            self.worker(v)
        
        
    def run_qcfit(self):
        tag = args.qc
        self.issue_command('python new_runCombineTTX.py -c --tag {}'.format(tag))
        def run_func(y,tag,func):
            func(y,tag)
        #[run_func(y,tag, self.run_fitDiag) for y in self.years + ['run2']]
        #[run_func(y,tag, self.run_diffn)   for y in self.years + ['run2']]
        run_func('run2',tag,self.run_fitDiag)
        run_func('run2',tag,self.run_diffn)

    def run_fitDiag(self,y,tag):
        #command = 'combine -M FitDiagnostics Higgs-Combine-Tool/datacard_{1}_{0}.txt --saveShapes --saveWithUncertainties -n _{1}_{0}'.format(y,tag)
        #command = "combine -M FitDiagnostics Higgs-Combine-Tool/datacard_{1}_{0}.txt --backgroundPdfNames='*ttH*,*ttZ*' --saveShapes --saveWithUncertainties -n _{1}_{0}".format(y,tag)
        ##command = "./new_runCombineTTX.py --t2w Higgs-Combine-Tool/datacard_{1}_{0}.txt --gbins 1".format(y,tag)
        ##self.issue_command(command)
        #command = "combine -M FitDiagnostics datacard_{1}_{0}.wp.root  --saveShapes --saveWithUncertainties -n _{1}_{0}".format(y,tag)
        command = 'combine -M FitDiagnostics Higgs-Combine-Tool/datacard_{1}_{0}.txt --saveShapes --saveWithUncertainties -n _{1}_{0} --setParameters r=1 --setParameterRanges r=.999,1.001'.format(y,tag)
        self.issue_command(command)
    def run_diffn(  self,y,tag):
        command = 'python test/diffNuisances.py fitDiagnostics_{1}_{0}.root --abs --all -g diffn_{1}_{0}.root'.format(y,tag)
        self.issue_command(command)

    @staticmethod
    def issue_command(c):
        print(c)
        try:
            os.system(c)
        except:
            pass

if __name__ == '__main__':

    runCombine()

