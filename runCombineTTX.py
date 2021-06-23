#!/usr/bin/env python
import argparse
import os
import re
from glob import glob
import numpy as np
from functools import partial
import time
import subprocess as sb
#
#combine -M FitDiagnostics datacard_run2.wp.root --plots --saveShapes --saveWithUncertainties -n _blind_run2 --setParameters r_ttZ=1,r_ttH=1 --setParameterRanges r_ttZ=.999,1.001:r_ttH=.999,1.001 -t -1
#combine -M FitDiagnostics datacard_run2.wp.root --plots --saveShapes --saveWithUncertainties -n _stxs_run2 --setParameters r_ttZ0=1,r_ttZ1=1,r_ttZ2=1,r_ttZ3=1,r_ttH0=1,r_ttH1=1,r_ttH2=1,r_ttH3=1 -t -1

#combine -M FitDiagnostics datacard_run2.inc.wp.root --saveShapes --saveWithUncertainties -n _inc_run2 --setParameters r_ttZ=1,r_ttH=1 -t -1

#text2workspace.py -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --channel-masks  --PO verbose --PO 'map=.*/ttZ0:r_ttX0[1,-10,10]' --PO 'map=.*/ttH0:r_ttX0[1,-10,10]' --PO 'map=.*/ttZ1:r_ttX1[1,-10,10]' --PO 'map=.*/ttH1:r_ttX1[1,-10,10]' --PO 'map=.*/ttZ2:r_ttX2[1,-10,10]' --PO 'map=.*/ttH2:r_ttX2[1,-10,10]' --PO 'map=.*/ttZ3:r_ttX3[1,-10,10]' --PO 'map=.*/ttH3:r_ttX3[1,-10,10]' Higgs-Combine-Tool/datacard_run2.txt -o datacard_run2.ttx.wp.root

dnn_ZH_vars = [
    'outZH_b1_pt','outZH_b2_pt','outZH_b1_score','outZH_b2_score',
    'outZh_q1_pt','outZh_q2_pt','outZh_q1_btag','outZh_q2_btag',
    'outZH_b1_q_mindr','outZH_b2_q_mindr','outZH_q_q_dr_nearb1','outZH_q_q_dr_nearb2',
    'outZH_qq_M_nearb1','outZH_qq_M_nearb2',#'outZH_b1q_M',#'outZH_b2q_M',
    'outZH_b1_qq_dr','outZH_b2_qq_dr','outZH_b1qq_M','outZH_b2qq_M',
    'ZH_b1qq_dr','ZH_b2qq_dr','ZH_lbb1qq_dr','ZH_lbb2qq_dr',
    'l_b2_mtb','Zh_closeb_invM',#'Zh_closeq_invM',
    'n_ak8jets', 'n_ak4jets','n_ak8_Zhbb','outZh_max_ak8sdM','outZh_b12_m', 'outZh_b12_dr', 
    'ht_b', 'ht_outZh', 
    'ak4_bestb_inZH', 'ak4_worstb_inZH',
    #'n_Zh_btag_sj','Zh_bestb_sj', 'Zh_worstb_sj',
    'nonZhbb_q1_dr', 'nonZhbb_b1_dr','inZhb_outZhb_dr','Zh_l_dr', 'Zh_l_invM_sd', 
    'l_b1_invM','l_b2_invM',
    'l_b1_dr','l_b2_dr','spher','aplan','n_b_inZh', 'n_q_inZh','n_b_outZh', 'n_q_outZh',
    'Zh_bbvLscore',
]
hl2_vars = ['NN_'+str(i) for i in range(64)]
qc_or_nn_vars = {'qcnn':dnn_ZH_vars, 'hl2':hl2_vars}
years = ['2016','2017','2018']

#
parser = argparse.ArgumentParser(description='Run Combine scripts and tools on analysis')
parser.add_argument('-c','--combine', dest='combine', action='store_true', required=False, 
                    help='run combineCards on datacards', default=False)
parser.add_argument('--ceft','--combineeft', dest='combineeft', action='store_true', required=False, 
                    help='run combineCards on eft datacards', default=False)
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
parser.add_argument('--impacts','-I', dest='impacts', action='store_true', required=False, 
                    help='run impacts', default=False)
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
parser.add_argument('--gof', type=str, required=False,
                    help='check gof for input datacard, and run many toys to compute the p-score', default=None)
parser.add_argument('--gofnn', type=bool, required=False,
                    help='run for many inputs, for many toys, to compute the p-score for each nn input', default=None)
parser.add_argument('--vars', type=str, required=False,
                    help='which variable set to run qcnn or gofnn over', default='qcnn')
parser.add_argument('--nogoft0', required=False,
                    help='do not run t0 for gof', action='store_true', default=False)
parser.add_argument('--uncbreak', dest='uncbreak', action='store_true', required=False,
                    help='display results with uncertainty breakdown', default=False)
#
parser.add_argument('--isnotblind', dest='isblind', action='store_false', required=False, 
                    help='run with Asimov (calling this argument stores false)', default=True)
#
args = parser.parse_args()

class runCombine():
    
    combine_dir = '/home/bcaraway/CombineTool/CMSSW_10_2_9/src/HiggsAnalysis/CombinedLimit'
    #
    dc_dir = 'Higgs-Combine-Tool/'
    eft_dir = 'Higgs-Combine-Tool/eftdatacards/'
    dc_namef = 'datacard_tag_year.txt'
    roo_namef = 'datacard_tag_year.root'
    #input_eft = 'EFT_Parameterization_v5.npy' # gen,  no ttbb
    input_eft = 'EFT_Parameterization_v7.npy' # full reco ttbb ####### this one is to be used for reco
    crab_dir  = 'crab_projects/'
    crab_store = '/cms/data/store/user/bcaraway/crab_test/Combine'
    #
    n_gen_bins = args.gbins
    wp = args.wp
    asimov_str = '-t -1 ' if args.isblind else ''
    #pois = ['ttZbb','ttHbb'] 
    pois = ['ttZ','ttH'] 
    wc_list  = ['cQei','cQl3i','cQlMi','cbW','cpQ3','cpQM','cpt','cptb','ctG','ctW','ctZ','ctei','ctlSi','ctlTi','ctli','ctp']
    relwc_list = ['cbW','cpQ3','cpQM','cpt','cptb','ctW','ctZ','ctp']
    wc_range = {
        'cQei' :[-200,200],
        'cQl3i':[-200,200],
        'cQlMi':[-200,200],
        'cbW'  :[-9,9],
        'cpQ3' :[-7,7],
        'cpQM' :[-20,20],
        'cpt'  :[-20,20],
        'cptb' :[-25,25],
        'ctG'  :[-1.0,1.0],
        'ctW'  :[-4,4],
        'ctZ'  :[-4,4],
        'ctei' :[-200,200],
        'ctlSi':[-200,200],
        'ctlTi':[-200,200],
        'ctli' :[-200,200],
        'ctp'  :[-20,50],
    }
    #poi_list = [p+str(i) for i in range(args.npoi) for p in pois]
    poi_list = [p+str(i) for i in range(args.npoi) for p in pois]
    if args.npoi == 1:
        poi_list = pois
    

    def __init__(self):
        #
        self.years = [args.year] if args.year != 'all' else years
        self.datacards = [self.dc_namef.replace('year', y).replace('tag',args.tag).replace('__','_') for y in self.years] # which datacards to combine
        # run selectred script
        if args.combine:   
            self.combine_datacards()
        elif args.combineeft:   
            self.combine_eft_datacards()
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
                import multiprocessing
                pool = multiprocessing.Pool(8)
                arg = "--wp {0} {1} --wcs".format(args.wp, '--floatpoi' if args.floatPOI==True else '')
                w_ = partial(worker,arg=arg)
                _ = pool.map(w_, self.relwc_list)
                #for wc in self.wc_list:
                #    wc_scan([wc], floatPOI=args.floatPOI, isfast=args.fastscan, makepdf=True)
            else:
                wc_scan(args.wcs, floatPOI=args.floatPOI, isfast=args.fastscan, makepdf=True)

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

        elif args.impacts:
            self.run_impacts()
        elif args.json is not None:
            self.run_impactplots()
        elif args.wcjson is not None:
            self.run_wcimpactplots()
        elif args.qcnn is not None:
            self.run_worker('--qc')
        elif args.qc is not None:
            self.run_qcfit()
        elif args.gofnn is not None:
            self.run_worker('--gof')
        elif args.gof is not None:
            self.run_gof()
        elif args.uncbreak:
            self.unc_breakdown()
        #
        else: print("Please select script mode from options: python runCombineTTX.py --help")

    def combine_datacards(self, alt_dir=None):
        dir_ = self.dc_dir if alt_dir is None else alt_dir
        out_datacard = 'datacard_tag_run2.txt'.replace('tag',args.tag).replace('__','_')# when combining datacards
        #
        command = 'cd {}; combineCards.py '.format(dir_)
        for dc in self.datacards:
            #command += re.search(r'201\d',dc).group()+'='+dc+' '
            command += 'y{}='.format(re.search(r'201\d',dc).group()) + dc+' '
        command += '> {}; cd -'.format(out_datacard)
        self.issue_command(command)

    def combine_eft_datacards(self):
        self.datacards = []
        for y in years:
            tmp_txt = 'tmptmp__.txt'
            out_datacard = 'datacard_tag_{}.txt'.format(y).replace('tag',args.tag).replace('__','_')# when combining datacards
            self.datacards.append(out_datacard)
            #
            command = 'cd {}; combineCards.py '.format(self.eft_dir)
            command += 'datacard_tag_*_{1}.txt'.format(self.eft_dir,y).replace('tag',args.tag).replace('__','_')
            command += ' > {}; cd -'.format(tmp_txt)
            self.issue_command(command)
            command = ''
            with open(self.eft_dir+tmp_txt,'r') as tmp_:
                with open(self.eft_dir+out_datacard,'w') as out_:
                    out_.writelines([re.sub(r'ch\d*_','',l) for l in tmp_.readlines()])
            self.issue_command('rm '+self.eft_dir+tmp_txt)
        #
        self.combine_datacards(alt_dir=self.eft_dir)
            

    def run_t2w(self, dc):
        if '/' in dc:
            out_wsp = dc.split('/')[1].split('.')[0]+'.wp.root'
        else:
            out_wsp = dc.split('.')[0]+'.wp.root'
        if self.n_gen_bins == 1:
            out_wsp = out_wsp.replace('.wp','.inc.wp')
        else:
            out_wsp = out_wsp.replace('.wp','.stxs.wp')
        #
        range_dict = {
            'ttZ0': '[1,-30,30]',
            'ttZ1': '[1,-30,30]',

            'ttH0': '[1,-30,30]',
            'ttH1': '[1,-30,30]',
        }
        #
        command = 'text2workspace.py -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --channel-masks  --PO verbose '
        for poi in self.poi_list:
            r_param = poi
            if self.n_gen_bins-1 < int(re.search(r'\d$',poi).group()): r_param = re.sub(r'\d$',str(self.n_gen_bins-1),poi)
            if self.n_gen_bins == 1:
                #    if '0' in poi: continue
                r_param = re.sub(r'\d$','',poi)
            command += '--PO \'map=.*/'+poi+':r_'+r_param+range_dict.get(poi,'[1,-10,10]')+'\' '
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
        command = 'combine -M MultiDimFit {} --floatOtherPOIs=1 --algo=grid --points=400 -n {} -t -1'.format(self.wp,'_'+'_'.join(self.poi_list))#--fastScan '
        p_str = ' '
        r_str = ' --setParameterRanges '
        sP_str= ' --setParameters '
        for poi in self.poi_list:
            if isfast: 
                command += ' --fastScan '
            p_str += '-P r_'+poi+' '
            r_str += 'r_'+poi+'=-2.0,4.0:'
            sP_str+= 'r_'+poi+'=1.0,'
        command += p_str+r_str.rstrip(':')+sP_str.rstrip(',') 
        self.issue_command(command)
        #os.system('sleep 5')

    ## need WC scanner
    def run_WCscan(self, wcs, floatPOI=False, isfast=False, dofreeze=True, makepdf=False):
        fl_or_fi = 'float' if floatPOI else 'fixed'
        #wc_list = set(self.wc_list)
        #command = 'combine -M MultiDimFit {} --floatOtherPOIs={} --algo=grid -v 0 --points=50 -n {} -t -1'.format(self.wp,int(floatPOI),'_'.join(wcs))#--fastScan ' 
        #command = 'combine -M MultiDimFit -d {} --floatOtherPOIs={} --freezeParameters {} -v 0 -n {} -t -1 '.format(self.wp,','.join([wc for wc in self.relwc_list if wc not in wcs]),'_'.join(wcs))#--fastScan ' 

        command = 'combine -M MultiDimFit -d {} --floatOtherPOIs={} -v 0 -n {} -t -1 '.format(self.wp,int(floatPOI),'_'.join(wcs)+'_'+fl_or_fi)#--fastScan ' 
        
        if args.wcs is not None:
            #command += ' --algo=grid --points=5 '
            command += ' --algo=grid --points=50 '
        elif args.wcs2d is not None:
            command += ' --algo=grid --points=2000 '
        if isfast: 
            command += ' --fastScan '
        
        p_str =  ' '.join([' -P '+wc for wc in wcs]) 
        r_str =  ' --setParameterRanges ' + ':'.join([wc+'={0},{1}'.format(*self.wc_range[wc]) for wc in wcs])
        sP_str = ' --setParameters ' +      ','.join([wc+'=0.0'.format(*self.wc_range[wc]) for wc in wcs])
        command += p_str+r_str+sP_str
        #command = ''
        if makepdf and len(wcs) == 1: # only do this for 1d scans
            pdf_name = "run2_{1}_{0}".format(wcs[0], fl_or_fi)
            command += '; plot1DScan.py higgsCombine{0}.MultiDimFit.mH120.root -o {1} --POI {2} --main-label Expected'.format(str(wcs[0])+'_'+fl_or_fi,pdf_name,wcs[0])
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
        

    def run_impacts(self):
        param_dict = {
            'stxs': '--setParameters '+','.join(['r_tt'+p+str(i)+'=1.0' for p in ['Z','H'] for i in range(self.n_gen_bins)]),
            'inc' : '--setParameters r_ttZ=1.0,r_ttH=1.0',
        }
        if args.isblind:
            asimov_str = '-t -1 ' #--expectSignal 1 '
            asimov_str += param_dict[re.search(r'(stxs|inc)',self.wp).group()]
        else:
            asimov_str = ''
        #command = 'combineTool.py -M Impacts -d {0} -m 125 --doInitialFit --robustFit 1 {1} --freezeNuisanceGroup=sig_rtheo'.format(self.wp, asimov_str) # why do i have to input stupid mass
        if 'inc' not in self.wp:
            add_freeze = '--freezeNuisanceGroup=sig_rtheo'
        else:
            add_freeze = ''
        command = 'combineTool.py -M Impacts -d {0} -m 125 --doInitialFit {1} --robustHesse=1 {2}'.format(self.wp, asimov_str, add_freeze)
        #command = 'combineTool.py -M Impacts -d {0} -m 125 --doInitialFit {1} --robustHesse=1 --freezeParameters CMS_ttbbnorm,tt2bxsec --freezeNuisanceGroup=sig_rtheo'.format(self.wp, asimov_str) # why do i have to input stupid mass

        #command = 'combineTool.py -M Impacts -d {0} -m 125 --doInitialFit --robustFit 1 {1} '.format(self.wp, asimov_str) # why do i have to input stupid mass
        self.issue_command(command)
        #
        if args.initial: return
        command = 'combineTool.py -M Impacts -d {0} -m 125 --doFits       --robustFit 1 {1} --parallel 60 {2}'.format(self.wp, asimov_str, add_freeze)
        self.issue_command(command)
        #
        command = 'combineTool.py -M Impacts -d {0} -m 125 -o impacts_{1}.json {2}'.format(self.wp, self.wp.replace('datacard_','').replace('.wp.root','').rstrip('_'), add_freeze)
        self.issue_command(command)
        
    def run_impactplots(self):
        for i in range(args.npoi):
            for j in ['Z','H']:
                #command = 'plotImpacts.py -i {0} -o {1} --POI r_tt{2}{3} --max-pages 1'.format(args.json, args.json.replace('.json','_tt{0}{1}'.format(j,i)), j,i)
                command = 'plotImpacts.py -i {0} -o {1} --POI r_tt{2}{3} '.format(args.json, args.json.replace('.json','_tt{0}{1}'.format(j,i)), j,i)
                self.issue_command(command)
        #command = 'for i in 0 1 2 3; do for j in Z H; do plotImpacts.py -i nomcstats_asi_scrun2_2018_impacts.json -o nomcstats_asi_scrun2_2018_tt$j\bb$i --POI r_tt$j\bb$i;  done; done'

    def run_wcimpactplots(self):
        for wc in self.relwc_list:
            command = 'plotImpacts.py -i {0} -o {1} --POI {2} --max-pages 1'.format(args.wcjson, args.wcjson.replace('wc.json',wc), wc)
            self.issue_command(command)
                                                                   

    def run_worker(self,arg):
        import multiprocessing
        pool = multiprocessing.Pool(8)
        w_ = partial(worker,arg=arg,nn=True)
        _ = pool.map(w_, qc_or_nn_vars[args.vars])
        #for v in dnn_ZH_vars:
        #    w_(v)
         
        
        
    def run_qcfit(self):
        tag = args.qc
        self.issue_command('python runCombineTTX.py -c --tag {}'.format(tag))
        def run_func(y,tag,func):
            func(y,tag)
        #[run_func(y,tag, self.run_fitDiag) for y in self.years + ['run2']]
        #[run_func(y,tag, self.run_diffn)   for y in self.years + ['run2']]
        run_func('run2',tag,self.run_fitDiag)
        run_func('run2',tag,self.run_diffn)
    
    def run_fitDiag(self,y,tag):
        #command = 'combine -M FitDiagnostics Higgs-Combine-Tool/datacard_{1}_{0}.txt --plots --saveShapes --saveWithUncertainties -n _{1}_{0} --setParameters r=1 --setParameterRanges r=.999,1.001'.format(y,tag)
        command = 'combine -M FitDiagnostics Higgs-Combine-Tool/datacard_{1}_{0}.txt --saveShapes --saveWithUncertainties -n _{1}_{0} --setParameters r=1 --setParameterRanges r=.999,1.001'.format(y,tag)
        self.issue_command(command)
    def run_diffn(  self,y,tag):
        command = 'python test/diffNuisances.py fitDiagnostics_{1}_{0}.root --abs --all -g diffn_{1}_{0}.root'.format(y,tag)
        self.issue_command(command)


    def run_gof(self):
        tag = args.gof

        self.issue_command('python runCombineTTX.py -c --tag {}'.format(tag))
        def run_func(y,tag,func):
            func(y,tag)
        if not args.nogoft0:
            run_func('run2',tag, self.get_t0)
        run_func('run2',tag, self.get_ttoys)
        # need to hadd the output
        run_func('run2',tag, self.hadd_toys)

    def get_t0(self,y,tag):
        command = 'combine -M GoodnessOfFit Higgs-Combine-Tool/datacard_{1}_{0}.txt -n _{1}_{0} --fixedSignalStrength=1 --algo=saturated'.format(y,tag)
        self.issue_command(command)
        
    def get_ttoys(self,y,tag):
        #command = 'combine -M GoodnessOfFit Higgs-Combine-Tool/datacard_{1}_{0}.txt -n _{1}_{0}_toy --fixedSignalStrength=1 --algo=saturated -t 250 -s {2}'.format(y,tag,int(time.time()))
        #self.issue_command(command)
        #---- 
        num_jobs_running = lambda: int(sb.check_output(
            "qstat -u $USER -w -f | grep 'Job_Name = generate_' | wc -l", shell=True).decode())
        for i in range(0,30): # 10,000 toys total
            command = 'qsub -l nodes=1:ppn=1 -N generate_toys_{1}_{0}_{2} -v tag={1}_{0},seed={3} runGOFtoys.sh'.format(y,tag,i,int(time.time())*i % 1000000)# -o test_{2}.out -e test_{2}.err
            time.sleep(1)
            while num_jobs_running()>50:
                time.sleep(30)
            self.issue_command(command)
            
    def hadd_toys(self,y,tag):
        num_jobs_running = lambda: int(sb.check_output(
            "qstat -u $USER -w -f | grep 'Job_Name = generate_toys_{1}_{0}' | wc -l".format(y,tag), shell=True).decode())
        while num_jobs_running()>0:
            time.sleep(30)
        command = 'hadd -f higgsCombine_{1}_{0}_toy.GoodnessOfFit.mH120.root higgsCombine_{1}_{0}_toy.GoodnessOfFit.mH120.*.root'.format(y,tag)
        self.issue_command(command)
        command = 'rm higgsCombine_{1}_{0}_toy.GoodnessOfFit.mH120.*.root'.format(y,tag)
        self.issue_command(command)
        # rm files

    def unc_breakdown(self):

        inc_or_stxs = 'inc' if 'inc' in self.wp else 'stxs'
        multi_settings = {
            #'inc' : '-P r_ttZ -P r_ttH --setParameters r_ttZ=1,r_ttH=1 --setParameterRanges r_ttZ=-3,4:r_ttH=-3,5',
            'inc' : '--redefineSignalPOIs r_ttZ,r_ttH --setParameters r_ttZ=1,r_ttH=1 {0}'.format(self.asimov_str),
            'stxs': '--redefineSignalPOIs r_ttZ0,r_ttH0,r_ttZ1,r_ttH1,r_ttZ2,r_ttH2,r_ttZ3,r_ttH3  --setParameters r_ttZ0=1,r_ttH0=1,r_ttZ1=1,r_ttH1=1,r_ttZ2=1,r_ttH2=1,r_ttZ3=1,r_ttH3=1 {0}'.format(self.asimov_str),
        }
        get_unc = (lambda str_ : sb.check_output("{0} | grep '(68%)'".format(str_), shell=True).decode())
        #
        total_str = 'combine -M MultiDimFit {0} -n .{1}.snapshot -m 125 --algo singles {2} --saveWorkspace'.format(self.wp, self.wp.rstrip('.root'), multi_settings[inc_or_stxs])
        total = get_unc(total_str)
        #
        freeze_scan_pre = 'combine -M MultiDimFit higgsCombine.{0}.snapshot.MultiDimFit.mH125.root  -n .{1}.dummy -m 125 --algo singles {2} --snapshotName MultiDimFit'.format(self.wp.rstrip('.root'), self.wp.rstrip('.root'), multi_settings[inc_or_stxs])
        #
        sys_frozen_str = '{0} --freezeParameters allConstrainedNuisances'.format(freeze_scan_pre)
        sys_fro = get_unc(sys_frozen_str)
        #
        mcstat_frozen_str = '{0} --freezeParameters rgx{{prop.*}}'.format(freeze_scan_pre)
        mcstat_fro = get_unc(mcstat_frozen_str)
        #
        theo_fro_str = '{0} --freezeNuisanceGroups=theo'.format(freeze_scan_pre)
        theo_fro = get_unc(theo_fro_str)
        #
        ttbb_fro_str = '{0} --freezeParameters CMS_ttbbnorm'.format(freeze_scan_pre)
        ttbb_fro = get_unc(ttbb_fro_str)
        #
        def parse_unc(unc_str):
            _out = {}
            for sstr in unc_str.split('\n')[:-1]:
                parsed = sstr.split()
                down, up = parsed[3].split('/')
                _out[parsed[0]] = [float(down),float(up)]
            return _out
        total_unc = parse_unc(total)
        sorted_keys = [k for k in sorted(total_unc)]
        get_unc_from_frozen = (lambda d: {k: np.sqrt( np.power(total_unc[k],2) - np.power(d[k],2) ) for k in total_unc})
        ttbb_unc = get_unc_from_frozen(parse_unc(ttbb_fro)) # wont keep
        mcstat_unc = get_unc_from_frozen(parse_unc(mcstat_fro)) # mcstat unc --> keep
        theo_unc   = get_unc_from_frozen(parse_unc(theo_fro)) # theo unc --> keep
        stat_unc = {k: np.sqrt(np.power(parse_unc(sys_fro)[k],2) - np.power(ttbb_unc[k],2)) for k in total_unc}  # stat unc --> keep
        exp_unc = {k: np.sqrt(np.power(total_unc[k],2) - np.power(mcstat_unc[k],2) - np.power(theo_unc[k],2) - np.power(stat_unc[k],2))for k in total_unc} # exp unc --> keep
        print('\n')
        for k in sorted_keys:
            print('{0:<10} Total unc. {1:.2f}/+{2:.2f} | -{3:.2f}(Stat.){4:.2f}(MC Stat.){5:.2f}(Exp.){6:.2f}(Theo.) / +{7:.2f}(Stat.){8:.2f}(MC Stat.){9:.2f}(Exp.){10:.2f}(Theo.)'.format(
                k, total_unc[k][0], total_unc[k][1], stat_unc[k][0], mcstat_unc[k][0], exp_unc[k][0], theo_unc[k][0], stat_unc[k][1], mcstat_unc[k][1], exp_unc[k][1], theo_unc[k][1]))
        # now look at theory systematics broken down
        # theory sys : 
        theory_sys = '''
        CMS_ttbbnorm tt2bxsec ttCxsec hdamp_ttbb hdamp UE toppt pdf_ttbb pdf
        mu_f_ttbb mu_r_ttbb mu_f_tt mu_r_tt mu_f_tth mu_r_tth mu_f_ttz mu_r_ttz
        isr_ttbb fsr_ttbb isr_tt fsr_tt isr_tth fsr_tth isr_ttz fsr_ttz
        tth_ggpdf ttz_ggpdf tth_qsc ttz_qsc ggpdf qqpdf qgpdf tt_qsc ttx_qsc singlet_qsc v_qsc
        '''
        t_sys_unc_dict = {k:{} for k in total_unc}
        for t_sys in theory_sys.split():
            fro_str = '{0} --freezeParameters {1}'.format(freeze_scan_pre, t_sys) 
            t_sys_unc = get_unc_from_frozen(parse_unc(get_unc(fro_str)))
            for k in t_sys_unc:
                t_sys_unc_dict[k][t_sys] = t_sys_unc[k]
                #print('{0:<10} {1:15} unc. {2:.2f}/+{3:.2f}'.format(k,t_sys,t_sys_unc[k][0],t_sys_unc[k][1]))

        for k in t_sys_unc_dict:
            sorted_t_sys = [t_sys for t_sys in sorted(t_sys_unc_dict[k], key= lambda _t: -1*max(t_sys_unc_dict[k][_t]))]
            print("\n")
            for t_sys in sorted_t_sys:
                print('{0:<10} {1:15} unc. {2:.3f}/+{3:.3f}'.format(k,t_sys,t_sys_unc_dict[k][t_sys][0],t_sys_unc_dict[k][t_sys][1]))
            
            
        
        

    @staticmethod
    def issue_command(c):
        print(c)
        try:
            os.system(c)
        except:
            pass



def worker(tag,arg='--qc',nn=False) :
    command = ''
    if nn:
        tag += '_NNcuts'
        #command += './runCombineTTX.py -c --tag {0} ; '.format(tag)
    command += './runCombineTTX.py {0} {1}'.format(arg,tag)
    #self.issue_command('./runCombineTTX.py --qc {}'.format(tag))
    os.system(command)


def unc_break_worker(t_sys, guff=None, pu=None, gu=None, fsp=None):
    fro_str = '{0} --freezeParameters {1}'.format(fsp, t_sys) 
    t_sys_unc = guff(pu(gu(fro_str)))
    return t_sys_unc

if __name__ == '__main__':
    runCombine()

