import argparse
import os
import re
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
#
parser.add_argument('--isnotblind', dest='isblind', action='store_false', required=False, 
                    help='run with Asimov (calling this argument stores false)', default=True)
#
args = parser.parse_args()

class runCombine():
    
    dc_dir = 'Higgs-Combine-Tool/'
    dc_namef = 'datacard_tag_year.txt'
    roo_namef = 'datacard_tag_year.root'
    input_eft = 'EFT_Parameterization_v3.npy'
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
                'cbW'  :[-10,10],
                'cpQ3' :[-10,10],
                'cpQM' :[-30,30],
                'cpt'  :[-25,25],
                'cptb' :[-25,25],
                'ctG'  :[-1.0,1.0],
                'ctW'  :[-3,3],
                'ctZ'  :[-3,3],
                'ctei' :[-200,200],
                'ctlSi':[-200,200],
                'ctlTi':[-200,200],
                'ctli' :[-200,200],
                'ctp'  :[-15,50]}
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
                    wc_scan([wc], args.fastscan, makepdf=True)
            else:
                wc_scan(args.wcs, args.fastscan)
        elif args.fitd:
            self.run_fitd()
        elif args.json is not None:
            self.run_impactplots()
        elif args.wcjson is not None:
            self.run_wcimpactplots()
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
        wc_list = set(self.wc_list)
        command = 'combine -M MultiDimFit {} --floatOtherPOIs={} --algo=grid -v 0 --points=50 -n {} -t -1'.format(self.wp,int(floatPOI),'_'.join(wcs))#--fastScan ' 
        if isfast: 
            command += ' --fastScan '
        for wc in wcs:
            #
            wc_list.remove(wc)
            #
            p_str = ' '
            r_str = ' --setParameterRanges '
            sP_str= ' --setParameters '
            p_str += '-P '+wc+' '
            #r_str += wc+'=-50.0,50.0'
            r_str += wc+'={0},{1}'.format(*self.wc_range[wc])
            sP_str+= wc+'=0.0'
            command += p_str+r_str+sP_str 
        #if dofreeze:
        #    command += ' --freezeParameters '
        #    command += ','.join(wc_list)+ ',' #+ ','.join(['r_'+poi for poi in self.poi_list])
        if makepdf and len(wcs) == 1: # only do this for 1d scans
            command += '; plot1DScan.py higgsCombine{}.MultiDimFit.mH120.root -o run2_{} --POI {} --main-label Expected'.format(wcs[0],wcs[0],wcs[0])
        self.issue_command(command)
        #os.system('sleep 5')

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
                command = 'plotImpacts.py -i {0} -o {1} --POI r_tt{2}{3}'.format(args.json, args.json.replace('.json','_tt{0}{1}'.format(j,i)), j,i)
                self.issue_command(command)
        #command = 'for i in 0 1 2 3; do for j in Z H; do plotImpacts.py -i nomcstats_asi_scrun2_2018_impacts.json -o nomcstats_asi_scrun2_2018_tt$j\bb$i --POI r_tt$j\bb$i;  done; done'

    def run_wcimpactplots(self):
        for wc in self.relwc_list:
            command = 'plotImpacts.py -i {0} -o {1} --POI {2}'.format(args.wcjson, args.wcjson.replace('.json',wc), wc)
            self.issue_command(command)
                                                                   

    def run_qcfit(self):
        tag = args.qc
        self.issue_command('python new_runCombineTTX.py -c --tag {}'.format(tag))
        def run_func(y,tag,func):
            func(y,tag)
        [run_func(y,tag, self.run_fitDiag) for y in self.years + ['run2']]
        [run_func(y,tag, self.run_diffn)   for y in self.years + ['run2']]

    def run_fitDiag(self,y,tag):
        command = 'combine -M FitDiagnostics Higgs-Combine-Tool/datacard_{1}_{0}.txt --saveShapes --saveWithUncertainties -n _{1}_{0}'.format(y,tag)
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

