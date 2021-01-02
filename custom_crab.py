def custom_crab(config):
  print '>> Customising the crab config'
  config.General.workArea = 'crab_projects'
  config.Site.storageSite = 'T3_US_Baylor'
  config.Site.blacklist = ['T2_US_Caltech']
  #config.Site.whitelist = ['T2_US_*','T3_US_Baylor']
  config.Site.whitelist = ['T2_US_*']
  config.JobType.allowUndistributedCMSSW = True
  config.Data.outLFNDirBase = '/store/user/bcaraway/crab_test'
