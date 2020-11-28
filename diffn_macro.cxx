void saveHist(TFile * roof, TString tag)
{
  TString f_name = roof->GetName();
  TString o_name = f_name.ReplaceAll(".root",tag+".pdf");
  //
  //nuisances->SaveAs(o_name);
  //nuisances->Clear();
  TCanvas* canvas = (TCanvas*) gROOT->FindObject("nuisances");
  canvas->SaveAs(o_name);
  
}

void diffn_macro(TString tag = "")
{
  gStyle->SetOptStat(0);

  _file0->cd();
  saveHist(_file0, tag);
  _file1->cd();
  saveHist(_file1, tag);
  _file2->cd();
  saveHist(_file2, tag);
  _file3->cd();
  saveHist(_file3, tag);
  

}

