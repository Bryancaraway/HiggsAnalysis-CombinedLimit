void limitmacro(){
  gStyle->SetOptStat(0);
  gStyle->SetLegendFillColor(0);
  
  TTree *tree = (TTree*) gFile->Get("limit") ; 
  TString f_name = gFile->GetName();
  TString process = f_name(12,7);
  TString draw_c  = "2*deltaNLL:r_"+process;
  tree->Draw(draw_c+">>h_full","2*deltaNLL<6",                "goff prof");
  tree->Draw(draw_c+">>h_Cl1","2*deltaNLL<=1.",                "goff prof");  
  tree->Draw(draw_c+">>h_Cl2","2*deltaNLL<=3.84;2*deltaNLL>1.","goff prof");
  //
  TGraph *gr_full   = (TGraph*) gROOT->FindObject("h_full")->Clone();
  TGraph *graph_Cl1 = (TGraph*) gROOT->FindObject("h_Cl1")->Clone();
  TGraph *graph_Cl2 = (TGraph*) gROOT->FindObject("h_Cl2")->Clone();
  //
  gr_full->SetTitle(process);
  gr_full->SetLineColor(0); graph_Cl1->SetLineColor(0); graph_Cl2->SetLineColor(0);
  gr_full->Draw("HIST");  
  graph_Cl2->SetFillColor(5);
  graph_Cl2->Draw("Hist Bar same"); 
  graph_Cl1->SetFillColor(3);
  graph_Cl1->Draw("Hist Bar same"); 
  gPad->RedrawAxis();
  //
  TLegend* legend = new TLegend(0.65,0.7,0.85,0.85);
  legend->AddEntry(graph_Cl1, "68% CL", "f");
  legend->AddEntry(graph_Cl2, "95% CL", "f");
  legend->SetBorderSize(1);
  legend->Draw();
  //
  TCanvas* canvas = (TCanvas*) gROOT->FindObject("c1");
  canvas->SaveAs("pdf_400inc_ttbb50/"+process+".pdf");
}
