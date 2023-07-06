#!/usr/bin/env python3
import os
import glob
import ROOT

INPUT_DIR = "/home/zhuxu/particle_transformer/downloads/JetPair/TRSM/trsm20230620" # Be absolute dir
OUTPUT_DIR = "/home/zhuxu/particle_transformer/downloads/JetPair/TRSM"
TREENAME="HHHNtuple"
OUT_SUBDIRS=["train_6jets", "test_6jets", "valid_6jets"] # Train, test, valid
OUT_RATIOS=[0.7, 0.15, 0.15] # SUM shuod be 1; last bin may not used
NJET = 8
COLUMN_LIST=['GenFiltHT', 'NPV',
             'actualInteractionsPerCrossing', 'averageInteractionsPerCrossing', 'bcid', 'cleanEvent',
             'coreFlags','correctedActualMu', 'correctedAndScaledActualMu', 'correctedAndScaledAverageMu',
             'correctedAverageMu', 'eventNumber', 'isCR', 'isSR',
             'jets_DL1dv00', 'jets_DL1dv00_pb', 'jets_DL1dv00_pc', 'jets_DL1dv00_pu',
             'jets_DL1dv01', 'jets_DL1dv01_pb', 'jets_DL1dv01_pc', 'jets_DL1dv01_pu',
             'jets_DL1r', 'jets_DL1r_pb', 'jets_DL1r_pc', 'jets_DL1r_pu',
             'jets_E', 'jets_GN1', 'jets_GN1_pb', 'jets_GN1_pc', 'jets_GN1_pu',
             'jets_HadronConeExclExtendedTruthLabelID', 'jets_HadronConeExclTruthLabelID',
             'jets_eta', 'jets_is_DL1dv01_FixedCutBEff_60', 'jets_is_DL1dv01_FixedCutBEff_70',
             'jets_is_DL1dv01_FixedCutBEff_77', 'jets_is_DL1dv01_FixedCutBEff_85',
             'jets_phi', 'jets_pt', 'lumiBlock', 'mH1', 'mH2', 'mH3', 'mHHH',
             'mcChannelNumber', 'mcEventNumber', 'mcEventWeight', 'muon_EnergyLoss', 'muon_EnergyLossSigma',
             'muon_MeasEnergyLoss', 'muon_MeasEnergyLossSigma', 'muon_ParamEnergyLoss',
             'muon_ParamEnergyLossSigmaMinus', 'muon_ParamEnergyLossSigmaPlus', 'muon_charge', 'muon_energyLossType',
             'muon_eta', 'muon_m', 'muon_phi', 'muon_pt', 'nBTags', 'njets', 'nmuon', 'ntruthBQuarks', 'ntruthHiggs',
             'passGRL', 'passedTriggerHashes', 'rand_lumiblock_nr', 'rand_run_nr', 'runNumber', 'truthBQuarks_E',
             'truthBQuarks_eta', 'truthBQuarks_is_bhad', 'truthBQuarks_is_higgs', 'truthBQuarks_pdgId',
             'truthBQuarks_phi', 'truthBQuarks_pt', 'truthBQuarks_status', 'truthHiggs_E', 'truthHiggs_eta',
             'truthHiggs_is_bhad', 'truthHiggs_is_higgs', 'truthHiggs_pdgId', 'truthHiggs_phi', 'truthHiggs_pt',
             'truthHiggs_status', 'truth_mHHH', 'truth_pT_H1', 'truthjet_eta', 'truthjet_m', 'truthjet_phi',
             'truthjet_pt', 'weight', 'weight_pileup', 'xsTimesFiltEff']

DSID_TO_mXmS={
    "521163":[325,200],
    "521164":[400,200],
    "521165":[475,200],
    "521166":[550,200],
    "521167":[350,225],
    "521168":[425,225],
    "521169":[500,225],
    "521170":[575,225],
    "521171":[375,250],
    "521172":[425,250],
    "521173":[475,250],
    "521174":[525,250],
    "521175":[570,250],
    "521176":[400,275],
    "521177":[450,275],
    "521178":[500,275],
    "521179":[550,275],
    "521180":[425,300],
    "521181":[450,300],
    "521182":[500,300],
    "521183":[540,300],
    "521184":[450,325],
    "521185":[485,325],
    "521186":[520,325],
    "521187":[475,350],
    "521188":[500,350]
}

# Prepare pair functions
# Wij * Wkl * Wmn , i==0, m > k > i; j>i, l>k>i, n>m>k>i
NJET_TO_PAIR=6
PAIR_COLUMN_EXPR={}
for j in list(range(1, NJET_TO_PAIR)):
    for k in list(range(1, NJET_TO_PAIR)):
        for l in list(range(k+1, NJET_TO_PAIR)):
            for m in list(range(k+1, NJET_TO_PAIR)):
                for n in list(range(m + 1, NJET_TO_PAIR)):
                    i_list = [j, k, l, m, n]
                    if (len(set(i_list)) != len(i_list)): continue  # Check all unique
                    column_name = f"Pair0{j}{k}{l}{m}{n}"
                    PAIR_COLUMN_EXPR[column_name] = f"PairWeight_0{j} * PairWeight_{k}{l} * PairWeight_{m}{n}"

PAIR_LESS3_EXPR= "-(("
for column_name, expr in PAIR_COLUMN_EXPR.items():
    PAIR_LESS3_EXPR += "+"
    PAIR_LESS3_EXPR += expr
PAIR_LESS3_EXPR += ")-1)"

PAIR_GREAT3_EXPR= "-(("
for column_name, expr in PAIR_COLUMN_EXPR.items():
    PAIR_GREAT3_EXPR += "+"
    PAIR_GREAT3_EXPR += expr
PAIR_GREAT3_EXPR += ")-1)"

sum_ntrain = 0;
sum_ntest = 0;
sum_nvalid = 0;

def get_last_file_id(filepath0):
    path_wildcard = filepath0.replace("_0.root", "_*.root")
    paths = glob.glob(path_wildcard)
    last_id = 0
    for path_str in paths:
        name = os.path.basename(path_str)
        this_id = int(name.replace(".root", "").split('_')[-1])
        if this_id > last_id:
            last_id = this_id
    return last_id

def process_tree(in_filepath:str, dsid:str, mX:int, mS:int):
    global sum_ntrain, sum_ntest, sum_nvalid
    #ROOT.EnableImplicitMT()
    file_id = 0;
    out_filename = f"mS{mS}_mX{mX}_{dsid}_{file_id}.root"
    temp_filepath = f"{OUTPUT_DIR}/{OUT_SUBDIRS[0]}/{out_filename}"
    if (os.path.exists(temp_filepath)):
        file_id = get_last_file_id(temp_filepath) + 1
    out_filename = f"mS{mS}_mX{mX}_{dsid}_{file_id}.root"
    train_filepath = f"{OUTPUT_DIR}/{OUT_SUBDIRS[0]}/{out_filename}"
    test_filepath  = f"{OUTPUT_DIR}/{OUT_SUBDIRS[1]}/{out_filename}"
    valid_filepath = f"{OUTPUT_DIR}/{OUT_SUBDIRS[2]}/{out_filename}"

    columnList = COLUMN_LIST.copy()
    rdf = ROOT.RDataFrame(TREENAME, in_filepath)
    rdf = rdf.Define("jets_px", "getJetsPx(jets_pt, jets_eta, jets_phi)") \
             .Define("jets_py", "getJetsPy(jets_pt, jets_eta, jets_phi)") \
             .Define("jets_pz", "getJetsPz(jets_pt, jets_eta, jets_phi)") \
             .Define("jets_index", "getJetsIndex(jets_E)")
    columnList += ["jets_px", "jets_py", "jets_pz", "jets_index"]
    for i in list(range(0, NJET - 1)):
        for j in list(range( i+1, NJET)):
            column_name = f"PairWeight_{i}{j}"
            rdf = rdf.Define(column_name, f"getPairWeight_{i}{j}(jets_pt, jets_eta, jets_phi, jets_E, truthBQuarks_pt, truthBQuarks_eta, truthBQuarks_phi, truthBQuarks_E, truthBQuarks_parent_barcode)")
            columnList.append(column_name)

    for column_name, expr in PAIR_COLUMN_EXPR.items():
        rdf = rdf.Define(column_name, expr)
        columnList.append(column_name)

    rdf = rdf.Define("PairLess3", PAIR_LESS3_EXPR)
    columnList.append("PairLess3")
    #rdf = rdf.Define("PairGreat3", PAIR_GREAT3_EXPR)
    #columnList.append("PairGreat3")
    rdf = rdf.Define("BMatchedJetNum", "getBMatchedJetNum(jets_pt, jets_eta, jets_phi, jets_E, truthBQuarks_pt, truthBQuarks_eta, truthBQuarks_phi, truthBQuarks_E)")
    columnList.append("BMatchedJetNum")

    nevent = rdf.Count().GetValue()
    print(f"Processed event: {nevent}")
    ntrain = int(nevent * OUT_RATIOS[0])
    ntest = int(nevent * OUT_RATIOS[1])

    rdf_train = rdf.Range(0, ntrain).Filter("BMatchedJetNum>=6")
    rdf_test = rdf.Range(ntrain,ntrain+ntest).Filter("BMatchedJetNum>=6")
    rdf_valid = rdf.Range(ntrain+ntest,0).Filter("BMatchedJetNum>=6")

    cur_ntrain = rdf_train.Count().GetValue()
    cur_ntest  = rdf_test.Count().GetValue()
    cur_nvalid = rdf_valid.Count().GetValue()
    print(f"Saved train event: {cur_ntrain}, {cur_ntest}, {cur_nvalid}")
    sum_ntrain += cur_ntrain
    sum_ntest  += cur_ntest
    sum_nvalid += cur_nvalid

    rdf_train.Snapshot(TREENAME, train_filepath, columnList)
    rdf_test.Snapshot(TREENAME, test_filepath, columnList)
    rdf_valid.Snapshot(TREENAME, valid_filepath, columnList)

def main():
    # Check old files
    for out_dir in OUT_SUBDIRS:
        if os.path.exists(f"{OUTPUT_DIR}/{out_dir}"):
            print(f"[ERROR] {OUTPUT_DIR}/{out_dir} already exisit.")
            exit()
    for out_dir in OUT_SUBDIRS:
        os.mkdir(f"{OUTPUT_DIR}/{out_dir}")
    # Declare functions
    ROOT.gInterpreter.Declare("""
    ROOT::VecOps::RVec<float> getJetsPx(ROOT::VecOps::RVec<float> jets_pt, ROOT::VecOps::RVec<float> jets_eta, ROOT::VecOps::RVec<float> jets_phi) {
        ROOT::VecOps::RVec<float> ret_vals;
        for (int i = 0; i < jets_pt.size(); i++)
            ret_vals.emplace_back(jets_pt.at(i) * TMath::Cos(jets_phi.at(i)));
        return ret_vals;
    }
    """)
    ROOT.gInterpreter.Declare("""
    ROOT::VecOps::RVec<float> getJetsPy(ROOT::VecOps::RVec<float> jets_pt, ROOT::VecOps::RVec<float> jets_eta, ROOT::VecOps::RVec<float> jets_phi) {
        ROOT::VecOps::RVec<float> ret_vals;
        for (int i = 0; i < jets_pt.size(); i++)
            ret_vals.emplace_back(jets_pt.at(i) * TMath::Sin(jets_phi.at(i)));
        return ret_vals;
    }
    """)
    ROOT.gInterpreter.Declare("""
    ROOT::VecOps::RVec<float> getJetsPz(ROOT::VecOps::RVec<float> jets_pt, ROOT::VecOps::RVec<float> jets_eta, ROOT::VecOps::RVec<float> jets_phi) {
        ROOT::VecOps::RVec<float> ret_vals;
        for (int i = 0; i < jets_pt.size(); i++)
            ret_vals.emplace_back(jets_pt.at(i) * sinh(jets_eta.at(i)));
        return ret_vals;
    }
    """)
    ROOT.gInterpreter.Declare("""
    ROOT::VecOps::RVec<int> getJetsIndex( ROOT::VecOps::RVec<float> jets_E) {
        ROOT::VecOps::RVec<int> ret_vals;
        for (int i = 0; i < jets_E.size(); i++)
            ret_vals.emplace_back(i);
        return ret_vals;
    }
    """)

    for label_i in list(range(0, NJET - 1)):
        for label_j in list(range(label_i+1, NJET)):
            # print(f"comipling {label_i} {label_j}")
            ROOT.gInterpreter.Declare(f"""
            double getPairWeight_{label_i}{label_j}(ROOT::VecOps::RVec<float> jets_pt, ROOT::VecOps::RVec<float> jets_eta, ROOT::VecOps::RVec<float> jets_phi, ROOT::VecOps::RVec<float> jets_E,
            ROOT::VecOps::RVec<float> truthBQuarks_pt, ROOT::VecOps::RVec<float> truthBQuarks_eta, ROOT::VecOps::RVec<float> truthBQuarks_phi, ROOT::VecOps::RVec<float> truthBQuarks_E,
            ROOT::VecOps::RVec<std::vector<int>> truthBQuarks_parent_barcode) {{
                const std::vector<int> empty_vec;
                int label_i = {label_i};
                int label_j = {label_j};
                assert(label_i < label_j);
                std::vector<TLorentzVector> lv_jets;
                std::vector<TLorentzVector> lv_Bs;
                std::vector<std::vector<int>> parent_barcodes;
                std::vector<bool> is_paired_B;
                int njets = jets_pt.size();
                if (njets > {NJET})
                    njets = {NJET};
                if (label_j >= njets)
                    return 0;
                for (int i = 0; i < njets; i++) {{
                    TLorentzVector jet;
                    jet.SetPtEtaPhiE(jets_pt.at(i), jets_eta.at(i), jets_phi.at(i), jets_E.at(i));
                    lv_jets.emplace_back(jet);
                    parent_barcodes.emplace_back(empty_vec);
                }}
                // Prepare B quarks
                int nBs = truthBQuarks_pt.size();
                for (int i = 0; i < nBs; i++) {{
                    TLorentzVector BQuark;
                    BQuark.SetPtEtaPhiE(truthBQuarks_pt.at(i), truthBQuarks_eta.at(i), truthBQuarks_phi.at(i), truthBQuarks_E.at(i));
                    lv_Bs.emplace_back(BQuark);
                    is_paired_B.emplace_back(false);
                }}
                // Loop jet, pair to b
                double min_dR, cur_dR;
                int pair_B_id;
                for (int jet_i = 0; jet_i < njets; jet_i++) {{
                    min_dR = 3;
                    pair_B_id = -1;
                    for (int B_i = 0; B_i < nBs; B_i++) {{
                        if (is_paired_B.at(B_i)) continue;
                        cur_dR = lv_jets.at(jet_i).DeltaR(lv_Bs.at(B_i));
                        if (cur_dR < min_dR) {{
                            min_dR = cur_dR;
                            pair_B_id = B_i;
                        }}
                    }}
                    if ( min_dR < 0.3 ) {{
                        parent_barcodes.at(jet_i) = truthBQuarks_parent_barcode.at(pair_B_id);
                        is_paired_B.at(pair_B_id) = true;
                    }}
                }}
                
                //// Normalize
                //double num_pairs=0;
                //for (int i = 0; i < njets - 1; i++) {{
                //    for (int j = i+1; j < njets ; j++) {{
                //        if (parent_barcodes.at(i).size() == 0 || parent_barcodes.at(j).size() == 0) continue;
                //        if (parent_barcodes.at(i) == parent_barcodes.at(j)) {{
                //            num_pairs += 1;
                //        }}
                //    }}
                //}}
                //if(num_pairs == 0) {{
                //    return -1; // -1 for filter
                //}}
                
                // Check if i and j jet are from the same Higgs
                if (parent_barcodes.at(label_i).size() > 0 && parent_barcodes.at(label_j).size() > 0) {{
                    if (parent_barcodes.at(label_i) == parent_barcodes.at(label_j)) {{
                        // return (1 / num_pairs);
                        return 1;
                    }}
                }}
                return 0;
            }}
            """)

    ROOT.gInterpreter.Declare(f"""
    int getBMatchedJetNum(ROOT::VecOps::RVec<float> jets_pt, ROOT::VecOps::RVec<float> jets_eta, ROOT::VecOps::RVec<float> jets_phi, ROOT::VecOps::RVec<float> jets_E,
    ROOT::VecOps::RVec<float> truthBQuarks_pt, ROOT::VecOps::RVec<float> truthBQuarks_eta, ROOT::VecOps::RVec<float> truthBQuarks_phi, ROOT::VecOps::RVec<float> truthBQuarks_E) {{
        const std::vector<int> empty_vec;
        std::vector<TLorentzVector> lv_jets;
        std::vector<TLorentzVector> lv_Bs;
        std::vector<bool> is_matched_B;
        int njets = jets_pt.size();
        if (njets > {NJET})
            njets = {NJET};
        for (int i = 0; i < njets; i++) {{
            TLorentzVector jet;
            jet.SetPtEtaPhiE(jets_pt.at(i), jets_eta.at(i), jets_phi.at(i), jets_E.at(i));
            lv_jets.emplace_back(jet);
        }}
        // Prepare B quarks
        int nBs = truthBQuarks_pt.size();
        for (int i = 0; i < nBs; i++) {{
            TLorentzVector BQuark;
            BQuark.SetPtEtaPhiE(truthBQuarks_pt.at(i), truthBQuarks_eta.at(i), truthBQuarks_phi.at(i), truthBQuarks_E.at(i));
            lv_Bs.emplace_back(BQuark);
            is_matched_B.emplace_back(false);
        }}
        // Loop jet, match to b
        double min_dR, cur_dR;
        int pair_B_id;
        for (int jet_i = 0; jet_i < njets; jet_i++) {{
            min_dR = 3;
            pair_B_id = -1;
            for (int B_i = 0; B_i < nBs; B_i++) {{
                if (is_matched_B.at(B_i)) continue;
                cur_dR = lv_jets.at(jet_i).DeltaR(lv_Bs.at(B_i));
                if (cur_dR < min_dR) {{
                    min_dR = cur_dR;
                    pair_B_id = B_i;
                }}
            }}
            if ( min_dR < 0.3 ) {{
                is_matched_B.at(pair_B_id) = true;
            }}
        }}
        int total_match_num = 0;
        for (auto matched : is_matched_B) {{
            total_match_num += matched;
        }}
        return total_match_num;
    }}
    """)

    for input_dir in glob.glob(f"{INPUT_DIR}/*.root"):
        dirname = os.path.basename(input_dir)
        dsid = dirname.split('.')[4]
        print(f"Reading DSID: {dsid}")
        mX, mS = DSID_TO_mXmS[dsid]
        for filepath in glob.glob(f"{input_dir}/*"):
            process_tree(filepath, dsid, mX, mS)
    print("Summary:")
    print(f"N Train: {sum_ntrain}")
    print(f"N Test: {sum_ntest}")
    print(f"N Valid: {sum_nvalid}")
if __name__ == '__main__':
    main()