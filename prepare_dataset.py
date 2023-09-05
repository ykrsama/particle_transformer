#!/usr/bin/env python3
import itertools
import os
import glob
import ROOT
from itertools import permutations
import numpy as np

INPUT_DIR = "/home/zhuxu/particle_transformer/downloads/JetPair/trsm20230620" # Be absolute dir
OUTPUT_DIR = "/home/zhuxu/particle_transformer/downloads/JetPair/TRSM"
TREENAME="HHHNtuple"
OUT_SUBDIRS=["train_6jets", "test_6jets", "valid_6jets"] # Train, test, valid
OUT_RATIOS=[0.7, 0.15, 0.15] # SUM shuod be 1; last bin may not used
NJET = 8
MAX_DR=0.4
COLUMN_LIST=['GenFiltHT', 'NPV', 'runNumber', 'eventNumber', 'passGRL',
             'actualInteractionsPerCrossing', 'averageInteractionsPerCrossing', 'bcid', 'cleanEvent',
             'coreFlags','correctedActualMu', 'correctedAndScaledActualMu', 'correctedAndScaledAverageMu',
             'correctedAverageMu', 'nmuon',
             'njets', 'jets_pt', 'jets_eta', 'jets_phi', 'jets_E', 'jets_DL1dv01',
             'jets_is_DL1dv01_FixedCutBEff_77', 'jets_is_DL1dv01_FixedCutBEff_85',
             'lumiBlock', 'mH1', 'mH2', 'mH3', 'mHHH',
             'ntruthHiggs', 'truth_mHHH', 'truth_pT_H1', 'ntruthBQuarks',
             'mcChannelNumber', 'mcEventNumber', 'mcEventWeight',
             'weight', 'weight_pileup', 'xsTimesFiltEff']
             # 'isCR', 'isSR',
             # 'jets_HadronConeExclExtendedTruthLabelID', 'jets_HadronConeExclTruthLabelID',
             # 'muon_EnergyLoss', 'muon_EnergyLossSigma',
             # 'muon_MeasEnergyLoss', 'muon_MeasEnergyLossSigma', 'muon_ParamEnergyLoss',
             # 'muon_ParamEnergyLossSigmaMinus', 'muon_ParamEnergyLossSigmaPlus', 'muon_charge', 'muon_energyLossType',
             # 'muon_eta', 'muon_m', 'muon_phi', 'muon_pt', 'nBTags',
             # 'passedTriggerHashes', 'rand_lumiblock_nr', 'rand_run_nr',  'truthBQuarks_E',
             # 'truthBQuarks_eta', 'truthBQuarks_is_bhad', 'truthBQuarks_is_higgs', 'truthBQuarks_pdgId',
             # 'truthBQuarks_phi', 'truthBQuarks_pt', 'truthBQuarks_status', 'truthHiggs_E', 'truthHiggs_eta',
             # 'truthHiggs_is_bhad', 'truthHiggs_is_higgs', 'truthHiggs_pdgId', 'truthHiggs_phi', 'truthHiggs_pt',
             # 'truthHiggs_status', , 'truthjet_eta', 'truthjet_m', 'truthjet_phi',
             # 'truthjet_pt']

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

PAIR_LESS3_EXPR= "1.0 * (("
for column_name, expr in PAIR_COLUMN_EXPR.items():
    PAIR_LESS3_EXPR += "+"
    PAIR_LESS3_EXPR += expr
PAIR_LESS3_EXPR += ")==0)"

PAIR_GREAT3_EXPR= "-(("
for column_name, expr in PAIR_COLUMN_EXPR.items():
    PAIR_GREAT3_EXPR += "+"
    PAIR_GREAT3_EXPR += expr
PAIR_GREAT3_EXPR += ")-1)"

sum_norigin = 0
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

def process_tree(in_filepath:str, jet_perm:list, dsid:str, mX:int, mS:int):
    global sum_ntrain, sum_ntest, sum_nvalid, sum_norigin
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

    # Permutation of jets
    if jet_perm:
        rdf = rdf.Define("jets_pt_perm",      f"permuteVector(jets_pt,      {jet_perm[0]}, {jet_perm[1]}, {jet_perm[2]}, {jet_perm[3]}, {jet_perm[4]}, {jet_perm[5]})")
        rdf = rdf.Define("jets_eta_perm",     f"permuteVector(jets_eta,     {jet_perm[0]}, {jet_perm[1]}, {jet_perm[2]}, {jet_perm[3]}, {jet_perm[4]}, {jet_perm[5]})")
        rdf = rdf.Define("jets_phi_perm",     f"permuteVector(jets_phi,     {jet_perm[0]}, {jet_perm[1]}, {jet_perm[2]}, {jet_perm[3]}, {jet_perm[4]}, {jet_perm[5]})")
        rdf = rdf.Define("jets_E_perm",       f"permuteVector(jets_E,       {jet_perm[0]}, {jet_perm[1]}, {jet_perm[2]}, {jet_perm[3]}, {jet_perm[4]}, {jet_perm[5]})")
        rdf = rdf.Define("jets_DL1dv01_perm", f"permuteVector(jets_DL1dv01, {jet_perm[0]}, {jet_perm[1]}, {jet_perm[2]}, {jet_perm[3]}, {jet_perm[4]}, {jet_perm[5]})")
        rdf = rdf.Define("jets_is_DL1dv01_FixedCutBEff_77_perm", f"permuteVector(jets_is_DL1dv01_FixedCutBEff_77, {jet_perm[0]}, {jet_perm[1]}, {jet_perm[2]}, {jet_perm[3]}, {jet_perm[4]}, {jet_perm[5]})")
        rdf = rdf.Define("jets_is_DL1dv01_FixedCutBEff_85_perm", f"permuteVector(jets_is_DL1dv01_FixedCutBEff_85, {jet_perm[0]}, {jet_perm[1]}, {jet_perm[2]}, {jet_perm[3]}, {jet_perm[4]}, {jet_perm[5]})")

        rdf = rdf.Redefine("jets_pt",      "jets_pt_perm")
        rdf = rdf.Redefine("jets_eta",     "jets_eta_perm")
        rdf = rdf.Redefine("jets_phi",     "jets_phi_perm")
        rdf = rdf.Redefine("jets_E",       "jets_E_perm")
        rdf = rdf.Redefine("jets_DL1dv01", "jets_DL1dv01_perm")
        rdf = rdf.Redefine("jets_is_DL1dv01_FixedCutBEff_77", "jets_is_DL1dv01_FixedCutBEff_77_perm")
        rdf = rdf.Redefine("jets_is_DL1dv01_FixedCutBEff_85", "jets_is_DL1dv01_FixedCutBEff_85_perm")

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
    #print(f"Processed event: {nevent}")
    ntrain = int(nevent * OUT_RATIOS[0])
    ntest = int(nevent * OUT_RATIOS[1])

    rdf_train = rdf.Range(0, ntrain).Filter("BMatchedJetNum>=6")
    rdf_test = rdf.Range(ntrain,ntrain+ntest).Filter("BMatchedJetNum>=6")
    rdf_valid = rdf.Range(ntrain+ntest,0).Filter("BMatchedJetNum>=6")

    cur_ntrain = rdf_train.Count().GetValue()
    cur_ntest  = rdf_test.Count().GetValue()
    cur_nvalid = rdf_valid.Count().GetValue()
    print(f"Permutation {jet_perm}: saved train/test/valid event: {cur_ntrain}, {cur_ntest}, {cur_nvalid}")
    sum_ntrain += cur_ntrain
    sum_ntest  += cur_ntest
    sum_nvalid += cur_nvalid
    sum_norigin += nevent

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
    ROOT::RVec<float> getJetsPx(ROOT::RVec<float> jets_pt, ROOT::RVec<float> jets_eta, ROOT::RVec<float> jets_phi) {
        ROOT::RVec<float> ret_vals;
        for (int i = 0; i < jets_pt.size(); i++)
            ret_vals.emplace_back(jets_pt.at(i) * TMath::Cos(jets_phi.at(i)));
        return ret_vals;
    }
    """)
    ROOT.gInterpreter.Declare("""
    ROOT::RVec<float> getJetsPy(ROOT::RVec<float> jets_pt, ROOT::RVec<float> jets_eta, ROOT::RVec<float> jets_phi) {
        ROOT::RVec<float> ret_vals;
        for (int i = 0; i < jets_pt.size(); i++)
            ret_vals.emplace_back(jets_pt.at(i) * TMath::Sin(jets_phi.at(i)));
        return ret_vals;
    }
    """)
    ROOT.gInterpreter.Declare("""
    ROOT::RVec<float> getJetsPz(ROOT::RVec<float> jets_pt, ROOT::RVec<float> jets_eta, ROOT::RVec<float> jets_phi) {
        ROOT::RVec<float> ret_vals;
        for (int i = 0; i < jets_pt.size(); i++)
            ret_vals.emplace_back(jets_pt.at(i) * sinh(jets_eta.at(i)));
        return ret_vals;
    }
    """)
    ROOT.gInterpreter.Declare("""
    ROOT::RVec<int> getJetsIndex( ROOT::RVec<float> jets_E) {
        ROOT::RVec<int> ret_vals;
        for (int i = 0; i < jets_E.size(); i++)
            ret_vals.emplace_back(i);
        return ret_vals;
    }
    """)

    ROOT.gInterpreter.Declare("""
    template<typename T>
    ROOT::RVec<T> permuteVector(ROOT::RVec<T>& input_vector,
    int p0, int p1, int p2, int p3, int p4, int p5) {
        std::vector<int> permutation = {p0, p1, p2, p3, p4, p5};
        ROOT::RVec<T> permuted_vector(input_vector.size());
        for (int i = 0; i < input_vector.size(); i++) {
            if ( i < permutation.size()) {
                permuted_vector[i] = input_vector[permutation[i]];
            } else {
                permuted_vector[i] = input_vector[i];
            }
        }
        return permuted_vector;
    }
    """)
    for label_i in list(range(0, NJET - 1)):
        for label_j in list(range(label_i+1, NJET)):
            # print(f"comipling {label_i} {label_j}")
            ROOT.gInterpreter.Declare(f"""
            double getPairWeight_{label_i}{label_j}(ROOT::RVec<float> jets_pt, ROOT::RVec<float> jets_eta, ROOT::RVec<float> jets_phi, ROOT::RVec<float> jets_E,
            ROOT::RVec<float> truthBQuarks_pt, ROOT::RVec<float> truthBQuarks_eta, ROOT::RVec<float> truthBQuarks_phi, ROOT::RVec<float> truthBQuarks_E,
            ROOT::RVec<std::vector<int>> truthBQuarks_parent_barcode) {{
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
                    if ( min_dR < {MAX_DR} ) {{
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
    int getBMatchedJetNum(ROOT::RVec<float> jets_pt, ROOT::RVec<float> jets_eta, ROOT::RVec<float> jets_phi, ROOT::RVec<float> jets_E,
    ROOT::RVec<float> truthBQuarks_pt, ROOT::RVec<float> truthBQuarks_eta, ROOT::RVec<float> truthBQuarks_phi, ROOT::RVec<float> truthBQuarks_E) {{
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
            if ( min_dR < {MAX_DR} ) {{
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

    jet_perms = list(permutations(range(NJET_TO_PAIR)))
    jet_perms_rand = np.random.permutation(jet_perms)
    for input_dir in glob.glob(f"{INPUT_DIR}/*.root"):
        dirname = os.path.basename(input_dir)
        dsid = dirname.split('.')[4]
        print(f"Reading DSID: {dsid}")
        mX, mS = DSID_TO_mXmS[dsid]
        for filepath in glob.glob(f"{input_dir}/*"):
            #for jet_perm in jet_perms_rand[0:64]:
            #    process_tree(filepath, jet_perm, dsid, mX, mS)
            process_tree(filepath, None, dsid, mX, mS)
    print("Summary:")
    print(f"N Processed: {sum_norigin}")
    print(f"N Train: {sum_ntrain}")
    print(f"N Test: {sum_ntest}")
    print(f"N Valid: {sum_nvalid}")
if __name__ == '__main__':
    main()