import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from sksurv.metrics import concordance_index_censored
import torch
import torchtuples as tt
from pycox.models import CoxPH


import random
random.seed(1)     # python random generator
np.random.seed(1)  # numpy random generator



get_target = lambda df_clinic: (df_clinic['OS.time'].values, df_clinic['OS.Status'].values)


def pairwise_fusion(y_test, surv_c, surv_m, surv_g, cindex_c, cindex_m, cindex_g):
    durations_test, events_test = get_target(pd.DataFrame(y_test))
    events_test = np.where(events_test == 1, True, False)

    # Combine clinical and mRNA
    # Weighted average based on c-index
    surv_cg = np.average([surv_c, surv_g], weights=[cindex_c, cindex_g], axis=0)
    # surv_cg = pd.DataFrame(surv_cg, index=surv_c.index)
    ev_cg = concordance_index_censored(events_test, durations_test, surv_cg.flatten())
    c_index_cg = round(ev_cg[0], 4)

    print("C-Index Clinical + Gene", c_index_cg)

    # Integrate 2 data types C+ fluxomic data
    surv_cm = np.average([surv_c, surv_m], weights=[cindex_c, cindex_m], axis=0)
    # surv_cm = pd.DataFrame(surv_cm, index=surv_c.index)
    ev_cm = concordance_index_censored(events_test, durations_test, surv_cm.flatten())
    c_index_cm = round(ev_cm[0], 4)
    print("C-Index Clinical + Meta", c_index_cm)


    # Integrate 2 data types Gene and fluxomic data
    surv_gm = np.average([surv_g, surv_m], weights=[cindex_g, cindex_m], axis=0)
    # surv_gm = pd.DataFrame(surv_gm, index=surv_c.index)
    ev_gm = concordance_index_censored(events_test, durations_test, surv_gm.flatten())
    c_index_gm = round(ev_gm[0], 4)
    print("C-Index  Gene + Meta", c_index_gm)
    return [c_index_cg, surv_cg, c_index_cm, surv_cm, c_index_gm, surv_gm]


def late_fusion(y_test, surv_c, surv_m, surv_g, cindex_c, cindex_m, cindex_g):
    durations_test, events_test = get_target(pd.DataFrame(y_test))
    events_test = np.where(events_test == 1, True, False)

    # Combine clinical, mRNA and fluxomic data
    surv_all = np.average([surv_c, surv_m, surv_g], weights=[cindex_c, cindex_m, cindex_g], axis=0)
    # surv_all = pd.DataFrame(surv_all, index=surv_c.index)
    ev_all = concordance_index_censored(events_test, durations_test, surv_all.flatten())
    c_index_all = round(ev_all[0], 4)

    print("C-Index late fusion Clinical + Gene + Flux", c_index_all)

    return [c_index_all, surv_all]

def hybrid_fusion(y_test, surv_uni, surv_pairwise, cindex_uni, cindex_pairwise,):
    durations_test, events_test = get_target(pd.DataFrame(y_test))
    events_test = np.where(events_test == 1, True, False)

    surv_hybrid = np.average([surv_pairwise, surv_uni], weights=[cindex_pairwise, cindex_uni], axis=0)
    # surv_hybrid = pd.DataFrame(surv_hybrid, index=surv_pairwise.index)
    ev_hybrid = concordance_index_censored(events_test, durations_test, surv_hybrid.flatten())
    c_index_hybrid = round(ev_hybrid[0], 4)

    return [c_index_hybrid, surv_hybrid]


# Intermediate fusion
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch.nn as nn
size=0.2
RANDOM_STATE =123

class ClinicalNet1(nn.Module):
    """Clinical data extractor.
      """
    def __init__(self, hid_out, input):
        super(ClinicalNet1, self).__init__()

        self.hid_out = hid_out
        self.input = input

        # Linear Layers
        self.fc1 = torch.nn.Linear(self.input, 2)
        self.relu1 = torch.nn.ReLU()
        self.batch1 = torch.nn.BatchNorm1d(2)
        self.drop1 = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(2, self.hid_out)
        self.relu2 = torch.nn.ReLU()
        self.batch2 = torch.nn.BatchNorm1d(self.hid_out)
        # self.drop2 = torch.nn.Dropout(0.2)
        # self.fc3 = torch.nn.Linear(self.hid_out, self.out_features)
       

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        try:
            self.batch1(x)
        except:
            self.batch1.eval()
            self.batch1(x)
            self.batch1.train()

        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        out = self.batch2(x)
        # x = self.drop2(x)
        # out = self.fc3(x)
        
        return out

class GeneNet1(nn.Module):
    """Transcriptomic data extractor.
      """
    def __init__(self, hid_out, input):
        super(GeneNet1, self).__init__()
        self.hid_out = hid_out
        self.input = input

        # Linear Layers
        self.fc1 = torch.nn.Linear(input, int(self.input/8))
        self.relu1 = torch.nn.ReLU()
        self.batch1 = torch.nn.BatchNorm1d(int(self.input/8))
        self.drop1 = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(int(self.input/8), self.hid_out)
        self.relu2 = torch.nn.ReLU()
        self.batch2 = torch.nn.BatchNorm1d(self.hid_out)
        # self.drop2 = torch.nn.Dropout(0.2)
        # self.fc3 = torch.nn.Linear(self.hid_out, self.out_features)
       

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        try:
            self.batch1(x)
        except:
            self.batch1.eval()
            self.batch1(x)
            self.batch1.train()

        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        out = self.batch2(x)
        # x = self.drop2(x)
        # out = self.fc3(x)
        
        return out

class MetaNet1(nn.Module):
    """Metabolic data extractor.
      """
    def __init__(self, hid_out, input):
        super(MetaNet1, self).__init__()
        self.hid_out = hid_out
        self.input = input

        # Linear Layers
        self.fc1 = torch.nn.Linear(input, int(self.input/8))
        self.relu1 = torch.nn.ReLU()
        # self.drop1 = torch.nn.Dropout(0.5)
        self.batch1 = torch.nn.BatchNorm1d(int(self.input/8))
        self.drop1 = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(int(self.input/8), self.hid_out)
        self.relu2 = torch.nn.ReLU()
        self.batch2 = torch.nn.BatchNorm1d(self.hid_out)
        # self.drop2 = torch.nn.Dropout(0.2)
        # self.fc3 = torch.nn.Linear(self.hid_out, self.out_features)
       

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.batch1(x)

        try:
            self.batch1(x)
        except:
            self.batch1.eval()
            self.batch1(x)
            self.batch1.train()

        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        out = self.batch2(x)
        # x = self.drop2(x)
        # out = self.fc3(x)
        
        return out


class ParentModel(nn.Module):
    def __init__(self, data_modalities, hid_out, dropout, feature_dims):
        super(ParentModel, self).__init__()
        self.data_modalities = data_modalities
        self.hid_out = hid_out
        self.dropout = dropout
        self.feature_dims = feature_dims
        self.n_clinic, self.n_gene, self.n_meta = self.feature_dims
        
        valid_mods = ['clinical', 'gene', 'meta']
        assert all(mod in valid_mods for mod in data_modalities), \
                f'Accepted input data modalitites are: {valid_mods}'

        self.submodels = {}        
        # Clinical -----------------------------------------------------------#
        if 'clinical' in self.data_modalities:
            self.clinical_submodel = ClinicalNet1(self.hid_out, self.n_clinic)
            self.submodels['clinical'] = self.clinical_submodel

        # Transcriptomic -----------------------------------------------------#
        if 'gene' in self.data_modalities:
            self.gene_submodel = GeneNet1(self.hid_out, self.n_gene)
            self.submodels['gene'] = self.gene_submodel

        # Metabolic -----------------------------------------------------------#
        if 'meta' in self.data_modalities:
            self.meta_submodel = MetaNet1(self.hid_out, self.n_meta)
            self.submodels['meta'] = self.meta_submodel

        self.fc1 = torch.nn.Linear(self.hid_out*len(self.data_modalities),8)
        # self.relu = torch.nn.Softmax(dim=-1)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(self.dropout)
        self.fc2 = torch.nn.Linear(8, 1)
        
    def forward(self, x):
        
        x1 = x[:, :self.n_clinic]
        x2 = x[:, self.n_clinic:self.n_gene+self.n_clinic]
        x3 = x[:, self.n_gene+self.n_clinic:]
        data_list = {'clinical': x1, 'gene':x2, 'meta':x3}
        outputs = []
        for modality in self.submodels:
            m = self.submodels[modality](data_list[modality])
            outputs.append(m)
        # print(len(outputs))
        
        # x1 = self.submodels['clinical'](x1)
        # x2 = self.submodels['gene'](x2)
        # x3 = self.submodels['meta'](x3)
        y = torch.cat(outputs, dim=1)
        
        fc1 = self.fc1(y)
        relu = self.relu(fc1)
        drop = self.drop(relu)
        output = self.fc2(drop)
        
        return output


class Intermediate_Wrap(BaseEstimator):
    def __init__(
        self,
        modality=['clinical', 'gene', 'meta'],
        feature_dims=(6, 6005, 2969),
        hid_out=32,
        learning_rate=0.001,
        dropout=0.1,
        batch_size=256,
        epochs=200,
    ):
        self.modality = modality
        self.feature_dims = feature_dims
        self.hid_out = hid_out
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y):
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        
        self.net_ = ParentModel(self.modality, self.hid_out, self.dropout, self.feature_dims)
        self.deepsurv_ = CoxPH(self.net_, tt.optim.Adam)
        self.deepsurv_.optimizer.set_lr(self.learning_rate)

        # Get Validation set
        X_train_, X_val_, y_train_, y_val_ = train_test_split(X, y, test_size=size,
                                                      random_state=RANDOM_STATE)
        y_train = get_target(pd.DataFrame(y_train_))
        y_val = get_target(pd.DataFrame(y_val_))
        val = X_val_, y_val


        callbacks = [tt.callbacks.EarlyStopping(patience=20)]
        # print(self.deepsurv_)
        self.deepsurv_.fit(
            X_train_,
            y_train,
            self.batch_size,
            self.epochs,
            callbacks=callbacks,
            verbose=False,
            val_data=val,
            val_batch_size=self.batch_size
        )
        # save weights
        torch.save(
            self.deepsurv_.net.state_dict(),
            "bc_intermediate_wrap_weights.pt"
        )
        return  self

    def predict(self, X):
        _ = self.deepsurv_.compute_baseline_hazards()
        return self.deepsurv_.predict(X)

    def score(self, X, y):
        _ = self.deepsurv_.compute_baseline_hazards()
        surv = self.deepsurv_.predict(X)
        durations_test, events_test  = get_target(pd.DataFrame(y))
        events_test =  np.where(events_test == 1, True, False)

        ev = concordance_index_censored(events_test, durations_test, surv.flatten())
        return ev[0]

# Early fusion
class BC_NeuralNet(nn.Module):
    """Network for early fusion model.
      """
    def __init__(self, input, dropout):
        super(BC_NeuralNet, self).__init__()

        self.dropout = dropout
        self.input = input

        # Linear Layers
        self.fc1 = torch.nn.Linear(self.input, int(self.input/8))
        self.relu1 = torch.nn.ReLU()
        self.batch1 = torch.nn.BatchNorm1d(int(self.input/8))
        self.drop1 = torch.nn.Dropout(self.dropout)
        self.fc2 = torch.nn.Linear(int(self.input/8), 4)
        self.relu2 = torch.nn.ReLU()
        self.batch2 = torch.nn.BatchNorm1d(4)
        self.drop2 = torch.nn.Dropout(self.dropout)
        self.fc3 = torch.nn.Linear(4, 1)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        try:
            self.batch1(x)
        except:
            self.batch1.eval()
            self.batch1(x)
            self.batch1.train()

        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.batch2(x)
        x = self.drop2(x)
        out = self.fc3(x)

        return out


class BC_Early_Wrap(BaseEstimator):
    def __init__(
        self,
        learning_rate=0.001,
        dropout=0.1,
        batch_size=256,
        epochs=200,
    ):
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y):
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)

        self.net_ = BC_NeuralNet(X.shape[1], self.dropout)
        self.deepsurv_ = CoxPH(self.net_, tt.optim.Adam)
        self.deepsurv_.optimizer.set_lr(self.learning_rate)

        # Check
        # X, y = check_X_y(X, y)
        # self.is_fitted_ = True

        # Get Validation set
        X_train_, X_val_, y_train_, y_val_ = train_test_split(X, y, test_size=size,
                                                      random_state=RANDOM_STATE)
        y_train = get_target(pd.DataFrame(y_train_))
        y_val = get_target(pd.DataFrame(y_val_))
        val = X_val_, y_val


        callbacks = [tt.callbacks.EarlyStopping(patience=20)]
        # print(self.deepsurv_)
        self.deepsurv_.fit(
            X_train_,
            y_train,
            self.batch_size,
            self.epochs,
            callbacks=callbacks,
            verbose=False,
            val_data=val,
            val_batch_size=self.batch_size
        )
        # save weights
        torch.save(
            self.deepsurv_.net.state_dict(),
            "bc_early_wrap_weights.pt"
        )
        return  self

    # def predict(self, X):
    #     _ = self.deepsurv_.compute_baseline_hazards()
    #     return self.deepsurv_.predict_surv_df(X)

    def predict(self, X):
        _ = self.deepsurv_.compute_baseline_hazards()
        return self.deepsurv_.predict(X)

    def score(self, X, y):
        _ = self.deepsurv_.compute_baseline_hazards()
        surv = self.deepsurv_.predict(X)
        durations_test, events_test  = get_target(pd.DataFrame(y))
        events_test =  np.where(events_test == 1, True, False)

        ev = concordance_index_censored(events_test, durations_test, surv.flatten())
        # print('Result', ev.concordance_td())
        return ev[0]