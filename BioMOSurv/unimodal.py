import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sksurv.metrics import concordance_index_censored
import torch
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import torch.nn as nn

class ClinicalNet(nn.Module):
    """Clinical data extractor.
      """
    def __init__(self, input, dropout):
        super(ClinicalNet, self).__init__()

        self.dropout = dropout
        self.input = input

        # Linear Layers
        self.fc1 = torch.nn.Linear(self.input, 2)
        self.relu1 = torch.nn.ReLU()
        self.batch1 = torch.nn.BatchNorm1d(2)
        self.drop1 = torch.nn.Dropout(self.dropout)
        self.fc2 = torch.nn.Linear(2, 1)
        # self.relu2 = torch.nn.ReLU()
        # self.batch2 = torch.nn.BatchNorm1d(int(self.input/4))
        # self.drop2 = torch.nn.Dropout(self.dropout)
        # self.fc3 = torch.nn.Linear(int(self.input/4), 1)


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
        out = self.fc2(x)
        # x = self.relu2(x)
        # x = self.batch2(x)
        # x = self.drop2(x)
        # out = self.fc3(x)

        return out

class GeneNet(nn.Module):
    """Transcriptomic data extractor.
      """
    def __init__(self, input, dropout):
        super(GeneNet, self).__init__()

        self.dropout = dropout
        self.input = input

        # Linear Layers
        self.fc1 = torch.nn.Linear(self.input, int(self.input/16))
        self.relu1 = torch.nn.ReLU()
        self.batch1 = torch.nn.BatchNorm1d(int(self.input/16))
        self.drop1 = torch.nn.Dropout(self.dropout)
        self.fc2 = torch.nn.Linear(int(self.input/16), 4)
        self.relu2 = torch.nn.ReLU()
        self.batch2 = torch.nn.BatchNorm1d(4)
        self.drop2 = torch.nn.Dropout(self.dropout+0.1)
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

class MetaNet(nn.Module):
    """Metabolic model.
      """
    def __init__(self, input, dropout):
        super(MetaNet, self).__init__()

        self.dropout = dropout
        self.input = input

        # Linear Layers
        self.fc1 = torch.nn.Linear(self.input, int(self.input/16))
        self.relu1 = torch.nn.ReLU()
        self.batch1 = torch.nn.BatchNorm1d(int(self.input/16))
        self.drop1 = torch.nn.Dropout(self.dropout)
        self.fc2 = torch.nn.Linear(int(self.input/16), 4)
        self.relu2 = torch.nn.ReLU()
        self.batch2 = torch.nn.BatchNorm1d(4)
        self.drop2 = torch.nn.Dropout(self.dropout+0.1)
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
    
get_target = lambda df_clinic: (df_clinic['OS.time'].values, df_clinic['OS.Status'].values)


size=0.20
RANDOM_STATE =123
class Clinical_Wrap(BaseEstimator):
    def __init__(
        self,
        learning_rate=0.001,
        dropout=0.1,
        batch_size=256,
        epochs=200,
        n_clinic=6,
    ):
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_clinic = n_clinic

    def fit(self, X, y):
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)

        self.net_ = ClinicalNet(self.n_clinic, self.dropout)
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
            "clinical_weights.pt"
        )
        return self

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

class Gene_Wrap(BaseEstimator):
    def __init__(
        self,
        learning_rate=0.001,
        dropout=0.1,
        batch_size=256,
        epochs=200,
        n_gene=6005
    ):
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_gene = n_gene

    def fit(self, X, y):
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)

        self.net_ = GeneNet(self.n_gene, self.dropout)
        self.deepsurv_ = CoxPH(self.net_, tt.optim.Adam)
        self.deepsurv_.optimizer.set_lr(self.learning_rate)

        # Get Validation set
        X_train_, X_val_, y_train_, y_val_ = train_test_split(X, y, test_size=size,
                                                      random_state=RANDOM_STATE)
        y_ = get_target(pd.DataFrame(y_train_))
        y_val_ = get_target(pd.DataFrame(y_val_))
        val = X_val_, y_val_

        callbacks = [tt.callbacks.EarlyStopping(patience=50)]

        self.deepsurv_.fit(
            X_train_,
            y_,
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
            "gene_weights.pt"
        )
        return self

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

class Meta_Wrap(BaseEstimator):
    def __init__(
        self,
        learning_rate=0.01,
        dropout=0.7,
        batch_size=256,
        epochs=200,
        n_meta=2969
    ):
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_meta = n_meta

    def fit(self, X, y):
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)

        self.net_ = MetaNet(self.n_meta, self.dropout)
        self.deepsurv_ = CoxPH(self.net_, tt.optim.Adam)
        self.deepsurv_.optimizer.set_lr(self.learning_rate)

        # Get Validation set
        X_train_, X_val_, y_train_, y_val_ = train_test_split(X, y, test_size=size,
                                                      random_state=RANDOM_STATE)
        y_ = get_target(pd.DataFrame(y_train_))
        y_val_ = get_target(pd.DataFrame(y_val_))
        val = X_val_, y_val_

        callbacks = [tt.callbacks.EarlyStopping(patience=50)]


        self.deepsurv_.fit(
            X_train_,
            y_,
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
            "flux_weights.pt"
        )
        return self

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


def optimise_model(model, para,
                   inner_cv, X_train, y_train,
                   X_test, y_test):
    grid = GridSearchCV(model, para, cv=inner_cv, n_jobs=-1)
    grid.fit(X_train, y_train)

    # Best model
    best_score = grid.best_score_
    best_param = grid.best_params_
    print("Best parameter", best_param)
    # print("Best training score", best_score)
    model = grid.best_estimator_
    cindex = model.score(X_test, y_test)
    print("Test score", cindex)
    return model, cindex, best_param

