# Load package
import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Load mapping file
kegg_gene =  pd.read_csv('Data/kegg_gene_all.csv')
map_id = pd.read_csv('Data/meta/mapped_gene_hugo_id.csv')

def hybrid_selection(df_gene, num_gene_mrmr=None, num_gene_kegg=100):
    # Select KEGG genes and mrmr genes
    kegg_gene =  pd.read_csv('Data/kegg_gene_all.csv')
    mrmr = pd.read_csv('Data/GE_3279_status.csv', index_col=-1)
    gene_list = kegg_gene.query('Benjamini  <0.05').nlargest(num_gene_kegg, columns='Benjamini' )
    mrmr = mrmr.iloc[:,:num_gene_mrmr]

    gene_list_kegg =[]
    for g in gene_list['Genes']:
        g_list = g.split(", ")
        gene_list_kegg.append(g_list)

    select_gene = [number for sublist in gene_list_kegg for number in sublist]
    gene_kegg = set(select_gene)
    print(f'Total KEGG selected gene: {len(gene_kegg)}')

    # Select mrmr gene
    mrmr_gene = set(mrmr.columns)

    union_gene = list(gene_kegg | mrmr_gene)
    # inter_gene = list(gene_kegg & mrmr_gene)
    print(f'Total combined selected gene: {len(union_gene)}')
    # valid_genes = [g for g in union_gene if g in df_gene.columns]
    # df_gene = df_gene[valid_genes]
    df_gene = df_gene[union_gene]

    # Map gene ID with hugo id
    map_id1 =  dict(zip(map_id["Entrez_Gene_Id"].astype(str), map_id["Hugo_Symbol"]))
    df_gene = df_gene.rename(columns=map_id1)
    return df_gene


def normalise_data(X_train, X_test):
    nm = MinMaxScaler()
    x_train = nm.fit_transform(X_train).astype('float32')
    x_test = nm.transform(X_test).astype('float32')
    # x_val = nm.transform(X_val).astype('float32')
    return [x_train, x_test]

def predict_omic(model, X_test, durations_test, events_test):
    _ = model.compute_baseline_hazards()
    surv = model.predict_surv_df(X_test)
    pred = model.predict(X_test)
    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    c_index = round(ev.concordance_td(), 4)
    print("C-Index ", c_index)

    return [c_index, surv, pred]