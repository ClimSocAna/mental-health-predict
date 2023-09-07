import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from ydata_profiling import ProfileReport
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.compose import ColumnTransformer
from load_prepare_data import load_prepare_data
import statsmodels.api as sm
from alibi.explainers import PartialDependence
from alibi.explainers import PermutationImportance
from pingouin import compute_effsize
import pickle

profile_data = False
baseline_check = False
recreate_data = False

if recreate_data:
    X, y, categorical, weight = load_prepare_data('sdq')
    file = open('data.p', 'wb')
    pickle.dump((X, y, categorical, weight),file)

else:
    file = open('data.p', 'rb')
    saveddat = pickle.load(file)
    X, y, categorical, weight = saveddat[0], saveddat[1], saveddat[2], saveddat[3]


if profile_data:
    profile = ProfileReport(X, interactions=None)
    profile.to_file("X_descr.html")


X[categorical] = OrdinalEncoder().fit_transform(X[categorical])
X[categorical] = X[categorical].astype('category')


if baseline_check:
    X = X[['FEMOTION']]
    categorical = []


binary = X.columns[X.nunique() == 2].to_list()
numerical = list(set(X.columns) - set(categorical) - set(binary))

lin_final_est = RidgeCV(alphas=(0.01, 250, 500, 1000, 1500, 2000))

def multimodel_nonlin(lX = None, ly = None, weight = None, random_state = 1):
    param_grid = {'learning_rate': (0.05, 0.1, 0.2, 0.3), 'max_leaf_nodes': (
            4, 8), 'min_samples_leaf': ([20, 30, 40])}
    multimodel = GridSearchCV(
        HistGradientBoostingRegressor(
            categorical_features=categorical,
            max_iter=400,
            early_stopping=True,
            tol=1e-10,
            random_state=random_state),
        param_grid=param_grid)
    if lX is not None and ly is not None:
        multimodel.fit(
                lX,
                stats.zscore(ly),
                sample_weight=weight)
    return multimodel

def multimodel_lin(lX = None, ly = None, weight = None):
    multimodel = Pipeline([('CatOneHot', ColumnTransformer([('CatOneHot', OneHotEncoder(), categorical), ('NumScale', StandardScaler(), numerical)], remainder='passthrough')),
                     ('Impute', KNNImputer()),
                     ('Predict', lin_final_est)])
    if lX is not None and ly is not None:
        multimodel.fit(
                lX,
                stats.zscore(ly),
                Predict__sample_weight=weight)
    return multimodel

def performCV(X, y, linear=True, nonlinear=True, weight=None, random_seed=0):
    linear_preds = []
    nonlinear_preds = []
    linear_models = []
    nonlinear_models = []
    truth = []
    kfold = KFold(10, shuffle=True, random_state=random_seed)
    for tri, tei in kfold.split(X):
        print('Next CV Fold')
        truth.append(y.iloc[tei])
        if nonlinear:
            gbm=multimodel_nonlin(X.iloc[tri], y.iloc[tri], weight.iloc[tri] if weight is not None else None, random_seed)
            nonlinear_models.append(gbm)
            nonlinear_preds.append(gbm.predict(X.iloc[tei]))
        if linear:
            linm = multimodel_lin(X.iloc[tri], y.iloc[tri], weight.iloc[tri] if weight is not None else None)
            linear_models.append(linm)
            linear_preds.append(linm.predict(X.iloc[tei]).ravel())
    truth = pd.concat(truth)
    nonlinear_acc = 'not estimated'
    linear_acc = 'not estimated'
    if nonlinear:
        nonlinear_preds = np.hstack(nonlinear_preds)
        nonlinear_acc_model = LinearRegression().fit(pd.DataFrame(stats.zscore(truth)),
                                                     pd.DataFrame(stats.zscore(nonlinear_preds)), sample_weight=weight[truth.index])
        nonlinear_acc = nonlinear_acc_model.coef_[0]
    if linear:
        linear_preds = np.hstack(linear_preds)
        linear_acc_model = LinearRegression().fit(pd.DataFrame(stats.zscore(truth)),
                                                  pd.DataFrame(stats.zscore(linear_preds)), sample_weight=weight[truth.index])
        linear_acc = linear_acc_model.coef_[0]

    return (linear_models, nonlinear_models), (linear_acc, nonlinear_acc)


acclist = []
modellist = []

# Perform cross validation with different random seeds governing the splits and
# the fitting of the GBM
for r in range(10):
    print(r)
    models, acc = performCV(X, y, weight=weight, random_seed=r)
    acclist.append(acc)
    modellist.append(models)

# Create a number of GBM models with different random seeds to account for random 
# fluctuations in feature importances
gbm_final=[]
for r in range(10):
    gbm_final.append(multimodel_nonlin(X, y, weight, random_state=r))
    
lin_final = multimodel_lin(X, y, weight)

def feature_importances(X, y, multimodel, weight=None):
    mean_fi=np.empty((X.shape[1], len(multimodel)))
    for m in range(len(multimodel)):
        predict_fn = lambda x: multimodel[m].predict(x)
        fi = PermutationImportance(predict_fn,loss_fns='mean_squared_error',feature_names=X.columns.to_list())
        fi_exp = fi.explain(X.values, stats.zscore(y.values),n_repeats=10, sample_weight=weight.values)
        mean_fi[:,m] = np.array([x['mean'] for x in fi_exp.feature_importance[0]])
    fitab = pd.DataFrame({'feature_importance':mean_fi.mean(1)})
    fitab['Variable_Description']=''
    fitab['Variable_Domain']=''
    fitab.index=X.columns
    codes = pd.read_excel(
        'predictors_desc.xlsx',
        header=None).set_index(0)
    for var in fitab.index.to_list():
        fitab.at[var,'Variable_Description'] = codes.loc[var,1]
        fitab.at[var,'Variable_Domain'] = codes.loc[var,2]
    return fitab

def resultsTableLin(X, y, multimodel, weight):
    Xoh = pd.get_dummies(X, columns=categorical, dummy_na=True)
    Xoh.drop(columns=Xoh.columns[Xoh.nunique() == 1], inplace=True)
    Xohcols = Xoh.columns
    Xidx = X.index
    Xoh[numerical] = StandardScaler().fit_transform(Xoh[numerical])
    Xoh = pd.DataFrame(KNNImputer().fit_transform(Xoh))
    Xoh.columns = Xohcols
    Xoh.index = Xidx
    ctrans = multimodel.named_steps['CatOneHot']
    codes = pd.read_excel(
        'predictors_desc.xlsx',
        header=None).set_index(0)
    fnames_prepro = ctrans.get_feature_names_out(ctrans.feature_names_in_)
    coefs = pd.Series(multimodel._final_estimator.coef_.ravel())
    coefs.index = pd.Series(fnames_prepro).str.removeprefix(
        'CatOneHot__').str.removeprefix('remainder__').str.removeprefix('NumScale__')
    restab = pd.DataFrame(
        columns=[
            'UniVar_Raw',
            'UniVar_Raw_CI',
            'UniVar_BaseCorrect',
            'MultiVar',
            'Variable_Description'],
        index=Xoh.columns)
    for var in restab.index.to_list():
        if weight is not None:
            model_raw = sm.WLS(stats.zscore(y), sm.add_constant(Xoh[var]), weights=weight).fit()
            model_basec = sm.WLS(stats.zscore(y), sm.add_constant(Xoh[[var, 'FEMOTION']]), weights=weight).fit()
        else:
            model_raw = sm.OLS(stats.zscore(y), sm.add_constant(Xoh[var])).fit()
            model_basec = sm.OLS(stats.zscore(y), sm.add_constant(Xoh[[var, 'FEMOTION']])).fit()            
        restab.at[var, 'UniVar_Raw'] = model_raw.params[1]
        restab.at[var, 'UniVar_Raw_CI'] = (model_raw.conf_int().iloc[1,0],model_raw.conf_int().iloc[1,1])
        restab.at[var, 'UniVar_BaseCorrect'] = model_basec.params[1]
        restab.at[var, 'MultiVar'] = coefs[var]
        try:
            restab.at[var, 'Variable_Description'] = codes.loc[var,1]
        except BaseException:
            pass
    return restab


def plotVarImpact(X, y, multimodel, var):
    bin_pos = pd.Series(X[var].unique()).sort_values().dropna()
    if len(bin_pos) > 20:
        bin_pos = np.arange(
            X[var].min(),
            X[var].max(),
            (X[var].max() - X[var].min()) / 20)
    sns.regplot(x=X[var], y=y, x_bins=bin_pos, ci=None, fit_reg=False)
    def predfun(x): return multimodel.predict(x)
    partdep = PartialDependence(predfun)
    pd_exp = partdep.explain(
        X=X.values, features=[
            X.columns.get_loc(var)], grid_points={
            X.columns.get_loc(var): bin_pos})
    plt.plot(np.array(pd_exp.feature_values).ravel(),
             pd_exp.pd_values[0].ravel(), 'o')
    codes = pd.read_excel(
        'predictors_desc.xlsx',
        header=None).set_index(0)
    try:
        plt.title(codes.loc[var, 1])  
    except:
        plt.title(var)
    plt.show()
        
def plot_univariate_results(values, confidence_intervals, labels, save=False):
    values = np.array(values)
    ci = values-np.array([confidence_intervals[x][0] for x in confidence_intervals.index])
    y_positions = range(len(values))
    fig = plt.figure(figsize=(20,len(values)*0.3))
    plt.barh(y_positions, values, xerr=ci, align='center', alpha=0.5,
             color='#005EB8',ecolor='black', capsize=5)
    plt.yticks(y_positions, labels)
    y_ticks = plt.yticks()[0]
    for y in y_ticks:
        plt.axhline(y, color='lightgray', linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=18)
    plt.xlabel('Univariate Coefficient',fontsize=20)
    plt.ylabel('Predictor',fontsize=20)
    plt.tight_layout()
    if save:
        fig.savefig('UniFig.png',dpi=300)
    else:
        fig.show()

restab= resultsTableLin(X,y,lin_final,weight)

# Select strongest significant univariate effects and plot results
lower_ci_neg=np.array([restab.at[x,'UniVar_Raw_CI'][0]<0 for x in restab.index])
upper_ci_neg=np.array([restab.at[x,'UniVar_Raw_CI'][1]<0 for x in restab.index])
ci_sig=lower_ci_neg==upper_ci_neg
eff_rel = restab['UniVar_Raw'].abs()>0.1
relevant_and_sig_uni = restab.loc[eff_rel&ci_sig,:]
relevant_and_sig_uni = relevant_and_sig_uni.reindex(relevant_and_sig_uni['UniVar_Raw'].abs().sort_values().index)
selected_effects = relevant_and_sig_uni.index[-20:]
plot_univariate_results(relevant_and_sig_uni.loc[selected_effects,'UniVar_Raw'],relevant_and_sig_uni.loc[selected_effects,'UniVar_Raw_CI'],relevant_and_sig_uni.loc[selected_effects,'Variable_Description'])

# Create table with feature importance of individual predictors based on list
# with trained nonlinear models
fitab=feature_importances(X, y, gbm_final, weight)

# Get importance for feature domains based on list
# with trained nonlinear models
dom_idxs=[]
for dom in fitab['Variable_Domain'].unique():
    dom_idxs.append(tuple(np.where(fitab['Variable_Domain']==dom)[0]))
mean_dom_fis=np.empty((len(gbm_final),len(dom_idxs)))
for m in range(len(gbm_final)):
    predict_fn = lambda x: gbm_final[m].predict(x)
    gfi = PermutationImportance(predict_fn,loss_fns='mean_squared_error',feature_names=X.columns.to_list())
    gfi_exp = gfi.explain(X.values, stats.zscore(y.values), dom_idxs, sample_weight=weight.values)
    mean_dom_fis[m,:]=[x['mean'] for x in gfi_exp.feature_importance[0]]
gfi_res=pd.Series(mean_dom_fis.mean(0))
gfi_res.index=fitab['Variable_Domain'].unique()
pickle.dump(gfi_res, open('gfi_mc.p', 'wb'))

def merge_export_tabs(lrestab, lfitab, missings):
    lrestab['UniVar_Raw']=lrestab['UniVar_Raw'].astype(float).round(4)
    lrestab['UniVar_Raw_CI']=[(np.round(x[0],4),np.round(x[1],4)) for x in lrestab['UniVar_Raw_CI']]
    lrestab=lrestab.join((lfitab['feature_importance']-1).round(4),how='outer')
    lrestab.drop(columns=['UniVar_BaseCorrect','MultiVar'],inplace=True)
    lrestab=lrestab[['Variable_Description','UniVar_Raw','UniVar_Raw_CI','feature_importance']]
    return lrestab

mtab = merge_export_tabs(restab.copy(), fitab.copy(), X.isnull().mean(0))

def counterfactualSim(multimodel, lX, intervention):
    basepred = multimodel.predict(lX)
    if intervention == 'exercise':
        targeted = lX['FCPHEX00']<-3
        lX.loc[targeted,'FCPHEX00']= lX.loc[targeted,'FCPHEX00']+1
        lX.loc[targeted,'Mean_Accelerometer']=lX.loc[targeted,'Mean_Accelerometer']+30
    if intervention == 'exercise_plus':
        targeted = lX['FCPHEX00']<-3
        lX.loc[targeted,'FCPHEX00']= lX.loc[targeted,'FCPHEX00']+1
        lX.loc[targeted,'FCORGA00']= lX.loc[targeted,'FCORGA00']+2
        lX.loc[lX['FCORGA00']>-1,'FCORGA00']= -1
        lX.loc[targeted,'Mean_Accelerometer']=lX.loc[targeted,'Mean_Accelerometer']+30
        lX.loc[targeted,'FPEER']= lX.loc[targeted,'FPEER']-2
        lX.loc[lX['FPEER']<0,'FPEER']= 0
        lX.loc[targeted,'FCSTFR0A']= lX.loc[targeted,'FCSTFR0A']+1
        lX.loc[lX['FCSTFR0A']>-1,'FCSTFR0A']= -1
    if intervention == 'ses_misstarget':
        targeted=lX['FOEDE000']>np.nanquantile(lX['FOEDE000'],0.75)
        lX.loc[targeted,'FOEDE000']=lX.loc[targeted,'FOEDE000']+115
        lX.loc[targeted,'FCSTYU00']=lX.loc[targeted,'FCSTYU00']+20
        lX.loc[lX['FCSTYU00']>100,'FCSTYU00']= 100
        lX.loc[targeted,'FCSTYY00']=lX.loc[targeted,'FCSTYY00']+30
        lX.loc[lX['FCSTYY00']>100,'FCSTYY00']= 100
    if intervention == 'ses':
        targeted=lX['FOEDE000']<np.nanquantile(lX['FOEDE000'],0.25)
        lX.loc[targeted,'FOEDE000']=lX.loc[targeted,'FOEDE000']+115
        lX.loc[targeted,'FCSTYU00']=lX.loc[targeted,'FCSTYU00']+20
        lX.loc[lX['FCSTYU00']>100,'FCSTYU00']= 100
        lX.loc[targeted,'FCSTYY00']=lX.loc[targeted,'FCSTYY00']+30
        lX.loc[lX['FCSTYY00']>100,'FCSTYY00']= 100
    if intervention == 'ses_plus':
        targeted=lX['FOEDE000']<np.nanquantile(lX['FOEDE000'],0.25)
        lX.loc[targeted,'FOEDE000']=lX.loc[targeted,'FOEDE000']+115
        lX.loc[targeted,'FCSTYU00']=lX.loc[targeted,'FCSTYU00']+20
        lX.loc[lX['FCSTYU00']>100,'FCSTYU00']= 100
        lX.loc[targeted,'FCSTYY00']=lX.loc[targeted,'FCSTYY00']+30
        lX.loc[lX['FCSTYY00']>100,'FCSTYY00']= 100
        lX.loc[targeted,'FCCINE00']=lX.loc[targeted,'FCCINE00']+1
        lX.loc[lX['FCCINE00']>-1,'FCCINE00']= -1
        lX.loc[targeted,'FCSPOR00']=lX.loc[targeted,'FCSPOR00']+1
        lX.loc[lX['FCSPOR00']>-1,'FCSPOR00']= -1
        lX.loc[targeted,'FDKESSLf']=lX.loc[targeted,'FDKESSLf']-3
        lX.loc[lX['FDKESSLf']<0,'FDKESSLf']= 0
        lX.loc[targeted,'FDKESSLm']=lX.loc[targeted,'FDKESSLm']-3
        lX.loc[lX['FDKESSLm']<0,'FDKESSLm']= 0
    if intervention == 'ses_plus_misstarget':
        targeted=lX['FOEDE000']>np.nanquantile(lX['FOEDE000'],0.75)
        lX.loc[targeted,'FOEDE000']=lX.loc[targeted,'FOEDE000']+115
        lX.loc[targeted,'FCSTYU00']=lX.loc[targeted,'FCSTYU00']+20
        lX.loc[lX['FCSTYU00']>100,'FCSTYU00']= 100
        lX.loc[targeted,'FCSTYY00']=lX.loc[targeted,'FCSTYY00']+30
        lX.loc[lX['FCSTYY00']>100,'FCSTYY00']= 100
        lX.loc[targeted,'FCCINE00']=lX.loc[targeted,'FCCINE00']+1
        lX.loc[lX['FCCINE00']>-1,'FCCINE00']= -1
        lX.loc[targeted,'FCSPOR00']=lX.loc[targeted,'FCSPOR00']+1
        lX.loc[lX['FCSPOR00']>-1,'FCSPOR00']= -1
        lX.loc[targeted,'FDKESSLf']=lX.loc[targeted,'FDKESSLf']-3
        lX.loc[lX['FDKESSLf']<0,'FDKESSLf']= 0
        lX.loc[targeted,'FDKESSLm']=lX.loc[targeted,'FDKESSLm']-3
        lX.loc[lX['FDKESSLm']<0,'FDKESSLm']= 0
    if intervention == 'some':
        screentime=lX['FCSOME00']/7+lX['FCTVHO00']+lX['FCCOMH00']
        targeted=screentime>np.nanquantile(screentime,0.75)        
        lX.loc[targeted,'FCSOME00']= lX.loc[targeted,'FCSOME00']-1
        lX.loc[lX['FCSOME00']<1,'FCSOME00']= 1
        lX.loc[targeted,'FCTVHO00']= lX.loc[targeted,'FCTVHO00']-1
        lX.loc[lX['FCTVHO00']<1,'FCTVHO00']= 1
        lX.loc[targeted,'FCCOMH00']= lX.loc[targeted,'FCCOMH00']-1
        lX.loc[lX['FCCOMH00']<1,'FCCOMH00']= 1
    if intervention == 'some_plus':
        screentime=lX['FCSOME00']/7+lX['FCTVHO00']+lX['FCCOMH00']
        targeted=screentime>np.nanquantile(screentime,0.75)        
        lX.loc[targeted,'FCSOME00']= lX.loc[targeted,'FCSOME00']-1
        lX.loc[lX['FCSOME00']<1,'FCSOME00']= 1
        lX.loc[targeted,'FCTVHO00']= lX.loc[targeted,'FCTVHO00']-1
        lX.loc[lX['FCTVHO00']<1,'FCTVHO00']= 1
        lX.loc[targeted,'FCCOMH00']= lX.loc[targeted,'FCCOMH00']-1
        lX.loc[lX['FCCOMH00']<1,'FCCOMH00']= 1
        lX['FCBTNG00']=lX['FCBTNG00']-1
        lX.loc[lX['FCBTNG00']<-4,'FCBTNG00']= -4
        lX['FCCYBU00']=lX['FCCYBU00']-1
        lX.loc[lX['FCCYBU00']<-6,'FCCYBU00']= -6       
        lX['FCWYLK00']=lX['FCWYLK00']+1
        lX.loc[lX['FCWYLK00']>-1,'FCWYLK00']= -1
        lX.loc[targeted,'FCSTFR0A']= lX.loc[targeted,'FCSTFR0A']+1
        lX.loc[lX['FCSTFR0A']>-1,'FCSTFR0A']= -1
        lX.loc[targeted,'FCMDSJ00']= lX.loc[targeted,'FCMDSJ00']-1
        lX.loc[lX['FCMDSJ00']<1,'FCMDSJ00']= 1
        lX['FPEER']= lX['FPEER']-1
        lX.loc[lX['FPEER']<0,'FPEER']= 0
        lX.loc[targeted,'FCSLLN00']= lX.loc[targeted,'FCSLLN00']-1
        lX.loc[lX['FCSLLN00']<1,'FCSLLN00']= 1
        lX.loc[targeted,'FCSLTR00']= lX.loc[targeted,'FCSLTR00']-1
        lX.loc[lX['FCSLTR00']<-6,'FCSLTR00']= -6
    intpred = multimodel.predict(lX)
    return basepred, intpred, targeted

# Compute simulated intervention effects based on list with trained nonlinear
# models
inters=['exercise', 'exercise_plus', 'ses', 'ses_plus', 'ses_misstarget', 'ses_plus_misstarget', 'some', 'some_plus']
interrestab=pd.DataFrame(index=inters,columns=['d','p'])
for i in inters:
    all_basepreds=pd.DataFrame(index=y.index,columns=range(len(gbm_final)))
    all_intpreds=pd.DataFrame(index=y.index,columns=range(len(gbm_final)))
    for m in range(len(gbm_final)):
        basepred, intpred, targeted = counterfactualSim(gbm_final[m], X.copy(), i)
        all_basepreds[m]=basepred
        all_intpreds[m]=intpred
    interrestab.at[i,'p']=stats.ttest_1samp(all_basepreds.loc[targeted,:].mean(1)-all_intpreds[targeted].mean(1), 0)[1]
    interrestab.at[i,'d']=compute_effsize(all_basepreds.loc[targeted,:].mean(1), all_intpreds[targeted].mean(1), paired=True)
