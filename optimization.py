import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint

from functions import ChangeDfPCA, ScalerTransform

import warnings
warnings.filterwarnings("ignore")

# возможно надо писать отдельные оптимизации для вне / внутри

def f(x, x_ddd, x_other, model_cols, ddd_columns, model, pca_model, scaler):        
    x_ddd_opt = x_ddd * x   
   
    X = pd.concat([x_other, x_ddd_opt], axis=1) 
    X = X[model_cols] # восстанавливаем порядок столобцов
    
    X_scaled = ScalerTransform(X, scaler) 
    X_pca_scaled = ChangeDfPCA(X_scaled, pca_model, ddd_columns) # преобразуем часть с ddd через PCA   
    y_pred = model.predict(X_pca_scaled)
    
    res = y_pred.mean()
    
    return res

# проверка
def makeY(W, X_ddd, X_others, scaler, model, pca_model, ddd_columns, model_columns):
    x_ddd_opt = X_ddd * np.array(W)  
       
    X = pd.concat([X_others, x_ddd_opt], axis=1) 
    X = X[model_columns] # восстанавливаем порядок стообцов
    
    X_scaled = ScalerTransform(X, scaler) 
    X_pca_scaled = ChangeDfPCA(X_scaled, pca_model, ddd_columns) # преобразуем часть с ddd через PCA   
    y_pred = model.predict(X_pca_scaled)
    
    return y_pred


def MakeOptimize(datapoint, dict_forec, model, pca_model, scaler):
    other_feat = dict_forec['common_cols'] # прочие параметры
    ddd_columns = dict_forec['ddd_columns'] # все колонки с ddd
    model_columns = dict_forec['model_columns'] # колонки для модели
    
    ddd_columns_cur = [t for t in ddd_columns if t.split(', ')[1] != 'PrevYear'] # ddd текущие
    ddd_columns_prev = [t for t in ddd_columns if t.split(', ')[1] == 'PrevYear'] # ddd прошлого года
    
    X_ddd = dict_forec['base'][ddd_columns_cur] # оптимизируемая часть - всегда берется последняя точка  известного потребления
    X_others = datapoint[other_feat + ddd_columns_prev] # остальные параметры
    
    W0_ddd = pd.Series(np.ones(len(X_ddd.columns))) # в качестве начальной берем текущие, следовательно 1-чный вектор
    
    # ограничения    
    matrix = np.eye(len(W0_ddd))
    lb = np.zeros(len(W0_ddd))  # нижние границы
    ub = 2 * np.ones(len(W0_ddd))  # верхние границы
    cons = LinearConstraint(matrix, lb, ub)
    
    result = minimize(f, W0_ddd, args=(X_ddd, X_others, model_columns, ddd_columns, model, pca_model, scaler), method='COBYLA', constraints=cons)   
    opt_ddd_coef = pd.DataFrame(result.x).rename(columns={0:'opt koeff'})
    opt_ddd_coef.index = ddd_columns_cur
    
    # приняем оптимальные коэффициенты к ddd
    X_ddd_opt = X_ddd * np.array(result.x)
    datapoint_opt = pd.concat([datapoint[dict_forec['Index']], X_others, X_ddd_opt], axis=1) 
    datapoint_opt = datapoint_opt[datapoint.columns] # восстанавливаем порядок стообцов
        
    return datapoint_opt
    



