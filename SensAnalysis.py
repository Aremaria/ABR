import pandas as pd
import numpy as np

from functions import ScalerTransform, ChangeDfPCA

import warnings
warnings.filterwarnings("ignore")

def MakeSensParam(df, model_columns, scaler, model, pca_model, pca_col, param): # прогон для одного параметра
    param_list = [t for t in pca_col if t.split(', ')[-1] == param]
    X_init = df[model_columns]
    X_init_pca_scaled = ChangeDfPCA(ScalerTransform(X_init, scaler), pca_model, pca_col)

    pred_init = model.predict_proba(X_init_pca_scaled)[:,1]
    sum_init = pred_init.mean()
        
    step_range = [i/100.0 for i in range(0, 210, 10)]
    result = pd.DataFrame()
    for step in step_range:
        X_sens = X_init.copy()
               
        for col in param_list:
            param_init = X_init[col]
            X_sens[col] = step*param_init
            
        # преобразование PCA и scaler
        X_sens_pca_scaled = ChangeDfPCA(ScalerTransform(X_sens, scaler), pca_model, pca_col)                                             
        pred = model.predict_proba(X_sens_pca_scaled)[:,1] 
        
        step_range = step - 1    
        sum_sens = pred.mean()
        
        row = pd.DataFrame({'param': [param], 'step': [step], 'step_range': [step_range],                         
                            'simulation': [sum_sens], 'simulation_delta': [sum_sens/sum_init-1]})  

        result = pd.concat([result, row])
        
    return result

def MakeSensParamOther(df, model_columns, scaler, model, pca_model, pca_col, param): # прогон для одного параметра
    X_init = df[model_columns]
    X_init_pca_scaled = ChangeDfPCA(ScalerTransform(X_init, scaler), pca_model, pca_col)

    pred_init = model.predict_proba(X_init_pca_scaled)[:,1]
    sum_init = pred_init.mean()
        
    step_range = [i/100.0 for i in range(0, 210, 10)]
    result = pd.DataFrame()
    for step in step_range:
        X_sens = X_init.copy()
               
        param_init = X_init[param]
        X_sens[param] = step*param_init
            
        # преобразование PCA и scaler
        X_sens_pca_scaled = ChangeDfPCA(ScalerTransform(X_sens, scaler), pca_model, pca_col)                                             
        pred = model.predict_proba(X_sens_pca_scaled)[:,1] 
        
        step_range = step - 1    
        sum_sens = pred.mean()
        
        row = pd.DataFrame({'param': [param], 'step': [step], 'step_range': [step_range],                         
                            'simulation': [sum_sens], 'simulation_delta': [sum_sens/sum_init-1]})  

        result = pd.concat([result, row])
        
    return result


def SensAn(df, model_columns, target, scaler, model, pca_model, param_list, pca_col): # прогон для всех
    df_sens = pd.DataFrame()
    for param in param_list:
        res = MakeSensParam(df, model_columns, scaler, model, pca_model, pca_col, param)
        # res = MakeSensParamOther(df, model_columns, scaler, model, pca_model, pca_col, param)
        df_sens = pd.concat([df_sens, res])
        
    y_actual_mean = df[target].mean()
    y_model_mean = np.mean(df_sens['simulation'].loc[df_sens['step']==1])
    print('Check: ', 'actual level:', round(y_actual_mean*100,2), '%,', 'model level:', round(y_model_mean*100,2), '%',)
    
    df_sens['actual base level'] = y_actual_mean
    df_sens['model base level'] = y_model_mean
    
    return df_sens