import pandas as pd
import numpy as np
import pickle
import json
import streamlit as st
import lightgbm as lgb


def LoadDataModel(): # загрузка данных для моделирования
    df_ABR_model = pd.read_csv('./DATA/ABR_model.csv', index_col=0)
    df_DDD_model = pd.read_csv('./DATA/DDDs_model.csv', index_col=0) 
    dict_data = json.loads(open('./DATA/dict_feat_init.txt').read())
    
    return df_ABR_model, df_DDD_model, dict_data

def LoadDataForecast(pairname, typeparam): # загрузка данных для моделирования
    data = pd.read_csv(f'''./results/data/{(pairname + '_' + typeparam + '.csv')}''', index_col=0)
    dict_model = json.loads(open(f'''./results/model/{(pairname + '_' + typeparam + '_dict.txt')}''').read())
    
    return data, dict_model

def LoadModelForecast(pairname, typeparam): # прогрузка данных для формирвоания прогноза    
    model = lgb.Booster(model_file=str(f'''./results/model/{(pairname + '_' + typeparam + '.txt')}''')) 
    scaler = pickle.load(open(f'''./results/scaler/{(pairname + '_' + typeparam + '.sav')}''', 'rb')) 
    pca_model = pickle.load(open(f'''./results/pca/{(pairname + '_' + typeparam + '.pkl')}''','rb'))
   
    return model, scaler, pca_model

def SaveDataModel(pairname, vecs, typeparam): # сохранение данных (с каким набором параметров обучали)
    stat_year, stat_region, df_sens, df_metrics, importance, model_final, scaler, pca_model_final, df_full, dict_model = vecs
    stat_year.to_excel(f'''./results/base/{(pairname + '_' + typeparam + '_year.xlsx')}''')
    stat_region.to_excel(f'''./results/base/{(pairname + '_' + typeparam + '_region.xlsx')}''')   
    df_sens.to_csv(f'''./results/sense/{(pairname + '_' + typeparam + '.csv')}''')
    df_sens.to_excel(f'''./results/sense/{(pairname + '_' + typeparam + '.xlsx')}''')      
    df_metrics.to_excel(f'''./results/acc/{(pairname + '_' + typeparam + '.xlsx')}''')    
    importance.to_excel(f'''./results/importance/{(pairname + '_' + typeparam + '.xlsx')}''')
    model_final.booster_.save_model(f'''./results/model/{(pairname + '_' + typeparam + '.txt')}''')
    pickle.dump(scaler, open(f'''./results/scaler/{(pairname + '_' + typeparam + '.sav')}''','wb'))
    pickle.dump(pca_model_final, open(f'''./results/pca/{(pairname + '_' + typeparam + '.pkl')}''','wb'))
    df_full.to_csv(f'''./results/data/{(pairname + '_' + typeparam + '.csv')}''')
    json.dump(dict_model, open(f'''./results/model/{(pairname + '_' + typeparam + '_dict.txt')}''','w')) 
   
    # base.to_csv(f'''./results/base/{(pairname + '_base.csv')}''')
    #json.dump(bestparams, open(f'''./results/model/{(pairname + '_dict.txt')}''','w')) 

@st.cache
def LoadDataAppNoDDD(pairname, appdir): # подгрузка данных для приложения no ddd
    pd_acc = pd.read_excel(f'''{appdir}/results/acc/{(pairname + '_noDDD.xlsx')}''', index_col=0)
    stat_year = pd.read_excel(f'''{appdir}/results/base/{(pairname + '_year_noDDD.xlsx')}''', index_col=0)    
    stat_region = pd.read_excel(f'''{appdir}/results/base/{(pairname + '_region_noDDD.xlsx')}''', index_col=0)
    
    return pd_acc, stat_year, stat_region


@st.cache
def LoadDataApp(pairname, appdir, typeparam): # подгрузка данных для приложения по метрикам
    df_sens = pd.read_csv(f'''{appdir}/results/sense/{(pairname + '_' + typeparam + '.csv')}''', index_col=0)
    importance = pd.read_excel(f'''{appdir}/results/importance/{(pairname + '_' + typeparam + '.xlsx')}''', index_col=0)
    
    pd_acc = pd.read_excel(f'''{appdir}/results/acc/{(pairname + '_' + typeparam + '.xlsx')}''', index_col=0)    
    stat_year = pd.read_excel(f'''{appdir}/results/base/{(pairname + '_' + typeparam + '_year.xlsx')}''', index_col=0)      
    stat_region = pd.read_excel(f'''{appdir}/results/base/{(pairname + '_' + typeparam + '_region.xlsx')}''', index_col=0)

    
    df_pair_ddd = pd.read_excel(f'''{appdir}/results/base/{(pairname + '_' + typeparam +  '_ddds.xlsx')}''', index_col=0)
    forecast = pd.read_csv(f'''{appdir}/results/forecast/{(pairname + '_' + typeparam + '.csv')}''', index_col=0)
    forecasts_ddd = pd.read_csv(f'''{appdir}/results/forecast/{(pairname + '_' + typeparam + '_ddd.csv')}''')
    
    # преобразования
    forecast['forecast weight'] = forecast['N'] * forecast['forecast']  # взвешенная R по регионам
    forecast = pd.pivot_table(forecast, values=['forecast weight', 'N'],
                                   index = ['OrganismName', 'AntibioticName', 'AntibioticClass','Year', 'Nosocomial', 'typeddd'],
                                   aggfunc='sum').reset_index()
    
    forecast['forecast'] = forecast['forecast weight'] / forecast['N']
           
    return df_sens, importance, pd_acc, stat_year, stat_region, df_pair_ddd, forecast, forecasts_ddd

@st.cache
def LoadTablesApp(appdir): # подгрузка данных для приложения 
    df_forecastDDD = pd.read_excel(f'''{appdir}/results/tables/DDDforecast.xlsx''', index_col=0)  
    ab_coeff =  pd.read_excel(f'''{appdir}/results/tables/DDDcoeff.xlsx''', index_col=0)     
    df_ABR_init = pd.read_excel(f'''{appdir}/results/tables/ABRstats.xlsx''', index_col=0)
    
    dict_molecula = pd.read_excel(f'''{appdir}/manual/dict_molecula.xlsx''')
    
    return ab_coeff, df_ABR_init, dict_molecula , df_forecastDDD


@st.cache
def LoadDataForecastApp(appdir, pairname, typeparam): # прогрузка данных для формирвоания прогноза - не используется
    # base = pd.read_csv(f'''{appdir}/results/base/{(pairname + '_base.csv')}''', index_col=0).reset_index(drop=True)
    dict_data = json.loads(open(f'''{appdir}/DATA/dict_feat.txt''').read()) 
        
    model = lgb.Booster(model_file=str(f'''{appdir}/results/model/{(pairname + '_' + typeparam + '.txt')}'''))    
    scaler = pickle.load(open(f'''{appdir}/results/scaler/{(pairname + '_' + typeparam + '.sav')}''', 'rb')) 
    
    return dict_data, model, scaler


    


