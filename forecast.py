import pandas as pd
import numpy as np
import copy

from sklearn.linear_model import LinearRegression

from LoadData import LoadModelForecast
from functions import MakeBaseForecast, ChangeDfPCA, ScalerTransform
from features import FeatMovAvgABR 
from optimization import MakeOptimize

import warnings
warnings.filterwarnings("ignore")

year_outliers = (2020,2021)

def DDDformula(df_temp, bd, ab): # формула для прогноза        
    y = np.array(df_temp['Actual']/df_temp['Actual'].mean()).reshape(len(df_temp),1) # для прогноза нужен коэфф, а не само значечение - нормированный ряд
    x = np.array(df_temp['Year']).reshape(len(df_temp),1)
    model = LinearRegression().fit(x, y)
    
    a = model.coef_[0][0]
    const = model.intercept_[0]
    n = 5
    x_long = np.array(range(df_temp['Year'].iloc[0], df_temp['Year'].iloc[-1] + n)).reshape(n + len(df_temp)-1, 1) 
    y_pred = model.predict(x_long) * df_temp['Actual'].mean()
    y_pred = pd.DataFrame(y_pred).rename(columns={0:'Forecast'})
    y_pred['Forecast'] = np.where(y_pred['Forecast'] < 0, 0, y_pred['Forecast']) 
    
    row = pd.DataFrame({'TypeBD': [bd], 'AntibioticClass': [ab], 'a': [a], 'const': [const]})
    
    # формируем прогноз и сохраняем таблицу для отрисовки картинки
    df_forecast = np.concatenate([x_long, y_pred], axis=1)
    df_forecast = pd.DataFrame(df_forecast).rename(columns={0:'Year',1:'Forecast'})
    df_forecast = pd.merge(df_forecast, df_temp, on=['Year'], how='left')
    df_forecast.insert(0,'TypeBD', bd)
    df_forecast.insert(0,'AntibioticClass', ab)
                    
    return row, df_forecast

def DDDForecast(): # прогнозирование всех потреблений (нормированный ряд, так как нет фильтрации по регионам)
    # прогнозируем коэфф, на сколько вырастит по отношению к предыдущему году
    DDDstat = pd.read_excel('./results/tables/DDDstats.xlsx', index_col=0)
    list_Ab = np.unique(DDDstat['AntibioticClass'])
    
    ab_coeff = ab_coeff_adj = pd.DataFrame()
    df_forecastDDD = df_forecastDDD_adj = pd.DataFrame()
    
    bd_list = ('HOSPITAL', 'RETAIL')
    for bd in bd_list:
        for ab in list_Ab:
            df_temp = DDDstat[['Year', 'AntibioticClass', bd]].loc[DDDstat['AntibioticClass'] == ab].reset_index(drop=True). \
                drop(columns={'AntibioticClass'}).rename(columns={bd:'Actual'}).fillna(0)
                                
            if len(df_temp) >= 5: # если ряд достаточно длинный
                row, df_forecast = DDDformula(df_temp, bd, ab) # прогноз по чистому ряду               
                df_forecastDDD = pd.concat([df_forecastDDD, df_forecast])
                ab_coeff = pd.concat([ab_coeff, row])
                
                # сделать таблицу с очисткой от выбросов
                df_adj = df_temp.copy()
                val_adj = (df_temp['Actual'].loc[df_temp['Year'] == 2019].values[0] + df_temp['Actual'].loc[df_temp['Year'] == 2021].values[0]) / 2
                df_adj['Actual'] = np.where(df_adj['Year'].isin(year_outliers), val_adj, df_adj['Actual'])
                
                row_adj, df_forecast_adj = DDDformula(df_adj, bd, ab) # прогноз по чистому ряду
                df_forecastDDD_adj = pd.concat([df_forecastDDD_adj, df_forecast_adj])
                ab_coeff_adj = pd.concat([ab_coeff_adj, row_adj])
                       
    ab_coeff_adj = ab_coeff_adj.rename(columns={'a': 'a_adj', 'const': 'const_adj'})
    df_forecastDDD_adj = df_forecastDDD_adj.rename(columns={'Forecast': 'Forecast_adj', 'Actual': 'Actual_adj'})
    
    df_forecastDDD = pd.merge(df_forecastDDD, df_forecastDDD_adj, on=['AntibioticClass', 'TypeBD', 'Year'], how='left')
    ab_coeff = pd.merge(ab_coeff, ab_coeff_adj, on=['AntibioticClass', 'TypeBD'], how='left')
        
    df_forecastDDD.to_excel('./results/tables/DDDforecast.xlsx') 
    ab_coeff.to_excel('./results/tables/DDDcoeff.xlsx') 
            

def PointForecast(last, df_ABR_last, dict_forec, model, scaler, pca_model, typeddd): # формирование базы для прогноза новой точки и самого прогноза
    datapoint = last.copy() 
    year = max(last['Year'])
    
    # обновлляем год, смещение на новый период
    datapoint['Year'] = year + 1 # смещение на новый период
    
    # обновляем ddd предыдущего года (смещение)
    bd_list = ('HOSPITAL', 'RETAIL')
    for bd in bd_list:
        for ab in dict_forec['DictDDD'][bd]['InitDDD']:
            col = bd + ', ' + ab
            col_prev = bd + ', PrevYear, ' + ab
        
            datapoint[col_prev] = last[col] # обновляем ddd предыдущего года
        
    # обновление ddd
    if typeddd in ('no adj', 'adj'): # если какой-то вариант прогноза по ddd (линейный, с выбросами или без) 
        ddd_mean = dict_forec['ddd_region_mean'] if typeddd == 'no adj' else dict_forec['ddd_region_mean_adj']
    
        datapoint = datapoint.drop(columns=set(ddd_mean.columns.to_list())) # обновление ddd на среднее по регионам для накрутки коэффициента
        datapoint = pd.merge(datapoint, ddd_mean, on=['RegionName'], how='left')
        
        bd_list = ('HOSPITAL', 'RETAIL')
        for bd in bd_list:
            ab_coeff_base = dict_forec['ab_coeff'].loc[dict_forec['ab_coeff']['TypeBD'] == bd]
            
            for ab in dict_forec['DictDDD'][bd]['InitDDD']:   
                col = bd + ', ' + ab
                a_name = ['a' if typeddd == 'no adj' else 'a_adj'][0]
                const_name = ['const' if typeddd == 'no adj' else 'const_adj'][0]
                
                a = ab_coeff_base[a_name].loc[ab_coeff_base['AntibioticClass'] == ab].values[0]
                const = ab_coeff_base[const_name].loc[ab_coeff_base['AntibioticClass'] == ab].values[0]
                
                coeff = max(a * (year + 1) + const, 0)
                datapoint[col] = datapoint[col] * coeff # ddd не менее нуля, прогнозирруем ddd для новой точки, домножать на среднеее по региону
                
    elif typeddd == 'opt': # подбор оптимального сета параметров ddd
        datapoint = MakeOptimize(datapoint, dict_forec, model, pca_model, scaler)
                                            
    # замена скользящего среднего
    try:
        datapoint = datapoint.drop(columns={'MovAvgRes'})
    except:
        pass
    cols = dict_forec['Index'] + dict_forec['DataTime'] + ['Nosocomial', ]
    df_abr = pd.concat([df_ABR_last, datapoint[cols]]) # временная сцепка с новым прогнозным блоком   
    datapoint = FeatMovAvgABR(datapoint, df_abr, dict_forec, lag=dict_forec['lag']) # считаем скольщяее среднее 
    
    # прогнозирование
    X = datapoint[dict_forec['model_columns']]
    X_scaled = ScalerTransform(X, scaler)    
    X_pca_scaled = ChangeDfPCA(X_scaled, pca_model, dict_forec['ddd_columns']) # преобразуем часть с ddd через PCA   
    y_pred = pd.DataFrame(model.predict(X_pca_scaled)).rename(columns={0:'forecast'})
    
    forecast = dict_forec['base'][cols]
    forecast['Year'] = year + 1
    forecast = pd.concat([forecast, y_pred], axis=1)
    forecast['N'] = 1
    
    # потребление по годам (для оптимального)
    ddd_columns_cur = [t for t in dict_forec['ddd_columns'] if t.split(', ')[1] != 'PrevYear'] # ddd текущие  

    piv_ddd = pd.pivot_table(datapoint, values=ddd_columns_cur, index=['Year', 'RegionName'], aggfunc='mean').reset_index() # усреднение по регионам, так как переход от классификатора
    piv_ddd = pd.pivot_table(piv_ddd, values=ddd_columns_cur, index=['Year'], aggfunc='sum').reset_index() # сумма по регионам
    
    return datapoint, forecast, piv_ddd

def HorizontForecast(horizont, df_ABR_pair, dict_forec, model, scaler, pca_model, typeddd, list_ab):
    result = pd.DataFrame() # для записи прогноза
    result_ddd = pd.DataFrame() # для записи расчетного ddd
    
    for point in range(1, horizont+1): 
        if point == 1:
            last = dict_forec['base'].copy()
            df_ABR_last = df_ABR_pair.copy()
            
        datapoint, forecast, piv_ddd = PointForecast(last, df_ABR_last, dict_forec, model, scaler, pca_model, typeddd)
        last = datapoint.copy()
        new_abr = forecast.copy().rename(columns={'forecast':'Resistance'})
        df_ABR_last = pd.concat([df_ABR_last, new_abr]) # добавление новой прогнозной точки для дальнейшего расчета скользящего по периоду
        
        result = pd.concat([result, forecast])
        
        piv_ddd.insert(0, 'typeddd', typeddd)
        result_ddd = pd.concat([result_ddd, piv_ddd])
                                   
    return result, result_ddd


def PairForecast(df_ABR_model, data, dict_model, ab, org, ab_coeff, horizont, typeddd, typeparam): # для пары
    pairname = str(org) + '_' + str(ab)
    model, scaler, pca_model = LoadModelForecast(pairname, typeparam) # подгрузка модели и пр
        
    df_ABR_pair = df_ABR_model.loc[(df_ABR_model[dict_model['model_unit_ab']] == ab) &
                                   (df_ABR_model[dict_model['model_unit_org']] == org)].reset_index(drop=True) 
    
    col_val = dict_model['DictDDD']['HOSPITAL']['CurYearDDD'] + dict_model['DictDDD']['RETAIL']['CurYearDDD']
    ddd_region_mean = pd.pivot_table(data, values=col_val, index=['RegionName'], aggfunc='mean')  # для прогноза ddd в разрезе регионов через единый коэффициент
    ddd_region_mean_adj = pd.pivot_table(data.loc[data['Year'].isin(year_outliers) == False], values=col_val, index=['RegionName'], aggfunc='mean') 
    #ddd_mean = pd.pivot_table(data.loc[(data['Year'] >= 2018) & (data['Year'] != 2020)], values=col_val, index=['RegionName'], aggfunc='mean') # среднее потребление по послединим нескольким годам
    
    base = MakeBaseForecast(data, dict_model, dict_model['model_columns']) # база для прогноза  
    
    dict_forec = copy.deepcopy(dict_model)  
    dict_forec.update({'ddd_region_mean': ddd_region_mean,  # 'ddd_mean': ddd_mean
                       'ddd_region_mean_adj': ddd_region_mean_adj, 
                       'ab_coeff': ab_coeff, 'base' : base})
    
    list_ab = np.unique(dict_forec['DictDDD']['HOSPITAL']['InitDDD'] +  dict_forec['DictDDD']['RETAIL']['InitDDD'])  # полный список АБ          
    ind_col = ['OrganismName', 'AntibioticName', 'AntibioticClass', 'RegionName', 'Year', 'Nosocomial']
    
    result, result_ddd = HorizontForecast(horizont, df_ABR_pair, dict_forec, model, scaler, pca_model, typeddd, list_ab) # прогноз в минимальной детализаци
    
    result_piv = pd.pivot_table(result, values=['forecast', 'N'], index=ind_col, aggfunc={'forecast': np.mean, 'N': 'sum'}).reset_index() # Надо взвешивать по регионам!!!
    result_piv['typeddd'] = typeddd
    
    return result_piv, result_ddd
