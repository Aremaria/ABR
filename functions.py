import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# вспомогательные функции

def data_scaler(X: pd.DataFrame):     
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(X)
    
    return scaler

def ScalerTransform(X, scaler):
    X_scaled = pd.DataFrame(scaler.transform(X))
    X_scaled.columns = X.columns
    
    return X_scaled

def PCA_method(df, n_reduce): 
    pca = PCA(n_reduce)
    
    pca_model = pca.fit(df)   
    explained_variance = pca_model.explained_variance_ratio_
    # factor_weight = pca_model.components_ # факторная нагрузка
    print(round(explained_variance.sum()*100,2), '% info saved')
    
    return pca_model


def ChangeDfPCA(X, pca_model, pca_col): # заменяем часть df на PCA
    df_pca = pca_model.transform(X[pca_col]) 
    X_changed = pd.concat([X.copy().drop(columns=set(pca_col)), pd.DataFrame(df_pca)], axis=1) 
    
    return X_changed

def MakeTrain(X_train, pca_col): # преобразование для сета параметров (трейн)
    scaler = data_scaler(X_train)
    X_train_scaled = ScalerTransform(X_train, scaler)
    pca_model = PCA_method(X_train_scaled[pca_col], n_reduce=5)
       
    X_train_pca_scaled = ChangeDfPCA(X_train_scaled, pca_model, pca_col) # преобразуем часть с ddd через PCA
    
    return X_train_pca_scaled, scaler, pca_model


def MakeTableDDD(data_series, dict_ddd): # формирование таблицы для общего потребления (без мэтча по регоионам)
    vals = dict_ddd['HOSPITAL']['CurYearDDD'] + dict_ddd['RETAIL']['CurYearDDD'] 
    df_ddd = pd.pivot_table(data_series, values=vals, index=['RegionName','Year'], aggfunc='mean')
    df_ddd = df_ddd.stack().reset_index().rename(columns={'level_2':'AB', 0: 'DDDs'})
    df_ddd[['TypeBD', 'AntibioticClass']] = df_ddd['AB'].str.split(', ', 1, expand=True)
    df_ddd = pd.pivot_table(df_ddd, values=['DDDs'], index=['Year','AntibioticClass','TypeBD'], aggfunc='sum').reset_index()
    
    return df_ddd

def MakeData(df_ABR_model, dict_data, AB, Org):
    model_unit_org = dict_data['model_unit_org'] # по какому срезу формируем ряд
    model_unit_ab = dict_data['model_unit_ab'] # по какому срезу формируем ряд
    
    df_full = df_ABR_model.loc[(df_ABR_model[model_unit_ab] == AB) & (df_ABR_model[model_unit_org] == Org)].reset_index(drop=True) 
    
    # отсекаем самые редкие регионы
    region_list = (df_full.groupby('RegionName')['Year'].count() / df_full['Year'].count()).reset_index()
    region_list = region_list.sort_values(by=['Year'], ascending=False).reset_index(drop=True)
    region_list['cumsum'] = region_list['Year'].cumsum()
    region_list = np.unique(region_list['RegionName'].loc[(region_list['cumsum'] <= 0.975)])
    df_full = df_full.loc[df_full['RegionName'].isin(region_list) == True].reset_index(drop=True)
                                
    return df_full


def MakeBaseForecast(df_full, dict_data, model_columns): # создание прогнозоной базы
    base = df_full.loc[df_full['Year'] == max(df_full['Year'])].reset_index(drop=True) 
    base = base[list(dict_data['Index']) + model_columns]
    
    return base
           
def MakeStats(X, df, cols, model, SetName):    
    y_model = model.predict_proba(X)[:,1]    
    stat = df[cols].reset_index(drop=True)
    stat = pd.concat([stat, pd.DataFrame(y_model).rename(columns={0:'Forecast'})], axis=1)
    stat = stat.rename(columns = {'OrganismName': 'Numbers'})
    stat['Nosocomial'] = np.where(stat['Nosocomial'] == 1, 'Nosocomial', 'NotNosocomial')
    
    # добавляем разрез по вне / внутри госпитальным
    stat_year = pd.pivot_table(stat, values=['Resistance','Forecast', 'Numbers'], index=['Year'], 
                          aggfunc={'Resistance': 'mean', 'Forecast': 'mean', 'Numbers': 'count'}).reset_index()
    
    stat_year_type = pd.pivot_table(stat, values=['Resistance','Forecast', 'Numbers'], index=['Year'], columns=['Nosocomial'],
                          aggfunc={'Resistance': 'mean', 'Forecast': 'mean', 'Numbers': 'count'}).reset_index()
    stat_year_type.columns = [t[0] + ', ' + t[1] if t[1] else t[0] for t in stat_year_type.columns]
    
    stat_year = pd.merge(stat_year, stat_year_type, on=['Year'], how='left')
    stat_year.index = stat_year['Year']
    stat_year = stat_year.drop(columns={'Year'})
    
    stat_year.insert(0, 'SetName', SetName)

    stat_region = pd.pivot_table(stat, values=['Resistance','Forecast', 'Numbers'], index=['RegionName'], 
                              aggfunc={'Resistance': 'mean', 'Forecast': 'mean', 'Numbers': 'count'})
    
    stat_region.insert(0, 'SetName', SetName)
    
    return stat_year, stat_region

def MakeStatCuts(df, X, SetName, dict_data, model_columns, model, step):  # для проверки фитирования по годам и регионам
    cols = list(dict_data['Index']) + [dict_data['Target'],] + dict_data['DataTime'] + ['Nosocomial',]
    stat_year, stat_region = MakeStats(X, df, cols, model, SetName)
    stat_year.insert(0, 'step', step)
    stat_region.insert(0, 'step', step)
       
    # stat_year_full, stat_region_full = MakeStats(df_dict['X_full_pca_scaled'], df_dict['df_full'], cols, model_final, SetName='Full')
    return stat_year, stat_region

@st.cache
def MakeListPair(dict_list):
    pair_list = []
    for org in list(dict_list.keys()):
        for ab in dict_list[org]:
            temp = str(org) + '_' + str(ab)
            pair_list.append(temp)
            
    return pair_list

@st.cache
def MakeDDD(df_all, df_pair):    
    list_ab_type = pd.pivot_table(df_pair, values=['DDDs'], index=['AntibioticClass', 'TypeBD'], aggfunc='sum').reset_index() # добавляем два года с известными данными по ddd
    df_ddd = pd.merge(df_all, list_ab_type, on=['AntibioticClass', 'TypeBD'], how='inner').drop(columns={'DDDs'})
    
    df_ddd = pd.merge(df_ddd, df_pair, on=['Year', 'AntibioticClass', 'TypeBD'], how='left').rename(columns={'DDDs': 'PairActual'})
    df_ddd.index = df_ddd['Year']
    df_ddd = df_ddd[['AntibioticClass', 'TypeBD', 'PairActual', 'Actual', 'Forecast','Forecast_adj', 'Actual_adj']]
    
    df_ddd_hospital = df_ddd.loc[(df_ddd['TypeBD'] == 'HOSPITAL')]
    df_ddd_retail = df_ddd.loc[(df_ddd['TypeBD'] == 'RETAIL')]
    
    ab_list = list(np.unique(df_ddd['AntibioticClass']))
    
    return df_ddd_hospital, df_ddd_retail, ab_list

def CountCorr(stat):
    df_model = stat.loc[stat['SetName'] == 'Model']
    df_valid= stat.loc[stat['SetName'] == 'Valid']
    
    model_corr = round(df_model['Forecast'].corr(df_model['Resistance']), 2)
    valid_corr = round(df_valid['Forecast'].corr(df_valid['Resistance']), 2)
    
    return  model_corr, valid_corr

def DataFilter(df_DDD_model, ab_list_filt): # фильтруем нужные ab            
    df_DDD_filt = df_DDD_model.loc[df_DDD_model['AntibioticClass'].isin(ab_list_filt)].reset_index(drop=True)
    
        
    return df_DDD_filt
        
        

    