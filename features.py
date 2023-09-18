import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

           
def funcMA(df_res, df_abr, ind_col, wind, min_per, lag, colName): # расчет скользящего по столбцу
    ind_col_point = ind_col + ['Year',]
    df_piv = pd.pivot_table(df_abr, values=['Resistance','N'], index=ind_col_point, aggfunc=np.sum).reset_index()
    
    # смещение на лаг по DDD, чтобы не дублировать влияние, если оно есть за предыдущие периоды
    df_piv['Resistance roll'] =  df_piv.groupby(ind_col)['Resistance'].shift(lag)
    df_piv['N roll'] =  df_piv.groupby(ind_col)['N'].shift(1)
    
    df_piv['MovAvgRes'] = df_piv.groupby(ind_col)['Resistance roll'].rolling(wind, min_per, closed='left').sum().reset_index(drop=True)
    df_piv['MovAvgNum'] = df_piv.groupby(ind_col)['N roll'].rolling(wind, min_per, closed='left').sum().reset_index(drop=True)   
    df_piv[colName] = df_piv['MovAvgRes'] / df_piv['MovAvgNum']
    cols = ind_col_point + [colName,]
    df_piv = df_piv[cols]
    
    df_res = pd.merge(df_res, df_piv, on=ind_col_point, how='left')
    
    return df_res

# средний уровень ABR за предыдущие периоды   
def FeatMovAvgABR(df_model, df_abr, dict_data, lag): # сборка финального показателя скользящего по всем столбцам
    df_res = df_model.copy()    
    df_res['N'] = 1;  df_abr['N'] = 1  

    wind = 10
    min_per = 3
    
    model_unit_org = dict_data['model_unit_org']
    model_unit_ab = dict_data['model_unit_ab']
    
    df_res = funcMA(df_res, df_abr, [model_unit_org,model_unit_ab,'RegionName','Nosocomial'], wind, min_per, lag, 'MovAvgResRegAB') # прошлое значение на уровне года-региона-АБ
    df_res = funcMA(df_res, df_abr, [model_unit_org,model_unit_ab,'RegionName','Nosocomial'], wind, min_per, lag, 'MovAvgResReg') # прошлое значение на уровне года-региона-класса
    df_res = funcMA(df_res, df_abr, [model_unit_org,model_unit_ab,'Nosocomial'], wind, min_per, lag, 'MovAvgResAllAB') # прошлое значение на уровне года-АБ  
    df_res = funcMA(df_res, df_abr, [model_unit_org,model_unit_ab,'Nosocomial'], wind, min_per, lag, 'MovAvgResAll') # прошлое значение на уровне года-класса 
    
           
    df_res['MovAvgRes'] = np.where(df_res['MovAvgResRegAB'].isna()==False, df_res['MovAvgResRegAB'], 
                                   np.where(df_res['MovAvgResAllAB'].isna()==False, df_res['MovAvgResAllAB'], 
                                         np.where(df_res['MovAvgResReg'].isna()==False, df_res['MovAvgResReg'], df_res['MovAvgResAll']))) 
    
    df_res['flag'] = np.where(df_res['MovAvgResRegAB'].isna()==False, 'MovAvgResRegAB', 
                                   np.where(df_res['MovAvgResAllAB'].isna()==False, 'MovAvgResAllAB', 
                                         np.where(df_res['MovAvgResReg'].isna()==False, 'MovAvgResReg',  
                                                  np.where(df_res['MovAvgResAll'].isna()==False, 'MovAvgResAll', 'null'))))
           
    df_res = df_res.drop(columns={'MovAvgResRegAB','MovAvgResReg','MovAvgResAllAB','MovAvgResAll','flag', 'N'}).reset_index(drop=True)
    
    return df_res


def FeatDDD(df_model, df_DDD_model, lag):
    df_res = df_model.copy()   
    dict_ddd = {}
    ddd_columns = []
    for Channel in ('HOSPITAL', 'RETAIL'):
        df = df_DDD_model.loc[(df_DDD_model['Channel'] == Channel)]            
        df = pd.pivot_table(df, values=['SumDDDs'], index=['Year','AntibioticClass','RegionName'], aggfunc=np.sum)
        
        df = df.sort_values(by=['AntibioticClass','RegionName', 'Year'])
        df['PrevYear'] = df.groupby(['AntibioticClass','RegionName'])['SumDDDs'].shift(lag)
        df = df.dropna()
        
        df_ddd = pd.pivot_table(df, values=['SumDDDs'], index=['RegionName','Year'], columns=['AntibioticClass'], aggfunc=np.sum) 
        ab_list = [t[1] if t[1] else t[0] for t in df_ddd.columns]
        df_ddd.columns = [Channel + ', ' + t[1] if t[1] else t[0] for t in df_ddd.columns]    
        df_ddd = df_ddd.dropna()
       
        df_shift = pd.pivot_table(df, values=['PrevYear'], index=['RegionName','Year'], columns=['AntibioticClass'], aggfunc=np.sum)
        df_shift.columns = [Channel + ', PrevYear, ' + t[1] if t[1] else t[0] for t in df_shift.columns]  
        
        df_res = pd.merge(df_res, df_ddd, on=['Year', 'RegionName'], how = 'inner')
        df_res = pd.merge(df_res, df_shift, on=['Year', 'RegionName'], how = 'inner')
        
        
        ddd_columns = ddd_columns + df_ddd.columns.to_list() + df_shift.columns.to_list()
        ChannelDict = {'InitDDD': ab_list, 'CurYearDDD': df_ddd.columns.to_list(), 'PrevYearDDD': df_shift.columns.to_list()}
        
        dict_ddd.update({Channel: ChannelDict})
    
    dict_ddd.update({'ddd_columns' : ddd_columns})
        
               
    return df_res, dict_ddd
    
       
def FeaturesCount(df, df_DDD_model, df_ABR_model, dict_data, lag):
    df = FeatMovAvgABR(df, df_ABR_model, dict_data, lag) # скользящий по годам
    df, dict_ddd = FeatDDD(df, df_DDD_model, lag) #  формируем данные ddd в нужном срезе
    df = df.dropna().reset_index(drop=True)
    
    return df, dict_ddd

def FeaturesCountNoDDD(df, df_ABR_model, dict_data, lag):
    df = FeatMovAvgABR(df, df_ABR_model, dict_data, lag) # скользящий по годам
    df = df.dropna().reset_index(drop=True)
    
    return df
    
  
    