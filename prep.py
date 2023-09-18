import numpy as np
import pandas as pd
import os
import json

import warnings
warnings.filterwarnings("ignore")


def xlsx_to_csv(): # перевод из экселя в csv
    typebd = 'Retail' # Retail, Hospital
    content = os.listdir(f'''./DATA/DDD/{typebd}''') 
    
    for filename in content:
        data = pd.read_excel(f'''./DATA/DDD/{typebd}/{filename}''')
        year_list = np.unique(data['Year'])
        month_list = np.unique(data['Month'])
            
        print(filename)
        print(year_list)
        print(month_list)
        
        dataname = typebd + '_' + filename.split('.')[0] + '.csv'
        data.to_csv(f'''./DATA/DDD/FullData/{dataname}''')
                  
def load_dict():
    dict_manual = {}
    dict_calendare = pd.read_excel('./manual/dict_calendare.xlsx')
    dict_molecula = pd.read_excel('./manual/dict_molecula.xlsx')
    dict_region = pd.read_excel('./manual/dict_region.xlsx')
    
    dict_manual.update({'dict_calendare': dict_calendare, 'dict_molecula': dict_molecula, 'dict_region': dict_region})
    
    return dict_manual

def MakeReportDDD(): # обработка исходной статистики DDD
    content = os.listdir('./DATA/DDD/FullData') 
    report = pd.DataFrame()
    
    for filename in content:
        data = pd.read_csv(f'''./DATA/DDD/FullData/{filename}''')
        
        cols = ['Molecule','Region', 'Year','Month','Channel', 'New Form Classification Lev 1']    
        add_cols = ['Segment',]
        
        data['SumDDDs'] = data['Sum Units'] * data['DDDs']
                        
        try:
            a = data[add_cols]
        except KeyError:
            for col in add_cols:
                data[col] = 'Retail'
           
        piv = pd.pivot_table(data, values=['SumDDDs','Sum Units'], index=(cols+add_cols), aggfunc=np.sum).reset_index()
        report = pd.concat([report, piv])
        
    report = report.reset_index(drop=True)
    report.to_csv('./DATA/DDD/fullDDD.csv')
        
def MakeFinalDDD(min_year, lag): # подготовка данных по  DDD
    report = pd.read_csv('./DATA/fullDDD.csv', index_col=0)
    report = report.loc[report['SumDDDs'] != 0]
    
    list_form_parenteral = ('F - PARENTERAL ORDINARY','G - PARENTERAL RETARD','R - LUNG ADMINISTRATION')
    list_form_oral = ('A - ORAL SOLID ORDINARY','B - ORAL SOLID RETARD','D - ORAL LIQUID ORDINARY','E - ORAL LIQUID RETARD')
    
    report = report.loc[report['SumDDDs'] != 0]
    report = report.loc[report['New Form Classification Lev 1'].isin(list_form_parenteral+list_form_oral) == True]
    
    report['Form'] = np.where(report['New Form Classification Lev 1'].isin(list_form_parenteral) == True, 'parenteral', 'oral')
    report['Month2'] = report['Month'].str[8:]
    report['Channel'] = np.where(report['Channel'] == 'РОЗНИЧНЫЙ КОММЕРЧЕСКИЙ РЫНОК', 'RETAIL', report['Channel'] )
    
    dict_manual = load_dict()
    
    report = pd.merge(report, dict_manual['dict_calendare'], on=['Month2'], how='left')
    report = pd.merge(report, dict_manual['dict_molecula'], on=['Molecule'], how='left')
    report = pd.merge(report, dict_manual['dict_region'], on=['Region'], how='left')
    
    ind_col = ['Year', 'MonthName', 'Channel', 'Segment', 'AntibioticName', 'AntibioticClass', 'Form', 'RegionName']
    df_DDD = pd.pivot_table(report, values=['Sum Units', 'SumDDDs'], index = ind_col, aggfunc=np.sum).reset_index()
    
    
    dict_month = {'Январь':1, 'Февраль':2,'Март':3, 'Апрель':4, 'Май':5, 'Июнь':6, 
                  'Июль':7, 'Август':8, 'Сентябрь':9, 'Октябрь':10, 'Ноябрь':11, 'Декабрь':12}
    
    df_DDD['Month'] = df_DDD['MonthName'] 
    df_DDD = df_DDD.replace({'Month': dict_month})  
    df_DDD['Channel'] = np.where(df_DDD['Channel'] == 'PUBLIC (EXCL. DLO AND RLO)', 'HOSPITAL', 'RETAIL') 
        
    # срезовая статистика
    piv_DDD = pd.pivot_table(df_DDD, values=['SumDDDs'], index=['Year','AntibioticClass'], 
                             columns = ['Channel'], aggfunc=np.sum).reset_index()
    piv_DDD.columns = [t[1] if t[1] else t[0] for t in piv_DDD.columns]   
    piv_DDD = piv_DDD.loc[piv_DDD['Year'] >= (min_year-lag)].reset_index(drop=True)
    piv_DDD.to_excel('./results/tables/DDDstats.xlsx')
    
    # убираем редко используемые AB в разрезе баз данных
    df_res = pd.DataFrame()
    for channel in ('HOSPITAL', 'RETAIL'):
        df = df_DDD.loc[(df_DDD['Channel'] == channel)]
               
        # отсев редко используемых АБ
        df_cut = df.loc[(df['Year'] >= (min_year-lag))] 
        AB_list = pd.pivot_table(df_cut, values=['SumDDDs'], index=['AntibioticClass'], aggfunc=np.sum).reset_index()
        AB_list['share'] = AB_list['SumDDDs']/AB_list['SumDDDs'].sum()
        AB_list = AB_list.sort_values(by=['share'], ascending=False).reset_index(drop=True)
        AB_list['cumsum'] = AB_list['share'].cumsum()
        AB_list = list(AB_list['AntibioticClass'].loc[AB_list['cumsum'] <= 0.95])
                
        df = df.loc[df['AntibioticClass'].isin(AB_list)]
        df_res = pd.concat([df_res, df])
               
    df_res = df_res.reset_index(drop=True)    
    df_res.to_csv('./DATA/DDDs_model.csv') # подготовленные модельные данные
    
    
def MakeReportABR(): # обработка исходной статистики ABR
    dict_manual = load_dict()
    df_ABR = pd.read_excel('./DATA/fullABR2.xlsx').rename(columns={'RoOrgTypeName':'OrganismName'})
    
    df_ABR = pd.merge(df_ABR, dict_manual['dict_molecula'], on=['Molecule'], how='left')
    df_ABR = pd.merge(df_ABR, dict_manual['dict_region'], on=['Region'], how='left')
    df_ABR['Resistance'] = np.where(df_ABR['SIR'] == 'R', 1, 0)
    
    df_ABR = df_ABR.loc[df_ABR['DiagType'].isin(('Внебольничные', 'Нозокомиальные')) == True]
    df_ABR['Nosocomial'] = np.where( df_ABR['DiagType'] == 'Нозокомиальные', 1, 0)
        
    df_ABR = df_ABR[['Year','AntibioticName', 'AntibioticClass', 'RegionName','OrganismName','Nosocomial','Resistance']].reset_index(drop=True) 
    df_ABR['count'] = 1       
    df_ABR.to_csv('./DATA/ABR2.csv') 
    
    # сохранение срезовой статистики
    piv_ABR = pd.pivot_table(df_ABR, values=['Resistance','count'], index=['Year','AntibioticName','AntibioticClass','OrganismName','Nosocomial'], aggfunc='sum').reset_index()
    piv_ABR['res'] = piv_ABR['Resistance'] / piv_ABR['count']
    piv_ABR.index = piv_ABR['Year']
    piv_ABR = piv_ABR.drop(columns={'Year'})
    piv_ABR.to_excel('./results/tables/ABRstats.xlsx') 
   
def MakeFinalABR(model_unit_org, model_unit_ab): # подготовка данных  по ABR
    df_ABR = pd.read_csv('./DATA/ABR2.csv') 
    
    # определяем самые частые микроорганизмы - фильтруем
    organism_list = (df_ABR.groupby(model_unit_org)['Year'].count() / df_ABR['Year'].count()).reset_index()
    organism_list = organism_list.sort_values(by=['Year'], ascending=False).reset_index(drop=True)
    organism_list['cumsum'] = organism_list['Year'].cumsum()
    organism_list = np.unique(organism_list[model_unit_org].loc[(organism_list['cumsum'] <= 0.9)])
    df_ABR = df_ABR.loc[df_ABR[model_unit_org].isin(organism_list) == True]
            
    df_ABR  = df_ABR[['OrganismName', 'Nosocomial', 'AntibioticName', 'AntibioticClass', 'RegionName', 'Year', 'Resistance']] 
    dict_feat = {'Index': ['OrganismName', 'AntibioticName', 'AntibioticClass', 'RegionName'],
                 'DataTime': ['Year',],
                 'Target': 'Resistance',
                 'DataOthers': ['Nosocomial',]}  
    
    
    dict_feat.update({'model_unit_org': model_unit_org, 'model_unit_ab': model_unit_ab})
    
    df_ABR.to_csv('./DATA/ABR_model.csv')  # сохраняем подготовленные данные для модели
    json.dump(dict_feat, open('./DATA/dict_feat_init.txt','w'))
    
def MakeDataModelInit(df_ABR_model, df_DDD_model): # формируем полные модельные данные (без отсечек)
    df_ABR_model = pd.read_csv('./DATA/ABR_model.csv', index_col=0)
    df_DDD_model = pd.read_csv('./DATA/DDDs_model.csv', index_col=0) 
    
    # соединяем ABR и DDD
    df_DDD_model_piv = pd.pivot_table(df_DDD_model, values=['SumDDDs'], 
                                      index=['RegionName','Year','AntibioticClass'], 
                                      columns=['Channel'], aggfunc=np.sum)
    
    df_DDD_model_piv.columns = df_DDD_model_piv.columns.get_level_values(1)
    
    df_model = (
        pd.merge(df_ABR_model, df_DDD_model_piv, on=['Year', 'RegionName','AntibioticClass'], how = 'left')
        .reset_index(drop=True)
        )
    
    df_model.to_csv('./DATA/DATA_model.csv')
    

    
    