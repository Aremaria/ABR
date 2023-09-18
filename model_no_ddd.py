import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler

from functions import ScalerTransform, MakeStatCuts, MakeData
from model import ModelDef, EstimationClass, search_params
from features import FeaturesCountNoDDD

import warnings
warnings.filterwarnings("ignore")

hyperparameters = {
    'n_estimators': {'n_estimators':range(5, 100, 5)}, #20, (5, 100, 5), (5, 500, 25)
        
    'tree':   {'max_depth': range(2, 6, 1), # из-за переобучения снизили 10  (4)
              'num_leaves':range(2, 6, 1), # 6
              'min_child_samples': range(1, 10, 1)}, # 10
        
    'sample': {'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], # из-за переобучения снизили
                'subsample': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
                                                         
    'bagging': {'feature_fraction': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], # из-за переобучения снизили
                'bagging_fraction': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
    
    'learning_rate': {'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]}            
    }


def MakeTrainNoDDD(X_train): # преобразование для сета параметров (трейн)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(X_train)
    X_train_scaled = ScalerTransform(X_train, scaler)
           
    return X_train_scaled, scaler

def calibration(df_model, model_columns, target, hyperparameters, scor): # настройка гиперпараметров, калибровка не нужна частая - переодически запускать и следить
    print('calibration in progres...')
       
    df_train, df_test = train_test_split(df_model, test_size=0.5, shuffle=True) # shuffle=False
    
    X_train = df_train[model_columns]; X_test = df_test[model_columns]
    y_train = df_train[target]; y_test = df_test[target]
        
    X_train_scaled, scaler = MakeTrainNoDDD(X_train)
    X_test_scaled = ScalerTransform(X_test, scaler)
               
    model = ModelDef() # инициализируем модель    
    model_best = search_params(model, X_train_scaled, y_train, hyperparameters, scor) # калибруем
    model_best.fit(X_train_scaled, y_train)
                
    recall_train, precision_train, acc_train  = EstimationClass(model_best, X_train_scaled, y_train)
    recall_test, precision_test, acc_test  = EstimationClass(model_best, X_test_scaled, y_test)

    print('train recall:', round(recall_train*100,2), '%, train acc:', round(acc_train*100,2), '%')
    print('test recall:', round(recall_test*100,2), '%, test acc:', round(acc_test*100,2), '%')
            
    bestparams = model_best.booster_.params
    accs = pd.DataFrame({'cros_val recall_train': [recall_train], 'cros_val precision_train': [precision_train], 'cros_val acc_train': [acc_train],
            'cros_val recall_test': [recall_test], 'cros_val precision_test': [precision_test], 'cros_val acc_test': [acc_test]})
    
    return bestparams, accs


def MakeModel(df, model_columns, target, bestparams, scor):
    X = df[model_columns]; y = df[target]        
    X_scaled, scaler = MakeTrainNoDDD(X) 
       
    model_lgb = ModelDef() 
    
    if bestparams != {}:
        model_lgb.set_params(**bestparams) 
        
    model_lgb.fit(X_scaled, y, eval_metric = scor)   
          
    return scaler, model_lgb, X_scaled

def PairResultnoDDD(df_ABR_model, dict_data, min_year, lag, ab, org, scor, StepNum): # расчет результата по паре     
    pairname = str(org) + '_' + str(ab)  
    print(pairname)
    
    common_cols = ['Year', 'Nosocomial', 'MovAvgRes']
    model_columns = common_cols.copy()
    df_full = MakeData(df_ABR_model, dict_data, ab, org) # подготовка среза ABR для AB и Org, отсечка редких регионов 
     
    # цикл для интервальной оценки
    target = dict_data['Target']
    df_bestparams = df_metrics = stat_year = stat_region = pd.DataFrame()   
    for i in range(StepNum + 1):
        print('step ', i)       
        # начальный расчет разбивки для ABR, каждый раз новый сет
        df_model, df_valid = train_test_split(df_full, test_size=0.5, shuffle=True)
                
        # переопределяем: добавляем расчетные признаки и удаляем пустые значения
        df_model = FeaturesCountNoDDD(df_model, df_model, dict_data, lag) # в качестве ABR выступает сам файл
        df_valid = FeaturesCountNoDDD(df_valid, df_valid, dict_data, lag)
        
        # отсекаем по году
        df_model = df_model.loc[df_model['Year'] >= min_year].reset_index()
        df_valid = df_valid.loc[df_valid['Year'] >= min_year].reset_index()
                   
        bestparams, accs = calibration(df_model, model_columns, target, hyperparameters, scor) # если нужна калибровка bestparams = {} # если без калибровки
        scaler, model, X_model_scaled = MakeModel(df_model, model_columns, target, bestparams, scor) # обучение на df_model   
        X_valid_scaled = ScalerTransform(df_valid[model_columns], scaler)
                    
        recall_train, precision_train, acc_train = EstimationClass(model, X_model_scaled, df_model[target])
        recall_test, precision_test, acc_test = EstimationClass(model, X_valid_scaled, df_valid[target])
        
        stat_year_model, stat_region_model = MakeStatCuts(df_model, X_model_scaled, 'Model', dict_data, model_columns, model, i)
        stat_year_valid, stat_region_valid = MakeStatCuts(df_valid, X_valid_scaled, 'Valid', dict_data, model_columns, model, i)
        stat_year = pd.concat([stat_year, stat_year_model, stat_year_valid])
        stat_region = pd.concat([stat_region, stat_region_model, stat_region_valid])
        
        row_acc = pd.DataFrame({'step': [i],'model recall': [recall_train], 'model acc': [acc_train], 'model precision': [precision_train],
                                'valid recall': [recall_test], 'valid acc': [acc_test], 'valid precision': [precision_test]})
        row_acc = pd.concat([row_acc, accs], axis = 1)
        row_bestparams = pd.DataFrame.from_dict([bestparams])
        
        df_metrics = pd.concat([df_metrics, row_acc])
        df_bestparams = pd.concat([df_bestparams, row_bestparams])
        
    # обновляем гиперпараметры - усредняем по всем расчетам
    bestparams_mean = df_bestparams.mean().dropna().reset_index()
    for i in range(len(bestparams_mean)):
        col = bestparams_mean['index'].iloc[i]
        col_type = type(bestparams[col])
        col_val_mean = bestparams_mean[0].iloc[i]
        if  col_type == float: bestparams[col] = float(round(col_val_mean, 3))
        elif  col_type == int: bestparams[col] = int(round(col_val_mean, 0))
        else: pass
            
    # апдейтим df_full на параметры
    df_full = FeaturesCountNoDDD(df_full, df_full, dict_data, lag)
    df_full = df_full.loc[df_full['Year'] >= min_year].reset_index()
        
    # переобучаем на полном сете
    scaler_final, model_final, X_full_scaled = MakeModel(df_full, model_columns, target, bestparams, scor)     
    recall, precision, acc = EstimationClass(model_final, X_full_scaled, df_full[target])
    print('ful recall:', round(recall*100,2), '%, full acc:', round(acc*100,2), '%')   
              
    stat_year_full, stat_region_full = MakeStatCuts(df_full, X_full_scaled, 'Full', dict_data, model_columns, model_final, 'full') # статистика по срезам 
    stat_year = pd.concat([stat_year, stat_year_full]) # добавляем срез для проверки по финальной модели на всем сете
    stat_region = pd.concat([stat_region, stat_region_full]) # добавляем срез для проверки по финальной модели на всем сете
    
    drop_label = 'noDDD'
    
    df_metrics.to_excel(f'''./results/acc/{(pairname + '_' + drop_label + '.xlsx')}''')  
    stat_year.to_excel(f'''./results/base/{(pairname + '_year_' + drop_label + '.xlsx')}''')
    stat_region.to_excel(f'''./results/base/{(pairname + '_region_' + drop_label + '.xlsx')}''')   
                   




