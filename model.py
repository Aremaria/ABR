import pandas as pd
import numpy as np
import copy

from sklearn.metrics import confusion_matrix
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split # mean_squared_error

from functions import ScalerTransform, ChangeDfPCA, MakeTrain, MakeStatCuts, MakeData, MakeTableDDD
from SensAnalysis import SensAn
from LoadData import SaveDataModel

from features import FeaturesCount

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


def ModelDef(): # определение модели    
    model = LGBMClassifier(boosting_type='gbdt', verbosity=-1, importance_type='gain')  # is_unbalance = True (плохо работает)
    
    return model

def MakeDictModel(dict_data, dict_ddd, model_columns, common_cols, lag): # формируем нужные столбцы для обучения, сохраняем словарь  
    dict_model = copy.deepcopy(dict_data)    
    dict_model.update({'model_columns': model_columns, 'ddd_columns': dict_ddd['ddd_columns'], 
                       'DictDDD': dict_ddd, 'common_cols': common_cols, 'lag': lag}) 
      
    return dict_model

def EstimationClass(model, X, y):
    y_pred = model.predict(X)    
    r = confusion_matrix(np.array(y), y_pred)
    TP = r[1][1]
    FN = r[1][0]
    FP = r[0][1]
    TN = r[0][0]
    
    acc = (TP + TN) / (TP + TN + FP + FN)    
    recall = TP / (TP + FN) # TP / (TP + FN) верно определенные из всех модельных 1-ц
    precision = TP / (TP + FP) # TP / (TP + FP) верно определенные из всех реальных 1-ц
       
    return recall, precision, acc 

def search_params(model, X_train, y_train, hyperparameters, scor):
    for key, params in hyperparameters.items():
        grid = GridSearchCV(model, params, scoring=scor, cv=5, n_jobs=-1) # сюда Kfold
        grid.fit(X_train, y_train)
        
        print(f"Best score: {grid.best_score_} with parameters: {grid.best_params_}")
        model.set_params(**grid.best_params_)
   
    return model


def calibration(df_model, model_columns, pca_col, target, hyperparameters, scor): # настройка гиперпараметров, калибровка не нужна частая - переодически запускать и следить
    print('calibration in progres...')
       
    df_train, df_test = train_test_split(df_model, test_size=0.5, shuffle=True)
    
    X_train = df_train[model_columns]; X_test = df_test[model_columns]
    y_train = df_train[target]; y_test = df_test[target]
        
    X_train_pca_scaled, scaler, pca_model = MakeTrain(X_train, pca_col)
    X_test_pca_scaled = ChangeDfPCA(ScalerTransform(X_test, scaler), pca_model, pca_col)
               
    model = ModelDef() # инициализируем модель    
    model_best = search_params(model, X_train_pca_scaled, y_train, hyperparameters, scor) # калибруем
    model_best.fit(X_train_pca_scaled, y_train)
                
    recall_train, precision_train, acc_train  = EstimationClass(model_best, X_train_pca_scaled, y_train)
    recall_test, precision_test, acc_test  = EstimationClass(model_best, X_test_pca_scaled, y_test)

    print('train recall:', round(recall_train*100,2), '%, train acc:', round(acc_train*100,2), '%')
    print('test recall:', round(recall_test*100,2), '%, test acc:', round(acc_test*100,2), '%')
            
    bestparams = model_best.booster_.params
    accs = pd.DataFrame({'cros_val recall_train': [recall_train], 'cros_val precision_train': [precision_train], 'cros_val acc_train': [acc_train],
            'cros_val recall_test': [recall_test], 'cros_val precision_test': [precision_test], 'cros_val acc_test': [acc_test]})
    
    return bestparams, accs


def MakeModel(df, model_columns, target, pca_col, bestparams, scor):
    X = df[model_columns]; y = df[target]        
    X_pca_scaled, scaler, pca_model = MakeTrain(X, pca_col) # преобразование под PCA
       
    model_lgb = ModelDef() 
    
    if bestparams != {}:
        model_lgb.set_params(**bestparams) 
        
    model_lgb.fit(X_pca_scaled, y, eval_metric = scor)    
    importance = pd.DataFrame({'features': X_pca_scaled.columns,  'importance': model_lgb.feature_importances_})
      
    # раскрываем веса признаков внутри компонент
    fact_weights = pca_model.components_
    df_importance = pd.DataFrame(abs(fact_weights)).T
    df_importance = df_importance/df_importance.sum()
    df_importance.index = pca_col
    comp_list = df_importance.columns.to_list()
   
    for col in comp_list:
        col_weih = importance['importance'].loc[importance['features'] == col].values[0]
        df_importance[col] = df_importance[col] * col_weih
        
    # переписать в статмент
    df_importance['importance'] = df_importance.sum(axis=1)
    df_importance['features'] = [t.split(', ')[-1] for t in pca_col]
    df_importance['flag'] = 'ABdetail'
    
    df_importance_piv = pd.pivot_table(df_importance, values=['importance'], index=['features'], aggfunc=np.sum).reset_index()
    df_importance_piv['flag'] = 'ABall'
    
    df_importance['features'] = df_importance.index
    df_importance = df_importance[['flag','features','importance']].reset_index(drop=True)
    df_importance_piv = df_importance_piv[['flag','features','importance']].reset_index(drop=True)
    
    importance =  importance.loc[importance['features'].isin(comp_list) == False].reset_index(drop=True)
    importance.insert(0, 'flag', 'init')    
    importance = pd.concat([importance, df_importance, df_importance_piv]).reset_index(drop=True)
       
    return scaler, model_lgb, importance, pca_model, X_pca_scaled

def PairResult(df_ABR_model, df_DDD_model, dict_data, min_year, lag, ab, org, scor, StepNum, typesave): # расчет результата по паре        
    pairname = str(org) + '_' + str(ab)
    print(pairname)
    
    common_cols = ['Year', 'Nosocomial', 'MovAvgRes']
    df_full = MakeData(df_ABR_model, dict_data, ab, org) # подготовка среза ABR для AB и Org, отсечка редких регионов 
     
    # цикл для интервальной оценки
    target = dict_data['Target']
    df_bestparams = df_metrics = stat_year = stat_region = pd.DataFrame()
    for i in range(StepNum + 1):
        print('step ', i)
                       
        # начальный расчет разбивки для ABR, каждый раз новый сет
        df_model, df_valid = train_test_split(df_full, test_size=0.5, shuffle=True)
                
        # добавляем расчетные признаки и удаляем пустые значения по каждому сету отдельно
        # скользящие средние считаем отдельно для valid (модель ничего не знает про valid)
        df_model, dict_ddd = FeaturesCount(df_model, df_DDD_model, df_model, dict_data, lag) # в качестве ABR выступает сам файл
        df_valid, _ = FeaturesCount(df_valid, df_DDD_model, df_valid, dict_data, lag)
        
        # отсекаем по году
        df_model = df_model.loc[df_model['Year'] >= min_year].reset_index()
        df_valid = df_valid.loc[df_valid['Year'] >= min_year].reset_index()
               
        # определяем порядок столбцов
        ddd_columns = dict_ddd['ddd_columns']
        model_columns = common_cols + ddd_columns
                        
        bestparams, accs = calibration(df_model, model_columns, ddd_columns, target, hyperparameters, scor) # если нужна калибровка bestparams = {} # если без калибровки
        scaler, model, importance, pca_model, X_model_pca_scaled = MakeModel(df_model, model_columns, target, ddd_columns, bestparams, scor) # обучение на df_model   
        X_valid_pca_scaled = ChangeDfPCA(ScalerTransform(df_valid[model_columns], scaler), pca_model, ddd_columns)
                    
        recall_train, precision_train, acc_train = EstimationClass(model, X_model_pca_scaled, df_model[target])
        recall_test, precision_test, acc_test = EstimationClass(model, X_valid_pca_scaled, df_valid[target])
        
        # точности по схлопнутым статистикам
        stat_year_model, stat_region_model = MakeStatCuts(df_model, X_model_pca_scaled, 'Model', dict_data, model_columns, model, i)
        stat_year_valid, stat_region_valid = MakeStatCuts(df_valid, X_valid_pca_scaled, 'Valid', dict_data, model_columns, model, i)
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
    df_full, dict_ddd = FeaturesCount(df_full, df_DDD_model, df_full, dict_data, lag)
    df_full = df_full.loc[df_full['Year'] >= min_year].reset_index()
    
    ddd_columns = dict_ddd['ddd_columns']
    model_columns = common_cols + ddd_columns
    
    scaler_final, model_final, importance, pca_model_final, X_full_pca_scaled = MakeModel(df_full, model_columns, target, ddd_columns, bestparams, scor)     
    recall, precision, acc = EstimationClass(model_final, X_full_pca_scaled, df_full[target])
    print('ful recall:', round(recall*100,2), '%, full acc:', round(acc*100,2), '%')   
            
    param_list = np.unique(dict_ddd['HOSPITAL']['InitDDD'] + dict_ddd['HOSPITAL']['InitDDD'])
    df_sens = SensAn(df_full, model_columns, target, scaler_final, model_final, pca_model_final, param_list, ddd_columns) # анализ чувствительности
    dict_model = MakeDictModel(dict_data, dict_ddd, model_columns, common_cols, lag) # формируем словарь с порядком столбцов и др переменными
        
    # сводная статистика
    stat_year_full, stat_region_full = MakeStatCuts(df_full, X_full_pca_scaled, 'Full', dict_data, model_columns, model_final, 'full') # статистика по срезам 
    stat_year = pd.concat([stat_year, stat_year_full]) # добавляем срез для проверки по финальной модели на всем сете
    stat_region = pd.concat([stat_region, stat_region_full]) # добавляем срез для проверки по финальной модели на всем сете                  
    df_ddd = MakeTableDDD(df_full, dict_ddd) # для статистики потребления (сравнение с общей)
    
    # сохранение  ----------------------------   
    df_ddd.to_excel(f'''./results/base/{(pairname + '_' + typesave + '_ddds.xlsx')}''') 
    vecs = stat_year, stat_region, df_sens, df_metrics, importance, model_final, scaler_final, pca_model_final, df_full, dict_model
    SaveDataModel(pairname, vecs, typesave)
    





