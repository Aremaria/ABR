import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statistics import mean

import warnings
warnings.filterwarnings("ignore")


def PicParam(df_sens, param): # отрисовка по одному параметру (сюда нужно добавить факт точкой)
    result = df_sens.loc[df_sens['param'] == param]
    
    df = result[['step_range','simulation']]
    df.index = result['step_range']
    df = df.drop(columns={'step_range'})
    fig, ax = plt.subplots(figsize=(15,9))
    
    title = 'Resistance simulation'
    ax.plot(df, linestyle = 'solid', linewidth = 2, label='R')
    ax.plot(df_sens['actual base level'].iloc[0], marker='o', linestyle='none', color = 'red', markerfacecolor = 'none', markersize = 10, label = 'actual base level')
    ax.plot(df_sens['model base level'].iloc[0], marker='o', linestyle='none', color = 'red', markersize = 3, label = 'model base level')
    ax.legend()    
    ax.set_ylabel('Resistance')
    ax.set_xlabel('Level Consumption')
    ax.set_title(title)
    
def PicParams(df_sens): # отрисовка по нескольким параметрам сразу
    result = df_sens.pivot(values=['simulation'], index=['step_range'], columns=['param'])
    result.columns = [t[1] for t in result.columns]
    
    fig, ax = plt.subplots(figsize=(15,9))
    
    title = 'Resistance simulation'
    labels = result.columns.to_list()
    ax.plot(result, linestyle = 'solid', linewidth = 1, label=labels)        
    ax.legend()    
    ax.set_ylabel('Resistance')
    ax.set_xlabel('Level Consumption')
    ax.set_title(title)

def PicImportance(importance): # рисуем важность факторов
    df = importance.sort_values(by=['importance']).reset_index(drop=True)
    x = df['importance'].values
    labels = df['feature'].values
    
    title = 'Features importance'
    fig, ax = plt.subplots(figsize=(15,9))
    ax.barh(labels, x)
    ax.set_title(title)

def PicSens(importance, df_sens, ab, perc):
    df = importance.loc[importance['flag'].isin(('init','ABall'))]
    df = df.sort_values(by=['importance']).reset_index(drop=True)
    x = df['importance'].values
    labels = df['features'].astype(str).values
        
    result = df_sens.pivot(values=['simulation'], index=['step_range'], columns=['param'])
    result.columns = [t[1] for t in result.columns]
    
    catparam_sign = result.iloc[-1] - result.iloc[1]   # знак влияния параметра   
    catparam_abs = result.max() - result.min() 
    catparam_abs = round(catparam_abs, 3)
    
    labelsMin = list(catparam_sign[(catparam_sign<0) & (abs(catparam_abs) > perc)].index)
    labelsMax = list(catparam_sign[(catparam_sign>0) & (abs(catparam_abs) > perc)].index)
    
       
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15,9)) 
    for idx, row in enumerate(axes):
        for jdx, ax in enumerate(row):
            if (idx == 0) & (jdx==0):  
                title = 'Features importance'
                ax.barh(labels, x)
                ax.set_title(title)
                
            if (idx == 0) & (jdx==1):            
                title = 'Resistance simulation Main AB'
                ax.plot(result[ab], linestyle = 'solid', linewidth = 2, label='Main AB')
                ax.plot(df_sens['actual base level'].iloc[0], marker='o', linestyle='none', color = 'red', markerfacecolor = 'none', markersize = 10, label = 'actual base level')
                ax.plot(df_sens['model base level'].iloc[0], marker='o', linestyle='none', color = 'red', markersize = 3, label = 'model base level')
                ax.legend()    
                ax.set_ylabel('Resistance')
                ax.set_xlabel('Level Consumption')
                ax.set_title(title)
                
            if (idx == 1) & (jdx==0): 
                title = 'Resistance Simulation Good Influence (Others)'
                ax.plot(result[labelsMin], linestyle = 'solid', linewidth = 1, label=labelsMin)
                ax.legend()    
                ax.set_ylabel('Resistance')
                ax.set_xlabel('Level Consumption')
                ax.set_title(title)
                
            if (idx == 1) & (jdx==1): 
                title = 'Resistance Simulation Bad Influence (Others)'
                ax.plot(result[labelsMax], linestyle = 'solid', linewidth = 1, label=labelsMax)
                ax.legend()    
                ax.set_ylabel('Resistance')
                ax.set_xlabel('Level Consumption')
                ax.set_title(title)
                            
    st.pyplot(fig)

def PicDDD(ab, df_ddd, forecasts_ddd, typebd): # отрисовка потребления + тренд
    n = 1000
    df_ab = df_ddd.loc[(df_ddd['AntibioticClass'] == ab)]
    
    
    try:
        s = typebd + ', ' + ab
        df_opt = forecasts_ddd[['Year', s]].loc[(forecasts_ddd['typeddd'] =='opt')]
        
        last_point = df_ab['PairActual'].dropna().iloc[-1]
        row = pd.DataFrame({'Year': [min(df_opt['Year'])-1], s: [last_point]})
        df_opt =  row.append(df_opt, ignore_index = True)        
        df_opt.index = df_opt['Year']
        opt = df_opt[s]/n # соединяем с предыдущей точкой
        
    except:
        pass
  
    title = str(ab) + ', ' + str(typebd)          
    fig, ax = plt.subplots(figsize=(15,9))
    ax.plot(df_ab['Actual']/n, linestyle = 'solid', linewidth = 2, color='green', label='Actual (full)')
    ax.plot(df_ab['Forecast']/n, linestyle = '--', color='green', linewidth = 2, label='Forecast')
    
    ax.plot(df_ab['PairActual']/n, linestyle = 'dotted', color='black', linewidth = 2, label='Actual (model)')
    try:
        ax.plot(opt, linestyle = 'dotted', color='magenta', linewidth = 2, label='optimal')
    except:
        pass
       
    ax.plot(df_ab['Actual_adj']/n, linestyle = 'solid', linewidth = 2, color='blue', label='Actual, adjusted')
    ax.plot(df_ab['Forecast_adj']/n, linestyle = '--', color='blue', linewidth = 2, label='Forecast, adjusted')
    
    ax.legend() 
    
    ax.set_ylabel('DDDs, 1000')
    ax.set_xlabel('Year')
    ax.set_title(title)
    #fig.show()
    
    st.pyplot(fig)
    
def PicForecast(forecast, stat_year): # предсказание на новый период 
    actual = stat_year['Resistance'].loc[stat_year['step'] == 'full']
    model = stat_year['Forecast'].loc[stat_year['step'] == 'full']
        
    predict = pd.pivot_table(forecast.loc[forecast['typeddd']=='no adj'], values=['forecast'], index=['Year'], aggfunc='mean')
    predict = pd.concat([pd.DataFrame(model[len(model)-1:].rename('forecast')), predict]) # добавляем последнюю точку модели для непрерывного графика
    
    predict_opt = pd.pivot_table(forecast.loc[forecast['typeddd']=='opt'], values=['forecast'], index=['Year'], aggfunc='mean')
    predict_opt = pd.concat([pd.DataFrame(model[len(model)-1:].rename('forecast')), predict_opt]) 
    
    predict_adj = pd.pivot_table(forecast.loc[forecast['typeddd']=='adj' ], values=['forecast'], index=['Year'], aggfunc='mean')
    predict_adj = pd.concat([pd.DataFrame(model[len(model)-1:].rename('forecast')), predict_adj]) # добавляем последнюю точку модели для непрерывного графика

    actual_NotNos = stat_year['Resistance, NotNosocomial'].loc[stat_year['step'] == 'full']
    model_NotNos = stat_year['Forecast, NotNosocomial'].loc[stat_year['step'] == 'full']
    
    predict_NotNos = pd.pivot_table(forecast.loc[(forecast['Nosocomial'] == 0) & (forecast['typeddd'] == 'adj')], values=['forecast'], index=['Year'], aggfunc='mean')
    predict_NotNos = pd.concat([pd.DataFrame(model_NotNos[len(model_NotNos)-1:].rename('forecast')), predict_NotNos]) # добавляем последнюю точку модели для непрерывного графика
 
    predict_NotNos_opt = pd.pivot_table(forecast.loc[(forecast['Nosocomial'] == 0) & (forecast['typeddd'] == 'opt')], values=['forecast'], index=['Year'], aggfunc='mean')
    predict_NotNos_opt = pd.concat([pd.DataFrame(model_NotNos[len(model_NotNos)-1:].rename('forecast')), predict_NotNos_opt])    
 
    actual_Nos = stat_year['Resistance, Nosocomial'].loc[stat_year['step'] == 'full']
    model_Nos = stat_year['Forecast, Nosocomial'].loc[stat_year['step'] == 'full']   
    
    predict_Nos = pd.pivot_table(forecast.loc[(forecast['Nosocomial'] == 1) & (forecast['typeddd'] == 'adj')], values=['forecast'], index=['Year'], aggfunc='mean')
    predict_Nos = pd.concat([pd.DataFrame(model_Nos[len(model_Nos)-1:].rename('forecast')), predict_Nos])
    
    predict_Nos_opt = pd.pivot_table(forecast.loc[(forecast['Nosocomial'] == 1) & (forecast['typeddd'] == 'opt')], values=['forecast'], index=['Year'], aggfunc='mean')
    predict_Nos_opt = pd.concat([pd.DataFrame(model_Nos[len(model_Nos)-1:].rename('forecast')), predict_Nos_opt])
 
    title = 'Actual&Model&Forecast'           
    fig, ax = plt.subplots(figsize=(15,9))    
    ax.plot(actual, linestyle = 'solid', linewidth = 3, color = 'black', label='actual')
    ax.plot(model, linestyle = 'solid', linewidth = 3, color = 'red', label='model')
    ax.plot(predict, linestyle = '--', linewidth = 3, color = 'red', label='forecast')
    ax.plot(predict_adj, linestyle = '--', linewidth = 3, color = 'magenta', label='forecast adj')
    ax.plot(predict_opt, linestyle = '--', linewidth = 3, color = 'darkmagenta', label='forecast opt')
    
    ax.plot(actual_NotNos, linestyle = 'solid', linewidth = 1.5, color = 'blue', label='actual, NotNosocomial (вне госпитальное)')
    ax.plot(model_NotNos, linestyle = 'solid', linewidth = 1.5, color = 'red')  # ='model, NotNosocomial'
    ax.plot(predict_NotNos, linestyle = '--', linewidth = 1.5, color = 'magenta') # label='forecast adj, NotNosocomial'
    ax.plot(predict_NotNos_opt, linestyle = '--', linewidth = 1.5, color = 'darkmagenta') 
    
    ax.plot(actual_Nos, linestyle = 'solid', linewidth = 1.5, color = 'green', label='actual, Nosocomial (госпитальное)')
    ax.plot(model_Nos, linestyle = 'solid', linewidth = 1.5, color = 'red')  # 'dashdot', 'dashed', 'model, Nosocomial'
    ax.plot(predict_Nos, linestyle = '--', linewidth = 1.5, color = 'magenta') # label='forecast adj, Nosocomial'
    ax.plot(predict_Nos_opt, linestyle = '--', linewidth = 1.5, color = 'darkmagenta') 
        
    ax.legend()    
    ax.set_ylabel('Resistance')
    ax.set_xlabel('Year')
    ax.set_title(title)
    ax.grid(True)
    
    st.pyplot(fig)
    
def PicForecastType(forecast, stat_year, delta, ab_ddd): # предсказание на новый период с усреднениями с выбором класса
    actual = stat_year['Resistance'].loc[stat_year['step'] == 'full']
    model = stat_year['Forecast'].loc[stat_year['step'] == 'full']
    
    predict = pd.pivot_table(forecast.loc[forecast['typeddd'] == 'adj'], values=['forecast'], index=['Year'], aggfunc='mean')
    predict = pd.concat([pd.DataFrame(model[len(model)-1:].rename('forecast')), predict]) 
            
    forecast_filt = forecast.loc[forecast['ab_ddd'] == ab_ddd]
    forecast_filt['typeddd'] =  forecast_filt['typeddd'].astype(float)

    predict_max = pd.pivot_table(forecast_filt.loc[forecast_filt['typeddd'] == delta], values=['forecast'], index=['Year'], aggfunc='mean')
    predict_max = pd.concat([pd.DataFrame(model[len(model)-1:].rename('forecast')), predict_max]) # добавляем последнюю точку модели для непрерывного графика
    
    predict_min = pd.pivot_table(forecast_filt.loc[forecast_filt['typeddd'] == -delta], values=['forecast'], index=['Year'], aggfunc='mean')
    predict_min = pd.concat([pd.DataFrame(model[len(model)-1:].rename('forecast')), predict_min]) # добавляем последнюю точку модели для непрерывного графика
 
    title = 'Actual&Model&Forecast'           
    fig, ax = plt.subplots(figsize=(15,9))    
    ax.plot(actual, linestyle = 'solid', linewidth = 3, color = 'black', label='actual')
    ax.plot(model, linestyle = 'solid', linewidth = 3, color = 'red', label='model')
    ax.plot(predict, linestyle = 'dashed', linewidth = 3, color = 'magenta', label='forecast adj')
    ax.plot(predict_max, linestyle = 'dashed', linewidth = 3, color = 'red', label=('forecast ' + str(delta)))
    ax.plot(predict_min, linestyle = 'dashed', linewidth = 3, color = 'blue', label=('forecast ' + str(-delta)))
            
    ax.legend()    
    ax.set_ylabel('Resistance')
    ax.set_xlabel('Year')
    ax.set_title(title)
    
    st.pyplot(fig)
    
def PicForecastTypeLoc(forecast, stat_year, delta, ab_ddd, loc): # предсказание на новый период с усреднениями с выбором класса и среза (вне / внутри)
    forecast_loc = forecast.loc[(forecast['Nosocomial'] == loc)]
    
    actual = stat_year['Resistance'].loc[stat_year['step'] == 'full']
    model = stat_year['Forecast'].loc[stat_year['step'] == 'full']
    
    predict = pd.pivot_table(forecast_loc.loc[forecast_loc['typeddd'] == 'adj'], values=['forecast'], index=['Year'], aggfunc='mean')
    predict = pd.concat([pd.DataFrame(model[len(model)-1:].rename('forecast')), predict]) 
            
    forecast_filt = forecast_loc.loc[forecast_loc['ab_ddd'] == ab_ddd]
    forecast_filt['typeddd'] =  forecast_filt['typeddd'].astype(float)

    predict_max = pd.pivot_table(forecast_filt.loc[forecast_filt['typeddd'] == delta], values=['forecast'], index=['Year'], aggfunc='mean')
    predict_max = pd.concat([pd.DataFrame(model[len(model)-1:].rename('forecast')), predict_max]) # добавляем последнюю точку модели для непрерывного графика
    
    predict_min = pd.pivot_table(forecast_filt.loc[forecast_filt['typeddd'] == -delta], values=['forecast'], index=['Year'], aggfunc='mean')
    predict_min = pd.concat([pd.DataFrame(model[len(model)-1:].rename('forecast')), predict_min]) # добавляем последнюю точку модели для непрерывного графика
 
    title = 'Nosocomial' if loc == 1 else 'NotNosocomial'   
    fig, ax = plt.subplots(figsize=(15,9))    
    ax.plot(actual, linestyle = 'solid', linewidth = 3, color = 'black', label='actual')
    ax.plot(model, linestyle = 'solid', linewidth = 3, color = 'red', label='model')
    ax.plot(predict, linestyle = 'dashed', linewidth = 3, color = 'magenta', label='forecast adj')
    ax.plot(predict_max, linestyle = 'dashed', linewidth = 3, color = 'red', label=('forecast ' + str(delta)))
    ax.plot(predict_min, linestyle = 'dashed', linewidth = 3, color = 'blue', label=('forecast ' + str(-delta)))
            
    ax.legend()    
    ax.set_ylabel('Resistance')
    ax.set_xlabel('Year')
    ax.set_title(title)
    ax.set_ylim([0, 1]) 
    
    st.pyplot(fig)

def PicFitting(df_stat, xlab): # отрисовка факта-прогноза на известных данных (фитирование без экстраполяции) - переписать как на внизу
    res = pd.pivot_table(df_stat.reset_index(), values=['Forecast', 'Numbers', 'Resistance'], index=[xlab, 'SetName'], aggfunc='mean').reset_index() 
    res.index = res[xlab]
    res = res.drop(columns={xlab})
    # сюда еще добавить колебания (отклоение между фактом и прогнозом по точкам - мин и макс)

    df_fullNum = res.loc[res['SetName'] == 'Full']
    df_fullNum = df_fullNum[['Numbers']].rename(columns={'Numbers': 'NumbersFull'})    

    if xlab == 'RegionName':
        df = pd.merge(res, df_fullNum, on=['RegionName'], how='left')
        df_model = df.loc[res['SetName'] == 'Model'].sort_values(by=['NumbersFull'], ascending=False)
        df_valid = df.loc[res['SetName'] == 'Valid'].sort_values(by=['NumbersFull'], ascending=False)
        df_full = df.loc[res['SetName'] == 'Full'].sort_values(by=['NumbersFull'], ascending=False)
    else:
        df_model = res.loc[res['SetName'] == 'Model']
        df_valid = res.loc[res['SetName'] == 'Valid']
        df_full = res.loc[res['SetName'] == 'Full']
           
    title = 'Actual&Model'           
    fig, ax = plt.subplots(figsize=(15,9))
    ax1 = ax.twinx()
    line1 = ax.plot(df_model['Resistance'], linestyle = 'solid', linewidth = 2, color = 'green', label='ModelSetActual')
    line2 = ax.plot(df_model['Forecast'], linestyle = 'dotted', linewidth = 2, color = 'green', label='ModelSetForecast')
    
    line3 = ax.plot(df_valid['Resistance'], linestyle = 'solid', linewidth = 2, color = 'red', label='ValidSetActual')
    line4 = ax.plot(df_valid['Forecast'], linestyle = 'dotted', linewidth = 2, color = 'red', label='ValidSetForecast')
    
    line5 = ax.plot(df_full['Resistance'], linestyle = 'solid', linewidth = 2, color = 'black', label='FullSetActual')
    line6 = ax.plot(df_full['Forecast'], linestyle = 'dotted', linewidth = 2, color = 'black', label='FullSetForecast')
    
    line7 = ax1.plot(df_model['Numbers'], marker='o', linestyle='none', color = 'purple', markerfacecolor = 'none', markersize = 10, label='ModelSetN')
    line8 = ax1.plot(df_valid['Numbers'], marker='o', linestyle='none', color = 'blue', markerfacecolor = 'none', markersize = 10, label='ValidSetN')
    

    leg = line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8
        
    labs = [l.get_label() for l in leg]
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.legend(leg, labs, loc='upper center')
    ax.set_xlabel(xlab)
    ax.set_ylabel('Resistance')
    ax1.set_ylabel('Number of observations')
    ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 10)
    plt.tight_layout(pad=1.5)
    ax.set_title(title)
    
    st.pyplot(fig)
        

def PicGroup(df_stat, xlab): # отрисовка факта-прогноза на известных данных (фитирование без экстраполяции)
    res = pd.pivot_table(df_stat.reset_index(), values=['Forecast', 'Numbers', 'Resistance'], index=[xlab, 'SetName'], aggfunc='mean').reset_index() 
    res.index = res[xlab]
    res = res.drop(columns={xlab})
    # сюда еще добавить колебания (отклоение между фактом и прогнозом по точкам - мин и макс)

    df_full = res.loc[res['SetName'] == 'Full']
    df_full = df_full[['Numbers']].rename(columns={'Numbers': 'NumbersFull'})    

    if xlab == 'RegionName':
        df = pd.merge(res, df_full, on=['RegionName'], how='left')
        df_model = df.loc[res['SetName'] == 'Model'].sort_values(by=['NumbersFull'], ascending=False)
        df_valid = df.loc[res['SetName'] == 'Valid'].sort_values(by=['NumbersFull'], ascending=False)
    else:
        df_model = res.loc[res['SetName'] == 'Model']
        df_valid = res.loc[res['SetName'] == 'Valid']
    
    title = 'Actual&Model'           
    fig, ax = plt.subplots(figsize=(15,9))
    ax1 = ax.twinx()
    line1 = ax.plot(df_model['Resistance'], linestyle = 'solid', linewidth = 2, color = 'green', label='ModelSetActual')
    line2 = ax.plot(df_model['Forecast'], linestyle = 'dotted', linewidth = 2, color = 'green', label='ModelSetForecast')
    
    line3 = ax.plot(df_valid['Resistance'], linestyle = 'solid', linewidth = 2, color = 'red', label='ValidSetActual')
    line4 = ax.plot(df_valid['Forecast'], linestyle = 'dotted', linewidth = 2, color = 'red', label='ValidSetForecast')
    
    line5 = ax1.plot(df_model['Numbers'], marker='o', linestyle='none', color = 'black', markerfacecolor = 'none', markersize = 10, label='ModelSetN')
    line6 = ax1.plot(df_valid['Numbers'], marker='o', linestyle='none', color = 'blue', markerfacecolor = 'none', markersize = 10, label='ValidSetN')
    
    leg = line1 + line2 + line3 + line4 + line5 + line6
    labs = [l.get_label() for l in leg]
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.legend(leg, labs, loc='upper center')
    ax.set_xlabel(xlab)
    ax.set_ylabel('Resistance')
    ax1.set_ylabel('Number of observations')
    ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 10)
    plt.tight_layout(pad=1.5)
    ax.set_title(title)
    
    valid_corr = round(df_valid['Forecast'].corr(df_valid['Resistance']), 2)
    print('correl valid:', valid_corr)
    # ax.scatter(res['Resistance'], res['forecast'])
    
def PicDistrMetric(df_metrics1, df_metrics2, col):
    a1 = round(df_metrics1[[col]],2); a2 = round(df_metrics2[[col]],2)
    a1['count'] = 1; a2['count'] = 1
    piv1 = pd.pivot_table(a1, values=['count'], index=[col], aggfunc='sum').reset_index()
    piv2 = pd.pivot_table(a2, values=['count'], index=[col], aggfunc='sum').reset_index()
    w = 0.5
    
    fig, ax = plt.subplots(figsize=(8,8))
    bars1 = plt.bar(piv1[col]*100 - w/2, piv1['count'], label='1', width=w)
    bars2 = plt.bar(piv2[col]*100 + w/2, piv2['count'], label='2', width=w)
    ax.legend()    
    plt.show()
    
def PicDistrMetricApp(df_metrics1, df_metrics2, df_metrics3):
    
    def makeformat(serrie):
        return str(round(mean(serrie*100),1))
    
    def makePic(df1, df2, df3, w, perc_lim, ind):
        piv1 = pd.pivot_table(df1, values=['count'], index=[ind], aggfunc='sum').reset_index()
        piv12 = pd.pivot_table(df2, values=['count'], index=[ind], aggfunc='sum').reset_index()
        piv13 = pd.pivot_table(df3, values=['count'], index=[ind], aggfunc='sum').reset_index()
        
        avg_lim = (mean(piv1[ind]) + mean(piv12[ind]) + mean(piv13[ind])) / 3
        
        ax.bar(piv1[ind]*100 + 2*w/3, piv1['count'], label=('ddd - ' + makeformat(df1[ind])), width=w/3)
        ax.bar(piv13[ind]*100 - w/3, piv13['count'], label=('ddd, filt - ' + makeformat(df3[ind])), width=w/3)
        ax.bar(piv12[ind]*100 - 2*w/3, piv12['count'], label=('no ddd - ' + makeformat(df2[ind])), width=w/3)
       
        ax.spines[['right', 'top', 'bottom']].set_visible(False)
        ax.set_ylabel('num of observations')
        ax.set_xlabel('metric, %')
        ax.legend() 
        ax.set_xlim([(avg_lim - perc_lim)*100, (avg_lim + perc_lim)*100])
        ax.set_title(ind)
        
    df1 = round(df_metrics1,2); df2 = round(df_metrics2,2); df3 = round(df_metrics3,2)
    df1['count'] = 1; df2['count'] = 1; df3['count'] = 1
    w = 0.6
    perc_lim = 0.1
       
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15,9)) 
    for idx, row in enumerate(axes):
        for jdx, ax in enumerate(row):
            if (idx == 0) & (jdx==0):  
                makePic(df1, df2, df3, w, perc_lim, ind = 'model recall')
                
            if (idx == 0) & (jdx==1):  
                makePic(df1, df2, df3, w, perc_lim, ind = 'model precision')
                                
            if (idx == 1) & (jdx==0): 
                makePic(df1, df2, df3, w, perc_lim, ind = 'valid recall')
                
            if (idx == 1) & (jdx==1): 
                makePic(df1, df2, df3, w, perc_lim, ind = 'valid precision')
                                           
    st.pyplot(fig)
  
def PicDistrMetricApp2(df_metrics):
    df = round(df_metrics,2)
    df['count'] = 1
    
    def makePic(df, title):
            piv1 = pd.pivot_table(df, values=['count'], index=[title], aggfunc='sum').reset_index()
            ax.bar(piv1[title]*100, piv1['count'])
            
            ax.spines[['right', 'top', 'bottom']].set_visible(False)
            ax.set_ylabel('num of observations')
            ax.set_xlabel('metric, %')
            ax.set_title(title)
    
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15,9)) 
    for idx, row in enumerate(axes):
        for jdx, ax in enumerate(row):
            if (idx == 0) & (jdx==0): 
                makePic(df, 'model recall')
               
            if (idx == 0) & (jdx==1):  
                makePic(df, 'model precision')
               
            if (idx == 1) & (jdx==0): 
                makePic(df, 'valid recall')
                
            if (idx == 1) & (jdx==1): 
                makePic(df, 'valid precision')
                            
    st.pyplot(fig)
    
def PicABRInit(df_ABR_init, ab, org): # предсказание на новый период
    df_Nos = df_ABR_init.loc[(df_ABR_init['AntibioticName'] == ab) & (df_ABR_init['OrganismName'] == org) & (df_ABR_init['Nosocomial'] == 1)]
    df_NotNos = df_ABR_init.loc[(df_ABR_init['AntibioticName'] == ab) & (df_ABR_init['OrganismName'] == org) & (df_ABR_init['Nosocomial'] == 0)]
    
    df = pd.pivot_table(df_ABR_init.loc[(df_ABR_init['AntibioticName'] == ab) & (df_ABR_init['OrganismName'] == org)], 
                        values=['Resistance','count'], index=['Year'], aggfunc='sum')
    
    df['res'] = df['Resistance'] / df['count']
    
    title = 'Resistance since 1998'           
    fig, ax = plt.subplots(figsize=(15,9))
    ax1 = ax.twinx()
    line1 = ax.plot(df_Nos['res'], linestyle = 'solid', linewidth = 2, color = 'green', label='Nosocomial')  
    line2 = ax1.plot(df_Nos['count'], marker='o', linestyle='none', color = 'green', markerfacecolor = 'none', markersize = 10, label='Nosocomial, N')
    
    line3 = ax.plot(df_NotNos['res'], linestyle = 'solid', linewidth = 2, color = 'blue', label='Not Nosocomial') 
    line4 = ax1.plot(df_NotNos['count'], marker='o', linestyle='none', color = 'blue', markerfacecolor = 'none', markersize = 10, label='Not Nosocomial, N')
    
    line5 = ax.plot(df['res'], linestyle = 'solid', linewidth = 2, color = 'black', label='total') 
    # line6 = ax1.plot(df['count'], marker='o', linestyle='none', color = 'black', markerfacecolor = 'none', markersize = 10, label='N')
   
    leg = line1 + line2 + line3 + line4 + line5
    labs = [l.get_label() for l in leg]
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.legend(leg, labs, loc='upper center')
    ax.set_xlabel('Year')
    ax.set_ylabel('Resistance')
    ax1.set_ylabel('Number of observations')
    ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 10)
    plt.tight_layout(pad=1.5)
    ax.set_title(title)
    
    st.pyplot(fig)
 
