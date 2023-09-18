#streamlit run D:/SpyderProjects/OTHERS/app.py

import streamlit as st
st.set_page_config(layout="wide", page_title='resistance', page_icon='https://w7.pngwing.com/pngs/984/359/png-transparent-antimicrobial-stewardship-antimicrobial-resistance-antibiotics-health-care-others-thumbnail.png')

from functions import MakeListPair, MakeDDD, CountCorr
from pic import PicSens, PicFitting, PicDDD, PicForecast, PicDistrMetricApp, PicDistrMetricApp2
from LoadData import LoadDataApp, LoadTablesApp, LoadDataAppNoDDD


appdir = 'D:/SpyderProjects/OTHERS'
dict_list = {'Escherichia coli' : ['CEFOTAXIME','CEFEPIME'],}
set_list = ('Full', 'Model', 'Valid')

pairs_list = MakeListPair(dict_list)
ab_coeff, df_ABR_init, dict_molecula, df_forecastDDD = LoadTablesApp(appdir) #  прогноз DDD, коэффициэнты для прогноза DDD

# --------------------------
tab0, tab1, tab2, tab3, tab4 = st.tabs(['Validation', 'Fitting', 'Simulation', 'Consumptions','Forecast']) 
pairname = st.sidebar.selectbox('Pair:', pairs_list, index=0)
ab = pairname.split('_')[1]
ab_class = dict_molecula['AntibioticClass'].loc[dict_molecula['AntibioticName'] == ab].values[0]
org = pairname.split('_')[0]

if pairname:  
    # анализ чуст-ти, важности факторов, точности и рез-ты фитирования, прогноз
    df_sens_all, importance_all, pd_acc_all, stat_year_all, stat_region_all, df_pair_ddd_all, forecast_all, forecasts_ddd_all = LoadDataApp(pairname, appdir, 'all')
    df_sens_filt, importance_filt, pd_acc_filt, stat_year_filt, stat_region_filt, df_pair_ddd_filt, forecast_filt, forecasts_ddd_filt = LoadDataApp(pairname, appdir, 'filt')
    pd_acc_noDDD, stat_year_noDDD, stat_region_noDDD = LoadDataAppNoDDD(pairname, appdir)
    
    df_ddd_hospital_all, df_ddd_retail_all, ab_list = MakeDDD(df_forecastDDD, df_pair_ddd_all)
    df_ddd_hospital_filt, df_ddd_retail_filt, _ = MakeDDD(df_forecastDDD, df_pair_ddd_filt)
   
        
# -------------------   Validation 
    with tab0:        
        try:
            PicDistrMetricApp(pd_acc_all, pd_acc_noDDD, pd_acc_filt)
        except:
            PicDistrMetricApp2(pd_acc_all)
        
# -------------------    Fitting
    with tab1:
        st.text('all factors ---------------------------------------------------------------------------------------------------------------------------------------------------------')
        col1, col2 = st.columns([1, 1])
        with col1:
            model_corr_all, valid_corr_all = CountCorr(stat_region_all)
           
            st.text('model_corr: ' + str(model_corr_all))
            st.text('valid_corr: ' + str(valid_corr_all))
            
            PicFitting(stat_year_all, 'Year')
            
        with col2:
            model_corr2_all, valid_corr2_all = CountCorr(stat_region_all)
           
            st.text('model_corr: ' + str(model_corr2_all))
            st.text('valid_corr: ' + str(valid_corr2_all))
            
            PicFitting(stat_region_all, 'RegionName')
            
        st.text('Filt ---------------------------------------------------------------------------------------------------------------------------------------------------------')
        col3, col4 = st.columns([1, 1])
        
        with col3:
            model_corr_filt, valid_corr_filt = CountCorr(stat_region_filt)
           
            st.text('model_corr: ' + str(model_corr_filt))
            st.text('valid_corr: ' + str(valid_corr_filt))
            
            PicFitting(stat_year_filt, 'Year')
            
        with col4:
            model_corr2_filt, valid_corr2_filt = CountCorr(stat_region_filt)
           
            st.text('model_corr: ' + str(model_corr2_filt))
            st.text('valid_corr: ' + str(valid_corr2_filt))
            
            PicFitting(stat_region_filt, 'RegionName')
            
        col5, col6 = st.columns([1, 5])
        
            
# -------------------   simulation    
    with tab2:
        perc = st.sidebar.number_input(label='Impact,%:', value=0.02, on_change=None)
        st.text('all factors ---------------------------------------------------------------------------------------------------------------------------------------------------------')
        PicSens(importance_all, df_sens_all, ab_class, perc)
        st.text('filt ---------------------------------------------------------------------------------------------------------------------------------------------------------')
        PicSens(importance_filt, df_sens_filt, ab_class, perc)

# -------------------   consumption   
# сюда добавить оптимальные ddd         
    with tab3:
        col1, col2 = st.columns([1, 5])
        with col1:
            abDDD = st.selectbox('Antibiotic Class:', ab_list, index=0)
        
        st.text('all factors')
        col1, col2 = st.columns([1, 1])
        with col1:
            PicDDD(abDDD, df_ddd_hospital_all, forecasts_ddd_all, 'HOSPITAL')
        with col2:
            PicDDD(abDDD, df_ddd_retail_all, forecasts_ddd_all, 'RETAIL')
            
        st.text('filt')
        col3, col4 = st.columns([1, 1])
        with col3:
            PicDDD(abDDD, df_ddd_hospital_filt, forecasts_ddd_filt, 'HOSPITAL')
        with col4:
            PicDDD(abDDD, df_ddd_retail_filt, forecasts_ddd_filt, 'RETAIL')


# -------------------  forecast              
    with tab4: 
        col1, col2 = st.columns([1, 4])
        with col1:    
            st.text('all factors ---------')
            
        with col2:  
            # написать сюда максимальную разницу - скорость падения в год
            PicForecast(forecast_all, stat_year_all)
            
        col3, col4 = st.columns([1, 4])
        with col3:    
            st.text('filt -----------------')
            
        with col4:   
            PicForecast(forecast_filt, stat_year_filt)
            
        

         



            
            

   

