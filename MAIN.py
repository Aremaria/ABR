import pandas as pd

# from prep import MakeReportDDD, MakeReportABR
# from forecast import DDDForecast

from LoadData import LoadDataModel, LoadDataForecast
from prep import MakeFinalDDD, MakeFinalABR
from functions import DataFilter
from model import PairResult
from model_no_ddd import PairResultnoDDD
from forecast import PairForecast

import warnings

warnings.filterwarnings("ignore")

dict_list = {'Escherichia coli': ['CEFOTAXIME', 'CEFEPIME'], }

# по каким срезам формируем ряд
model_unit_org = 'OrganismName'
model_unit_ab = 'AntibioticName'

min_year = 2013  # минимальный год рассмотрения данных (в модель идет с 2012, так как есть лаг)
lag = 1  # максимальный лаг

# урезанный лист препаратов для другой версии расчета
ab_list_filt = ('Aminoglycosides', 'Carbapenems', 'Cephalosporin 3', 'Cephalosporins  inhibitor',
                'Fluoroquinolones', 'Nitrofuran derivatives', 'Nitrofuran derivatives',
                'Other antibacterials', 'Penicillins inhibitor', 'Tetracyclines')


def DataPrep():  # блок с обработкой исходных данных, выбор полей, подтаскивание справочников
    # MakeReportDDD() # если менялся исходник DDD
    # MakeReportABR() # если менялся исходник ABR
    MakeFinalDDD(min_year, lag)  # подготовка DDD для модели
    MakeFinalABR(model_unit_org, model_unit_ab)  # подготовка ABR для модели


def Model():  # блок с моделированием
    StepNum = 100  # сколько раз обсчитываем калибровку для взвешенного расчета гиперпараметров и интревальной оценки
    scor = 'balanced_accuracy'  # оптимизируемая метрика, прочие не ок ('recall', 'precision', 'accuracy')

    df_ABR_model, df_DDD_model, dict_data = LoadDataModel()  # общие данные для всех пар
    df_DDD_filt = DataFilter(df_DDD_model, ab_list_filt)  # данные с фильтрацией ab

    for org in list(dict_list.keys()):
        for ab in dict_list[org]:
            PairResult(df_ABR_model, df_DDD_model, dict_data, min_year, lag, ab, org, scor, StepNum, typesave='all')
            PairResult(df_ABR_model, df_DDD_filt, dict_data, min_year, lag, ab, org, scor, StepNum, typesave='filt')
            PairResultnoDDD(df_ABR_model, dict_data, min_year, lag, ab, org, scor, StepNum)  # без ddd  


def MakeForecast():  # блок с прогнозированием
    horizont = 30
    # DDDForecast()  # создаем линейный прогноз DDD (если данные обновлялись)

    df_ABR_model, _, _ = LoadDataModel()
    ab_coeff = pd.read_excel('./results/tables/DDDcoeff.xlsx', index_col=0)

    for org in list(dict_list.keys()):
        for ab in dict_list[org]:
            print(org, ab)
            pairname = str(org) + '_' + str(ab)

            for typeparam in ('all', 'filt'):  # типы прогнозов - без/с фильтрами
                print('------------------------ ', typeparam)
                data, dict_model = LoadDataForecast(pairname, typeparam)

                forecasts = pd.DataFrame();
                forecasts_ddd = pd.DataFrame()

                typeddd_list = ['no adj', 'adj', 'opt']  # типы прогноза DDD (линейный с/без выбросов, оптимальный)
                for typeddd in typeddd_list:
                    print('------', typeddd)
                    forecast, result_ddd = PairForecast(df_ABR_model, data, dict_model, ab, org, ab_coeff, horizont,
                                                        typeddd, typeparam)
                    forecasts = pd.concat([forecasts, forecast])
                    forecasts_ddd = pd.concat([forecasts_ddd, result_ddd])

                forecasts.to_csv(f'''./results/forecast/{(pairname + '_' + typeparam + '.csv')}''')
                forecasts_ddd.to_csv(f'''./results/forecast/{(pairname + '_' + typeparam + '_ddd.csv')}''')


if __name__ == '__main__':
    # DataPrep()
    # Model()
    MakeForecast()
