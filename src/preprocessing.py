import pandas as pd
import numpy as np
import warnings
import datetime

warnings.filterwarnings('ignore')

from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

from hugging_classifier import HuggingClassifier, model_param_bert_pt, NEWS_OM_MODEL, logger
clf = HuggingClassifier(modelParam=model_param_bert_pt, train_mode=False)
clf.load_prediction_model(model_dir=NEWS_OM_MODEL, num_categories=3, labels=['-1','0','1'])

def preprocessing_folder(folder:str):
    """
    폴더이름을 입력받아 해당 폴더의 모든 엑셀파일을 읽어와서
    원하는 컬럼만 남긴후 종목을 컬럼을 추가한 데이터 프레임을 반환하는 함수
    """
    import pandas as pd
    import os

    file_list = os.listdir(folder)

    # 빈데이터 프레임 생성
    blank = pd.DataFrame()

    for file in file_list:
        path = folder + "/" + str(file)
        temp = pd.read_excel(path)
        temp.sort_values('일자', inplace=True)
        temp = temp[temp['통합 분류1'].str.startswith('경제')]
        temp = temp[['일자', '제목','언론사']]
        temp['code'] = folder
        temp.rename(columns={'제목': 'headline', '언론사': 'press', '일자': 'date'}, inplace=True)

        data = pd.concat([blank, temp])
        print(f"{file} 전처리 완료")

    save_name = folder + ".csv"
    # 파일로 저장할거라면 주석해제
    # data.to_csv(save_name, encoding='utf-8') 

    blank.reset_index(drop=True, inplace=True)
    
    return blank

def preprocessing_file(file:str, company:str):
    """
    빅카인즈 뉴스 엑셀 파일과 해당 뉴스의 종목 이름을 입력받아 
    원하는 컬럼들만 남기는 함수
    """
    import pandas as pd

    print('파일 전처리 시작')

    file_name = file + '.xlsx'

    temp = pd.read_excel(file_name)
    temp.sort_values('일자', inplace=True)
    # temp['id'] = range(0,len(temp))

    # id = temp[['id', '일자']]
    temp = temp[['일자', '제목','언론사']]
    temp['code'] = company

    temp.rename(columns={'제목': 'headline', '언론사': 'press', '일자': 'date'}, inplace=True)

    print(f"{file} 전처리 완료")

    return temp

def regex(df_news_data):
    """
    뉴스 데이터 헤드라인의 [문자열] 등 형태와 같이 
    분석과 상관없는 특수문자 및 문자열, 즉 정규표현식을 찾아 제거하는 전처리 함수
    """

    import re
    import pandas as pd

    print('정규표현식 전처리 시작')

    for i in range(len(df_news_data)):
        if i % 1000 == 0:
            now = datetime.datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(f'{current_time} {i}번 정규표현식 전처리 완료')
        df_news_data['headline'][i] = re.sub(r'\[[^)]*\]', '', df_news_data['headline'][i])
        df_news_data['headline'][i] = re.sub(r'\([^)]*\)', '', df_news_data['headline'][i])
        df_news_data['headline'][i] = re.sub(r'\<[^)]*\>', '', df_news_data['headline'][i])
    
    print('정규표현식 전처리 종료')

    return df_news_data

    # 이후 to_csv 변환
    # df_news_data.to_csv("news_data.csv")

def one_hot(df, col_name:str):
    """
    데이터 프레임과 컬럼명을 입력받아
    해당 컬럼을 원핫 인코딩 하는 함수
    """
    # 컬럼의 유니크한 값을 리스트로 만들어둠
    col_items = df[col_name].unique().tolist()

    onehot = OneHotEncoder(sparse=False)
    onehot_encoded_arr = onehot.fit_transform(df[col_name].values.reshape(-1, 1))
    onehot_encoded_label = onehot.categories_[0]
    onehot_encoded_df = pd.DataFrame(onehot_encoded_arr, columns=onehot_encoded_label)
    df.drop(col_name, axis=1, inplace=True)
    df = pd.concat([df, onehot_encoded_df], axis=1)

    return df

def make_sentiment(df):
    """
    뉴스데이터를 입력받아
    헤드라인의 긍부정 컬럼을 생성하는 함수
    """
    import pandas as pd
    
    print('헤드라인 긍부정 전처리 시작')
    title = df['headline']
    title_list = list(title)

    sentiment_col = []
    proba_col = []

    index = 0
    # 긍부정 컬럼 리스트 생성하기
    for sentence in title_list:
        ret = clf.prediction(sentence)
        sentiment_col.append(ret[0])
        proba_col.append(ret[1])
        if index % 1000 == 0:
            now = datetime.datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(f"{current_time} {index}번째 데이터 처리 완료")
        index += 1

    df['senti'] = sentiment_col
    df['senti_proba'] = proba_col
    # df_s = df.dropna(subset=['senti']).reset_index(drop=True)
    # df_sh = df_s.dropna(subset=['headline']).reset_index(drop=True)
    df.dropna(axis = 0, inplace=True)
    id_len  = len(df)

    df['id'] = range(0,id_len)

    news_df = df[['id', 'headline', 'press', 'senti', 'senti_proba']]
    id_df = df[['id', 'date', 'code']]

    news_df.to_csv('news_db_result.csv', encoding = 'utf-8', index=False)
    id_df.to_csv('id_db_result.csv', encoding = 'utf-8', index=False)
    
    print('전체 전처리 완료')
    return df

if __name__ == "__main__":
    
    # folder = '뉴스데이터/'
    # company = '네이버'
    # data_dir = folder + company
    
    # data = preprocessing_folder(data_dir)

    data = pd.read_csv('네이버.csv')

    data = regex(data)
    make_sentiment(data)
    
    # data['company'] = '삼성전자'
    # data.to_csv('삼성전자_result.csv', encoding='utf-8')
    