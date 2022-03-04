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

def preprocessing_folder(folder:str, code:str):
    """
    폴더이름과 해당 종목이름을 입력받아 해당 폴더의 모든 엑셀파일을 읽어와서
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
        temp['code'] = code
        temp.rename(columns={'제목': 'headline', '언론사': 'press', '일자': 'date'}, inplace=True)

        blank = pd.concat([blank, temp])
        print(f"{file} 전처리 완료")

    blank.reset_index(drop=True, inplace=True)

    save_name = folder + ".csv"
    # 파일로 저장할거라면 주석해제
    # blank.to_csv(save_name, encoding='utf-8') 

    return blank

def regex(df_news_data):
    """
    뉴스 데이터 헤드라인의 [문자열] 등 형태와 같이 
    분석과 상관없는 특수문자 및 문자열, 즉 정규표현식을 찾아 제거하는 전처리 함수
    """

    import re
    import pandas as pd

    print('정규표현식 전처리 시작')

    for i in range(len(df_news_data)):
        if i % 100 == 0:
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
        if index % 100 == 0:
            now = datetime.datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(f"{current_time} {index}번째 데이터 처리 완료")
        index += 1

    df['senti'] = sentiment_col
    df['senti_proba'] = proba_col
    df.dropna(axis = 0, inplace=True)
    print('전체 전처리 완료')

    return df

def toDB(df):
    import pandas as pd
    import sqlite3

    # db 경로는 로컬에 맞게 설정해야함
    conn = sqlite3.connect("C:/Python_workspace/2jo_Final_Python/DB/final.db")

    # 커서 바인딩
    c = conn.cursor()

    # 데이터 프레임 각 행을 DB에 입력
    for row in df.itertuples():
        temp = c.execute('SELECT seq from sqlite_sequence')
        seq = int(temp.fetchone()[0]) + 1
        sql1 = "insert into news_db(headline, press, senti, senti_proba) values (?,?,?,?)"
        c.execute(sql1, (row[2],row[3],row[5],row[6]))
        sql2 = "insert into news_id(id, date, code) values (?,?,?)"
        c.execute(sql2, (seq, row[1],row[4]))
    conn.commit()

    print('DB 입력 완료')

if __name__ == "__main__":
    
    folder = '뉴스데이터/'
    company = 'temp'
    data_dir = folder + company
    
    data = preprocessing_folder(data_dir, '임시')
    data = regex(data)
    data = make_sentiment(data)
    # data.to_csv('temp.csv', encoding='utf-8')
    toDB(data)

