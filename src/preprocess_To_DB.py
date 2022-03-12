import pandas as pd
import numpy as np
import warnings
import datetime
from konlpy.tag import Okt

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
        temp = temp[['일자', '제목']]
        temp['code'] = code
        temp.rename(columns={'제목': 'headline', '일자': 'date'}, inplace=True)
        
        okt = Okt()
        headline = temp['headline']
        keyword = []
        for line in headline:
            keyword.append(okt.nouns(line))
        temp['keyword'] = keyword

        keyword_temp_list = temp['keyword']
        keyword_result_list = []

        stopwords = ['코스피', '전자', '에스케이', '코스닥', '주식', '한국', '경제', '뉴스', '기자', '기업', '증권',
        '종목', '마감', '예컨대', '가격', '새해', '시즌', '그룹', '동영상', '올해', '개월']
        stopwords.append(code)

        for k in keyword_temp_list:
            keyword_result_list.append(' '.join(filter(lambda x : (len(x) >= 2) & (x not in stopwords), k)))

        temp['keyword'] = keyword_result_list
        
        blank = pd.concat([blank, temp])
        print(f"{file} 전처리 완료")

    blank.reset_index(drop=True, inplace=True)

    # 파일로 저장할거라면 주석해제
    # save_name = folder + ".csv"
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
    conn = sqlite3.connect("DB/2jo.db")

    # 커서 바인딩
    c = conn.cursor()

    # 데이터 프레임 각 행을 DB에 입력
    for row in df.itertuples():
        temp = c.execute('SELECT seq from sqlite_sequence')
        seq = int(temp.fetchone()[0]) + 1
        sql1 = "insert into news_db(headline, keyword, senti, senti_proba) values (?,?,?,?)"
        c.execute(sql1, (row[2],row[4],row[5],row[6]))
        sql2 = "insert into news_id(id, date, code) values (?,?,?)"
        c.execute(sql2, (seq, row[1],row[3]))
    conn.commit()

    print('DB 입력 완료')

def StocktoDB(start, end, code:str):
    '''
    조회할 시작,끝 날짜와 종목이름을 입력받아
    prkrx에서 조회한 OHLCV데이터를 DB에 입력하는 함수
    '''
    import pandas as pd
    import sqlite3
    from datetime import date
    from pykrx import stock
    from pykrx import bond
    
    # db 경로는 로컬에 맞게 설정해야함
    conn = sqlite3.connect("D:/2jo_Final_Python2/DB/final.db")

    # 커서 바인딩
    c = conn.cursor()

    # 종목명 설정, 추후에 딕셔너리 추가 필요!
    ticker_dict = {'삼성전자' : '005930', 'SK하이닉스' : '000660', '네이버' : '035420'}

    df_temp = stock.get_market_ohlcv(start, end, ticker_dict[code], adjusted=False)
    f_rate = df_temp['등락률']

    df = stock.get_market_ohlcv(start, end, ticker_dict[code])
    df = pd.concat([df,f_rate],axis=1)

    df.reset_index(inplace=True)
    df['날짜'] = df['날짜'].apply(lambda x: x.strftime('%Y%m%d'))
    df['날짜'] = df['날짜'].astype(int)
    
    # 데이터 프레임 각 행을 DB에 입력
    for row in df.itertuples():
        sql1 = "insert into stock_db(s_date, s_code, open, high, low, close, volume, f_rate) values (?,?,?,?,?,?,?,?)"
        c.execute(sql1, (row[1], code, row[2], row[3], row[4], row[5], row[6], round(row[7], 2)))
    conn.commit()

    print('DB 입력 완료')

if __name__ == "__main__":
    
    # folder = '뉴스데이터/'
    # company = '하이닉스'
    # data_dir = folder + company
    
    # data = preprocessing_folder(data_dir, 'SK하이닉스')
    # data = data[:1000]
    # data = regex(data)
    # data = make_sentiment(data)
    # # data.to_csv('temp.csv', encoding='utf-8')
    # toDB(data)
    StocktoDB('20170101', '20211231', 'SK하이닉스')

