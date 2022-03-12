import pandas as pd
import sqlite3
from datetime import datetime, timedelta, date
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import selectStock_datetime

def scaler(result_df:pd.DataFrame) -> pd.DataFrame:
    """
    date를 제외한 나머지 컬럼 0과 1사이로 정규화하는 함수
    result_df      : 정규화할 데이터 프레임 데이터
    """
    date_c = list(result_df['date'])

    x = result_df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    result_df = pd.DataFrame(x_scaled, columns=result_df.columns)

    result_df['date'] = date_c

    return result_df

def service(code:str, start_day:int, end_day:int, period:int, drop_holi = 0, stock_moving_avg = 1, day_shift = 0) -> pd.DataFrame:
    '''
    각 옵션을 입력받아 해당 기간의 데이터를 DB로부터 조회하여 원하는 형태로 가공하여 리턴하는 함수
    
    -- 옵션 설명 --
    code                : 조회할 종목이름
    start_day           : 조회를 시작 날짜
    end_day             : 조회 종료 날짜
    period              : 뉴스 긍부정과 주가를 이동평균 낼 기간
    drop_holi           : 주말 혹은 공휴일 뉴스를 사용할지 여부. 0 (디폴트) - 다음 영업일 주가로 채워서 사용 / 1 - 주말 및 공휴일 데이터는 drop
    stock_moving_avg    : 주가를 이동평균 낼지 여부. 1 (디폴트) - 주가 이동평균 사용 / 0 - 이동평균 사용안함
    day_shift           : 뉴스와 주가의 몇 일의 텀을 두고 분석할 지.  0(디폴트) | +x - 해당일의 뉴스와 다음날의 주가 분석 | -x - 해당일의 뉴스와 전날의 주가 분석 
    
    -- 리턴 설명--
    result_df           : 날짜 / 긍부정 / 주가 등락 결과를 정제한 데이터프레임
    all_keyword         : 조회한 기간 중 전체의 키워드
    pos_keyword         : 조회한 기간 중 주가가 오른날의 키워드
    neg_keyword         : 조회한 기간 중 주가가 내린날의 키워드
    df_length           : 조회한 기간의 데이터프레임 길이
    '''

    # 이동평균을 고려하여 DB에서 조회할 날짜 설정
    inq_day = (datetime.strptime(str(start_day), "%Y%m%d").date() - timedelta(days = period - 1)).strftime('%Y%m%d')
    end_day = str(end_day)

    # db 경로는 로컬에 맞게 설정해야함
    conn = sqlite3.connect("DB/2jo.db")
    
    # 커서 바인딩
    c = conn.cursor()

    # 뉴스데이터 조회
    # query = c.execute(f"select a.id, a.date, a.code, b.senti, b.senti_proba from news_db b join news_id a on b.id = a.id where a.date BETWEEN {inq_day} and {end_day};")
    query = c.execute(f"select a.id, a.date, a.code, b.keyword, b.senti, b.senti_proba from news_db b join news_id a on b.id = a.id where a.code = \'{code}\' and (a.date BETWEEN {inq_day} and {end_day});")
    # 컬럼명 조회
    cols = [column[0] for column in query.description]
    # 데이터 프레임으로 만들기
    news_result_df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)

    # 키워드 데이터프레임 만들어 놓기
    keyword_result_df = news_result_df.groupby('date')['keyword'].apply(lambda x: x.sum())

    # 커서 닫기 - 일단 주석처리함
    # conn.close()

    # 주가 데이터 조회
    query = c.execute(f"select s_date, s_code, f_rate from stock_db where s_code = \'{code}\' and (s_date BETWEEN {inq_day} and {end_day});")
    # 컬럼명 조회
    cols = [column[0] for column in query.description]
    # 데이터 프레임으로 만들기
    stock_result_df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
    stock_result_df.rename(columns={'s_date': 'date', 's_code': 'code', 'f_rate': 'UpDown'}, inplace=True)
    
    # 데이터프레임 길이 반환
    df_length = len(stock_result_df)

    # 주말 및 공휴일 drop 여부는 옵션에 따라; 디폴트는 드랍안함
    if drop_holi:
        # 주말이나 공휴일 등으로 주가가 빠진 날은 drop
        merge_outer_df = pd.merge(news_result_df,stock_result_df, how='outer',on='date')
        merge_outer_df = merge_outer_df.dropna(subset=['UpDown'])
    else:
        # 주말이나 공휴일 등으로 주가가 빠진 날은 다음 Business Day의 주가로 채워줌
        merge_outer_df = pd.merge(news_result_df,stock_result_df, how='outer',on='date').fillna(method='bfill')

    dateg = merge_outer_df.groupby(['date']).mean()
    dateg = dateg[['senti', 'UpDown']]
    dateg = dateg.reset_index()

    # 설정한 기간에 따라 뉴스 긍부정 이동평균
    dateg['senti_moving_avg'] = dateg['senti'].rolling(window=period).mean()

    # 주가의 이동평균은 옵션에 따라 결정; 디폴트는 이동평균 사용
    if stock_moving_avg:
        dateg['UpDown_moving_avg'] = dateg['UpDown'].rolling(window=period).mean()
    else:
        dateg['UpDown_moving_avg'] = dateg['UpDown']

    dateg = dateg[['date', 'senti_moving_avg', 'UpDown_moving_avg']]

    # 뉴스와 주가사이의 텀 설정
    dateg['UpDown_moving_avg'] = dateg['UpDown_moving_avg'].shift(day_shift)

    # 키워드랑 병합하기
    result = pd.merge(dateg,keyword_result_df, how='outer',on='date')
    result.dropna(inplace=True)

    # 전체 키워드
    all_keyword_list = result['keyword']
    all_keyword_result_list = []

    for k in all_keyword_list:
        all_keyword_result_list.append(k)

    all_keyword = ' '.join(all_keyword_result_list)

    # 오른날 키워드
    pos_day = result[result['UpDown_moving_avg'] >= 0]
    
    pos_keyword_list = pos_day['keyword']
    pos_keyword_result_list = []

    for k in pos_keyword_list:
        pos_keyword_result_list.append(k)

    pos_keyword = ' '.join(pos_keyword_result_list)

    # 떨어진날 키워드
    neg_day = result[result['UpDown_moving_avg'] < 0]

    neg_keyword_list = neg_day['keyword']
    neg_keyword_result_list = []

    for k in neg_keyword_list:
        neg_keyword_result_list.append(k)

    neg_keyword = ' '.join(neg_keyword_result_list)

    result = result[['date', 'senti_moving_avg', 'UpDown_moving_avg']]

    result=scaler(result)
    return result, all_keyword, pos_keyword, neg_keyword, df_length


def correlation_df_to_list(corr_df:pd.DataFrame)-> list: 
    """
    date, 뉴스 긍부정 평균값, 주가 등락률의 평균값
    list 형식으로 반환

    dateFrame의 value값만 -> list로 반환
    """
    # 날짜 데이터 list로 변환
    date_list = corr_df['date'].values.tolist() # 일반적인 날짜 -> 20210101
    # 날짜를 unixtime으로 변환
    date_list=selectStock_datetime.list_datetime_to_unixtime(date_list)

    # 뉴스 긍부정 데이터 list로 변환
    news_moving_avg_list=corr_df['senti_moving_avg'].round(2).values.tolist()

    # 주가 등락률 데이터 list로 변환
    stock_moving_avg_list=corr_df['UpDown_moving_avg'].round(2).values.tolist()

    return date_list, news_moving_avg_list, stock_moving_avg_list

def correlation_value_to_list(result_df:pd.DataFrame) -> list:
    """
    피어슨 상관계수 값 list로 반환
    ex) 2*2 상관 그래프일 때
    [[0,0, value], [0,1, value], [1,0, value], [1,1, value]]
    """
    # 데이터 프레임에서 피어슨 상관계수로 상관계수 뽑아내기
    pairplot = result_df.iloc[:, 1:3] # date 제외
    corr_value=pairplot.corr(method='pearson').iloc[0]

    # 0이 아닌 값만 뽑아내기
    for index in corr_value:
        if index != 0:
            corr_value=index

    # 히트맵 값 들어가는 형식 맞추기
    corr_value=round(corr_value, 2)
    pair_list=[[0,0,1],[0,1,corr_value],[1,0,corr_value],[1,1,1]]
    return list(pair_list)


################## 번외 함수 ####################

def date_corNews_list(corr_df:pd.DataFrame)-> list: 
    """
    corr 결과
    [날짜, 뉴스 긍부정] 형태의 list로 반환
    """

    import datetime
    import time

    date_list= corr_df['date'].values.tolist()
    for index in range(len(corr_df['date'])):

        # string 타입의 날짜 형식을 datetime 타입으로 변환
        datetime_date=datetime.datetime.strptime(str(date_list[index]), '%Y%m%d')

        # datetime 타입을 unixtime으로 변환
        # *1000 : 분, 초 포함하도록
        unix_date=(time.mktime(datetime_date.timetuple()))*1000

        # unixtime을 int값으로 변환 후 데이터 프레임에 넣기
        date_list[index] = int(float(unix_date))
    
    date_df=pd.DataFrame(date_list,  columns=['date'])
    corr_df=pd.concat([date_df, corr_df['senti_moving_avg']], axis=1)

    corr_list=corr_df.values.tolist()
    return corr_list

