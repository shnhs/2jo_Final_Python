
def select_stock_db(stock:str, startdate:int, enddate:int)->list:
    """
    주가 데이터만
    stock_db에서 조회해서 날짜, 시가, 고가, 저가, 종가, 거래량 반환하는 함수
    """
    import sqlite3
    con = sqlite3.connect('DB/2jo.db')
    cur = con.cursor()

    stock_query = """
    select      s_date, open, high, low, close, volume
    from        stock_db 
    where       s_code = :stock 
    and         s_date 
    between     :startdate and :enddate
    """
    
    cur.execute(stock_query, (stock, startdate, enddate))
    stock_list = list(cur.fetchall())
    
    # 날짜 데이터를 유닉스 날짜 데이터로 변환
    stock_list=sql_datetime_to_unixtime(stock_list)

    cur.close()
    return stock_list


def sql_datetime_to_unixtime(query_result:list)-> list:
    """    
    db에서 조회된 컬럼 중 string형의 날짜데이터를 unixtime으로 바꿔주는 함수  
    query_result        : list type
    """
    import datetime
    import time
    import pandas as pd
    # list 타입의 sql 조회된 데이터를 데이터 프레임화
    sql_df = pd.DataFrame(query_result)

    # db로부터 조회한 데이터를 데이터프레임에 씌우면 컬럼값이 0, 1, 2 ...
    # 보통 첫번째 컬럼이 date컬럼인 경우가 많아 0으로 설정
    for index in range(len(sql_df[0])):

        # string 타입의 날짜 형식을 datetime 타입으로 변환
        datetime_date=datetime.datetime.strptime(str(sql_df[0][index]), '%Y%m%d')

        # datetime 타입을 unixtime으로 변환
        # *1000 : 분, 초 포함하도록
        unix_date=(time.mktime(datetime_date.timetuple()))*1000

        # unixtime을 int값으로 변환 후 데이터 프레임에 넣기
        sql_df[0][index] = int(float(unix_date))

    # sql_df의 값을 다시 리스트로 반환
    query_result=sql_df.values.tolist()

    return query_result


def list_datetime_to_unixtime(date_list:list)-> list:
    """
    날짜데이터를 unixtime으로 바꿔주는 함수  
    date_list        : list type 
    """
    import datetime
    import time

    # db로부터 조회한 데이터를 데이터프레임에 씌우면 컬럼값이 0, 1, 2 ...
    # 보통 첫번째 컬럼이 date컬럼인 경우가 많아 0으로 설정
    for index in range(len(date_list)):

        # string 타입의 날짜 형식을 datetime 타입으로 변환
        datetime_date=datetime.datetime.strptime(str(date_list[index]), '%Y%m%d')

        # datetime 타입을 unixtime으로 변환
        # *1000 : 분, 초 포함하도록
        unix_date=(time.mktime(datetime_date.timetuple()))*1000

        # unixtime을 int값으로 변환 후 데이터 프레임에 넣기
        date_list[index] = int(float(unix_date))

    return date_list




