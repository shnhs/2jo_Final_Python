from collections import OrderedDict
from datetime import date
from flask import Flask, make_response, request
import json
import selectStock_datetime # db 조회하는 함수 집합
import analysis
import datetime

app = Flask(__name__)


print('2조 파이널 프로젝트 입니다.')


@app.route("/", methods=["POST"])
def java_connection():

    # 받아온 변수를 순서대로 json에 담아 보내기 위해 orderedDict 이용
    response_dict=OrderedDict()

    # 서버 통신 확인
    print('')
    print(f'서버 통신 요청 : {request.method}, 요청 시간 : {datetime.datetime.now()}')
    print('')

    # 비동기방식으로 넘어온 request 객체에서 종목, 조회 시작날짜, 조회 종료날짜, 뉴스 
    stock_name = request.form.get("stock")
    startdate = request.form.get("startdate")
    enddate = request.form.get("enddate")
    news_period = request.form.get("day")

    # default 값이 있는 변수
    drop_holiday = request.form.get("holiday")
    stock_moving_avg = request.form.get("moveavg")
    day_shift = request.form.get("shift")

    print('받아오기 성공 : ',stock_name, startdate, enddate, news_period,drop_holiday,stock_moving_avg,day_shift)


    # 상관관계 데이터 조회
    # corr_df=analysis.service(stock_name, int(startdate), int(enddate), int(news_period), int(drop_holiday), int(stock_moving_avg), int(day_shift))
    corr_df, all_day_keyword, pos_day_keyword, neg_day_keyword, date_length= analysis.service(stock_name, int(startdate), int(enddate), int(news_period), int(drop_holiday), int(stock_moving_avg), int(day_shift))

    # 상관관계 데이터 값 반환
    date_list, news_avg_list, stock_avg_list=analysis.correlation_df_to_list(corr_df)
    
    print('상관관계 조회 반환 성공', type(date_list), type(news_avg_list), type(stock_avg_list))

    # stock db에서 주가 데이터 조회해서 반환
    stock_data=selectStock_datetime.select_stock_db(stock_name, startdate, enddate)

    # 상관계수 값 받아오기, 히트맵 
    corr_list = analysis.correlation_value_to_list(corr_df)

    # scatter plot 그리기 위한 값 받아오기
    # scatter_list = analysis.scatter_value_to_list(corr_df)

    # 딕셔너리 키-값 형태로 매치
    response_dict['s_code']=stock_name
    response_dict['stock_data']=stock_data
    response_dict['cor_date']=date_list
    response_dict['cor_news']=news_avg_list
    response_dict['cor_stock']=stock_avg_list
    response_dict['cor_value']=corr_list
    response_dict['all_keyword']= all_day_keyword
    response_dict['pos_keyword']= pos_day_keyword
    response_dict['neg_keyword']= neg_day_keyword
    response_dict['date_len']= date_length
    # response_dict['scatter_value']=scatter_list
    

    # response 객체 담아 post로 전송
    # json 형식으로 변환, ensure_ascii : 한글 깨짐 방지
    response = make_response(json.dumps(dict(response_dict), ensure_ascii=False)) 
    response.headers.add("Access-Control-Allow-Origin", "*") # 비동기 요청 처리 

    return response
    
if __name__ == '__main__':
    app.debug = True
    app.run()