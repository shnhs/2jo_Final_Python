from flask import Flask, jsonify, make_response, render_template, request
import sqlite3
import json

app = Flask(__name__)


print('솔트룩스 2조 파이널 프로젝트 입니다.')

testlist = []


def select_sqlite(stock, startdate, enddate):
    global testlist, daylist
    con = sqlite3.connect('final.db')
    cur = con.cursor()
    query = """
    select      * from stock_db 
    where       s_code=:stock 
    and         s_date=:startdate
    """

    datequery = """
    select      * 
    from        stock_db 
    where       s_code= ?
    and         s_date 
    between     ? and ?
    """
    cur.execute(datequery, (stock, startdate, enddate))
    daylist = cur.fetchall()
    print(daylist)
    return daylist

def default_select_data(stock):
    """
    기본적인 데이터 조회하는 함수
    """
    global defaultlist
    con = sqlite3.connect('final.db')
    cur = con.cursor()
    defaultquery = """
    select      * 
    from        stock_db 
    where       s_code= ?
    """
    cur.execute(defaultquery, (stock, ))
    defaultlist = cur.fetchall()
    print(defaultlist)
    return defaultlist

@app.route("/", methods=["POST", "GET"])
def javaconn():
    print('요청하는 방식 : ', request.method)
    global stock, startdate, enddate, news_period, answer
    if request.method == 'POST':
        stock = request.form.get("stock")
        startdate = int(request.form.get("startdate"))
        enddate = int(request.form.get("enddate"))
        news_period = int(request.form.get("day"))
        print(news_period)
        answer = select_sqlite(stock, startdate,enddate)

    elif request.method == 'GET':
        stock = request.args.get("stock")
        answer = default_select_data(stock)
        # with open('aapl-ohlcv.json', 'r') as jsonf:
        #     json_data=json.load(jsonf)

    response = make_response(jsonify(answer))
    response.headers.add("Access-Control-Allow-Origin", "*")
    
    # response2=make_response(json_data)
    # response2.headers.add("Access-Control-Allow-Origin", "*")

    return response

if __name__ == '__main__':
    app.debug = True
    app.run()