import requests
import jsons
import numpy as np
def getCandleData(ttime="days",cname="BTC"):# 거래데이터 raw 정보 가져오기
        get_addr = r"https://api.bithumb.com/v1/candles/{}?market=KRW-{}&count=200"\
            .format(ttime,cname)
        response = requests.get(get_addr)
        candle_data = jsons.load(response.json())#load>dict 변형 dump>json 
        return candle_data
def creatX(dataset,transCount):# 문제데이터 추출
    dataset.reverse()
    #최근 첫번째 데이터 삭제
    del dataset[-1]
    #오래된 순서부터 낮은 인덱스
    xlist = []
    ylist = []
    for i in range(len(dataset)-transCount):
        tmp = dataset[i:transCount+i]
        for data in tmp:
            if "market" in data:
                del data["market"];del data["candle_date_time_utc"];
            if "timestamp" in data:
                del data["timestamp"];
            if "change_price" in data:
                del data["change_price"];
            if "change_rate" in data:
                del data["change_rate"];
            if "candle_date_time_kst" in data:
                del data["candle_date_time_kst"];
            if "prev_closing_price" in data:
                del data["prev_closing_price"];
            if "first_day_of_period" in data:
                del data["first_day_of_period"];
        xlist.append(tmp)
        ylist.append([dataset[transCount+i]["opening_price"],
                     dataset[transCount+i]["high_price"],
                     dataset[transCount+i]["low_price"],
                     dataset[transCount+i]["trade_price"]])
    return xlist,np.array(ylist)
def integraion_xdata(xdata):
    keylist = xdata[0][0].keys()
    xlist =[]# [[[1 2 3 4 5],[2 3 4 5 6],[3 4 5 6 7]],[2 3 4 5 6],[ 3 4 5 6 7]]
    for d in xdata:
        tmp=[]
        for f in d:
            tmp.append(list(f.values()))
        xlist.append(tmp)
    return np.array(xlist),keylist
    