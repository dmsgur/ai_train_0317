import json
import requests
import numpy as np
from datetime import datetime
import datetime as dt
def get_coindata(coinname,getunit="days",timm="",to=""):#to 는 날짜 포맷스트링
    if getunit=="minutes" and not timm:
        timm=60
    conv_date = datetime.now()
    cut_date = datetime.strptime(to,"%Y-%m-%d %H:%M:%S")
    datasets=[]
    if to :
        while cut_date<conv_date:
            # 날짜 포맷 : yyyy-MM-dd HH:mm:ss
            to = conv_date.strftime("%Y-%m-%d %H:%M:%S")
            url = f"https://api.bithumb.com/v1/candles/{('minutes/'+str(timm) if getunit=='minutes' else getunit)}?market=KRW-{coinname}&count=200{'&to='+to if to else ''}"
            headers = {"accept": "application/json"}    
            response = requests.get(url, headers=headers)
            # if(datasets):
            #     print(datetime.strptime(datasets[-1]["candle_date_time_kst"],"%Y-%m-%dT%H:%M:%S"))
            jData = json.loads(response.text)
            if datasets:
                highDate = datetime.strptime(datasets[-1]["candle_date_time_kst"],"%Y-%m-%dT%H:%M:%S")
                lowDate = datetime.strptime(jData[0]["candle_date_time_kst"],"%Y-%m-%dT%H:%M:%S")
                if highDate<=lowDate:
                    for ix in range(len(jData)):
                        lowDate = datetime.strptime(jData[ix]["candle_date_time_kst"],"%Y-%m-%dT%H:%M:%S")
                        if highDate<=lowDate:
                            jData.pop(ix)
                        else:break
            datasets.extend(jData)
            conv_date = conv_date-dt.timedelta(days=200)
    else :
        url = f"https://api.bithumb.com/v1/candles/{('minutes/'+str(timm) if getunit=='minutes' else getunit)}?market=KRW-{coinname}&count=200"
        headers = {"accept": "application/json"}    
        response = requests.get(url, headers=headers) 
        datasets.extend(json.loads(response.text))
    return datasets#dumps to json (load, dump  파일컨트롤)

def create_rnndata(dataset,timestep=30):
    dataset.reverse()#오름차순 정렬 변경
    x_data=[];y_data=[]
    for ix in range(len(dataset)-timestep):
        slice_data = [[d["opening_price"],d["high_price"],d["low_price"],d["trade_price"],\
                       d["prev_closing_price"]] for d in dataset[ix:timestep+ix]]
        x_data.append(slice_data)
        y_data.append(dataset[timestep+ix]) # y        
    y_data = [int((d["opening_price"]+d["trade_price"])/2*10000)/10000 for d in y_data]
    return np.array(x_data),np.array(y_data)

#BTC_days_T30 , 어제데이터도 출력,, 어제하고 갭을 출력해서 오늘 확률의 편차
def get_xpred(coinname="BTC",getunit="days",timm="",timestep=30):
    url = f"https://api.bithumb.com/v1/candles/{('minutes/'+str(timm) if getunit=='minutes' else getunit)}?market=KRW-{coinname}&count={timestep+1}"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers) 
    datasets=json.loads(response.text)
    if len(datasets)<timestep+1:
       return np.array([])
    datasets.reverse()
    y_curprice  = int((datasets[-1]["opening_price"]+datasets[-1]["trade_price"])/2*10000)/10000
    slice_data = [[d["opening_price"],d["high_price"],d["low_price"],d["trade_price"],\
                       d["prev_closing_price"]] for d in datasets[-timestep:]]
    x_data=(slice_data)
    slice_data = [[d["opening_price"],d["high_price"],d["low_price"],d["trade_price"],\
                       d["prev_closing_price"]] for d in datasets[-timestep-1:-1]]
    x_predata=(slice_data)
    return np.array(x_data),np.array(x_predata),y_curprice
def get_highxpred(coinname="BTC",getunit="days",timm="",timestep=30):
    url = f"https://api.bithumb.com/v1/candles/{('minutes/'+str(timm) if getunit=='minutes' else getunit)}?market=KRW-{coinname}&count={timestep+1}"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers) 
    datasets=json.loads(response.text)
    if len(datasets)<timestep+1:
       return np.array([])
    datasets.reverse()
    y_curprice  = datasets[-1]["high_price"]*1
    slice_data = [d["high_price"] for d in datasets[-timestep:]]
    x_data= slice_data
    slice_data = [d["high_price"] for d in datasets[-timestep-1:-1]]
    x_predata= slice_data 
    return np.array(x_data),np.array(x_predata),y_curprice
def high_create_rnndata(dataset,timestep=30):
    print(len(dataset))
    dataset.reverse()#오름차순 정렬 변경
    # print(dataset[-1])
    x_data=[];y_data=[]
    for ix in range(len(dataset)-timestep):
        # ["opening_price","high_price","low_price","trade_price","prev_closing_price"]])# x
        # 산점도 분석후 데이터
        slice_data = [[d["high_price"]] for d in dataset[ix:timestep+ix]]
        x_data.append(slice_data)
        y_data.append(dataset[timestep+ix]) # y
    # print(y_data[-1])
    # print(x_data[-1][-1])
    y_data = [d["high_price"] for d in y_data]
    return np.array(x_data),np.array(y_data)
    