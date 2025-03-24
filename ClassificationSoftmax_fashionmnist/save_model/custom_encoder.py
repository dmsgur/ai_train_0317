class CustomEncoder():
    def __init__(self,mylist=None):
        if mylist:
            self.__mylist  = sorted(list(set(mylist)))# 중복데이터 제거
            self.__convInt = [ix for ix,_ in enumerate(self.__mylist)]# 인덱스를 정수로 바꾸는 코드
        else :
            self.__mylist=None
            self.__convInt=None        
    def label_to_integer(self,y_dataset):
        if self.__mylist :
            print("mylist 를 먼저 입력해야합니다.")
            return None
        return [self.__mylist.index(d) for d in y_dataset]
        
    def label_to_one_hot(self,y_dataset):
        if self.__mylist :
            print("mylist 를 먼저 입력해야합니다.")
            return None
        tmp = y_dataset.copy()
        tmp = self.convertInteger(tmp)#모두 정수로 변환됨
        print(tmp)
        rettmp = []
        retlist = [0 for i in range(len(self.__mylist))]       
        for d in tmp:
            rtmp = retlist.copy()
            rtmp[d]=1
            rettmp.append(rtmp)
        return self.__mylist,tmp,rettmp
    def integer_to_one_hot(self,y_intdataset,labeldata=None):
        if labeldata:
            self.__mylist = labeldata.copy()
            self.__convInt = [ix for ix,_ in enumerate(self.__mylist)]
        maxdata = max(y_intdataset)
        retlist = [0 for i in range(maxdata+1)]
        rettmp=[]
        for d in y_intdataset:
            rtmp = retlist.copy()
            rtmp[d]=1
            rettmp.append(rtmp)
        return rettmp
    def one_hot_to_label(self,oharr,label_list=None):#리스트 목록이 없을때 ?
        import numpy as np
        if label_list:
            temp = [label_list[np.argmax(ohdata)] for ohdata in oharr]
        else :
            print(self.__mylist)
            temp = [self.__mylist[np.argmax(ohdata)] for ohdata in oharr]
            
        return temp
#print(__name__)        
if __name__=="__main__":
    pass
#1. 생성자 이용시
#  - 라벨(문자형 리스트 정답 데이터)파라미터 존재시 - 라벨을 저장하고 정수형 변수를 저장해 놓는다.
#  - 라벨 파라미터 미 존재시 - 라벨과 정수형 변수는 관리하지 않는다. 
#2. label_to_integer(y_dataset) -  라벨이 존재하는 경우 y_dataset 라벨데이터를 정수형으로 변환한다.
#    ret - integer list type
#3. label_to_one_hot(y_dataset) - 라벨이 존재하는 경우 라벨리스트를 받아서 원핫인코딩 반환한다.
#4. integer_to_one_hot(self,y_intdataset,labeldata=None) 
#                     - 정수형 데이터를 받아서 원핫인코딩을 리턴한다.
#5. one_hot_to_label(self,oharr,label_list=None) 
#                      -  원핫인코딩된 데이터 및 라벨리스트를 받아서 라벨데이터리스트를 반환한다.