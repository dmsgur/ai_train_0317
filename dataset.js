
//메뉴 생성기 종료 E==============================
//데이터 아키텍처{sub_title:"",sub_content:"",sub_img:[],user_fill:""}
let data_sets=[]
class DataSet{
	constructor(sub_title,menuNum){this.sub_title=sub_title}
	user_fill=""
	sub_content=[]
	sub_img=[]
	set_content(content){this.sub_content.push(content)}
	set_img(num,obj){
		if(!this.sub_img[num]){this.sub_img[num]=[]}
		this.sub_img[num].push(obj)
	}
	set_fill(ufill){this.user_fill=ufill}	 
}

//d1.set_img(0,{imgtitle:"",imglog:"",imgurl:"",sourceurl:"https://github.com/dmsgur/ai_train_0317/blob/codeone/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})
let d1 = new DataSet("선형회귀모델")//메인 타이틀 //메뉴번호
d1.set_content("보스턴 주택 가격 예측 선형 회귀모델")//서브 타이틀
d1.set_img(0,{imgtitle:"보스턴 데이터 수신",imgurl:"https://drive.google.com/file/d/1bd1SORiBCOAMP3Jphnyt4VVN34QX9xIZ/view?usp=drive_link",imglog:"텐서플로우 보스턴 데이터셋 수신 코드",sourceurl:"https://github.com/dmsgur/ai_train_0317/blob/codeone/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"데이터 특성 파악",imglog:"각 필드별 데이터의 특성의 의미 및 값을 확인",imgurl:"https://drive.google.com/file/d/1lzgrydUhink3M0uZubeSkK3HaPmQIZDM/view?usp=drive_link",sourceurl:"https://github.com/dmsgur/ai_train_0317/blob/codeone/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"데이터 연관성 확인",imglog:"가격정답과 데이터의 특성별 상호 연관도를 파악",imgurl:"https://drive.google.com/file/d/19wJIHlNZWoqzmdLQBRXk2A0u0Rfzfj_B/view?usp=drive_link",sourceurl:"https://github.com/dmsgur/ai_train_0317/blob/codeone/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"데이터 분포도 파악",imglog:"히스토그램을 이용하여 데이터의 분포와 이상치 데이터 확인",imgurl:"https://drive.google.com/file/d/1_ldRd5N1IKHVltmlVDzkBQeM9wk2SQHi/view?usp=drive_link",sourceurl:"https://github.com/dmsgur/ai_train_0317/blob/codeone/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"데이터 정규분포 전환",imglog:"훈련 전처리를 위한 연관성이 있는 데이터를 평균 0 표준편차 1로 구성된 정규분포로 변환",imgurl:"https://drive.google.com/file/d/1Db-jQvLzJ5uyjYrku7ov4sQqTHDraNHt/view?usp=drive_link",sourceurl:"https://github.com/dmsgur/ai_train_0317/blob/codeone/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"순차모델구성 및 훈련실행",imglog:"평균제곱오차법을 이용한 손실함수와 경사하강법을 이용한 최적화 함수로 컴파일 및 최적화된 훈련 15회 실행",imgurl:"https://drive.google.com/file/d/1z9nhfN-No9tulS8w7mJYomi6oAzDMrlS/view?usp=drive_link",sourceurl:"https://github.com/dmsgur/ai_train_0317/blob/codeone/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"훈련 결과 시각화",imglog:"훈련 결과인 mse 손실값의 변화를 시각화 표현",imgurl:"https://drive.google.com/file/d/1LoHOqHtJvWK9OpItPBvFToEy16N1pvKr/view?usp=drive_link",sourceurl:"https://github.com/dmsgur/ai_train_0317/blob/codeone/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"모델의 예측 결과 측정",imglog:"테스트 데이터를 주입하여 예측결과를 인출하고 실제 정답과 차이를 정확률로 표시",imgurl:"https://drive.google.com/file/d/1yt6jxXD6ioA0qd6cDZGtZLqeYMheSEtC/view?usp=drive_link",sourceurl:"https://github.com/dmsgur/ai_train_0317/blob/codeone/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀

d1.set_content("캘리포니아 주택 가격 예측 선형 회구모델")
d1.set_img(1,{imgtitle:"캘리포니아 주택 특성데이터 수신 및 분석",imglog:"사이킷런에서 제공하는 캘리포니아 주택 가격에 따른 데이터 특성(X)들의 모음과 그에 따른 가격정보(Y)",imgurl:"https://drive.google.com/file/d/1vebA4ww2Bl4bSAAySpp7nliMTy-bfAFl/view?usp=drive_link",sourceurl:"https://github.com/dmsgur/ai_train_0317/blob/codeone/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})
d1.set_img(1,{imgtitle:"주택 특성과 가격의 연관성 분석",imglog:"주택의 특성별 산점도 분석으로 가격에 따른 선형성 확인",imgurl:"https://drive.google.com/file/d/19AWzTrD3WL47NLPjy34O2e8hdNh0Ukyv/view?usp=drive_link",sourceurl:"https://github.com/dmsgur/ai_train_0317/blob/codeone/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})
d1.set_img(1,{imgtitle:"데이터 통계정보 분석",imglog:"판다스 데이터프레임으로 전환 후 평균치,표준편차등의 데이터 통계정보 분석",imgurl:"https://drive.google.com/file/d/1cHs9OtbnM3vFznWhy1EG-MnuOoEkTgqk/view?usp=drive_link",sourceurl:"https://github.com/dmsgur/ai_train_0317/blob/codeone/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})
d1.set_img(1,{imgtitle:"데이터 분포 확인",imglog:"히스토그램으로 데이터 분포 시각화와 이상데이터 또는 범위를 벗어난 데이터 설정",imgurl:"https://drive.google.com/file/d/1uvwIm7Jr4D9ZBRUwC4LFkRqu58oDme-d/view?usp=drive_link",sourceurl:"https://github.com/dmsgur/ai_train_0317/blob/codeone/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})
d1.set_img(1,{imgtitle:"이상 데이터제거",imglog:"범위를 벗어나거나 이상치 데이터는 성능에 치명적인 영향을 줄 수 있으므로 이상치 및 범위를 벗어난 데이터를 제거하여 데이터 정제를 수행",imgurl:"https://drive.google.com/file/d/1PvaxU4UXyHLyZvENkGQtcbqQEtM80DyJ/view?usp=drive_link",sourceurl:"https://github.com/dmsgur/ai_train_0317/blob/codeone/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})
d1.set_img(1,{imgtitle:"데이터정제후 분포확인",imglog:"데이터 전처리 시행후 데이터 범위등의 히스토그램으로 이상데이터 분포확인",imgurl:"https://drive.google.com/file/d/1k_hJc-XfqIzquGcOEu56cs_ojF2KoAJr/view?usp=drive_link",sourceurl:"https://github.com/dmsgur/ai_train_0317/blob/codeone/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})
d1.set_img(1,{imgtitle:"훈련데이터와 테스트데이터 분할",imglog:"훈련데이터 80%, 테스트 데이터 20%를 사이킷런 라이브러리를 이용하여 분할 및 데이터 정규분포화 실행",imgurl:"https://drive.google.com/file/d/1VE0HbdA_d0_yACDEGm5zU3TU-3Yn5m7y/view?usp=drive_link",sourceurl:"https://github.com/dmsgur/ai_train_0317/blob/codeone/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})
d1.set_img(1,{imgtitle:"선형회귀 기계학습 모델 구성과 훈련",imglog:"은닉층이 존재하지 않는 머신러닝 모델을 구성하고 평균절대오차 손실함수 설정과 경사하강법으로 최적화 함수를 설정한후 훈련 100회 실행",imgurl:"https://drive.google.com/file/d/13oEq18jHHxzgW6JjAdu4vLe5Sr2HrLWz/view?usp=drive_link",sourceurl:"https://github.com/dmsgur/ai_train_0317/blob/codeone/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})
d1.set_img(1,{imgtitle:"훈련결과 시각화",imglog:"훈련시 저장된 손실값을 이용하여 시각화 그래프 표현",imgurl:"https://drive.google.com/file/d/1vwV4T0iFJUHLROM2Z_6hDewzVkuvoNcJ/view?usp=drive_link",sourceurl:"https://github.com/dmsgur/ai_train_0317/blob/codeone/LinearRegression/Examp_LinearRegression_CaliforniaHousing.py"})

d1.set_content("당뇨상태 1년후 예측 선형 회귀모델")
d1.set_img(2,{imgtitle:"1년후 당뇨상태 예측",imgurl:"",imglog:"",sourceurl:"https://github.com/dmsgur/ai_train_0317/blob/codeone/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀

d1.set_fill("선형 회귀모델은 단일 데이터를 이용하거나 다중 데이터를 이용하여 연속적인 값을 출력하여 예측한다.")//사용자 에필로그
data_sets.push(d1)

// menu2 =============================================================
let d2 = new DataSet("공통모듈구현")//메인타이틀

data_sets.push(d2)

// menu3 =============================================================
let d3 = new DataSet("서버프로그램구현")//메인타이틀

data_sets.push(d3)

// menu4 =============================================================
let d4 = new DataSet("배치프로그램구현")//메인타이틀

data_sets.push(d4)
