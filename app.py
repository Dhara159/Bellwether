from flask import Flask, request, render_template,session,make_response
from flaskext.mysql import MySQL
from werkzeug import generate_password_hash, check_password_hash
from flask import jsonify
from pytz import timezone
import datetime, requests
from datetime import timedelta
import json
import threading
import os, csv, quandl
import math
import operator
import pandas as pd
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import numpy as np


app = Flask(__name__)

app.secret_key = 'super secret key'

mysql = MySQL()
 
# MySQL configurations
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'bi1'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)

@app.route('/')
def hello():
	return 'hello'

@app.route('/signup',methods = ['POST', 'GET'])
def signup():
	return render_template("client_signup.html")
	

@app.route('/registered',methods = ['POST', 'GET'])
def registered():
	userName = request.args['userName']
	userEmail = request.args['userEmail']
	userPassword = request.args['userPassword']
	#virtualMoney = 1000000
	
	print(userName)
	print(userEmail)
	print(userPassword)
	return insertNewUser(userName, userEmail, userPassword)

	#cursor.execute("select * from users")
	#data = cursor.fetchone()
	#return render_template("client_signin.html")
def insertNewUser(userName, userEmail, userPassword):
	conn = mysql.connect()
	cursor = conn.cursor()
	try:
		cursor.execute("INSERT INTO users (userName, userEmail, userPassword, virtualMoney) VALUES (%s, %s, %s, %s)", (str(userName), str(userEmail), str(userPassword), '1000000'))
		conn.commit()
		return render_template("client_login.html")
	except:
		return("Failed to insert values")
	finally:
		cursor.close()
	return render_template("client_login.html")

@app.route('/signin')
def signin():
	return render_template("client_login.html")

# @app.after_request
# def add_header(response):
#     response.cache_control.max_age = 300
#     return response

@app.route('/client_dashboard')
def client_dashboard():
	userEmail = request.args['userEmail']
	userPassword = request.args['userPassword']
	print(userEmail)
	print(userPassword)
	conn = mysql.connect()
	cursor = conn.cursor()
	cursor.execute("SELECT userEmail,userPassword FROM users WHERE userEmail ='" + userEmail + "' ")
	data = cursor.fetchone()
	if data == None:
		return render_template("client_login.html")
	print(data)

	if userEmail == data[0] and userPassword == data[1]:
		conn.commit()
		# session['userEmail'] = request.args['userEmail']
		# session['loged_in'] = True
		# print(session)
		#redirect(url_for('client_dashboard.html'))
		#return gotoclient()
		return render_template("client_dashboard.html")

	#return render_template("client_dashboard.html")

#def gotoclient():
#	if session['userEmail']==None:
#		return('session null')
#	else:
#		return render_template("client_dashboard.html")

@app.route('/client_logout')
def client_logout():
	return render_template('client_login.html')
	#session.pop('userEmail', None)
	#session.pop('loged_in',False)
	# app.secret_key = os.urandom(32)
	#for key in session.keys():
	#	session.pop(key)
	#session.clear()
	#print(session)
	#return signin()

@app.route("/historic_data")
def historic_data():
	return render_template("historic_data.html")

@app.route('/historic_data_search')
def historic_data_search():
	trade = request.args['trade']
	sdate = request.args['sdate']	
	edate = request.args['edate']
	print(trade)
	print(sdate)
	print(edate)
	fName = str(trade + '.csv')
	df = quandl.get("NSE/"+trade.upper(), authtoken="5GGEggAyyGa6_mVsKrxZ",start_date=sdate,end_date=edate)
	datalist = df.values.tolist()
	# print(df.index)
	# print(datalist)
	# with open(fName,'rt') as csvfile:
	# 	data = list(csv.reader(csvfile))
	# 	print(data)
	# df = pd.read_csv(fName)
	# print(df.loc[(df['Date'] == sdate)])
	# for x in df['Date']:
	# 	if (x<sdate) & (x>edate):
	# 		df1 = df(x)
	# print(df1)
	return render_template("historic_data_search.html",datalist=datalist, df=df.to_html(classes=["table", "thead-dark","table-bordered", "table-striped", "table-hover"]))



@app.route('/algorithms')
def algorithms():
	return render_template("algorithms.html")


@app.route('/historic_graph')
def historic_graph():
	return render_template("historic_graph.html")



@app.route('/historic_graph_search')
def historic_graph_search():
	trade = request.args['trade']
	attribute = request.args['attribute']
	sdate = request.args['sdate']
	edate = request.args['edate']
	print(trade)
	print(attribute)
	print(sdate)
	print(edate)
	fName = str(trade + '.csv')
	df = quandl.get("NSE/"+trade.upper(), authtoken="5GGEggAyyGa6_mVsKrxZ",start_date=sdate,end_date=edate)
	fig = df[[attribute]].plot()
	file_path = "static/images/mytable1.png"
	data = plt.savefig(file_path)
	return render_template("historic_graph_search.html")

@app.route('/buy_sell')
def buy_sell():
	return render_template("buy_sell.html")

@app.route('/buy_sell_confirm')
def buy_sell_confirm():
	order_type=request.args['buy']
	trade = request.args['trade']
	volume = request.args['volume']
	cprice = request.args['cprice']
	total = request.args['total']

	print(order_type)
	print(trade)
	print(volume)
	print(cprice)
	print(total)
	if(order_type == "buy"):
		sellingPrice =0
		purchasePrice = total

	else:
		purchasePrice =0
		sellingPrice = total

	conn = mysql.connect()
	cursor = conn.cursor()
	cursor.execute("INSERT INTO ORDERDETAILS (id,userId,tradeName,dates,purchasePrice,sellingPrice,volume) VALUES (%s,%s,%s,%s,%s,%s,%s)", ('9','2',trade,'2018-03-14',float(purchasePrice),float(sellingPrice),float(volume)))
	conn.commit()
	return render_template("buy_sell_confirm.html", trade=trade,volume=volume,total=total,cprice=cprice,purchasePrice=purchasePrice,sellingPrice=sellingPrice)


@app.route('/profile')
def profile():
	return render_template("profile.html")


@app.route("/order_details")
def order_details():
	conn = mysql.connect()
	cursor = conn.cursor()

	cursor.execute('SELECT id,userId,tradeName,dates,purchasePrice,sellingPrice,volume FROM orderdetails')
	conn.commit()
	return render_template("order_details.html", items=cursor.fetchall())

@app.route("/order_details_search",methods=['POST','GET'])	
def order_details_search():
	conn = mysql.connect()
	cursor = conn.cursor()
	trade = request.form['trade']
	cursor.execute("SELECT id,userId,tradeName,dates,purchasePrice,sellingPrice,volume FROM orderdetails WHERE tradeName ='" + trade + "' ")
	conn.commit()
	return render_template("order_details.html", items=cursor.fetchall())
	

@app.route('/print_items')
def print_items():
    cursor = db.execute('SELECT column1,column2,column3,column4 FROM tablename')
    return render_template('print_items.html', items=Units.query.all())

@app.route("/getLiveData")
def getLiveData():	
	trades = ["HDFC","BIOCON","PNB","AJANTAPHARM","AKZOINDIA","ASHOKLEY","ASIANPAINT","ASTRAZEN","AUROPHARMA","AXISBANK","BAJAJCORP","BPCL","CENTRALBK","DENABANK","DISHTV","DLF","GAIL","GLENMARK","GODREJCP","GODREJIND","GPPL","HAVELLS","HDFCBANK","HEROMOTOCO","ICICIBANK","IDBI","NAUKRI","JETAIRWAYS","JUSTDIAL","ONGC"]
	inLoop = threading.Timer(900.0,getLiveData).start()
	for i in range(len(trades)):
		url = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=%s&interval=15min&apikey=KPEFHIZF3S02LQF1'%(trades[i]))
		jsonData = json.loads(url.text)
		finalTime = getUsTime()
		openValue = jsonData['Time Series (1min)'][finalTime]['1. open']
		highValue = jsonData["Time Series (1min)"][finalTime]['2. high']
		lowValue = jsonData["Time Series (1min)"][finalTime]['3. low']
		closeValue = jsonData["Time Series (1min)"][finalTime]['4. close']
		volumeValue = jsonData["Time Series (1min)"][finalTime]['5. volume']
		insertLiveData(openValue, highValue, lowValue, closeValue, volumeValue, finalTime)
	return(NULL)

@app.route("/knn")
def knn():
    k = 3
    startDate = (datetime.datetime.now()).strftime("%Y-%m-%d")
    endDate = ((datetime.date.today()-relativedelta(months=+3))).strftime("%Y-%m-%d")
    df = quandl.get("NSE/DLF", authtoken="JMRWwixg-zh5jGHGnKzn",start_date=endDate,end_date=startDate)
    df1 = df[['Open','Close']]
    print(df1)
    dataList = df1.values.tolist()
    testInstance = dataList[len(dataList)-2]
    neighbors = getNeighbors(testInstance, dataList, k)
    idwPrediction, meanPrediction  = prediction(neighbors)
    return("Inverse Distance Weighted Average Prediction: %f <br/> Mean Average Prediction: %f" % (idwPrediction, meanPrediction))

@app.route("/mlp")
def mlp():
    fi = pd.read_csv(r'AAPL.csv', index_col=['Date'], header=0,usecols=['Open', 'High', 'Low', 'Volume', 'Adj. Close', 'Date'])
    x = 30
    mydates = [d.strftime('%d-%m-%Y') for d in pd.date_range('01-01-2014', '31-12-2014')]
    for i in mydates[:60]:
        f = []
        openP = []
        closeP = []
        date = []
        features = []
        dateS = i.split('-')
        try:
            f.append(to_integer(dateS))
            f.append(fi.loc[i]['Open'])
            f.append(fi.loc[i]['High'])
            f.append(fi.loc[i]['Low'])
            f.append(fi.loc[i]['Volume'])
            date.append(to_integer(dateS))
            features.append(f)
            openP.append(fi.loc[i]['Open'])
            closeP.append(fi.loc[i]['Adj. Close'])
        except KeyError:
            x += 1
    we = [20130201,	459.110001,	459.479996,	448.350021, 134871100]
    predictedPrice = predict_prices(date, closeP, we, features)
    return("MPLPredicted Price: %f" % (predictedPrice))

@app.route("/svrlinear")
def svrlinear():
    dates = []
    prices = []
    with open('AAPL_last_month.csv','r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[4]))
    predictedPrice=predictPricesSvrlinear(dates,prices, 167)
    return("SVRLinear predicted price: %f" % (predictedPrice))

@app.route("/svrpoly")
def svrpoly():
    dates = []
    prices = []
    with open('AAPL_last_month.csv','r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[4]))
    predictedPrice=predictPricesSvrpoly(dates,prices, 167)
    return("SVRPolynomial predicted price: %f" % (predictedPrice))

def insertLiveData(openValue, highValue, lowValue, closeValue, volumeValue, finalTime):
	conn = mysql.connect()
	cursor = conn.cursor()
	try:
		cursor.execute("INSERT INTO liveData (stockName, open, close, high, low, volume, dateAndTime) VALUES (%s, %s, %s, %s, %s, %s, %s)", ('biocon',  float(openValue), float(closeValue), float(highValue), float(lowValue), float(volumeValue), finalTime))
		conn.commit()
		return("Data inserted successfully")
	except MySQLdb.IntegrityError:
		return("Failed to insert values")

def getIndTime():
	indTime = datetime.datetime.now()
	saveInd = indTime.strftime("%Y-%m-%d %H:%M:00")
	return saveInd

def getUsTime():
	usa = timezone("US/Eastern")
	usTime = datetime.datetime.now(usa)
	saveUs = usTime.strftime("%Y-%m-%d %H:%M:00")
	return saveUs

def predictPricesSvrpoly(dates,prices,x):
    dates = np.reshape(dates,(len(dates),1))
    svr_poly=SVR(kernel='poly', C=0.02, degree=2)
    svr_poly.fit(dates,prices)
    plt.scatter(dates,prices, color='red', label='Data')
    plt.plot(dates,svr_poly.predict(dates),color='blue',label='Polynomial Model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    return svr_poly.predict(x)[0]

def predictPricesSvrlinear(dates,prices,x):
    dates = np.reshape(dates,(len(dates),1))
    svr_lin=SVR(kernel='linear', C=1)
    svr_lin.fit(dates,prices)
    plt.scatter(dates,prices, color='black', label='Data')
    plt.plot(dates,svr_lin.predict(dates),color='red',label='Linear Model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    return svr_lin.predict(x)[0]

def to_integer(dt_time):
    return 10000 * int(dt_time[2]) + 100 * int(dt_time[1]) + int(dt_time[0])    

def predict_prices(dates, prices, x, features):
    dates = np.reshape(dates, (len(dates), 1))
    cutoff = int(len(features)*3/4)
    training = features[:cutoff]
    test = features[cutoff:]
    mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(5, 5),max_iter=100, random_state=1)
    mlp.fit(training, prices[:cutoff])
    testResult = mlp.predict(test)
    ax, pl = plt.subplots()
    pl.scatter(dates, prices, color='black', label='Data')
    pl.scatter(dates[cutoff:], testResult, color='red', label='Test')
    pl.scatter(dates[:cutoff], mlp.predict(training), color='blue', label='Training')
    pl.set_ylim(50, 100)
    avg = 0.0
    lms = 0.0
    le = len(testResult)
    for i in range(0, le):
        print(testResult[i], prices[cutoff+i])
        oavg = sum(prices)/len(prices)
        pavg = sum(testResult)/len(testResult)
        avg += abs((testResult[i] - prices[cutoff+i])/prices[cutoff+i])
        lms += (testResult[i] - prices[cutoff+i])**2
    avg /= le
    lms /= 2
    print((pavg))
    print()
    print(avg*100, lms)
    plt.xlabel('Scaled Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()
    return mlp.predict(x)

def prediction(neighbors):
    countAll,openSum, closeSum =  0, 0, 0
    for x in range(len(neighbors)):
        countAll += (neighbors[x][0]*neighbors[x][1])
        openSum += neighbors[x][0]
        closeSum += neighbors[x][1]
    idwPrediction = (countAll/openSum)
    meanPrediction = (closeSum/float(3))
    return(idwPrediction, meanPrediction)

def getNeighbors(testInstance, dataList, k):
    length = len(testInstance)
    distances = []
    for x in range(len(dataList)-2):
        dist = euclideanDistance(testInstance, dataList, length, x)
        distances.append((dataList[x], dist))
        distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#Find euclidean distance between two instance                    
def euclideanDistance(instance1, dataList, length, check):
    distance = 0
    for x in range(length):
        if x == 0:
            distance += pow((instance1[x]- dataList[check][x]),2)
        else:
            distance += pow((instance1[x] - dataList[check+2][x]),2)
    return math.sqrt(distance)




if __name__ == "__main__":
	
	app.run()
