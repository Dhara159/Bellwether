from flask import Flask, request, render_template,session,make_response,url_for,redirect, jsonify
from flaskext.mysql import MySQL
from werkzeug import generate_password_hash, check_password_hash
from pytz import timezone
from datetime import timedelta
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from dateutil.relativedelta import relativedelta
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import json, threading, os, csv, quandl, math, operator, io
import pandas as pd
import datetime, requests
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

app.secret_key = 'super secret key'

mysql = MySQL()
 
app.config['MYSQL_DATABASE_USER'] = 'sql12230853'
app.config['MYSQL_DATABASE_PASSWORD'] = 'J5sp7m74qk'
app.config['MYSQL_DATABASE_DB'] = 'sql12230853'
app.config['MYSQL_DATABASE_HOST'] = 'sql12.freemysqlhosting.net'
mysql.init_app(app)

def preLoad():
	usTime = datetime.datetime.now()
	# currenttime = usTime.strftime("%H:%M:%S")
	today9am = usTime.replace(hour=9, minute=30, second=0, microsecond=0)
	if usTime > today9am:
		todayDate = usTime.date()
		conn = mysql.connect()
		cursor = conn.cursor()
		cursor.execute("SELECT * FROM dpdata WHERE rdate >='" + str(todayDate) + "' ")
		checkData = cursor.fetchall()
		checkData = list(checkData)
		conn.commit()
		cursor.execute("DELETE FROM dpdata WHERE pdate <'" + str(todayDate) + "' ")
		conn.commit()
		for x in range(len(checkData)):
			singleList = list(checkData[x])
			userId = singleList[0]
			buyPrice = cursor.execute("SELECT purchasePrice from orderdetails WHERE userId ='" + str(userId) + "' ")
			url = 'https://www.google.co.in/search?q=nse%3A'+ singleList[2] +'&oq=nse%3A'+ singleList[2] +'&aqs=chrome..69i57j69i60j69i58.6479j0j1&sourceid=chrome&ie=UTF-8'
			user_agent = 'Mozilla/5.0 (iPhone; CPU iPhone OS 5_0 like Mac OS X) AppleWebKit/534.46'
			req = urlopen(Request(str(url), data=None, headers={'User-Agent': user_agent}))
			soup = BeautifulSoup(req, 'html.parser')
			currentVal = soup.find('span',attrs={'class':'IsqQVc'})
			current = float((currentVal.text.strip()).replace(',', ''))
			if abs(current-singleList[6]) < 20:
				print("Time to sell")
			elif abs(current-singleList[7]) < 20:
				print("Time to buy")
		conn.close()
	app.run()

@app.after_request
def after_request(response):
    response.headers.add('Cache-Control', 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0')   
    return response

@app.route('/')
def hello():
	return 'hello'

@app.route('/signup',methods = ['POST', 'GET'])
def signup():
	return render_template("client_signup.html")
	
@app.route('/registered',methods = ['POST', 'GET'])
def registered():
	userName = request.form['userName']
	userEmail = request.form['userEmail']
	userPassword = request.form['userPassword']
	return insertNewUser(userName, userEmail, userPassword)

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

@app.route('/signin',methods=['GET','POST'])
def signin():
	if "userEmail" in session:
		return(redirect(url_for("client_page")))
	else :
		return render_template("client_login.html")

@app.route('/client_dashboard', methods = ["POST", "GET"])
def client_dashboard():
	userEmail = request.form['userEmail']
	userPassword = request.form['userPassword']
	conn = mysql.connect()
	cursor = conn.cursor()
	cursor.execute("SELECT userEmail,userPassword FROM users WHERE userEmail ='" + userEmail + "' ")
	data = cursor.fetchone()
	if data == None:
		return render_template("client_login.html")

	if userEmail == data[0] and userPassword == data[1]:
		conn.commit()
		session['userEmail'] = userEmail
		session['loged_in'] = True
		return(redirect(url_for("client_page")))

@app.route('/client_page')
def client_page():
	if "userEmail" in session:
		return render_template("client_dashboard.html")
	else:
		return("You are logged out!")
	
@app.route('/client_logout')
def client_logout():
	session.pop('userEmail', None)
	session.pop('loged_in',False)
	return(redirect(url_for("logout_page")))

@app.route('/logout_page')
def logout_page():
	return render_template("client_logout.html")

@app.route('/historic_data',methods=['GET','POST'])
def historic_data():
	if "userEmail" in session:
		return render_template("historic_data.html")
	else:
		return("You are logged out!")

@app.route('/historic_data_search',methods=['GET','POST'])
def historic_data_search():
	trade = request.form['trade']
	sdate = request.form['sdate']	
	edate = request.form['edate']
	fName = str(trade + '.csv')
	df = quandl.get("NSE/"+trade.upper(), authtoken="5GGEggAyyGa6_mVsKrxZ",start_date=sdate,end_date=edate)
	datalist = df.values.tolist()
	return render_template("historic_data_search.html",datalist=datalist, df=df.to_html(classes=["table", "thead-dark","table-bordered", "table-striped", "table-hover"]))

@app.route('/algorithms')
def algorithms():
	if "userEmail" in session:
		return render_template("algorithms.html")
	else:
		return("You are logged out!")

@app.route('/algorithm_prediction',methods=['GET','POST'])
def algorithmsrithm_prediction():
	if "userEmail" in session:
		trade = request.form['trade']
		return(knn("NSE/"+trade.upper()))

@app.route('/profit',methods=['GET','POST'])
def profit():
	if "userEmail" in session:
		conn = mysql.connect()
		cursor = conn.cursor()
		userEmail = str(session['userEmail'])
		cursor.execute("SELECT prediction,ssp,sbp from profit WHERE userEmail ='" + userEmail + "' ")
		data = list(cursor.fetchall())
		df = pd.DataFrame(data)
		print(df)
		fig = df[[0,1,2]].plot()
		file_path = "static/images/graph1.png"
		data = plt.savefig(file_path)
		conn1 = mysql.connect()
		cursor1 = conn1.cursor()
		cursor1.execute("SELECT prediction,purchasePrice,sellingPrice from profit WHERE userEmail ='" + userEmail + "' ")
		data1 = list(cursor1.fetchall())
		df1 = pd.DataFrame(data1)
		print(df1)
		fig1 = df1.plot.scatter(x=0, y=1,c=2)
		file_path1 = "static/images/graph2.png"
		data1 = plt.savefig(file_path1)
		return render_template("profit.html", data = data,df=df,file_path=file_path, data1 = data1,df1=df1,file_path1=file_path1)
	else:
		return("You are logged out!")

@app.route('/historic_graph',methods=['GET','POST'])
def historic_graph():
	if "userEmail" in session:
		return render_template("historic_graph.html")
	else:
		return("You are logged out!")
	
@app.route('/historic_graph_search',methods=['GET','POST'])
def historic_graph_search():
	trade = request.form['trade']
	attribute = request.form['attribute']
	sdate = request.form['sdate']
	edate = request.form['edate']
	fName = str(trade + '.csv')
	df = quandl.get("NSE/"+trade.upper(), authtoken="5GGEggAyyGa6_mVsKrxZ",start_date=sdate,end_date=edate)
	fig = df[[attribute]].plot()
	file_path = "static/images/mytable.png"
	data = plt.savefig(file_path)
	return render_template("historic_graph_search.html",df=df,data=data,file_path=file_path)

@app.route('/buy_sell')
def buy_sell():
	if "userEmail" in session:
		trades = ["HDFC","BIOCON","PNB","DLF","AKZOINDIA","ASHOKLEY","ASIANPAINT","ASTRAZEN","AUROPHARMA","AXISBANK","BAJAJCORP","BPCL","CENTRALBK","DENABANK","DISHTV","GAIL","GLENMARK","GODREJCP","GODREJIND","GPPL","HEROMOTOCO","IDBI","JETAIRWAYS","JUSTDIAL","ONGC"]
		mat = {}
		for trade in trades:
			todayTime = datetime.datetime.now()
			yesterday = todayTime - timedelta(1)
			latest = yesterday.strftime("%Y-%m-%d")
			old = ((datetime.date.today()-relativedelta(months=+3))).strftime("%Y-%m-%d")
			df = quandl.get("NSE/"+trade.upper(), authtoken="5GGEggAyyGa6_mVsKrxZ",start_date=old, end_date=latest)
			datalist = df.values.tolist()	
			oldData = datalist[0][4]
			latestData = datalist[len(datalist)-1][4]
			finalResult = oldData-latestData
			if finalResult > 0:
				mat[trade] = finalResult
		maxValTrade = max(mat.items(), key=operator.itemgetter(1))[0]
		trade ="NSE/"+(maxValTrade.upper())
		return render_template("buy_sell.html",tradeToBuy=maxValTrade)
	else:
		return("You are logged out!")

@app.route('/buy_sell_confirm', methods=['GET','POST'])
def buy_sell_confirm():
	if "userEmail" in session:
		trade = request.form['trade']
		volume = int(request.form['volume'])
		cprice = request.form['cprice']
		total = request.form['total']
		predictedPrice = knn("NSE/"+trade.upper())
		# print(predictedPrice)
		conn1 = mysql.connect()
		cursor1 = conn1.cursor()
		userEmail = str(session['userEmail'])
		cursor1.execute("SELECT virtualMoney,userId from users WHERE userEmail ='" + userEmail + "' ")
		virtualMoney = cursor1.fetchone()
		userId = int(virtualMoney[1])
		conn1.commit()
		purchasePrice = total
		sellingPrice = 0
		virtualMoney = int(virtualMoney[0]) - int(float(total))
		cursor1.execute("UPDATE users SET virtualMoney='" + str(virtualMoney) + "', userId= '"+ str(userId) +"' WHERE userEmail='"+ str(userEmail) +"' ")
		conn1.commit()
		usTime = datetime.datetime.now()
		currenttime = usTime.strftime("%Y-%m-%d")
		conn = mysql.connect()
		cursor = conn.cursor()
		cursor.execute("INSERT INTO orderdetails (userId,tradeName,dates,purchasePrice,sellingPrice,volume) VALUES (%s,%s,%s,%s,%s,%s)", (str(userId),trade,str(currenttime),float(purchasePrice),float(sellingPrice),float(volume)))
		cursor.execute("INSERT INTO profit (userEmail,trade,prediction,ssp,sbp,purchasePrice,sellingPrice) VALUES (%s,%s,%s,%s,%s,%s,%s)", (userEmail,trade,str(predictedPrice[0]),float(predictedPrice[1]),float(predictedPrice[2]),float(purchasePrice),float(sellingPrice)))
		conn.commit()
		return render_template("buy_sell_confirm.html", trade=trade,volume=volume,total=total,cprice=cprice,purchasePrice=purchasePrice,sellingPrice=sellingPrice)
	else:
		return("You are logged out!")


@app.route('/sell_buy')
def sell_buy():
	userEmail = str(session["userEmail"])
	conn = mysql.connect()
	cursor = conn.cursor()
	cursor.execute("SELECT userId FROM users WHERE userEmail ='" + userEmail +"' ")
	userId = cursor.fetchone()
	cursor.execute("SELECT tradeName FROM orderdetails WHERE userId = '" + str(userId[0]) + "'")
	tradeName = list(cursor.fetchall())
	sellMatrix = []
	for trade in tradeName:
		tradeData = []
		tradeData.append(trade[0])
		cursor.execute("SELECT purchasePrice FROM orderdetails WHERE tradeName = '" +trade[0]+ "'")
		price = list(cursor.fetchone())
		purchasePrice = np.mean(price)
		tradeData.append(purchasePrice)
		url = 'https://www.google.co.in/search?q=nse%3A'+trade[0]+'&oq=nse%3A'+trade[0]+'&aqs=chrome..69i57j69i60j69i58.6479j0j1&sourceid=chrome&ie=UTF-8'
		user_agent = 'Mozilla/5.0 (iPhone; CPU iPhone OS 5_0 like Mac OS X) AppleWebKit/534.46'
		req = urlopen(Request(str(url), data=None, headers={'User-Agent': user_agent}))
		soup = BeautifulSoup(req, 'html.parser')
		currentVal = soup.find('span',attrs={'class':'IsqQVc'})
		currentSell = float((currentVal.text.strip()).replace(',', ''))
		tradeData.append(currentSell)
		sellMatrix.append(tradeData)
	sellList = []
	for everyTrade in sellMatrix:
		tradeList = []
		tName = everyTrade[0]
		buyPrice = everyTrade[1]
		sellPrice = everyTrade[2]
		userEmail = str(session['userEmail'])
		cursor.execute("SELECT ssp from knnprediction WHERE userEmail = '"+ userEmail +"' AND trade = 'NSE/"+ tName+"' ORDER BY ssp ASC ")
		ssp = cursor.fetchone()
		if sellPrice > buyPrice and sellPrice >= ssp[0]:
			priceDiff = sellPrice - buyPrice
			tradeList.append(tName)
			tradeList.append(priceDiff)
			sellList.append(tradeList)
			sellTrade = (max((a,b) for (a,b) in sellList))
		else:
			sellTrade = ['-', 0]
	return render_template("sell.html", trade=sellTrade[0],sellPrice=sellTrade[1])

@app.route('/sell_buy_confirm', methods=['GET','POST'])
def sell_buy_confirm():
	if "userEmail" in session:
		trade = request.form['trade']
		volume = int(request.form['volume'])
		cprice = request.form['cprice']
		total = request.form['total']
		predictedPrice = knn("NSE/"+trade.upper())
		conn1 = mysql.connect()
		cursor1 = conn1.cursor()
		userEmail = str(session['userEmail'])
		cursor1.execute("SELECT virtualMoney,userId from users WHERE userEmail ='" + userEmail + "' ")
		virtualMoney = cursor1.fetchone()
		userId = int(virtualMoney[1])
		conn1.commit()
		purchasePrice =0
		sellingPrice = total
		virtualMoney = int(virtualMoney[0]) + int(total)
		cursor1.execute("UPDATE users SET virtualMoney='" + str(virtualMoney) + "', userId= '"+ str(userId) +"' WHERE userEmail='"+ str(userEmail) +"' ")
		conn1.commit()
		usTime = datetime.datetime.now()
		currenttime = usTime.strftime("%Y-%m-%d")
		conn = mysql.connect()
		cursor = conn.cursor()
		cursor.execute("INSERT INTO orderdetails (userId,tradeName,dates,purchasePrice,sellingPrice,volume) VALUES (%s,%s,%s,%s,%s,%s)", (str(userId),trade,str(currenttime),float(purchasePrice),float(sellingPrice),float(volume)))
		cursor.execute("INSERT INTO profit (userEmail,trade,prediction,ssp,sbp,purchasePrice,sellingPrice) VALUES (%s,%s,%s,%s,%s,%s,%s)", (userEmail,trade,str(predictedPrice[0]),float(predictedPrice[1]),float(predictedPrice[2]),float(purchasePrice),float(sellingPrice)))
		conn.commit()
		return render_template("buy_sell_confirm.html", trade=trade,volume=volume,total=total,cprice=cprice,purchasePrice=purchasePrice,sellingPrice=sellingPrice)
	else:
		return("You are logged out!")

@app.route('/profile')
def profile():
	if "userEmail" in session:
		userEmail=str(session['userEmail'])
		conn =mysql.connect()
		cursor = conn.cursor()
		cursor.execute("SELECT userId,userName,userEmail,virtualMoney FROM users WHERE userEmail = '"+ userEmail +"' ")
		data = cursor.fetchone()
		conn.commit()
		money_spend=1000000 - data[3]
		userId = str(data[0])
		cursor =conn.cursor()
		cursor.execute("SELECT SUM(volume) from orderdetails WHERE userId= '"+ userId +"' ")
		volume = cursor.fetchone()
		return render_template("profile.html",data=data,volume=volume[0], money_spend=money_spend)
	else:
		return("You are logged out!")
	
@app.route('/profile_edit')
def profile_edit():
	if "userEmail" in session:
		return render_template("profile_edit.html")
	else:
		return("You are logged out!")

@app.route("/order_details",methods=['POST','GET'])
def order_details():
	if "userEmail" in session:
		userEmail=str(session['userEmail'])
		conn = mysql.connect()
		cursor = conn.cursor()
		cursor.execute(" SELECT userId from users WHERE userEmail ='" + userEmail + "' ")
		userId = cursor.fetchone()
		print(userId)
		cursor.execute("SELECT id,userId,tradeName,dates,purchasePrice,sellingPrice,volume FROM orderdetails WHERE userId = '"+ str(userId[0]) +"' ")
		conn.commit()
		return render_template("order_details.html", items=cursor.fetchall())
	else:
		return("You are logged out!")

@app.route("/order_details_search",methods=['POST','GET'])	
def order_details_search():
	if "userEmail" in session:
		userEmail = str(session["userEmail"])
		trade = request.form['trade']
		conn = mysql.connect()
		cursor = conn.cursor()
		cursor.execute("SELECT userId FROM users WHERE userEmail ='" + userEmail +"' ")
		userId = cursor.fetchone()
		cursor.execute("SELECT id,userId,tradeName,dates,purchasePrice,sellingPrice,volume FROM orderdetails WHERE tradeName ='" + trade + "' AND userId = '" + str(userId[0]) + "'")
		conn.commit()
		return render_template("order_details.html", items=cursor.fetchall())
	else:
		return("You are logged out!")

@app.route('/print_items')
def print_items():
    cursor = db.execute('SELECT column1,column2,column3,column4 FROM tfablename')
    return render_template('print_items.html', items=Units.query.all())

@app.route("/live_feeding", methods=['GET', 'POST'])
def live_feeding():
	if "userEmail" in session:
		trades = ["HDFC","BIOCON","PNB","DLF","AKZOINDIA","ASHOKLEY","ASIANPAINT","ASTRAZEN","AUROPHARMA","AXISBANK","BAJAJCORP","BPCL","CENTRALBK","DENABANK","DISHTV","GAIL","GLENMARK","GODREJCP","GODREJIND","GPPL","HEROMOTOCO","IDBI","NAUKRI","JETAIRWAYS","JUSTDIAL","ONGC"]
		w, h = 5, len(trades);
		Matrix = [[0 for x in range(w)] for y in range(h)]
		for i in range(len(trades)):
			url = 'https://www.google.co.in/search?q=nse%3A'+trades[i]+'&oq=nse%3A'+trades[i]+'&aqs=chrome..69i57j69i60j69i58.6479j0j1&sourceid=chrome&ie=UTF-8'
			user_agent = 'Mozilla/5.0 (iPhone; CPU iPhone OS 5_0 like Mac OS X) AppleWebKit/534.46'
			req = urlopen(Request(str(url), data=None, headers={'User-Agent': user_agent}))
			soup = BeautifulSoup(req, 'html.parser')
			currentVal = soup.find('span',attrs={'class':'IsqQVc'})
			name_box = soup.find_all('td', attrs={'class':'iyjjgb'})
			if len(name_box) == 0:
				Matrix[i][4] = trades[i]
			else:
				openVal = name_box[0].text.strip()
				highVal = name_box[1].text.strip()
				lowVal = name_box[2].text.strip()
				current = currentVal.text.strip()
				Matrix[i][0] = openVal
				Matrix[i][1] = highVal
				Matrix[i][2] = lowVal
				Matrix[i][3] = current
				Matrix[i][4] = trades[i]
		return(render_template("live_feeding.html", matrix = Matrix, matLen = len(Matrix)))		

@app.route("/live_feeding_search", methods=['GET', 'POST'])
def live_feeding_search():
	if "userEmail" in session:
		trades = request.form["trade"]
		print(trades)
		w = 5
		Matrix = [0 for x in range(w)] 
		for i in range(len(Matrix)):
			url = 'https://www.google.co.in/search?q=nse%3A'+trades+'&oq=nse%3A'+trades+'&aqs=chrome..69i57j69i60j69i58.6479j0j1&sourceid=chrome&ie=UTF-8'
			user_agent = 'Mozilla/5.0 (iPhone; CPU iPhone OS 5_0 like Mac OS X) AppleWebKit/534.46'
			req = urlopen(Request(str(url), data=None, headers={'User-Agent': user_agent}))
			soup = BeautifulSoup(req, 'html.parser')
			currentVal = soup.find('span',attrs={'class':'IsqQVc'})
			name_box = soup.find_all('td', attrs={'class':'iyjjgb'})
			if len(name_box) == 0:
				Matrix[i][4] = trades
			else:
				openVal = name_box[0].text.strip()
				highVal = name_box[1].text.strip()
				lowVal = name_box[2].text.strip()
				current = currentVal.text.strip()
				Matrix[0] = openVal
				Matrix[1] = highVal
				Matrix[2] = lowVal
				Matrix[3] = current
				Matrix[4] = trades
		return(render_template("live_feeding_search.html", matrix = Matrix))	

@app.route("/knn")
def knn(tradeName):
    k = 3
    startDate = (datetime.datetime.now()).strftime("%Y-%m-%d")
    endDate = ((datetime.date.today()-relativedelta(months=+3))).strftime("%Y-%m-%d")
    df = quandl.get(tradeName, authtoken="JMRWwixg-zh5jGHGnKzn",start_date=endDate,end_date=startDate)
    df1 = df[['Open','Close']]
    dataList = df1.values.tolist()
    testInstance = dataList[len(dataList)-2]
    neighbors = getNeighbors(testInstance, dataList, k)
    idwPrediction, meanPrediction  = prediction(neighbors)
    return(bb(tradeName,endDate,startDate, idwPrediction,meanPrediction))

@app.route("/bb")
def bb(tradeName,endDate,startDate, idwPrediction, meanPrediction):
	df = quandl.get(tradeName, authtoken="5GGEggAyyGa6_mVsKrxZ",start_date=endDate)
	window=20
	no_of_std=2
	rolling_mean = df['Close'].rolling(window).mean()
	rolling_std = df['Close'].rolling(window).std()
	df['Bollinger High'] = rolling_mean + (rolling_std * no_of_std)
	df['Bollinger Low'] = rolling_mean - (rolling_std * no_of_std)
	df['Short'] = None
	df['Long'] = None
	df['Position'] = None
	for row in range(len(df)):
		if (df['Close'].iloc[row] > df['Bollinger High'].iloc[row]) and (df['Close'].iloc[row-1] < df['Bollinger High'].iloc[row-1]):
			df['Position'].iloc[row] = -1
		if (df['Close'].iloc[row] < df['Bollinger Low'].iloc[row]) and (df['Close'].iloc[row-1] > df['Bollinger Low'].iloc[row-1]):
			df['Position'].iloc[row] = 1
	df['Position'].fillna(method='ffill',inplace=True)
	df['Market Return'] = np.log(df['Close'] / df['Close'].shift(1))
	df['Strategy Return'] = df['Market Return'] * df['Position']
	df = df[np.isfinite(df['Strategy Return'])]
	df1 = df.loc[df['Strategy Return'] == max(df['Strategy Return'])]
	df2 = df.loc[df['Strategy Return'] == min(df['Strategy Return'])]
	df['Strategy Return'].cumsum().plot()
	df[['Close','Bollinger High','Bollinger Low']].plot()
	userEmail=session["userEmail"]
	resultDate = datetime.date.today() + datetime.timedelta(days=1)
	conn =mysql.connect()
	cursor = conn.cursor()
	suggestedBuying = ((df1['Close'].values)[0])
	suggestedSelling = ((df2['Close'].values)[0])
	cursor.execute("INSERT INTO knnprediction (userEmail,trade,pdate,rdate,prediction,ssp,sbp) VALUES (%s, %s, %s, %s, %s, %s, %s)", (userEmail,  tradeName, startDate, resultDate, idwPrediction,str(suggestedSelling), str(suggestedBuying)))
	conn.commit()
	cursor = conn.cursor()
	pdata = cursor.execute("SELECT * FROM knnprediction WHERE userEmail = '"+ userEmail +"' ")
	conn.commit()
	result = [idwPrediction,str(suggestedSelling), str(suggestedBuying)]
	return result
	# return render_template("algorithm_prediction.html",items=list(cursor.fetchall()))

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
    # dates = []
    # prices = []
    # with open('AAPL_last_month.csv','r') as csvfile:
    #     csvFileReader = csv.reader(csvfile)
    #     next(csvFileReader)
    #     for row in csvFileReader:
    #         dates.append(int(row[0].split('-')[0]))
    #         prices.append(float(row[4]))
    # predictedPrice=predictPricesSvrlinear(dates,prices, 167)
    # return("SVRLinear predicted price: %f" % (predictedPrice))
    dates = []
    cprices = []
    oprices = []
    startDate = (datetime.datetime.now()).strftime("%Y-%m-%d")
    endDate = ((datetime.date.today()-relativedelta(days=+10))).strftime("%Y-%m-%d")
    print(startDate)
    print(endDate)
    df = quandl.get('NSE/PNB', authtoken="5GGEggAyyGa6_mVsKrxZ",start_date=endDate,end_date=startDate)
    print(df)
    cprices = df['Close']
    oprices = df['Open']
    print(cprices)
    print(oprices)
    print(dates)
    predictedPrice = predictPricesSvrlinear(cprices,oprices, 99)
    return(str(predictedPrice))

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

@app.route("/getCurrentPrice",methods=['POST'])	
def getCurrentPrice():
	tradeName = (request.form["tradeName"]).upper()
	url = 'https://www.google.co.in/search?q=nse%3A'+tradeName+'&oq=nse%3A'+tradeName+'&aqs=chrome..69i57j69i60j69i58.6479j0j1&sourceid=chrome&ie=UTF-8'
	user_agent = 'Mozilla/5.0 (iPhone; CPU iPhone OS 5_0 like Mac OS X) AppleWebKit/534.46'
	req = urlopen(Request(str(url), data=None, headers={'User-Agent': user_agent}))
	soup = BeautifulSoup(req, 'html.parser')
	currentVal = soup.find('span',attrs={'class':'IsqQVc'})
	current = float((currentVal.text.strip()).replace(',', ''))
	return json.dumps(float(current))

# @app.route("/getbest")
# def getbest():
# 	trades = ["HDFC","BIOCON","PNB","DLF","AKZOINDIA","ASHOKLEY","ASIANPAINT","ASTRAZEN","AUROPHARMA","AXISBANK","BAJAJCORP","BPCL","CENTRALBK","DENABANK","DISHTV","GAIL","GLENMARK","GODREJCP","GODREJIND","GPPL","HEROMOTOCO","IDBI","NAUKRI","JETAIRWAYS","JUSTDIAL","ONGC"]
# 	mat = {}
# 	for trade in trades:
# 		todayTime = datetime.datetime.now()
# 		yesterday = todayTime - timedelta(1)
# 		latest = yesterday.strftime("%Y-%m-%d")
# 		old = ((datetime.date.today()-relativedelta(months=+3))).strftime("%Y-%m-%d")
# 		df = quandl.get("NSE/"+trade.upper(), authtoken="5GGEggAyyGa6_mVsKrxZ",start_date=old, end_date=latest)
# 		datalist = df.values.tolist()	
# 		oldData = datalist[0][4]
# 		latestData = datalist[len(datalist)-1][4]
# 		finalResult = oldData-latestData
# 		if finalResult > 0:
# 			mat[trade] = finalResult
# 	print(mat)
# 	maxValTrade = max(mat.items(), key=operator.itemgetter(1))[0]
# 	return("trades")



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

def predictPricesSvrlinear(cprices,oprices,x):
	cprices= np.reshape(cprices,(len(cprices),1))
	#dates = dates.values.reshape(8,1)
	svr_lin=SVR(kernel='linear', C=1)
	#svr_poly=SVR(kernel='poly', C=0.02, degree=2)
	svr_lin.fit(cprices,oprices)
	plt.scatter(cprices,oprices, color='black', label='Data')
	plt.plot(cprices,svr_lin.predict(cprices),color='red',label='Linear Model')
	plt.xlabel('Close Prices')
	plt.ylabel('Open prices')
	plt.legend()
	plt.show()
	return(svr_lin.predict(x)[0])

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
                  
def euclideanDistance(instance1, dataList, length, check):
    distance = 0
    for x in range(length):
        if x == 0:
            distance += pow((instance1[x]- dataList[check][x]),2)
        else:
            distance += pow((instance1[x] - dataList[check+2][x]),2)
    return math.sqrt(distance)

if __name__ == "__main__":
	preLoad()
