import csv
import gzip
import numpy as np
import pandas as pd
import datetime as datetime
from collections import deque
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.finance import candlestick2_ohlc


class PredictExchange:
    def __init__(self, fig, ax, NPred, delay, s, Test):
        # predicted data
        self.ND = NPred
        self.delay = delay
        # uration of season (in days)
        self.s = s
        # setup test data
        self.Test = Test

        self.fig = fig
        self.ax = ax
        self.fig.suptitle('Machine Learning')

        self.Ndot = 20

        self.alphaArr = np.logspace(-2.95, -1.05, num = self.Ndot)
        self.Q1d = np.zeros((self.Ndot,1), dtype=float)
        self.Q2d = np.zeros((self.Ndot,self.Ndot), dtype=float)
        self.Q3d = np.zeros((self.Ndot,1), dtype=float)


    def init_data(self):
#        dt = np.dtype([('Date', 'datetime64[D]', 1), ('Price', np.float, 5)])
        # load data from csv
        try:
            csv_date = np.genfromtxt('btcDayDate.csv', delimiter=",", dtype='datetime64[D]')
        except IOError:
            csv_date = np.array([])
            print('Load error! (')

        try:
            csv_price = np.genfromtxt('btcDayPrice.csv', delimiter=",", dtype=np.float)
        except IOError:
            csv_price = np.array([])
            print('Load error!')

        # in numpy datetime64 format
        DayDate = np.asarray(csv_date[:], dtype='datetime64[D]')
        # Day price has 5 columns in numpy array:
        # [average, opens, closes, highs, lows]
        self.DayPrice = csv_price.astype(np.float)
        self.Nt = self.DayPrice.shape[0]
        self.Ndel = self.Nt - self.delay 
        self.Y = self.DayPrice[0:self.Nt,0]
        print('{} days'.format(self.Y.shape[0]))

        self.DatePred = np.array([], dtype='datetime64[D]')
        TempDate = DayDate[0]
        for i in range(0, self.Nt+self.ND):
            TempDate = TempDate + np.timedelta64(1, 'D')
            self.DatePred = np.append(self.DatePred, TempDate)

        if self.Test == True:
            self.set_theory_data()


    def get_iter(self, x, pos):
        try:
            return self.StrDate[int(x)]
        except IndexError:
            return ''


    # plot functions for each model
    def draw_stock_plot(self):
        # numpy datetime64 ->(with pandas)-> datetime.datetime
        self.StrDate = []
        for i in np.nditer(self.DatePred):
            self.StrDate.append(pd.to_datetime(str(i)).strftime('%d.%m.%Y'))
        # stock graph
        candlestick2_ohlc(self.ax[0,0], self.DayPrice[:,1], self.DayPrice[:,3], self.DayPrice[:,4], self.DayPrice[:,2], width=0.6)
        self.ax[0,0].xaxis.set_minor_locator(ticker.MaxNLocator(30))
        self.ax[0,0].xaxis.set_major_locator(ticker.MaxNLocator(7))
        self.ax[0,0].xaxis.set_major_formatter(ticker.FuncFormatter(self.get_iter))
        self.fig.autofmt_xdate()
        self.fig.tight_layout()
        self.ax[0,0].set_ylabel('Btc (RUB)', size=20)
        self.ax[0,0].set_ylim([50, 1500])
        self.fig.subplots_adjust(top=0.9)


#    def draw_simple_model(self, YT1, YT2):
#        self.ax[0,1].plot(self.DatePred[0:self.Nt], self.DayPrice[:,0], 'r', label="History")
#        self.ax[0,1].plot(self.DatePred[self.Nt-self.ND:self.Nt], YT1, 'g', label="Averg")
#        self.ax[0,1].plot(self.DatePred[self.Nt-self.ND:self.Nt], YT2, 'b', label="Exp")   
#        self.ax[0,1].set_ylim([50,1500])  
#        self.ax[0,1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=3, mode="expand", borderaxespad=0.)


    def draw_predict_model(self, YT1, YT2):
        self.ax[0,1].plot(self.DatePred[self.Ndel:self.Nt], self.DayPrice[self.Ndel:self.Nt,0], 'r', label="History")
        self.ax[0,1].plot(self.DatePred[self.Ndel:self.Nt+self.ND], YT1[self.Ndel:self.Nt+self.ND], 'b', label="Brown")    
        self.ax[0,1].plot(self.DatePred[self.Ndel:self.Nt+self.ND], YT2[self.Ndel:self.Nt+self.ND], 'g', label="Holt")  
        self.ax[0,1].set_ylim([600,1400])  
        self.ax[0,1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)        


    def draw_season_model(self, YT1, YT2, YT3):
        self.ax[1,0].plot(self.DatePred[self.Ndel:self.Nt], self.DayPrice[self.Ndel:self.Nt,0], 'r', label="History")
        self.ax[1,0].plot(self.DatePred[self.Ndel:self.Nt+self.ND], YT1[self.Ndel:self.Nt+self.ND], 'b', label="HoltAdd")  
        self.ax[1,0].set_ylim([600,1400])  
        self.ax[1,0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)  

        self.ax[1,1].plot(self.DatePred[self.Ndel:self.Nt], self.DayPrice[self.Ndel:self.Nt,0], 'r', label="History")
        self.ax[1,1].plot(self.DatePred[self.Ndel:self.Nt+self.ND], YT2[self.Ndel:self.Nt+self.ND], 'b', label="WintLin")    
        self.ax[1,1].plot(self.DatePred[self.Ndel:self.Nt+self.ND], YT3[self.Ndel:self.Nt+self.ND], 'g', label="WintExp")    
        self.ax[1,1].set_ylim([600,1400])  
        self.ax[1,1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)  


    # most simple average model
    def simple_model(self):
        N = self.Nt - self.ND
        YT = np.zeros((self.ND,), dtype=np.float)
        YT[0] = 1/(1 + N)*np.sum(self.Y[N:self.Nt])
        for i in range(1, self.ND):
            YT[i] = YT[i-1] + 1/(1 + N) * (self.Y[N+i-1] - YT[i-1])
        return YT


    # simple exponential model
    def exp_model(self, alpha):
        N = self.Nt - self.ND
        YT = np.zeros((self.ND,), dtype=np.float)
        YT[0] = self.Y[N]
        for i in range(1, self.ND):
            YT[i] = YT[i-1] + alpha * (self.Y[N+i-1] - YT[i-1])
        return YT


    # find argmin for 1 alpha coefficient
    def exp_contrl(self):
        N = self.Nt - self.ND
        for i in range(0, self.Ndot):
            YT = self.exp_model(self.alphaArr[i])
            self.Q1d[i] = np.linalg.norm(YT - self.Y[N:self.Nt])
        alphaMin = self.alphaArr[np.argmin(self.Q1d)]
        print('Alpha in Exp model control = {}'.format(alphaMin))
        return alphaMin


    # find argmin for 1 alpha coefficient
    def brown_contrl(self):
        for i in range(0, self.Ndot):
            YT = self.holt_model(self.alphaArr[i], 1)
            self.Q3d[i] = np.linalg.norm(YT[self.Ndel:self.Nt] - self.Y[self.Ndel:self.Nt])
        alphaMin = self.alphaArr[np.argmin(self.Q3d)]
        print('Alpha in Brown model control = {}; Q = {}'.format(alphaMin, self.Q3d[np.argmin(self.Q3d)]))
        return alphaMin

    # Holt model without seasons 
    def holt_model(self, alpha1, alpha2):
        YT = np.zeros((self.Nt+self.ND,), dtype=np.float)
        a = np.zeros((self.Nt,), dtype=np.float)
        b = np.zeros((self.Nt,), dtype=np.float)

        YT[0:self.ND] = self.Y[0:self.ND]
        a[0] = 0
        b[0] = 0

        for i in range(1, self.Nt):
            a[i] = alpha1*self.Y[i] + (1 - alpha1)*(a[i-1] + b[i-1])
            b[i] = alpha2*(a[i] - a[i-1]) + (1 - alpha2)*b[i-1]
        for i in range(0,self.Nt):
            YT[i+self.ND] = a[i] +  b[i] * self.ND
        return YT


    # find argmin for alpha1 & alpha2 coefficients
    def holt_contr(self):
        for i in range(0, self.Ndot):
            for j in range(0, self.Ndot):
                YT = self.holt_model(self.alphaArr[i], self.alphaArr[j])
                self.Q2d[i, j] = np.linalg.norm(YT[self.Ndel:self.Nt] - self.Y[self.Ndel:self.Nt])
        alphaMin1 = self.alphaArr[np.unravel_index(self.Q2d.argmin(), self.Q2d.shape)[0]]
        alphaMin2 = self.alphaArr[np.unravel_index(self.Q2d.argmin(), self.Q2d.shape)[1]]

        print('Holt model: alpha1 = {}; alpha2 = {}'.format(alphaMin1, alphaMin2))
        return alphaMin1, alphaMin2    


    # Holt linear model with seasons
    def holt_season(self, alpha1, alpha2, alpha3):
        YT = np.zeros((self.Nt+self.ND,), dtype=np.float)
        a = np.zeros((self.Nt,), dtype=np.float)
        b = np.zeros((self.Nt,), dtype=np.float)
        tau = np.zeros((self.Nt,), dtype=np.float)

        YT[0:self.Nt] = self.Y[0:self.Nt]

        for i in range(1, self.Nt):
            a[i] = alpha1*self.Y[i] + (1 - alpha1)*(a[i-1] + b[i-1])
            b[i] = alpha2*(a[i] - a[i-1]) + (1 - alpha2)*b[i-1]
        for i in range (self.s, self.Nt):
            tau[i] = alpha3*(self.Y[i] - a[i]) + (1 - alpha3)*tau[i - self.s]
        for i in range(0,self.Nt):
            YT[i+self.ND] = a[i] +  b[i] * self.ND + tau[i + (self.ND % self.s) - self.s]
            
        return YT  


    # find argmin for alpha3 coefficient
    def holt_season_contr(self, alpha1, alpha2):
        for i in range(0, self.Ndot):
            YT = self.holt_season(alpha1, alpha2, self.alphaArr[i])
            self.Q2d[i, 0] = np.linalg.norm(YT[self.Ndel:self.Nt] - self.Y[self.Ndel:self.Nt])
        alphaMin = self.alphaArr[np.argmin(self.Q2d[:,0])]

        print('Holt lin season model: alpha = {}'.format(alphaMin))
        return alphaMin


    # Winters exponential model with seasons
    def winters_exp(self, alpha1, alpha2, alpha3):
        YT = np.zeros((self.Nt+self.ND,), dtype=np.float)
        a = np.ones((self.Nt,), dtype=np.float)
        r = np.ones((self.Nt,), dtype=np.float)
        tau = np.ones((self.Nt+self.ND+self.s,), dtype=np.float)

        YT[0:self.Nt] = self.Y[0:self.Nt]

        for i in range(1, self.Nt):
            a[i] = alpha1*self.Y[i]/tau[i] + (1 - alpha1)*(a[i-1] * r[i-1])
            r[i] = alpha2*a[i]/a[i-1] + (1 - alpha2)*r[i-1]
            tau[i+self.s] = alpha3*self.Y[i-1]/a[i-1] + (1 - alpha3)*tau[i]
        for i in range(0,self.Nt):
            YT[i+self.ND] = a[i] * (r[i] ** (self.ND)) * tau[i + (self.ND % self.s)]
        return YT          


    # find argmin for alpha3 coefficient
    def winters_exp_contr(self, alpha1, alpha2):
        for i in range(0, self.Ndot):
            YT = self.winters_exp(alpha1, alpha2, self.alphaArr[i])
            self.Q2d[i, 0] = np.linalg.norm(YT[self.Ndel:self.Nt] - self.Y[self.Ndel:self.Nt])
        alphaMin = self.alphaArr[np.argmin(self.Q2d[:,0])]

        print('Winters exponential season model: alpha = {}'.format(alphaMin))
        return alphaMin


    # Winters linear model with seasons
    def winters_lin(self, alpha1, alpha2, alpha3):
        YT = np.zeros((self.Nt*self.ND,), dtype=np.float)
        a = np.ones((self.Nt,), dtype=np.float)
        b = np.ones((self.Nt,), dtype=np.float)
        tau = np.ones((self.Nt+self.ND+self.s,), dtype=np.float)

        YT[0:self.Nt] = self.Y[0:self.Nt]

        for i in range(1, self.Nt):
            a[i] = alpha1*self.Y[i]/tau[i] + (1 - alpha1)*(a[i-1] + b[i-1])
            b[i] = alpha2*(a[i]-a[i-1]) + (1 - alpha2)*b[i-1]
            tau[i+self.s] = alpha3*self.Y[i-1]/a[i-1] + (1 - alpha3)*tau[i]
        for i in range(0,self.Nt):
            YT[i+self.ND] = (a[i] + b[i]*self.ND) * tau[i + (self.ND % self.s)]
        return YT      


    # find argmin for alpha3 coefficient
    def winters_lin_contr(self, alpha1, alpha2):
        for i in range(0, self.Ndot):
            YT = self.winters_lin(alpha1, alpha2, self.alphaArr[i])
            self.Q2d[i, 0] = np.linalg.norm(YT[self.Ndel:self.Nt] - self.Y[self.Ndel:self.Nt])
        alphaMin = self.alphaArr[np.argmin(self.Q2d[:,0])]

        print('Winters linear season model: alpha = {}'.format(alphaMin))
        return alphaMin


    # replace stock data with line+sin oscillations
    def set_theory_data(self):
        Arr = np.linspace(0, self.Nt/2, num=self.Nt)
        self.DayPrice[:,0] = 40000 + 500*Arr + 750*np.sin(Arr)


    def Main(self):
        self.init_data()
        self.draw_stock_plot()
        print(self.Nt, self.ND)
        #print(len(self.StrDate), self.DatePred.shape, self.DayPrice.shape)
        # simple models
        YT_sim = self.simple_model()
        alphaMin = self.exp_contrl()
        YT_exp = self.exp_model(alphaMin)
        
        # without season     
        YT_brown = self.holt_model(1 - (1-alphaMin) ** 2, 1) 
        #Ybr = self.holt_model(1 - (1-alphaMin) ** 2, 1)
        Ybr = self.holt_model(0.9, 1)
        Qbr = np.linalg.norm(Ybr[self.Ndel:self.Nt] - self.Y[self.Ndel:self.Nt])
        print('My brown alpha = {}; Q = {}'.format(1-(1-alphaMin)**2, Qbr))

        alphaMinHolt = self.brown_contrl()
        YT_bestb = self.holt_model(0.9, 1)

        alphaMin1, alphaMin2 = self.holt_contr()
        YT_holt = self.holt_model(alphaMin1, alphaMin2)

        # lin holt season
        alphaMinHolt = self.holt_season_contr(alphaMin1, alphaMin2)
        YT_holt_season = self.holt_season(alphaMin1, alphaMin2, alphaMinHolt)

        # lin winters season
        alphaMinWintLin = self.winters_lin_contr(alphaMin1, alphaMin2)
        YT_winters_lin = self.winters_lin(alphaMin1, alphaMin2, alphaMinWintLin)

        # exp winters season
        alphaMinWintExp = self.winters_exp_contr(alphaMin1, alphaMin2)
        YT_winters_exp = self.winters_exp(alphaMin1, alphaMin2, alphaMinWintExp)
    
    #    self.draw_simple_model(YT_sim, YT_exp)
        self.draw_predict_model(YT_brown, YT_holt)
        self.draw_season_model(YT_holt_season, YT_winters_lin,YT_winters_exp)
        plt.show()



# converting data in Day time format
def convert_datetime(delta, is_count):
    dt = '|S32'
    # list
    data_list = []
    # deque
    data_deque = deque([], maxlen=int(delta)+1)
    row_count = 0
    # number of lines in csv
    row_N = 29599492
    #row_N = 5661
    print('Open csv file')
    with gzip.open('btceUSD.csv.gz', 'rt') as f:
        # row_N counts size csv, work faster
        if is_count:
            row_N = sum(1 for line in f)
            print(row_N)
        else:
            for line in f:
                if (row_count == row_N - delta):
                    print('Writing in data_list')
                if (row_count >= row_N - delta):
                    if (row_count % 5 == 0):
                        data_deque.append(line.rstrip('\n').split(','))
                row_count += 1

    N = delta
    print('{} rows in data list'.format(N))
    print('Rows in csv: {} - sum (1 for line in f); {} - row count'.format(row_N, row_count))
    
    StartDateTime = np.array(data_deque.popleft()[0], dtype=np.int64) \
                    .astype(np.int64).astype('datetime64[s]').astype('datetime64[D]')
    CurrentDateTime = np.array(data_deque.pop()[0], dtype=np.int64) \
                        .astype('datetime64[s]').astype('datetime64[D]')

    print('Start time: {}; Current time: {}'.format(StartDateTime, CurrentDateTime))
    # huge array, memory overflow
    DayPrice = np.asarray([0,0,0,0,0], dtype=np.float)
    DayDate = np.asarray([], dtype='datetime64[D]')

    # get mean and peaks
    day_iter = StartDateTime
    while day_iter != CurrentDateTime:
        pop_deque = data_deque.popleft()
        TempRow = np.array(pop_deque[1], dtype=np.float)

        while day_iter == np.array(pop_deque[0], dtype=np.int64).astype('datetime64[s]').astype('datetime64[D]'):

            # pop_deque is ['unix_time', 'price']
            pop_deque = data_deque.popleft()
            TempRow = np.append(TempRow, np.array(pop_deque[1], dtype=np.float))
            if not data_deque:
                break 

        # smooth Date
        if TempRow.size > 20:
            sz = TempRow.size // 20 + 1
            sort = np.sort(TempRow)
            max_ = np.mean(sort[-1-sz:-1])
            min_ = np.mean(sort[0:sz])
            open_ = np.mean(TempRow[0:sz])
            close_ = np.mean(TempRow[-1-sz:-1])
            mean_ = np.mean(TempRow)
            values = np.array([mean_, open_, close_, max_, min_])

        else:
            values = np.array([np.mean(TempRow)]*5)


    #    SmoothArrUp = np.array([mean_, mean_*1.3, mean_*1.3, open_*1.5, close_*1.5])
    #    SmoothArrDown = np.array([mean_, mean_*0.7, mean_*0.7, open_*0.5, close_*0.5])
    #    values = np.median(np.vstack((np.array([mean_, open_, close_, max_, min_]), 
    #                        np.vstack((SmoothArrUp, SmoothArrDown)))), axis=0) 
        
        DayPrice = np.vstack((DayPrice, values))
        DayDate = np.append(DayDate, day_iter)
        
        day_iter += np.timedelta64(1, 'D')
        if not data_deque:
            break

    DayPrice = np.delete(DayPrice, (0), axis=0)
    print('Prices of {} last days'.format(DayDate.size))

    # check range between opens and closes prices
#    for i in range(1, DayPrice.shape[0]-1):
#        if DayPrice[i,1] > DayPrice[i,3]:
#            DayPrice[i,2] = DayPrice[i,1]
#        if DayPrice[i,2] < DayPrice[i,4]:
#            DayPrice[i,2] = DayPrice[i+1,1]

    # Create two csv (with data in d64 format and Price array)
    try:
        np.savetxt('btcDayDate.csv', DayDate, fmt='%s', delimiter=',')
    except IOError:
        print('IOerror! (Date)') 
    else:   
        print('Save date64 in csv')

    try:
        np.savetxt('btcDayPrice.csv', DayPrice, fmt='%f', delimiter=',')
    except IOError:
        print('IOerror! (Price)')  
    else:
        print('Save price in csv')


# ##################
# ####run script####
# ##################
if __name__ == "__main__":
	# load data from bitcoincharts API
	size = 30000000
	convert_datetime(size, False)
	# create figure and axis
	fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, squeeze=False)

	# Create and run our class
	Top = PredictExchange(fig, ax, 4, 120, 360, False)
	Top.Main()

