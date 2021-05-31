from matplotlib.legend import Legend
import pandas as pd
import pickle
from numpy import percentile
from fbprophet import Prophet
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
import os

class Rice_forcasting:

    def __init__(self,file_path):
        self.data = pd.read_csv(file_path)

    def data_preprocessing(self):
        ## Data Cleaning
        self.data['date_time'] = pd.to_datetime(self.data['Price Date']) # converting into date time formate
        self.data['month_year'] = self.data['date_time'].dt.to_period('M')
        self.data.groupby('month_year').mean()
        self.data = self.data[['month_year','Max Price (Rs./Quintal)']] # keeping only required columns 
        self.data.rename(columns={'month_year':'date','Max Price (Rs./Quintal)':'price'},inplace=True) # renaming columns names

        ## Outliers
        q25, q75 = percentile(self.data['price'], 25), percentile(self.data['price'], 75)
        iqr = q75 - q25
        # calculate the outlier cutoff
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off
        # identify outliers
        outliers = [x for x in self.data['price'] if x < lower or x > upper]
        outliers = list(dict.fromkeys(outliers))
        Removing_outliers = [self.data.drop(self.data[self.data['price']==x].index,axis=0,inplace=True) for x in outliers]
        if len(Removing_outliers)==0:print(f'No outliers in the  dataframe')
        else:print(f'{len(Removing_outliers)} outliers has been detected in  dataframe and removed')  

        self.data = self.data.groupby('date').mean()# grouping into month level
        self.data['price'] = self.data['price'].round()
        self.data = self.data.reindex(pd.period_range(self.data.index[0],self.data.index[-1],freq='M'))# adding the missing rows into the data
        self.data = self.data.reset_index()
        self.data['index']= self.data['index'].astype(str)
        self.data['index'] = pd.to_datetime(self.data['index'])
        self.data = self.data.set_index('index')

        # Filling missing values
        self.data['price'] = self.data['price'].fillna(method='ffill')

        # changing column names because prophet model wants date column as ds, and price column as y
        self.data.reset_index(inplace=True)
        self.data.rename(columns = {'index':'ds','price':'y'},inplace=True)



    def Train(self):
        train = self.data.iloc[:197] #data till 2018 goes to the train
        Drought = pd.DataFrame({
        'holiday': 'Drought',
        'ds': pd.to_datetime([
        '2002-03-01',
        '2002-04-01',
        '2002-05-01',
        '2002-06-01',
        '2002-07-01',
        '2002-08-01',
        '2002-09-01',      
        '2002-10-01',      
        '2004-03-01',
        '2004-04-01',
        '2005-06-01',
        '2005-07-01',
        '2005-08-01',
        '2005-09-01',
        '2005-10-01',     
        '2009-02-01',
        '2009-03-01',
        '2009-04-01',
        '2009-05-01',
        '2009-07-01',
        '2009-08-01',
        '2009-09-01',
        '2009-10-01',
        '2009-11-01',          
        '2014-04-01',
        '2014-05-01',
        '2014-06-01',
        '2014-07-01',
        '2014-08-01',
        '2014-09-01',
        '2015-01-01',
        '2015-02-01',
        '2015-03-01',      
        '2015-04-01',
        '2015-05-01',
        '2015-06-01', 
            
        ]),
        'lower_window': 0,
        'upper_window': 1,
        })

        Floods = pd.DataFrame({
        'holiday': 'flood',
        'ds': pd.to_datetime(['2010-02-01',
        '2010-03-01',
        '2010-04-01',
        '2010-05-01',
        '2019-07-01',
        '2019-03-01',
        '2019-04-01',
        '2019-05-01',
        '2019-06-01',
        '2019-07-01',
        '2016-09-01',
        '2016-10-01',
        '2016-11-01',
        '2013-02-01',
        '2013-03-01',
        '2013-04-01',
        '2013-05-01',
        '2013-06-01',
        '2013-07-01',
        '2013-08-01']),
        'lower_window': 0,
        'upper_window': 1,
        })

        Harvesting_period = pd.DataFrame({
        'holiday': 'flood',
        'ds': pd.to_datetime(['2010-02-01',
        '2001-02-01',
        '2001-05-01',
        '2001-06-01',
        '2001-07-01',
        '2001-11-01',
        '2001-12-01',])
        ,
        'lower_window': -1,
        'upper_window': 0,
        })
        holidays = pd.concat((Drought, Floods, Harvesting_period))

        # instantiate the model and fit the timeseries
        model = Prophet(interval_width=.95,holidays=holidays,changepoint_range=.2,seasonality_mode='multiplicative')
        model.fit(train)

        # save the model to disk
        self.filename = 'Rice_forcasting.sav'
        pickle.dump(model, open(self.filename, 'wb'))

        forecast = model.predict()

        # ploting original vs test_forcast_data 
        plt.figure(figsize=(20,5))
        plt.title('Original vs forcast of trained model')
        plt.plot(train['ds'],train['y'],label = 'Original')
        plt.plot(forecast['ds'],forecast['yhat'],label ='forcast')
        plt.xlabel('year')
        plt.ylabel('price')
        plt.show()

    def Evaluate(self):
        test = self.data.iloc[197:]

        model = pickle.load(open(self.filename, 'rb'))

        #creteing future dataframe with date te test model
        future = pd.DataFrame()
        future['ds'] = test['ds']
        
        #predictions
        forcast = model.predict(future)

        #mean absolute error
        print(f'Mean Absolute Error of the model = {mean_absolute_error(test["y"],forcast["yhat"])}')
          

        #ploting original vs forcasted data
        plt.figure(figsize=(20,5))
        plt.plot(test['ds'],test['y'],label='Original')
        plt.plot(forcast['ds'],forcast['yhat'],label='forcast')
        plt.title('Original vs Forcast')
        plt.xlabel('date')
        plt.ylabel('Price')
        plt.show()

if __name__ == "__main__":
    model = Rice_forcasting(os.path.join(os.getcwd(), "data/Rice.csv"))
    model.data_preprocessing()
    model.Train()
    model.Evaluate()

