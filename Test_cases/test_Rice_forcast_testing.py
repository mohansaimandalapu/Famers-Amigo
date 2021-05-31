import unittest
import pickle
import pandas as pd


class Rice_forcast_testing(unittest.TestCase):

    def test_case1(self):
        model = pickle.load(open('Rice_forcasting.sav', 'rb'))
        dataframe = pd.DataFrame()
        dataframe['ds'] = [pd.to_datetime('2020-12-1')]
        result = model.predict(dataframe)['yhat'][0].round()    
        print(result)
        self.assertAlmostEqual(first=result,second=5400,delta=300)

    def test_case2(self):
        model = pickle.load(open('Rice_forcasting.sav', 'rb'))
        dataframe = pd.DataFrame()
        dataframe['ds'] = [pd.to_datetime('2002-1-1')]
        result = model.predict(dataframe)['yhat'][0].round()    
        print(result)
        self.assertAlmostEqual(first=result,second=1500,delta=300)


    def test_case3(self):
        model = pickle.load(open('Rice_forcasting.sav', 'rb'))
        dataframe = pd.DataFrame()
        dataframe['ds'] = [pd.to_datetime('2004-1-1')]
        result = model.predict(dataframe)['yhat'][0].round()    
        print(result)
        self.assertAlmostEqual(first=result,second=1200,delta=300)


    def test_case4(self):
        model = pickle.load(open('Rice_forcasting.sav', 'rb'))
        dataframe = pd.DataFrame()
        dataframe['ds'] = [pd.to_datetime('2010-1-1')]
        result = model.predict(dataframe)['yhat'][0].round()    
        print(result)
        self.assertAlmostEqual(first=result,second=2500,delta=300)


    def test_case5(self):
        model = pickle.load(open('Rice_forcasting.sav', 'rb'))
        dataframe = pd.DataFrame()
        dataframe['ds'] = [pd.to_datetime('2015-8-1')]
        result = model.predict(dataframe)['yhat'][0].round()    
        print(result)
        self.assertAlmostEqual(first=result,second=4000,delta=300)


    def test_case6(self):
        model = pickle.load(open('Rice_forcasting.sav', 'rb'))
        dataframe = pd.DataFrame()
        dataframe['ds'] = [pd.to_datetime('2018-5-1')]
        result = model.predict(dataframe)['yhat'][0].round()    
        print(result)
        self.assertAlmostEqual(first=result,second=4300,delta=300)


    def test_case7(self):
        model = pickle.load(open('Rice_forcasting.sav', 'rb'))
        dataframe = pd.DataFrame()
        dataframe['ds'] = [pd.to_datetime('2021-2-1')]
        result = model.predict(dataframe)['yhat'][0].round()    
        print(result)
        self.assertAlmostEqual(first=result,second=5100,delta=300)


    def test_case8(self):
        model = pickle.load(open('Rice_forcasting.sav', 'rb'))
        dataframe = pd.DataFrame()
        dataframe['ds'] = [pd.to_datetime('2011-11-1')]
        result = model.predict(dataframe)['yhat'][0].round()    
        print(result)
        self.assertAlmostEqual(first=result,second=3000,delta=300)


    def test_case9(self):
        model = pickle.load(open('Rice_forcasting.sav', 'rb'))
        dataframe = pd.DataFrame()
        dataframe['ds'] = [pd.to_datetime('2006-10-1')]
        result = model.predict(dataframe)['yhat'][0].round()    
        print(result)
        self.assertAlmostEqual(first=result,second=1700,delta=300)



    def test_case10(self):
        model = pickle.load(open('Rice_forcasting.sav', 'rb'))
        dataframe = pd.DataFrame()
        dataframe['ds'] = [pd.to_datetime('2019-8-1')]
        result = model.predict(dataframe)['yhat'][0].round()    
        print(result)
        self.assertAlmostEqual(first=result,second=5000,delta=300)


    

if __name__ == '__main__':
    unittest.main()

    