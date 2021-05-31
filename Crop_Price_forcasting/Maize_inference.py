import pickle
import numpy as np
import pandas as pd
import datetime

class Rice_inference:
    """
     In this class you can give a particular date in the datetime format and you will get the price of the rice at that date

    """

    def __init__(self,Saved_pickle_file_name):
        self.Saved_pickle_file_name = Saved_pickle_file_name


    def inference(self,year,month,date):
        """
        input must be in formate (year,month,date).
        example - 2021,1,25

        """
        model = pickle.load(open(self.Saved_pickle_file_name, 'rb'))
        dataframe = pd.DataFrame()
        dataframe['ds'] = [pd.to_datetime(f'{year}-{month}-{date}')]
        result = model.predict(dataframe)
        print(f'price of the rice on {year}-{month}-{date} =  {result["yhat"][0].round()}')



if __name__ == "__main__":
    model = Rice_inference('Maize_forcasting.sav')

    model.inference(2021,2,1)