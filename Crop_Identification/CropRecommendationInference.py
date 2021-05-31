import pickle
import numpy as np

class CroprecommendationInference:
    """
     In this class you can give a set of values for the soil composition and Rainfall humidity and temperature values
     and it will predict a crop for the sowing

    """

    def __init__(self,Saved_pickle_file_name):
        self.Saved_pickle_file_name = Saved_pickle_file_name

    def inference(self,Soil_composition_list:list):
        """
            Soil composition is a list that must conatin minimum and maximum values mentioned in squred brackets.
                Nitrogrn[0,140],Phosporous[5,145], Potassium[5,205], Temperature[9,44] , Humidity[15,99], pH[3.5,9.9] , Rainfall[20,298]

        """
        loaded_model = pickle.load(open(self.Saved_pickle_file_name, 'rb'))
        result = loaded_model.predict(np.array(Soil_composition_list).reshape(1,7))
        print(f'Predicted crop is {result[0]}')



if __name__ == "__main__":
    model = CroprecommendationInference('Crop_recommendation_model.sav')

    model.inference([15,25,35,40,50,8,200])