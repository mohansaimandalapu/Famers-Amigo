import unittest
import pickle
import numpy as np
from pathlib import Path

class CropIdentificationTesting(unittest.TestCase):
    
    def test_apple1(self):


        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [20,134,199,22,92,5,112]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'apple')

    

    def test_apple2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [20,134,199,22,92,5,125]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'apple')

    def test_apple3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [40,145,205,24,94,6,124]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'apple')

    # For banana class

    def test_bananna1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [100,82,50,27,80,5,104]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'banana')

    def test_bananna2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [80,70,45,25,75,5,90]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'banana')

    def test_bananna3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [120,95,55,29,84,6,119]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'banana')

    
    def test_blackgram_1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [40,67,19,29,65,7,67]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'blackgram')

    def test_blackgram_2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [20,55,15,25,60,6,60]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'blackgram')

    def test_blackgram_3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [45,55,18,28,62,7,74]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'blackgram')

    def test_chickpea1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [40,67,79,18,16,7,80]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'chickpea')

    def test_chickpea2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [20,55,75,17,14,5,65]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'chickpea')

    def test_chickpea3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [60,80,85,21,19,8,94]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'chickpea')


    def test_coconut1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [21,16,30,27,94,5,175]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'coconut')

    def test_coconut2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [1,5,25,25,90,5,131]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'coconut')

    def test_coconut3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [40,30,35,29,99,6,225]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'coconut')

    def test_coffee1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [101,28,29,25,58,6,158]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'coffee')

    def test_coffee2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [80,15,25,23,50,6,115]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'coffee')

    def test_coffee3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [120,40,35,27,69,7,199]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'coffee')


    def test_cotton1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [117,46,19,23,79,6,80]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'cotton')

    def test_cotton2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [117,46,19,25,79,6,90]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'cotton')

    def test_cotton3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [140,60,25,25,84,7,99]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'cotton')


    def test_grapes1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [23,132,200,23,81,6,69]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'grapes')

    def test_grapes2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [0,120,195,8,80,5,65]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'grapes')

    def test_grapes3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [40,145,205,41,83,6,74]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'grapes')



    def test_jute1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [78,46,39,24,79,6,174]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'jute')

    def test_jute2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [60,35,35,23,70.88,6,150]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'jute')

    def test_jute3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [100,60,45,26,89,7,199]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'jute')


    def test_kidneybeans1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [20,67,20,20,21,5,105]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'kidneybeans')

    def test_kidneybeans2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [1,55,15,15,18,5,60]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'kidneybeans')

    def test_kidneybeans3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [40,80,25,24,24,6,149]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'kidneybeans')


    def test_lentil1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [18,68,19,24,64,6,45]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'lentil')

    def test_lentil2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [0,55,15,18,60,5,35]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'lentil')

    def test_lentil3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [40,80,25,29,69,7,54]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'lentil')


    def test_maize1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [100,60,25,26,74,7,109]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'maize')

    def test_maize2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [77,48,19,22,65,6,84]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'maize')

    def test_maize3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [60,35,15,18,55,5,60]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'maize')


    def test_mango1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [40,40,35,35,54,6,100]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'mango')

    def test_mango2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [0,15,25,27,45,4,89]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'mango')

    def test_mango3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [20,27,29,31,50,5,94]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'mango')


    def test_mothbeans1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [40,60,25,32,64,9,74]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'mothbeans')

    def test_mothbeans2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [0,35,15,24,40,3,30.9]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'mothbeans')

    def test_mothbeans3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [21,48,20,28,53,6,51]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'mothbeans')


    def test_mungbean1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [40,60,25,29,90,7,59]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'mungbean')

    def test_mungbean2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [0,35,15,27,80,6,36]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'mungbean')

    def test_mungbean3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [20,47,19,28,85,6,48]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'mungbean')


    def test_muskmelon1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [80,5,45,27,90,6,20]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'muskmelon')

    def test_muskmelon2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [110,25,55,29,94,6,29]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'muskmelon')

    def test_muskmelon3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [80,5,45,27,90,6,20]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'muskmelon')


    def test_orange1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [40,30,15,34,94,8,119]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'orange')

    def test_orange2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [1,5,5,10,90,6,100]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'orange')

    def test_orange3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [19,16,10,22,92,7,110]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'orange')


    def test_papaya1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [70,70,55,43,94,6,248]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'papaya')

    def test_papaya2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [31,46,45,24,91,6.5,41]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'papaya')

    def test_papaya3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [49,59,50,33,92,6,142]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'papaya')

    def test_pigeonpeas1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [40,80,25,36,69,7,198]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'pigeonpeas')

    def test_pigeonpeas2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [0,55,15,18,30,4,90]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'pigeonpeas')

    def test_pigeonpeas3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [20,67,20,27,48,5,149]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'pigeonpeas')


    def test_pomegranate1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [40,30,45,24,95,7,112]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'pomegranate')

    def test_pomegranate2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [0,5,35,18,85,5,102]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'pomegranate')

    def test_pomegranate3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [18,18,40,21,90,6,107]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'pomegranate')

    def test_rice1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [99,60,45,26,84,7,298]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'rice')

    def test_rice2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [60,35,35,20,80,5,182]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'rice')

    def test_rice3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [79,47,39,23,82,6,236]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'rice')


    def test_watermelon1(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case1 = [120,30,55,26,89,6,59]   
        result = model.predict(np.array(test_case1).reshape(1,7))
        
        self.assertEqual(result, 'watermelon')

    def test_watermelon2(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case2 = [80,5,45,24,80,6,40]
        result = model.predict(np.array(test_case2).reshape(1,7))
        
        self.assertEqual(result, 'watermelon')

    def test_watermelon3(self):
        model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
        test_case3 = [99,17,50,25,85,6,50]
        result = model.predict(np.array(test_case3).reshape(1,7))
        
        self.assertEqual(result, 'watermelon')


    
    

if __name__ == '__main__':
    unittest.main()

