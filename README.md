**Current Model Version: v2.0**
A machine learning based weather forecasting system that predicts whether it will rain tomorrow using historical meteorological data.
- This project trains a Random Forest classifier to predict RainTomorrow (Yes/No) based on daily weather observations

  
- The model prioritises high recall for rain events. In other words it would rather wrongly predict that rain is coming to allow people to be prepared in real world 
scenarios:

Recall (Rain): 88%

Precision (Rain): 45%
	
Accuracy: ~72%

There is a trade off to precision allowingthe model to catch as many rainy days as possible.

Algorithm: Random Forest Classifier
	•	Target Variable: Will it Rain Tomorrow 
	•	Decision Threshold: 30%
	•	Primary Objective: Maximise recall for rain events


**How to use**
1. Install Dependencies
   pip install pandas scikit-learn joblib
2. Run the latest model
   python RunSavedModel.py


Use python main.py to retrain
