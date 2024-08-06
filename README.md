## Introduction
This project contains the algorithms that were used to predict the price and perceived safety of a property. A Siamese Neural Network was used to predict the perceived safety, while XGBoost was used to predict the property price.

## Getting started
1. Have Python v3.10.8 installed [here](https://www.python.org/)
2. Clone the repository

```
git clone https://github.com/dyitsme/realestate_backend
```

3. Install the necessary libraries
```
pip install -r requirements.txt
```

4. Run the server
```
flask --app app run --debug
```

5. Open at http://localhost:5000/


## Routes
`/` is the home page 
`/predict_xgb` is used to predict property prices
`/predict_snn` is used to predict perceived safety

