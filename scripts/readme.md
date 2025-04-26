# How to run scripts

##Train departure
python train_departure_model.py \
    --kaggle_json_path /path/to/your/kaggle.json

##Predict departure
python predict_departure_model.py \
    --model_path /path/to/stacked_departure_delay_model.pkl \
    --new_data_path /path/to/your_new_unseen_data.csv

##Train arrival
python train_arrival_model.py \
    --kaggle_json_path /path/to/your/kaggle.json

##Predict arrival
python predict_arrival_model.py \
    --model_path /path/to/arrival_delay_xgboost_model.pkl \
    --new_data_path /path/to/your_new_unseen_data.csv
