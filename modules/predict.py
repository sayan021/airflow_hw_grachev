import json

import pandas as pd
import dill
import os
from datetime import datetime
import glob
import logging


path = os.environ.get('PROJECT_PATH', '.')

def predict():
    file_pattern = f'{path}/data/models/cars_pipe_*.pkl'

    matching_files = glob.glob(file_pattern)

    if matching_files:
        file_path = matching_files[0]

        with open(file_path, 'rb') as file:
            model = dill.load(file)
            logging.info(f'Model found: {model}')

        folder_path = f'{path}/data/test/'

        file_list = os.listdir(folder_path)

        df_preds = pd.DataFrame()

        for filename in file_list:
            file_path = os.path.join(folder_path, filename)
            with open(file_path) as item:
                form = json.load(item)
                df = pd.DataFrame.from_dict([form])
                y = model.predict(df)
                preds = {'car_id': df.id, 'pred': y}
                df2 = pd.DataFrame(preds)
                df_preds = pd.concat([df_preds, df2], axis=0)

        df_preds.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)

    else:
        logging.error(f'No models found')


if __name__ == '__main__':
    predict()
