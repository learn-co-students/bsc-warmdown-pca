from requests import get
import pandas as pd
from io import BytesIO
import numpy as np
from tqdm import tqdm



def load_data(): 
    with tqdm(total=5) as pbar:
        # Download dataset
        pbar.set_description('Downloading data from seattle.gov')
        response = get('https://data.seattle.gov/api/views/28ny-9ts8/rows.csv?accessType=DOWNLOAD')
        pbar.set_description('Decoding bytes object.')
        pbar.update(1)
        decoded = BytesIO(response.content)
        police_data = pd.read_csv(decoded)
        pbar.update(1)
        # Encode categoricals
        pbar.set_description('Reformatting variables.')
        police_data['arrest_count'] = police_data['Officer ID'].map(
                pd.qcut(police_data['Officer ID'].value_counts(), 5, labels=[0, 1, 2, 3, 4])\
                .to_dict())

        police_data['officer_age'] = ((pd.to_datetime(police_data['Reported Date'])\
                                       - pd.to_datetime(police_data['Officer YOB'], format='%Y'))\
                                      .dt.days/365).astype(int)

        police_data['time_of_day'] = pd.qcut(pd.to_datetime(police_data['Reported Time'])\
                                             .dt.hour, 5, [0, 1, 2, 3, 4])

        police_data['initial_call_type_frequency'] = police_data['Initial Call Type'].map(
            pd.qcut(np.log(police_data['Initial Call Type'].value_counts()), 3, [0,1,2]).to_dict())

        police_data['final_call_type_frequency'] = police_data['Final Call Type'].map(
            pd.qcut(police_data['Final Call Type'].value_counts(), 3, [0,1,2]).to_dict())

        police_data['call_type_frequency'] = police_data['Call Type'].map(
            pd.qcut(police_data['Call Type'].value_counts(), 3, [0,1,2]).to_dict()
                                                )

        police_data['officer_squad_frequency'] = police_data['Officer Squad'].map(
            pd.qcut(police_data['Officer Squad'].value_counts(), 3, [0,1,2]).to_dict())
        
        pbar.update(1)
        # Drop Nulls
        pbar.set_description('Dropping nulls.')
        police_data = police_data.dropna()
        pbar.update(1)
        drop = ['Subject ID', 'GO / SC Num', 
                'Terry Stop ID', 'Officer ID', 'Reported Time',
               'Officer YOB', 'Reported Date', 'Initial Call Type', 
                'Call Type', 'Final Call Type', 'Officer Squad',
               'Arrest Flag', 'Frisk Flag']
        police_data = police_data.drop(columns=drop)
        pbar.set_description('Complete!')
        pbar.update(1)
    return police_data