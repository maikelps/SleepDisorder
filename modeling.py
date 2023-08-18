import pandas as pd
import numpy as np
import joblib
from sklearn.naive_bayes import GaussianNB

def cleaned_fr(avoid_systolic = False, insomnia_cat = 1) -> pd.DataFrame:
    """
    Creates the cleaning pipeline according to the exploration notebook.
        avoid_systolic -> drops the systolic column generated
        insomnia_cat -> necessary when needing to separate the type of sleep disorder and give insomnia its own category
    """

    # Reading and fixing column names
    df = (
        pd.read_csv('Data/Sleep_health_and_lifestyle_dataset.csv', index_col=[0])
            .rename( columns=str.lower )
            .rename( columns=lambda x: x.replace(' ', '_') )
            )
    
    # Simplifying class
    cast_cat = {
        'None': 0,
        'Sleep Apnea': 1,
        'Insomnia' : insomnia_cat
    }

    # Mapping acording to the above dictionaries
    full_fr = (df
                    .assign( 
                        systolic_bp = df['blood_pressure'].str.split('/', expand=True)[0].astype('int64'),
                        is_male = np.where((df['gender'] == 'Male'), 1, 0).astype('uint8'),
                        elevated_bmi = np.where(df['bmi_category'].isin(['Overweight', 'Obese']), 1, 0).astype('uint8'),
                        wf_technical = np.where( df['occupation'].isin(['Software Engineer', 'Engineer', 'Accountant', 'Scientist']), 1,0 ).astype('uint8'),
                        sleep_issue = df['sleep_disorder'].astype(str).map(cast_cat)
                        )
                    .drop(columns=['gender', 'occupation', 'bmi_category', 
                                   'blood_pressure', 'sleep_disorder', 'quality_of_sleep',
                                   'physical_activity_level', 'stress_level'])
                    )
    
    if avoid_systolic:
        return full_fr.drop(columns=['systolic_bp'])
    else:
        return full_fr

def train_and_save_model(MODEL_PATH, model_selected, avoid_systolic = False):

    # 1. Load Data according to the type of model
    df = cleaned_fr(avoid_systolic = avoid_systolic)

    # Separating
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    # 2. Train the Model accordingly
    model_selected.fit(X, y)

    # 3. Serialize the Model
    joblib.dump(model_selected, MODEL_PATH + '.pkl')

def load_model(MODEL_PATH):
    return joblib.load(MODEL_PATH)