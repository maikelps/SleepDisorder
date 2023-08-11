import pandas as pd
import numpy as np

def cleaning(df, insomnia_cat = 1) -> pd.DataFrame:
    """
    First cleaning pipeline
    """
    
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
    
    return full_fr

# Reading and fixing column names
df = (pd.read_csv('Data/Sleep_health_and_lifestyle_dataset.csv', index_col=[0])
        .rename( columns=str.lower )
        .rename( columns=lambda x: x.replace(' ', '_') )
        )

print( cleaning(df) )