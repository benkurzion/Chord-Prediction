import pandas as pd
import ast
import re
import numpy as np
import csv

def clean_dataset(save=False) : 
    '''
    Grabs all the pop songs and removes the pointless features. Saves to new csv
    '''
    df = pd.read_csv('full_chordonomicon_data.csv', low_memory=False)
    df = df[df['main_genre'] == 'pop']
    df.drop(columns=['id', 'main_genre','release_date', 'genres', 'decade', 'rock_genre','artist_id', 'spotify_song_id', 'spotify_artist_id'], inplace=True)
    # remove inversions
    df['chords'] = df['chords'].apply(lambda s: re.sub(r"/[^/]*$", "", s))
    print(f"Cleaned data has shape {df.shape}")
    if save:
        print("Saving data to {cleaned_data.csv}")
        df.to_csv('cleaned_data.csv', index=False, header=False)



def get_data(type: str) -> np.ndarray:
    '''
    Returns the cleaned dataset. Chords are converted 

    Parameters
    ----------

    type : str
        Determines how to encode the chords. 
    '''
    if type not in ['degrees', 'notes']:
        raise ValueError("Invalid input type. Can only be either 'degrees' or 'notes'")
    
    data = []
    chord_relations = pd.read_csv('chords_mapping.csv', low_memory=False)

    encoding = {}
    if type == 'degrees':
        encoding = dict(zip(chord_relations['Chords'], chord_relations['Degrees']))
        for key, value in encoding.items():
            encoding[key] = ast.literal_eval(value)
    elif type == 'notes':
        encoding = dict(zip(chord_relations['Chords'], chord_relations['Notes']))
        for key, value in encoding.items():
            encoding[key] = ast.literal_eval(value)

    # Read and convert
    with open('cleaned_data.csv', 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            # Filter out empty strings and <...> markers
            clean_row = []
            for item in row:
                if item and '<' not in item and '>' not in item and item in encoding:
                    item = item.split('/', 1)[0]
                    clean_row.append(encoding[item])
            data.append(clean_row)
    return np.array(data, dtype=object)
