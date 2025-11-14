import pandas as pd
import ast
import re
import numpy as np
import csv
import torch
from torch.utils.data import Dataset, DataLoader

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



def get_data_as_list(type: str) -> list:
    '''
    Returns the cleaned dataset as a list of lists. Chords are converted 

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
    return data

class SequencePredictionDataset(Dataset):
    def __init__(self, padded_sequences, mask, max_len):
        # 1. Pad the sequences
        self.padded_sequences = padded_sequences
        self.mask = mask
        self.max_len = max_len 

        self.X = self.padded_sequences[:, :-1, :] 
        self.Y = self.padded_sequences[:, 1:, :] 
        
        self.mask = self.mask[:, :-1] 

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.mask[idx]


def get_data_as_torch(type: str, batch_size=8) -> DataLoader:
    '''
    Returns the cleaned dataset as a pytorch dataloader object. Chords are converted 

    Parameters
    ----------

    type : str
        Determines how to encode the chords. 
    batch_size : int
        Determines how many samples are processed through the network before loss is backpropagated
    '''
    data = get_data_as_list(type=type)
    TOKEN_VECTOR_SIZE = 12
    PADDING_TOKEN = [0] * TOKEN_VECTOR_SIZE
    max_len = max(len(seq) for seq in data)
    
    padded_sequences = []
    # Mask to keep track of actual data vs. padding
    masks = [] 
    
    for seq in data:
        padding_needed = max_len - len(seq)
        # Pad the sequence
        padded_seq = seq + [PADDING_TOKEN] * padding_needed
        padded_sequences.append(padded_seq)

        # 1 for real tokens, 0 for padding tokens
        mask = [1] * len(seq) + [0] * padding_needed
        masks.append(mask)

    X = torch.tensor(padded_sequences, dtype=torch.float32)
    M = torch.tensor(masks, dtype=torch.bool)

    dataloader = DataLoader(SequencePredictionDataset(X, M, max_len), batch_size=batch_size, shuffle=True)

    return dataloader