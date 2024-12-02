# src/data_preparation.py

import pandas as pd
from datasets import Dataset
import os

def create_custom_dataset():
    data = {
        'incorrect': [
            'I goes to the store yesterday.',
            'She dont like apples.',
            'They was playing football.',
            'He have many books.'
        ],
        'correct': [
            'I went to the store yesterday.',
            'She doesn\'t like apples.',
            'They were playing football.',
            'He has many books.'
        ]
    }
    
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    
    # Create the data directory if it doesn't exist
    os.makedirs('../data/custom_grammar_dataset', exist_ok=True)
    
    # Save the dataset
    dataset.save_to_disk('../data/custom_grammar_dataset')
    print("Custom dataset created and saved.")

if __name__ == "__main__":
    create_custom_dataset()