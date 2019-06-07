################
# Check the onsets / offsets files
# Correct them if necessary
################

from os.path import join
import pandas as pd
import os
import argparse
import glob
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


    


def transform(df, path):
    # standardize the dataframe
    data = []
    # inserting new rows at the first position  
    [onset, offset, word] = sorted(tuple(df.columns))
    data.insert(0, {'word': word, 'onsets': onset, 'offsets': offset})
    df.columns = ['word', 'onsets', 'offsets']
    result = pd.concat([pd.DataFrame(data), df], ignore_index=True)
    result.to_csv(path, index=False)

def get_data(language):
    base_path = '/neurospin/unicog/protocols/LePetitPrince_Pallier_2018/LePetitPrince/data/text/{}/onsets-offsets'.format(language)
    file_pattern = 'text_{}_onsets-offsets_run*.csv'.format(language)
    data = sorted(glob.glob(join(base_path, file_pattern)))
    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Correct the columns header of onsets / offsets files.""")
    parser.add_argument("--language", type=str, default='en', help="Language of the text.")

    args = parser.parse_args()
    
    files = get_data(args.language)

    for path in files:
        df = pd.read_csv(path)
        columns = sorted(tuple(df.columns))
        standard = ['offsets', 'onsets', 'word']

        if columns == standard:
            print('File: {} is in order.'.format(os.path.basename(path)))
        else:
            transform(df, path)
