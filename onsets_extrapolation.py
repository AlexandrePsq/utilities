import os
import sys

import argparse
import warnings
warnings.simplefilter(action='ignore')

# from nltk.tokenize import sent_tokenize 
# from nltk.tokenize import word_tokenize
from tokenizer import tokenize
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import re
import inflect

########################## PARSING ##########################

parser = argparse.ArgumentParser(description="Create the onsets for a given tokenizer relatively to an initial set of onsets")

parser.add_argument('--model_name', type=str, help='Name of the model to consider')
parser.add_argument('--language', type=str, help='Language of the text being tokenized')
parser.add_argument('--onsets', help='path to initial onsets folder')
parser.add_argument('--text', help='path to the folder containing the text to tokenize and for which we need the onsets')
parser.add_argument('--convert_numbers', default=False, action='store_true', help='Boolean precising if numerical values need to be converted into alphabetical values.')
parser.add_argument('--save', type=str, help='path to the folder in which to save the new onsets files')

args = parser.parse_args()




########################## PREPROCESSING ##########################

special_words = {
    'english':
    {
        'grown-ups': 'grownups',
        'hasn\'t': 'hasnt',
        'hasn’t': 'hasnt',
        'grown-up':'grownup'
    },
    'french':
    {

    }
                
}


def preprocess(text, special_words, convert_numbers=False):
    for word in special_words.keys():
        text = text.replace(word, special_words[word])
    if convert_numbers:
        transf = inflect.engine()
        numbers = re.findall('\d+', text)
        for number in numbers:
            text = text.replace(number, transf.number_to_words(number))
    return text







########################## MAIN ##########################


if __name__ == '__main__':
    text_files = sorted(glob.glob(os.path.join(args.text, 'text_{}_run*.txt'.format(args.language))))
    onsets_files = sorted(glob.glob(os.path.join(args.onsets, 'word_run*.csv')))
    
    try:
        assert len(onsets_files)==len(text_files)
        try:
            assert len(onsets_files)==9
        except AssertionError:
            print('You are working with {} runs instead of 9. Be careful to the data you are using and check the code/utilities/settings.py file to specify the new number of runs'.format(len(onsets_files)))
        
        for run in tqdm(range(len(onsets_files))):
            result = []
            ref_df = pd.read_csv(onsets_files[run])
            ref_words = list(ref_df.word)
            text = open(text_files[run], 'r').read().lower()
            text = preprocess(text, special_words[args.language], convert_numbers=args.convert_numbers)
            tmp_text = None
            for index in range(len(ref_words)):
                new_index = text.find(ref_words[index].lower())
                tmp_text = text[: new_index]
                text = text[new_index + len(ref_words[index]):]
                # Extrapolating onset-offset values
                words = tokenize(tmp_text, args.language)
                onsets = np.linspace(ref_df.offsets.iloc[max(0, index-1)], ref_df.onsets.iloc[index], len(words))
                offsets = np.hstack([onsets[1:], np.array(ref_df.onsets.iloc[index])]) if onsets.size > 0 else []
                result += list(zip(onsets, offsets, words))
                result.append(((ref_df['onsets'].iloc[index], ref_df['offsets'].iloc[index], ref_df['word'].iloc[index])))
            df = pd.DataFrame(result, columns=['onsets', 'offsets', 'word'])
            saving_path = os.path.join(args.save, 'word+punctuation_run{}.csv'.format(run + 1))
            df.to_csv(saving_path, index=False)
            
    except AssertionError:
        print('You do not have the same number of onsets files and text-to-tokenize files ...')
