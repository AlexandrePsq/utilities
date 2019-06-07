import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

import argparse
import warnings
warning.simplefilter(action='ignore')

from tokenizer import tokenizerimport torch
import pandas as pd
import numpy as np
import glob

########################## PARSING ##########################

parser = argparse.ArgumentParser(description="Create the onsets for a given tokenizer relatively to an initial set of onsets")

parser.add_argument('--model_name', type=str, help='Name of the model to consider')
parser.add_argument('--language', type=str, help='Language of the text being tokenized')
parser.add_argument('--onsets', help='path to initial onsets folder')
parser.add_argument('--onsets', help='path to the folder containing the text to tokenize and for which we need the onsets')
parser.add_argument('--save', type=str, help='path to the folder in which to save the new onsets files')

args = parser.parse_args()


if __name__ == '__main__':
    text_files = sorted(glob.glob(os.path.join(args.text, 'text_{}_run*.txt'.format(args.language))))
    onsets_files = sorted(glob.glob(os.path.join(args.text, 'text_{}_onsets-offsets_run*.txt'.format(args.language))))

    try:
        assert len(onsets_files)==len(text_files)
        try:
            assert len(onsets_files)==9
        except AssertionError:
            print('You are working with {} runs instead of 9. Be careful to the data you are using and cjeck the code/utilities/settings.py file to specify the new number of runs'.format(len(onsets_files)))
        
        for run in range(len(onsets_files)):
            result = []
            ref_df = pd.read_csv(onsets_files[run])
            ref_words = list(ref_df.word)
            text = text_files[run]
            tmp_text = None
            last_index = 0
            for index in range(len(ref_words)):
                new_index = text.find(ref_words[index])
                tmp_text = text[last_index: new_index]
                text = text[new_index:]
                last_index = new_index
                # Extrapolating onset-offset values
                words = tokenize(tmp_text, args.language)
                onsets = np.linspace(ref_df.onsets.iloc[max(0, index-1)], ref_df.onsets.iloc[index], len(words))
                offsets = np.linspace(ref_df.offsets.iloc[max(0, index-1)], ref_df.offsets.iloc[index], len(words))
                result += zip(onsets, offsets, words)
            df = pd.DataFrame(result, columns=['onsets', 'offsets', 'word'])
            saving_path = os.path.join(args.save, 'text_{}_{}_onsets-offsets_run{}.csv'.format(args.language, args.model_name, run + 1))
            df.to_csv(saving_path)



    except AssertionError:
        print('You do not have the same number of onsets files and text-to-tokenize files ...')