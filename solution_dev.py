# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 14:55:01 2020

@author: Steven
"""

##################
## GET THE DATA ##
##################

# load in data
import helper
data_dir = './data/Seinfeld_Scripts.txt'
text = helper.load_data(data_dir)




######################
## EXPLORE THE DATA ##
######################

view_line_range = (0, 10)

import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

print()
print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))




########################################
## Implement Pre-processing Functions ##
########################################

import problem_unittests as tests


# create a list of words
words = text.split()
words.index("jerry:")
# /n is not kept when splitting with split function. When running, token_lookup() will be ran first, replacing /n by ||Return||, it's how we will capture
# the line breaks
# ALSO, why do we need "text = text[81:] in the helper file?" put that in comments?

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    
    # return tuple
    return (None, None)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)
