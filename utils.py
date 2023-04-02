#!/usr/bin/python3
from dateutil.parser import parse
import math

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    Args:
        string (str): string to check for date
        fuzzy (bool): ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False
    
def tf(term_frequency):
    if term_frequency == 0:
        return 0
    return float(1 + math.log(term_frequency, 10))


def idf(N, doc_frequency):
    if doc_frequency == 0:
        return 0
    return float(math.log(float(N / doc_frequency), 10))


