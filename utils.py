#!/usr/bin/python3
import math
    
def tf(term_frequency):
    return 0 if term_frequency == 0 else float(1 + math.log(term_frequency, 10))


def idf(N, doc_frequency):
    if doc_frequency == 0:
        return 0
    return float(math.log(float(N / doc_frequency), 10))


