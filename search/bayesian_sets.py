import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import os
from nlp.pre_process.clean_text import TextCleaner
import re

class BayesianSets:

	def __init__(self, clean_text = True):
		self.clean_text = clean_text

	def score(X, x, c = 2):
	    m = X.mean(0)
	    N = x.shape[0]
	    alfa = c * m
	    beta = c * (1 - m)
	    alfa_ = alfa + x.sum(0)
	    beta_ = beta + N - x.sum(0)
	    nc = (np.log(alfa + beta) - np.log(alfa + beta + N) + np.log(beta_) - np.log(beta)).sum(1)
	    q = np.log(alfa_) - np.log(alfa) + np.log(beta) - np.log(beta_)
	    s = nc + (X * q.T)
	    
	    return s

	def search_query(self, query, corpus):    
	    # clean text
	    if self.clean_text:
	        cleaner = TextCleaner(clean_regex = ".*")
	        query = cleaner.clean(query)
	        corpus = cleaner.clean(corpus)
	    
	    # create DTM
	    vec = CountVectorizer().fit(corpus)
	    DTM = vec.transform(corpus)
	    DTM_query = vec.transform(query)
	    
	    # calculate scores
	    s = self.score(DTM, DTM_query)
	    
	    return s