#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:54:01 2024

@author: nileshgupta
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ["London Paris London", "Paris Paris London"]
cv = CountVectorizer()

count_matrix = cv.fit_transform(text)


similarity_scores = cosine_similarity(count_matrix)


print(similarity_scores)