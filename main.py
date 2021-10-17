#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 01:44:06 2021

@author: samwestlake
"""

from lxml import html
from common.utilities import Classifier, FeatureExtractor, extractCandidates
import requests

url = "https://www.taste.com.au/recipes/bobotie-south-african-curried-mince-pie/6fa6bf12-8aca-4509-80d4-dd624d691f2b?r=recipes/familyrecipes&c=s4vstj32/Family%20recipes"
parser = html.HTMLParser(remove_comments=True)
response = requests.get(url)
tree = html.fromstring(response.content)
cands = extractCandidates(tree)
classifier = Classifier()
featureExtractor = FeatureExtractor()

#NB Include features: 'to serve' feature to pick out ingredients with no quantity (usually at end)
#                       use of 'I'/'we', adjectives, date/time (and names?) to exclude comments
#                       more involved grammar analysis?
#                       no. capital letters to exclude other recipe titles
#                       'for', ':' or h# tags to exclude ingredient list subheadings
#                       'recipe' (/'ideas') to exclude branching links to other recipe groups
#NB Consider weighting data cases: non-ingredients e.g. Save button between ingredient list and method (both containing ingreds),
#                                       ingredient located at end of ingredient list
# Cutout elements whose text resides solely in a hyperlink, from viable candidates
# Cutout 'save' buttons
# More data e.g. skinnytaste.com
# Headers to prevent access denied e.g. https://www.tasteofhome.com/
# Filter out candidates with parent/child relationship (text duplicates) before further operations so don't corrupt neighbour analysis
# Filter ingredients matched, removing subset ingredients e.g. 'cheese' in 'cream cheese'
# Consider registering duplicate template matches i.e. capture template repetition in text

X = []
for cand in cands:
    text = cand['text']
    data = featureExtractor.getData(text)
    vec = featureExtractor.getVector(data)
    X.append(vec)
X = featureExtractor.addNeighbourFeatures(X)
y = classifier.predict(X)
bestCands = [cands[i] for i,x in enumerate(X) if y[i] == 1]
for c in bestCands:
    print(c)
    