from datetime import date
from lxml.html import HtmlElement
from lxml import html
from common.data import loadFeatureTemplates, loadTrainingData
import requests
from bisect import bisect_left
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import string
import datefinder

class Params:
    ELEMENT_NEIGHBOUR_RANGE = 5
    NO_RANDOM_NEGATIVE_SAMPLES = 20
    RANGE_TRAINING_CROP_FACTOR = 4
    
    URLS_DATA_FP = "./data/urls.text"
    URLS_INGREDNO_DATA_FP = "./data/urlsIngredNo.txt"
    
    WORD_COUNT_FEATURE_NAME = "wordCount"
    INGREDIENTS_FEATURE_NAME = "ingredients"
    NUMBERS_FEATURE_NAME = "numbers"
    UNITS_FEATURE_NAME = "units"
    
    IMPERATIVES_FEATURE_NAME = "imperatives"
    UTENSILS_FEATURE_NAME = "utensils"
    
    NUTR_INFO_FEATURE_NAME = "nutrInfo"
    
    DATE_TIME_FEATURE_NAME = "dateTime"
    
    FEATURE_NAMES = [INGREDIENTS_FEATURE_NAME, IMPERATIVES_FEATURE_NAME, UNITS_FEATURE_NAME, UTENSILS_FEATURE_NAME, NUTR_INFO_FEATURE_NAME]
    
    LABELLED_CANDS_DATA_FOLDER = "./data/labelledCandidates/"
    FEATURE_TEMPLATE_DATA_FOLDER = "./data/featureTemplates/"
    
    TRAINING_DATA_FP = "./data/trainingData.txt"
    FEATURE_TEMPLATE_DATA_FPS = {
        INGREDIENTS_FEATURE_NAME: FEATURE_TEMPLATE_DATA_FOLDER + "ingredients.txt",
        IMPERATIVES_FEATURE_NAME: FEATURE_TEMPLATE_DATA_FOLDER + "imperatives.txt",
        UNITS_FEATURE_NAME: FEATURE_TEMPLATE_DATA_FOLDER + "units.txt",
        UTENSILS_FEATURE_NAME: FEATURE_TEMPLATE_DATA_FOLDER + "utensils.txt",
        NUTR_INFO_FEATURE_NAME: FEATURE_TEMPLATE_DATA_FOLDER + "nutritionalInformation.txt",
        DATE_TIME_FEATURE_NAME: FEATURE_TEMPLATE_DATA_FOLDER + "dateTimes.txt"
    }

def treeFromUrl(url):
    response = requests.get(url)
    # response.encoding = 'utf-8'
    parser = html.HTMLParser(remove_comments=True) # encoding='utf-8'
    # head = bytes("<!ENTITY nbsp ' '>", response.encoding)
    tree = html.fromstring(response.content)
    return tree

def extractCandidates(tree):
    validTags = ["p", "div", "li", "td"]
    minCharNo = 200
    
    def isValidNode(node):
        if not isinstance(node, HtmlElement):
            return False
        if node.tag not in validTags:
            return False
        text = refineText(node.text_content())
        if text is None or not text or len(text) > minCharNo:
            return False
        return True
    
    def refineText(text):
        spaceChars = ['\n', '\xa0']
        for c in spaceChars:
            text = text.replace(c, ' ')
        punctuation = '&(),-./:;'
        fractions = '½¼¾'
        chars = string.ascii_lowercase + string.ascii_uppercase + string.digits + punctuation + fractions
        result = ''
        spaceCount = 0
        for c in text:
            if c == " ":
                if spaceCount == 0:
                    result += " "
                spaceCount += 1
            elif c in chars:
                result += c
                spaceCount = 0
        return result.strip()
    
    cands = []
    for node in tree.iter():
        if isValidNode(node):
            cands.append({
                'node': node,
                'text': refineText(node.text_content())
            })
    return cands

class FeatureExtractor:
    def __init__(self):
        self.initFeatureTemplates()
        
    def initFeatureTemplates(self):
        self.templates = templates = {}
        for name in Params.FEATURE_NAMES:
            templates[name] = loadFeatureTemplates(Params.FEATURE_TEMPLATE_DATA_FPS[name])
    
    def getData(self, text):
        data = {}
        data['text'] = text
        data[Params.WORD_COUNT_FEATURE_NAME] = text.count(" ") + 1
        data[Params.INGREDIENTS_FEATURE_NAME] = self.getIngredients(text)
        data[Params.NUMBERS_FEATURE_NAME] = self.getNumbers(text)
        data[Params.UNITS_FEATURE_NAME] = self.getUnits(text)
        
        data[Params.IMPERATIVES_FEATURE_NAME] = self.getImperatives(text)
        data[Params.UTENSILS_FEATURE_NAME] = self.getUtensils(text)
        
        data[Params.DATE_TIME_FEATURE_NAME] = self.getDateTimes(text)
        
        data[Params.NUTR_INFO_FEATURE_NAME] = self.getNutrInfo(text)
        return data
        
    def getVector(self, data=None, text=None):
        if data is None:
            if text is None:
                print("Can't obtain vector no data provided")
                return None
            else:
                data = self.getData(text)
            
        # Basic features
        noWords = data[Params.WORD_COUNT_FEATURE_NAME]
        noIngreds = len(data[Params.INGREDIENTS_FEATURE_NAME])
        noNumbers = len(data[Params.NUMBERS_FEATURE_NAME])
        noUnits = len(data[Params.UNITS_FEATURE_NAME])
        
        # Method exclusion features
        noImps = len(data[Params.IMPERATIVES_FEATURE_NAME])
        noUtens = len(data[Params.UTENSILS_FEATURE_NAME])
        
        # Nutr info exclusion features
        noNutrInfo = len(data[Params.NUTR_INFO_FEATURE_NAME])
        
        # Comment exclusion features
        noDateTime = len(data[Params.DATE_TIME_FEATURE_NAME])
        
        vector = [noWords, noIngreds, noNumbers, noUnits, noImps, noUtens, noNutrInfo]
        return vector
    
    def searchTextForTemplate(self, text, template, allowPlural, allowNumberHead):
        def isValidHead(head):
            numbers = [str(x) for x in range(10)]
            if head[-1] == " ":
                return True
            if allowNumberHead and head[-1] in numbers:
                rest = head[:-1]
                if not rest:
                    return True
                return isValidHead(rest)
            return False
        
        def isValidTail(tail):
            if tail[0] == " " or tail[0] == ",": 
                return True
            if allowPlural and tail[0] == "s" and (len(tail) == 1 or tail[1] == " "):
                return True
            return False
        
        ind = text.find(template)
        if ind > -1:
            if ind > 0:
                head = text[:ind]
                if not isValidHead(head):
                    return -1
            
            indT = ind + len(template) 
            if indT < len(text):
                tail = text[indT:]
                if not isValidTail(tail):
                    return -1
            return ind
        return -1
    
    def insertMatch(self, temp, ind, matches, inds):
        i = bisect_left(inds, ind)
        if i < len(inds):
            if inds[i] == ind:
                if len(matches[i]) > len(temp):
                    i += 1
        inds.insert(i, ind)
        matches.insert(i, temp)
            
    def searchTextForTemplates(self, text, templates, lowerCase=True, allowPlural=False, allowNumberHead=False):      
        if lowerCase:
            text = text.lower()
        matches = []
        inds = []
        for temp in templates:
            ind = self.searchTextForTemplate(text, temp, allowPlural=allowPlural, allowNumberHead=allowNumberHead)
            if ind > -1:
                if inds:
                    self.insertMatch(temp, ind, matches, inds)
                else:
                    matches.append(temp)
                    inds.append(ind)
        return matches
            
    def getIngredients(self, text):
        ingredients = self.templates[Params.INGREDIENTS_FEATURE_NAME]
        ingredsFound = self.searchTextForTemplates(text, ingredients, allowPlural=True)
        return ingredsFound
    
    def getNumbers(self, text):
        digits = [str(x) for x in range(10)]
        number = ''
        numbers = []
        inds = []
        for i in range(len(text)):
            c = text[i]
            if c in digits:
                number += c
            elif len(number):
                ind = bisect_left(inds, i)
                numbers.insert(ind, number)
                inds.insert(ind, i)
                number = ''
        return numbers   
    
    def getUnits(self, text):
        units = self.templates[Params.UNITS_FEATURE_NAME]
        unitsFound = self.searchTextForTemplates(text, units, allowNumberHead=True)
        return unitsFound 
    
    def getImperatives(self, text):
        imperatives = self.templates[Params.IMPERATIVES_FEATURE_NAME]
        impsFound = self.searchTextForTemplates(text, imperatives)
        return impsFound
    
    def getUtensils(self, text):
        utensils = self.templates[Params.UTENSILS_FEATURE_NAME]
        utensilsFound = self.searchTextForTemplates(text, utensils)
        return utensilsFound 
    
    def getNutrInfo(self, text):
        nutrInfoWords = self.templates[Params.NUTR_INFO_FEATURE_NAME]
        wordsFound = self.searchTextForTemplates(text, nutrInfoWords)
        return wordsFound
    
    def searchTextforNumericalDate(text):
        dateSeps = ["/","-"]
        for sep in dateSeps:
            inds = [i for i, c in enumerate(text) if c == sep]
            for i in range(len(inds)):
                ind = inds[i]
                if inds[i+1] == ind+2 or inds[i+1] == ind+3:
                    dates = datefinder.find_dates(text)
                    return dates
        return []
            
    
    def getDateTimes(self, text):
        textDTs = self.templates[Params.DATE_TIME_FEATURE_NAME]
        textDTsFound = self.searchTextForTemplates(text, textDTs)
        
        numDTsFound = self.searchTextForNumericalDate(text)
        
        return textDTsFound + numDTsFound
    
    def addNeighbourFeatures(self, X):
        def deriveNeighbourVec(subX, n):
            vec = [0]*(n-1)
            for v in subX:
                for i, val in enumerate(v[1:]):
                    vec[i] += val    
            return vec
    
        n = Params.ELEMENT_NEIGHBOUR_RANGE
        # preVecs, postVecs = [], []
        addVecs = []
        for i in range(len(X)):
            i1 = max(i-n,0)
            i2 = min(i+n, len(X)-1)
            preX = X[i1:i]
            postX = X[i+1:i2+1]
            addVec = deriveNeighbourVec(preX + postX, len(X[0]))
            addVecs.append(addVec)
            # preVec = deriveNeighbourVec(preX, len(X[0]))
            # postVec = deriveNeighbourVec(postX, len(X[0]))
            # preVecs.append(preVec)
            # postVecs.append(postVec)
        
        # X = [X[i] + preVecs[i] + postVecs[i] for i in range(len(X))]
        X = [X[i] + addVecs[i] for i in range(len(X))]
        return X
        
class Classifier:
    def __init__(self):
        self.X, self.y = loadTrainingData(Params.TRAINING_DATA_FP)
        self.clf = svm.SVC(class_weight={0: 1, 1: 2}) # class_weight={0: 1, 1: 5}
        # self.clf = LogisticRegression(class_weight={0: 1, 1: 5})
        self.clf.fit(self.X, self.y)
    
    def test(self, onlyPositive=False):
        X, y, clf = self.X, self.y, self.clf
        
        testX, testY = X[:1000], y[:1000]
        trainX, trainY = X[1000:], y[1000:]
        
        if onlyPositive:
            iArr = [i for i, y in enumerate(testY) if y == 1]
            testX, testY = [testX[i] for i in iArr], [testY[i] for i in iArr]
        
        clf.fit(trainX, trainY)
        accuracy = clf.score(testX, testY)
        clf.fit(X, y)
        return accuracy
        
    def predict(self, X):
        y = self.clf.predict(X)
        return y