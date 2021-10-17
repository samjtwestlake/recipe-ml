import os
import tkinter as tk
from common.utilities import treeFromUrl, extractCandidates, Params, FeatureExtractor, Classifier
from common.data import loadUrlsFromFile, loadLabelledCandidates
import urllib
import random

def testModel(onlyPositive=False):
    classifer = Classifier()
    accuracy = classifer.test(onlyPositive=onlyPositive)
    print(accuracy)
 
def saveTrainingData():
    def cropPageTrainingData(pageX, pageY):
        iArr = [i for i,y in enumerate(pageY) if y == 1]
        iMin, iMax = iArr[0], iArr[-1]
        r, cf = Params.ELEMENT_NEIGHBOUR_RANGE, Params.RANGE_TRAINING_CROP_FACTOR
        i1, i2 = iMin - r*cf, iMax + r*cf
        X, y = pageX[i1:i2+1], pageY[i1:i2+1]
        
        iArr = []
        a, n, s = 0, Params.NO_RANDOM_NEGATIVE_SAMPLES, 10
        iMin, iMax = max(i1 - s, 0), i2
        while a < n//s:
            i = random.randint(0, len(pageX)-1)
            if i < iMin or i > iMax:
                X += pageX[i:i+s+1]
                y += pageY[i:i+s+1]
                a += 1
        return X, y
    
    pagesCands = loadLabelledCandidates(Params.LABELLED_CANDS_DATA_FOLDER)
    featureExtractor = FeatureExtractor()
    X, y = [], []
    for labelledCands in pagesCands:
        pageX, pageY = [], []
        labelledCands = [x for x in labelledCands if len(x) == 2]
        for text, label in labelledCands:
            data = featureExtractor.getData(text)
            vec = featureExtractor.getVector(data)
            pageX.append(vec)
            pageY.append(int(label))
        pageX = featureExtractor.addNeighbourFeatures(pageX)
        pageX, pageY = cropPageTrainingData(pageX, pageY)
        X += pageX
        y += pageY
        
    fp = Params.TRAINING_DATA_FP
    with open(fp, 'w') as f:
        for i in range(len(X)):
            f.write(str(X[i]) + "|" + str(y[i]) + "\n")

def labelCandidatesByGUI():
    class labellingGUI:
        def __init__(self, url, ingredNo, texts):
            self.texts = texts
            self.shiftHeld = False
            self.lastCheckRow = None
            self.totalChecked = 0
            
            def setTitle():
                self.window.title(url + " | " + str(self.totalChecked) + "/" + str(ingredNo))
            self.window = window = tk.Tk()
            setTitle()
            
            scrollbar = tk.Scrollbar(window)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            canvas = tk.Canvas(window, yscrollcommand=scrollbar.set, borderwidth=2, relief=tk.SOLID, width=800, height=600)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.TRUE)
            scrollbar.config(command=canvas.yview)
            frame = tk.Frame(canvas)
            frame_id = canvas.create_window(0, 0, window=frame, anchor=tk.NW) 
            
            def onCheckBox(row):
                if self.shiftHeld:
                    i1, i2 = sorted((row, self.lastCheckRow))
                    for i in range(i1, i2+1):
                        checkButton = frame.grid_slaves(i, 9)[0]
                        checkButton.select()
                self.totalChecked = sum([var.get() for var in self.labelVars])
                setTitle()
                self.lastCheckRow = row
            
            self.labelVars = labelVars = []
            for i, text in enumerate(texts):
                labelVars.append(tk.IntVar())
                frame.columnconfigure(0, weight=1)
                tk.Label(frame, text=text).grid(columnspan=9, row=i) 
                button = tk.Checkbutton(frame, variable=labelVars[i], command=lambda i=i: onCheckBox(i))
                button.grid(column=9, columnspan=1, row=i)
                
            def onConfigure(e):
                canvas.configure(scrollregion=canvas.bbox('all'))
                canvas.itemconfig(frame_id, width=e.width)
                row = self.deriveStartRow(texts)
                canvas.yview_moveto(float(row) / len(texts))
                
            canvas.bind("<Configure>", onConfigure)      
                
            def onShiftDown(e):
                self.shiftHeld = True
                
            def onShiftUp(e):
                self.shiftHeld = False
                
            window.bind("<KeyPress-Shift_L>", onShiftDown)
            window.bind("<KeyRelease-Shift_L>", onShiftUp)
            
        def deriveStartRow(self, texts):
            featureExtractor = FeatureExtractor()
            strongCandRows = []
            objs = []
            # data = featureExtractor.getData(texts[247])
            for i, text in enumerate(texts):
                featureData = featureExtractor.getData(text)
                if featureData[Params.INGREDIENTS_FEATURE_NAME] and featureData[Params.UNITS_FEATURE_NAME]:
                    objs.append(featureData)
                    strongCandRows.append(i)
            iAvg = sum(strongCandRows) // len(strongCandRows)
            return iAvg
        
        def run(self):
            self.window.mainloop()
            labels = [var.get() for var in self.labelVars]
            zipped = list(zip(self.texts, labels))
            # zipped.sort(key=lambda x: x[1], reverse=True)
            texts, labels = zip(*zipped)
            return texts, labels
    
    def displayGUI(url, ingredNo, texts):
        gui = labellingGUI(url, ingredNo, texts)
        texts, labels = gui.run()
        return texts, labels
        
    
    urls, ingredNos = loadUrlsFromFile(Params.URLS_INGREDNO_DATA_FP)
    for url, ingredNo in list(zip(urls, ingredNos))[13:]:
        tree = treeFromUrl(url)
        cands = extractCandidates(tree)
        texts = [cand["text"] for cand in cands]
        texts, labels = displayGUI(url, ingredNo, texts)
        filename = urllib.parse.quote(url, '')
        fp = "./data/labelledCandidates/" + filename
        with open(fp, 'w') as f:
            for text, isIngredient in zip(texts, labels):
                f.write("{0}|{1}\n".format(text, isIngredient))
                
# labelCandidatesByGUI()
# saveTrainingData()
# NB Try training using fraction of negative labels: extract section around min/max positive label for each page for training
# Using less training improves performance, maybe due to overfitting with excessive number of negative labelss
testModel(onlyPositive=True)