from os import listdir

def loadLinesFromFile(fp):
    with open(fp, 'r') as f:
        lines = f.readlines()
        return lines
    
def loadUrlsFromFile(fp):
    lines = loadLinesFromFile(fp)
    lines = [tuple(l.strip().split("|")) for l in lines]
    urls, ingredNos = zip(*lines)
    return urls, ingredNos
    
def loadFeatureTemplates(fp):
    with open(fp, 'r') as f:
        lines = f.readlines()
        lines = [l.lower().strip() for l in lines]
    return lines

def loadLabelledCandidates(directory):
    fps = [directory + fn for fn in listdir(directory)]
    labelledCands = []
    for fp in fps:
        with open(fp, 'r') as f:
            lines = f.readlines()
            cands = [tuple(l.strip().split("|")) for l in lines]
            labelledCands.append(cands)
    return labelledCands

def loadTrainingData(fp):
    with open(fp, 'r') as f:
        lines = f.readlines()
        data = [tuple(l.strip().split("|")) for l in lines]
        data = [(eval(vec), int(label)) for vec, label in data]
        X, y = zip(*data)
    return X, y