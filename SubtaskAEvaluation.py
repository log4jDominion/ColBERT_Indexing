# SubtaskAEvaluation.py Version 1.2 as of June 30, 2024
# Douglas W. Oard, oard@umd.edu
# Changes from Version 1.1:
# - Replaced hardcoded full path names with a prefix that will be used for all files.
# - Added computation of confidence intervals
# - Added detail to terrier package installation comments
# - Fixed a bug that made the index number unchanging for the random index

import json
import math
import os
import random
import re
import shutil

import PyPDF2
import numpy as np
import pandas as pd
import pyterrier as pt  # For this to work you should pip install python-terrier
import pytrec_eval  # For this to work, you should pip install pytrec-eval-terrier, which requires a version of numpy before 2.0 (such as 1.26.4)

import train_model


### ...................Multi-Label classification...........................###


###....................End Multi-Label Classification........................###



def readExperimentControlFile(fileName):
    with open(fileName) as ecfFile:
        ecf = json.load(ecfFile)
    return ecf

def getSushiFiles(dir):
    fullCollection = {}
    for box in os.listdir(dir):
        print(f'Reading SUSHI collection box {box}')
        fullCollection[box] = {}
        for folder in os.listdir(os.path.join(dir,box)):
            fullCollection[box][folder] = []
#            print(f'Read box {box}, folder {folder}')
            for file in os.listdir(os.path.join(dir,box,folder)):
#                print(f'Read file {os.path.join(dir,box,folder,file)}')
                fullCollection[box][folder].append(file)
    for box in fullCollection:
        fullCollection[box] = sorted(fullCollection[box])
    return fullCollection

def trainRandomModel(trainingDoucments): # This simple index identifies the top 1000 folders, in decreasing order of number of digitized files
    global prefix
    global seq
    dir = prefix + 'sushi-files/'
    folders = {}
    for box in os.listdir(dir):
        if not box.startswith("."):
            for folder in os.listdir(os.path.join(dir, box)):
                if not folder.startswith("."):
                    folders[folder] = []
                    for file in os.listdir(os.path.join(dir, box, folder)):
                        folders[folder].append(file)
    seq += 1
    index = list(folders.keys())
    random.shuffle(index)
    return index[0:1000]

def create_trainingSet(trainingDocs, searchFields):
    noShortOcr = False  # Set to true if you want to replace OCR text that is nearly empty with the document title
    global prefix
    global seq  # Used to control creation of a separate index for each training set
    global unix  # Used to accommodate Terrier's use of var for indexes with Unix
    trainingSet = []

    # Read the Sushi Medadata and SNC excel files
    try:
        xls = pd.ExcelFile(prefix + 'SubtaskACollectionMetadataV1.1.xlsx')
        fileMetadata = xls.parse(xls.sheet_names[0])
        xls = pd.ExcelFile(prefix + 'SncTranslationV1.1.xlsx')
        sncExpansion = xls.parse(xls.sheet_names[0])
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        exit(-1)

    # Build the data structure that Terrier will index (list of dicts, one per indexed item)
    for trainingDoc in trainingDocs:

        # Read the box/folder/file directory structure
        sushiFile = trainingDoc[
                    -10:]  # This extracts the file name and ignores the box and folder labels which we will get from the medatada
        file = sushiFile
        folder = str(fileMetadata.loc[fileMetadata['Sushi File'] == sushiFile, 'Sushi Folder'].iloc[0])
        box = str(fileMetadata.loc[fileMetadata['Sushi File'] == sushiFile, 'Sushi Box'].iloc[0])

        # Construct the best available folder label (either by SNC lookup or by using the folder label text)
        brownLabel = str(fileMetadata.loc[fileMetadata['Sushi File'] == sushiFile, 'Brown Folder Name'].iloc[0])
        naraLabel = str(fileMetadata.loc[fileMetadata['Sushi File'] == sushiFile, 'NARA Folder Name'].iloc[0])
        if naraLabel != 'nan':
            start = naraLabel.find('(')  # Strip part markings
            if start != -1:
                naraLabel = naraLabel[:start]
            naraLabel = naraLabel.replace('BRAZ-A0', 'BRAZ-A 0')  # Fix formatting error
            naraLabel = naraLabel.replace('BRAZ-E0', 'BRAZ-E 0')  # Fix formatting error
            naraLabelElements = naraLabel.split()
            if len(naraLabelElements) in [3, 4]:
                if len(naraLabelElements) == 3:
                    naraSnc = naraLabelElements[0]
                else:
                    naraSnc = ' '.join(naraLabelElements[0:2])
                naraCountryCode = naraLabelElements[-2]
                naraDate = naraLabelElements[-1]
                #                print(f'parsed {naraLabel} to {naraSnc} // {naraCountryCode} // {naraDate}')
                if naraSnc in sncExpansion['SNC'].tolist():
                    label1965 = str(sncExpansion.loc[sncExpansion['SNC'] == naraSnc, 1965].iloc[0])
                    label1963 = str(sncExpansion.loc[sncExpansion['SNC'] == naraSnc, 1963].iloc[0])
                    if label1965 != 'nan':
                        label = label1965
                    elif label1963 != 'nan':
                        label = label1963
                    else:
                        print(f'Unable to translate {naraSnc} for file {sushiFile} in folder {folder}')
                        label = naraSnc
                else:
                    print(f'No expansion for {naraSnc}')
                    label = naraSnc

            else:
                print(f"NARA Folder Title doesn't have four parts: {naraLabel}")
                label = 'Bad NARA Folder Title'
        else:
            if brownLabel != 'nan':
                label = brownLabel.replace('_', ' ')
            else:
                print(f'Missing both NARA and Brown folder labels for file {sushiFile} in folder {folder}')
                label = 'No NARA or Brown Folder Title'
        #        print(f'File {file} Folder {folder} has expanded label {label}')

        # Construct the best available title (either Brown, or trimmed NARA)
        brownTitle = str(fileMetadata.loc[fileMetadata['Sushi File'] == sushiFile, 'Brown Title'].iloc[0])
        naraTitle = str(fileMetadata.loc[fileMetadata['Sushi File'] == sushiFile, 'NARA Title'].iloc[0])
        if brownTitle != 'nan':
            title = brownTitle
        else:
            start = naraTitle.find('Concerning')
            if start != -1:
                naraTitle = naraTitle[start + 11:]
            end1 = naraTitle.rfind(':')
            end2 = naraTitle.rfind('(')
            end = min(end1, end2)
            if end != -1:
                naraTitle = naraTitle[:end]
            title = naraTitle

        # Extract OCR text from the PDF file
        f = open(prefix + 'sushi-files/' + box + '/' + folder + '/' + file, 'rb')
        reader = PyPDF2.PdfReader(f)
        pages = len(reader.pages)
        maxPages = 1  # Increase this number if you want to index more of the OCR text
        fulltext = ''
        for i in range(min(pages, maxPages)):
            page = reader.pages[i]
            text = page.extract_text().replace('\n', ' ')
            fulltext = fulltext + text
        ocr = fulltext

        # Optionally replace any hopelessly short OCR with the document title
        if noShortOcr and len(ocr) < 5:
            print(f'Replaced OCR: //{ocr}// with Title //{title}//')
            ocr = title

        trainingSet.append(
            {'docno': file, 'folder': folder, 'box': box, 'title': title, 'ocr': ocr, 'folderlabel': label})



    return trainingSet

def trainColbertModel(trainingDocs, searchFields):
    trainingSet = create_trainingSet(trainingDocs, searchFields)

    train_model.test_colbert(trainingSet)

def trainTerrierModel(trainingDocs, searchFields):

    trainingSet = create_trainingSet(trainingDocs, searchFields)

    # Create the Terrier index for this training set and then return a Terrier retriever for that index
    seq += 1 # We create one Terrier index per training set
    indexDir = prefix + 'terrierindex/'+str(seq) # Be careful here -- this directory and all its contents will be deleted!
    if 'index' in indexDir and os.path.isdir(indexDir):
        print(f'Deleting prior index {indexDir}')
        shutil.rmtree(indexDir) # This is required because Terrier fails to close its index on completion
    if not pt.started(): pt.init()
    indexer = pt.IterDictIndexer(indexDir, meta={'docno': 20, 'folder':20, 'box': 20, 'title':16384, 'ocr':16384, 'folderlabel': 1024}, meta_reverse=['docno', 'folder', 'box'], overwrite=True)
    indexref = indexer.index(trainingSet, fields=searchFields)
    index = pt.IndexFactory.of(indexref)
    BM25 = pt.BatchRetrieve(index, wmodel="BM25", metadata=['docno', 'folder', 'box'], num_results=1000)
    return BM25

def trainModel(trainingDocuments, searchFields):
    global seq
    global model
    print(f'Training Called, preparing index for experiment set {seq+1}')
    if model == 'random':
        return trainRandomModel(trainingDocuments)
    elif model == 'terrier':
        return trainTerrierModel(trainingDocuments, searchFields)
    else:
        return trainColbertModel(trainingDocuments, searchFields)

def randomSearch(query, index): # This search ignores the query and returns the same ranked list of folders every time
    return index

def terrierSearch(query, engine):
    if not pt.started(): pt.init()
    query = re.sub(r'[^a-zA-Z0-9\s]', '', query) # Terries fails if punctuation is found in a query
    result = engine.search(query)
    rankedList = result['folder']
    rankedList.drop_duplicates(inplace=True)
    return rankedList.tolist()

def search(query, index):
    global model
    if model=='random':
        return randomSearch(query, index)
    elif model=='terrier':
        return terrierSearch(query, index)
    else:
        return train_model.colbert_search(query)

def generateSearchResults(ecf, searchFields):
    results = []
    i = 0
    for experimentSet in ecf['ExperimentSets']:
        index = trainModel(experimentSet['TrainingDocuments'], searchFields)
        topics = list(experimentSet['Topics'].keys())
        for j in range(len(topics)):
            results.append({})
            results[i]['Id'] = topics[j]
            query = experimentSet['Topics'][topics[j]]['TITLE']
            rankedFolderList = search(query, index)
            results[i]['RankedList'] = rankedFolderList
            i+=1
    return results

def writeSearchResults(fileName, results, runName):
    with open(fileName, 'w') as f:
        for topic in results:
            for i in range(len(topic['RankedList'])):
                print(f'{topic["Id"]}\t{topic["RankedList"][i]}\t{i+1}\t{1/(i+1):.4f}\t{runName}', file=f)
    f.close()

def createFolderToBoxMap(dir):
    boxMap = {}
    for box in os.listdir(dir):
        if not box.startswith("."):
            for folder in os.listdir(os.path.join(dir,box)):
                if folder in boxMap:
                    print(f'Duplicate folder {folder} in boxes {box} and {boxMap[folder]}')
                if not folder.startswith("."):
                    boxMap[folder] = box
    return boxMap

def makeBoxRun(folderRun):
    global prefix
    boxMap = createFolderToBoxMap(prefix+'sushi-files/')
    boxRun = {}
    for topicId in folderRun:
        boxRun[topicId]={}
        for folder in folderRun[topicId]:
            if boxMap[folder] not in boxRun[topicId]:
               boxRun[topicId][boxMap[folder]] = folderRun[topicId][folder]
    return boxRun

def stats(results, measure):
    sum=0
    squaredev=0
    n = len(results)
    for topic in results:
        sum += results[topic][measure]
    mean = sum / n
    for topic in results:
        squaredev += (results[topic][measure]-mean)**2
    variance = squaredev / (n-1)
    conf = 1.96 * math.sqrt(variance) / math.sqrt(n)
    return mean, conf

def evaluateSearchResults(runFileName, folderQrelsFileName, boxQrelsFileName):
#    print(pytrec_eval.supported_measures)
    measures = {'ndcg_cut', 'map', 'recip_rank', 'success'} # Generic measures for configuring a pytrec_eval evaluator
    measureNames = {'ndcg_cut_5': 'NDCG@5', 'map': '   MAP', 'recip_rank': '   MRR', 'success_1': '   S@1'} # Spedific measures for printing in pytrec_eval results

    with open(runFileName) as runFile, open(folderQrelsFileName) as folderQrelsFile, open(boxQrelsFileName) as boxQrelsFile:
        folderRun = {}
        for line in runFile:
            topicId, folderId, rank, score, runName = line.split('\t')
            if topicId not in folderRun:
                folderRun[topicId] = {}
            folderRun[topicId][folderId] = float(score)
        boxRun = makeBoxRun(folderRun)
        folderQrels = {}
        for line in folderQrelsFile:
            topicId, unused, folderId, relevanceLevel = line.split('\t')
            if topicId not in folderQrels:
                folderQrels[topicId] = {}
            folderQrels[topicId][folderId] = int(relevanceLevel.strip())  # this deletes the \n at end of line
        folderEvaluator = pytrec_eval.RelevanceEvaluator(folderQrels, measures)
        folderTopicResults = folderEvaluator.evaluate(folderRun)  # replace run with folderQrels to see perfect evaluation measures

        boxQrels = {}
        for line in boxQrelsFile:
            topicId, unused, folderId, relevanceLevel = line.split('\t')
            if topicId not in boxQrels:
                boxQrels[topicId] = {}
            if folderId in boxQrels[topicId]:
                boxQrels[topicId][folderId] = max(boxQrels[topicId][folderId],int(relevanceLevel.strip()))  # strip() deletes the \n at end of line
            else:
                boxQrels[topicId][folderId] = int(relevanceLevel.strip())
        boxEvaluator = pytrec_eval.RelevanceEvaluator(boxQrels, measures)
        boxTopicResults = boxEvaluator.evaluate(boxRun) # replace run with qrels to see perfect evaluation measures

        pm='\u00B1'
        print(f'          Folder          Box')
        for measure in measureNames.keys():
            folderMean, folderConf = stats(folderTopicResults, measure)
            boxMean, boxConf = stats(boxTopicResults, measure)
            print(f'{measureNames[measure]}: {folderMean:.3f}{pm}{folderConf:.2f}    {boxMean:.3f}{pm}{boxConf:.2f}' )

if __name__ == '__main__':
    # Set JAVA_HOME so that Terrier will work correctly
    os.environ["JAVA_HOME"] = "C:/Program Files/Java/jdk-22/" # Install Java if you don't already have it (tested with JDK 22) and then set this to where you have Java installed

    # Set global variables
    prefix = '/Users/shashank/Research/sushi/' # Absolute path for the sushi directory where all files and indexes will be.  Don't use relative paths; doing so alters terrier's behavior in a way that breaks this code.
    seq=0 # Controls index segments
    unix = False # Set to false for Windows, true for Unix.  This adapts the code to the locations where Terrier writes its index.
    model = 'colbert' # Set to 'random' for the random model or to 'terrier' for the Terrier model

    # Run the experiment
    searchFields = ['title', 'ocr', 'folderlabel'] # Used only with Terrier.  Edit this list to index fewer fields if desired
    ecf = readExperimentControlFile(prefix+'Ntcir18SushiDryRunExperimentControlFileV1.1.json')
    results = generateSearchResults(ecf, searchFields)
    writeSearchResults(prefix+'Ntcir18SushiDryRunResultsV1.1.tsv', results, 'Baseline-0')
    evaluateSearchResults(prefix+'Ntcir18SushiDryRunResultsV1.1.tsv', prefix+'Ntcir18SushiDryRunFolderQrelsV1.1.tsv', prefix+'Ntcir18SushiDryRunBoxQrelsV1.1.tsv')

