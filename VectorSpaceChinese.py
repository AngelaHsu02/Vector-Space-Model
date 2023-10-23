from pprint import pprint
from Parser import Parser
import util
import os
import jieba
import re
import numpy as np


class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentTFVectors = {} #member variables成員函式
    documentTFIDFvectors = {}
    #Mapping of vector index to keyword
    vectorKeywordIndex=[] #member variables

    #Tidies terms
    parser=None #member variables


    def __init__(self, documents=[]): #constructor
        self.documentTFVectors={} #member variables
        self.documentTFIDFvectors={}        
        self.parser = Parser() #member variables
        if(len(documents)>0): 
            self.build(documents)

    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents.values()) #member variables
        self.documentTFVectors = {name : self.makeVector(document) for name, document in documents.items()}

        getdocumentTFvectors = np.array([self.makeVector(document) for document in documents.values()])
        # print(getdocumentTFvectors)
        getdocumentDFvectors = np.array([self.makeDFVector(document) for document in documents.values()])
        sumdocumentDFvectors = np.sum(np.array(getdocumentDFvectors), axis=0)
        documentIDFvector = np.log(txtcount / (1 + sumdocumentDFvectors))  # 避免分母為0，加1避免log(0)錯誤
        # print(documentIDFvector)
        documentTFIDFvector = getdocumentTFvectors*documentIDFvector
        # print(documentTFIDFvector)        
        self.documentTFIDFvectors={}
        for idx, (docname, TFIDFvector) in enumerate(zip(documents.keys(), documentTFIDFvector)):
            self.documentTFIDFvectors[docname] = TFIDFvector.tolist()
        # print(self.documentTFIDFvectors)

    def makeDFVector(self, wordString):
        # print(wordString)
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = jieba.cut(wordString, cut_all=False)
        wordList = [word for word in wordList]
        # print(wordList)
        for keyword in self.vectorKeywordIndex.keys():
            if keyword in wordList:
                vector[self.vectorKeywordIndex[keyword]] = 1; #Use simple Term Count Model
        # print(vector)
        return vector
        

    def getVectorKeywordIndex(self, documentList):
        vocabularyString = " ".join(documentList)
        # print(vocabularyString)
        vocabularyList = jieba.cut(vocabularyString, cut_all=False)
        vocabularyList = [vocabulary for vocabulary in vocabularyList]
        # print(vocabularyList)

        vectorIndex={}
        offset=0
        for word in set(vocabularyList): #重複也會設index  初始documentVector=len(index)沒有重複少
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)

    def makeVector(self, wordString):
        # print(wordString)
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = jieba.cut(wordString, cut_all=False)
        wordList = [word for word in wordList]

        for word in wordList:
            if word in self.vectorKeywordIndex:
                vector[self.vectorKeywordIndex[word]] += 1 #Use simple Term Count Model
        return vector


    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        query = self.makeVector(" ".join(termList))
        return query
   
    def searchCosine(self,searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)
        ratings = {name : util.cosine(queryVector, documentTFVector) for name, documentTFVector in self.documentTFVectors.items()}  #TF + Cosine Similarity Ranking
        # ratings = {name : util.cosine(queryVector, documentTFIDFvector) for name, documentTFIDFvector in self.documentTFIDFvectors.items()}  #TFIDF + Cosine Similarity Ranking
        return sorted(ratings.items(), key=lambda x:x[1], reverse=True)[:10]



if __name__ == '__main__':
    txtcount=0
    documents={}
    # root_folder='./cnsample'
    root_folder='./ChineseNews'
    for root, dirs, files in os.walk(root_folder):
        for txt in files:
                txtcount+=1
                with open(os.path.join(root, txt), encoding="utf8") as file:
                    txtdata = file.read()
                    txtdata = re.sub(r'[^\u4e00-\u9fa5]', '', txtdata)
                    documents[txt] = txtdata      
    print(f"讀入{txtcount}本txt")
    # query="蘋果 大會"   
    query="迪士尼 元宇宙"   

    vectorSpace = VectorSpace(documents)

    #print(vectorSpace.vectorKeywordIndex)
    print(f"TF + Cosine Similarity Ranking:{vectorSpace.searchCosine([query])}") #7(3th order error)
    # print(f"TFIDF + Cosine Similarity Ranking:{vectorSpace.searchCosine([query])}") #7(4th order error)


###################################################
