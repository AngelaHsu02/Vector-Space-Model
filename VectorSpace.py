from pprint import pprint
from Parser import Parser
import util
import os
import glob
import nltk
# pos_tag
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

import numpy as np
import math

class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentTFIDFvectors = {}

    #Mapping of vector index to keyword
    vectorKeywordIndex=[]

    #Tidies terms
    parser=None


    def __init__(self, documents=[]):
        self.documentTFIDFvectors={}
        self.parser = Parser()
        if(len(documents)>0):
            self.build(documents)

    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents.values())
        # print(self.vectorKeywordIndex)
        # with open('./index.txt', 'w') as f:
        #     print(self.vectorKeywordIndex, file=f)
        
        getdocumentTFvectors = np.array([self.makeVector(document) for document in documents.values()])
        # print(getdocumentTFvectors)
  
        getdocumentDFvectors = np.array([self.makeDFVector(document) for document in documents.values()])
        sumdocumentDFvectors = np.sum(np.array(getdocumentDFvectors), axis=0)
        documentIDFvector = np.log(len(documents) / (1 + sumdocumentDFvectors))  # 避免分母為0，加1避免log(0)錯誤
        # print(documentIDFvector)

        documentTFIDFvector = getdocumentTFvectors*documentIDFvector
        # print(documentTFIDFvector)        
        self.documentTFIDFvectors={}
        for idx, (docname, TFIDFvector) in enumerate(zip(documents.keys(), documentTFIDFvector)):
            self.documentTFIDFvectors[docname] = TFIDFvector.tolist()
        # print(self.documentTFIDFvectors)
    

        
        

    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string	
        
        vocabularyString = " ".join(documentList)
        # vocabularyString+=query
        # print(vocabularyString)

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        # print(vectorIndex)
        return vectorIndex  #(keyword:position)
    
    def makeDFVector(self, wordString):
        # print(wordString)
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        # print(wordList)
        for keyword in self.vectorKeywordIndex.keys():
            if keyword in wordList:
                vector[self.vectorKeywordIndex[keyword]] = 1; #Use simple Term Count Model
        # print(vector)
        return vector


    def makeVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        # print(wordString)
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        # print(wordList)
        # print(wordList)
        for word in wordList:
            if word in self.vectorKeywordIndex:
                vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model
        # print(vector)
        return vector


    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        # print(termList)
        query = self.makeVector(" ".join(termList)) ###" ".join(termList)
        # print(query) 
        return query
    
    def buildPseudoQueryVector(self, termList):
        for name, cosine in self.searchCosine(termList)[:1]:
            documentFirstname=name
        # print(documentFirstname)
       
        txtdata=''
        with open(f'./EnglishNews/{documentFirstname}', encoding="utf8") as file:
        # with open(f'./sample/{self.documentFirstname}', encoding="utf8") as file:
            data = file.read()
            txtdata += data
        # print(txtdata)
        # print(self.nounAndVerb(txtdata))
        documentFirstVector=self.makeVector((self.nounAndVerb(txtdata)))
        # print(documentFirstVector)
        # print(self.buildQueryVector(termList))
        pseudoQueryVector=[ x+0.5*y for x, y in zip(self.buildQueryVector(termList) , documentFirstVector)]
        # print(pseudoQueryVector)
        return pseudoQueryVector



    def nounAndVerb(self, txtdata):
        # print(txtdata)
        tokens = nltk.word_tokenize(txtdata)
        taggedtokens = nltk.pos_tag(tokens)  
        nounAndVerb = [word for word, pos in taggedtokens if pos.startswith('NN') or pos.startswith('VB')]
        # print(nounAndVerb)
        nounAndVerbString = ' '.join(nounAndVerb)
        # print(nounAndVerbString)
        return nounAndVerbString     



    # def related(self,documentId):
    #     """ find documents that are related to the document indexed by passed Id within the document Vectors"""
    #     ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
    #     #ratings.sort(reverse=True)
    #     return ratings


    def searchCosine(self,searchList):
        """ search for documents that match based on a list of terms """
        # print(searchList)
        queryVector = self.buildQueryVector(searchList)
        # print(queryVector)      
        ratings = {name : util.cosine(queryVector, documentTFIDFvectors) for name, documentTFIDFvectors in self.documentTFIDFvectors.items()}
        return sorted(ratings.items(), key=lambda x:x[1], reverse=True)[:10]


    def searchEuclideandistance(self,searchList):
        queryVector = self.buildQueryVector(searchList)
        ratings = {name : util.euclideandistance(queryVector, documentTFIDFvectors) for name, documentTFIDFvectors in self.documentTFIDFvectors.items()}
        return sorted(ratings.items(), key=lambda x:x[1], reverse=False)[:10]


    def searchPseudo(self,searchList):
        pseudoQueryVector = self.buildPseudoQueryVector(searchList)
        ratings = {name : util.cosine(pseudoQueryVector, documentTFIDFvectors) for name, documentTFIDFvectors in self.documentTFIDFvectors.items()}
        return sorted(ratings.items(), key=lambda x:x[1], reverse=True)[:10]





if __name__ == '__main__':

    txtcount=0
    documents={}

    root_folder='./ensample'
    # root_folder='./EnglishNews'
    for root, dirs, files in os.walk(root_folder):
        for txt in files:
                txtcount+=1
                with open(os.path.join(root, txt), encoding="utf8") as file:
                    txtdata = file.read()
                    documents[txt]=txtdata
      
    # print(f"讀入{txtcount}本txt")
    # print(len(documents))
    # query="Ukraine leader Volodymyr Zelensky finds an idea"
    # query="Trump Taiwan travel"   

    #test data
    # documents = ["The cat in the hat disabled",
    #              "A cat is a fine pet ponies.",
    #              "Dogs and cats make good pets.",
    #              "I haven't got a hat."]
    #             #  "cat eat fish"]

    # query="cat eat fish" 
    vectorSpace = VectorSpace(documents)

    # print(vectorSpace.vectorKeywordIndex)
    # print(vectorSpace.documentVectors)
    # print(vectorSpace.related(1))
    # print(f"Cosine Similarity Ranking:{vectorSpace.searchCosine([query])}") #9
    # print(f"Euclidean Distance Ranking:{vectorSpace.searchEuclideandistance([query])}") #5
    # print(f"Relevance Feedback Ranking:{vectorSpace.searchPseudo([query])}")



###################################################
