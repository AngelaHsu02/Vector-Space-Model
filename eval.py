from VectorSpace import VectorSpace
import os
import pandas as pd
import numpy as np


def recall(queries, labeldict):
    recallAt10List = []

    for queryid in queries.keys():
        # print(queryid)
        relevantDocsid = [str(id) for id in labeldict[queryid]]
        # print(relevantDocsid)
        retrievedDocs = vectorspace.searchCosine(queries[queryid])
        # print(retrievedDocs)
        # print(queries[queryid])

        retrievedDocsid = [doc[0].split('.')[0].strip('d') for doc in retrievedDocs]
        # print(retrievedDocsid)

        retrievedrelevantDocs = sum([1 for id in retrievedDocsid if id in relevantDocsid])
        # print(retrievedrelevantDocs)
        recallAt10 = retrievedrelevantDocs / len(relevantDocsid)
        # print(recallAt10)
        recallAt10List.append(recallAt10)
        # print(recallAt10List)
        

    # print(f"{sum(recallAt10List)} / {len(queries.keys())}")
    avgRecallAt10 = sum(recallAt10List) / len(queries.keys())
    return avgRecallAt10


def map(queries, labeldict):
    apAt10List = []

    for queryid in queries.keys():
        # print(queryid)
        relevantDocsid = [str(id) for id in labeldict[queryid]]
        # print(relevantDocsid)
        retrievedDocs = vectorspace.searchCosine(queries[queryid])
        retrievedDocsid = [doc[0].split('.')[0].strip('d') for doc in retrievedDocs]
        # print(retrievedDocsid)

        idx=0
        retrievedrelevantDocs=0
        precisionList = []
        for id in retrievedDocsid:
             idx += 1
             if id in relevantDocsid:
                  retrievedrelevantDocs += 1
                  precision = retrievedrelevantDocs / idx
                  precisionList.append(precision)
        # print(precisionList)
        if retrievedrelevantDocs > 0:
            apAt10 = sum(precisionList) / retrievedrelevantDocs
            apAt10List.append(apAt10)
            # print(apAt10List)
    
    MapAt10 = sum(apAt10List) / len(queries.keys())
    return MapAt10


def mrr(queries, labeldict):
    rrAt10List = []
    for queryid in queries.keys():
        # print(queryid)
        relevantDocsid = [str(id) for id in labeldict[queryid]]
        # print(relevantDocsid)
        retrievedDocs = vectorspace.searchCosine(queries[queryid])
        retrievedDocsid = [doc[0].split('.')[0].strip('d') for doc in retrievedDocs]
        # print(retrievedDocsid)
        
        idx = 0
        for id in retrievedDocsid:
            idx += 1
            if id in relevantDocsid:
                  rr = 1/int(idx)
                  rrAt10List.append(rr)
                #   print(rrAt10List)
                  break
    MRRAt10 = sum(rrAt10List) / len(queries.keys())
    return MRRAt10          



if __name__ == '__main__':
    documentcount = 0
    querycount = 0
    documents = {}
    queries = {}

    # root_folder='./smaller_dataset/collectionsamplesmaller'
    # root_folder='./smaller_dataset/collectionsample'
    root_folder='./smaller_dataset/collections'
    for root, dirs, files in os.walk(root_folder):
        for txt in files:
                documentcount+=1
                with open(os.path.join(root, txt), encoding="utf8") as file:
                    txtdata = file.read()
                    key = os.path.splitext(txt)[0]
                    documents[txt]=txtdata
    print(f"讀入document : {documentcount}本")
    # print(documents)

    # root_folder='./smaller_dataset/querysamplesmaller'
    # root_folder='./smaller_dataset/querysample'
    root_folder='./smaller_dataset/queries'
    for root, dirs, files in os.walk(root_folder):
        for txt in files:
                querycount+=1
                with open(os.path.join(root, txt), encoding="utf8") as file:
                    txtdata = file.read()
                    key = os.path.splitext(txt)[0]
                    queries[key]=[txtdata]
    print(f"讀入query : {querycount}本")
    # print(queries)

    labeledf = pd.read_csv('./smaller_dataset/rel.tsv', sep='\t', header=None, names=['Query', 'Documents'])
    labeledf['Documents'] = labeledf['Documents'].apply(lambda x: eval(x))  # eval將字串轉列表
    labeldict = labeledf.set_index('Query').to_dict()['Documents']
    # print(labeldict)
    
    vectorspace = VectorSpace(documents)
    print(f"Recall@10: {recall(queries, labeldict)}")
    print(f"MAP@10: {map(queries, labeldict)}")
    print(f"MRR@10: {mrr(queries, labeldict)}")


