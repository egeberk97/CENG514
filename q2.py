import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import csv

def firstPassSON(partition):
    #pass of son return is the frrequent sets
    encoder = TransactionEncoder()
    transactions = encoder.fit(partition).transform(partition)
    df = pd.DataFrame(transactions, columns=encoder.columns_)
    x = apriori(df, min_support=0.02, use_colnames=True)
    x['setofpartition'] = x['itemsets'].apply(lambda x: sorted(x))
    return list(x['setofpartition'])

def originalApriori(grocery):
    #original apriori return is the number of sets
    encoder = TransactionEncoder()
    transactions = encoder.fit(grocery).transform(grocery)
    df = pd.DataFrame(transactions, columns=encoder.columns_)
    x = apriori(df, min_support=0.02,use_colnames=True)
    print("Number of frequent items generated in the original Apriori : %d"  %(len(x)))
    return len(x)


if __name__ == '__main__':
    ##read the data and turn to list
    with open('groceries.csv', newline='') as f:
        reader = csv.reader(f, delimiter=";")
        grocery = list(reader)

    numberOfOriginalApriori = originalApriori(grocery)
    rations = []
    itemsetsOfSON = []
    for i in range(5,11):
        chunksize = len(grocery)/i
        itemset = []
        ##partion the data
        for j in range(i):
            start = int(j*chunksize)
            end = int(min((j+1)*chunksize,len(grocery)))
            partion = grocery[start:end]
            setSon = firstPassSON(partion)
            #union the set
            for x in setSon:
                if x not in itemset:
                    itemset.append(x)
        itemsetsOfSON.append(len(itemset))
        rations.append(len(itemset)/numberOfOriginalApriori)
        print("Number of chunks =", i,", and number of frequent itemset =",len(itemset))
        print("Ratio for", i, "chunks =", len(itemset)/numberOfOriginalApriori)

