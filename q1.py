import itertools
import copy
import random
#import xlsxwriter

random.seed(42) # I give a seed to have same result each run
# returns the subsets with k elements of the given set with
def findsubsets(sets, k):
    return list(map(list,itertools.combinations(sets, k)))

def randomdatagenerator(n = 50):
    """
    :param n:  n is the number of k-item frequent set
    :return: a 2D array
    """
    ## creating a list that contains first 6 letters to use as items
    alphabetlist = ['a','b','c','d','e','f']
    data = []
    for j in range(n):
        setJ = []
        numberofItemset = random.randint(6,20) # how many sets will be occured
        numberofK = random.randint(2,3) # number of k
        for i in range(numberofItemset):
            itemset = random.sample(alphabetlist, k=numberofK)
            itemset = sorted(itemset) # sort the values
            if itemset not in setJ: #to uniqueness
                setJ.append(itemset)
        setJ=sorted(setJ)
        data.append(setJ)
    return data


def generationAlgo1(sets):
    C=[]
    for i in sets:
        for j in sets:
            c = copy.deepcopy(i)
            if i != j and i[:-1]==j[:-1]: ## if k-1 items are same add last element of j
                c.append(j[-1])
                setc = findsubsets(c, len(c)-1)
                check = all(item in sets for item in setc) # it should contains all subsets
                if check and c not in C:
                    C.append(c)
    return len(C)

def generationAlgo2(sets):
    items = []
    ##itemleri ayrı bir yere ekleyelim
    for i in sets:
        for j in i:
            if j not in items:
                items.append(j)

    k = len(sets[0]) # k valueyu ilk itemset üzerinden belirleyelim
    C=[]
    for j in items:
        for i in sets:
            c = copy.deepcopy(i)
            if j not in c:
                c.append(j)
                c = sorted(c)
            if c not in C and len(c) == k+1 : # unique olması ve her bir setin k+1 item taşıması için
                C.append(c)
    C = sorted(C)
    return len(C)


if __name__ == '__main__':
    n=50 #number of random data
    data = randomdatagenerator(n)
    reductions = []
    ## tablo halinde kayıt etmek için runları
    ##workbook = xlsxwriter.Workbook('runs.xlsx')
    ##worksheet = workbook.add_worksheet()

    for i in range(n):
        numberofAlgo1 = generationAlgo1(data[i])
        numberofAlgo2 = generationAlgo2(data[i])
        reduction = numberofAlgo1/numberofAlgo2
        print("Run ",i+1," Reduction = ", reduction)
        reductions.append(reduction)
    ##    worksheet.write(i, 0, i+1)
    ##    worksheet.write(i, 1, reduction)
    print("Average reduction = ", sum(reductions)/len(reductions))
    ##workbook.close()

