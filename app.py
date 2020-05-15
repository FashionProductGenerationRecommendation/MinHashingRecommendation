import re
import hashlib
import numpy as np
from pyspark import SparkContext, SparkConf

def formatRow(sentence):
    sentence = sentence.replace(",", " ")
    if sentence is not None:
        sentence = sentence.lower()
    pattern = r'((?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))'
    words = re.findall(pattern, sentence)
    productId = int(words[0])
    productDiscription = words[1:]
    productDiscription = list(set(productDiscription))
    return (productId, productDiscription) 

def hash_(word, productId, a, b, numberOfHashFunctions):
    uniqueWordNumber = int(hashlib.sha256(word.encode('utf-8')).hexdigest(), 16)
    output = []
    for i in range(0,numberOfHashFunctions):
        hashName = i + 1
        hashValue = ((uniqueWordNumber) + b[i]) % ((10**5) + a[i])
        output.append(((productId, hashName),hashValue))
    return output     

numberOfHashFunctions = 500
b = [(2*i) + 7 for i in range(0, numberOfHashFunctions)]
a = [i + 3 for i in range(0, numberOfHashFunctions)]
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)
numberOfHashFunctions = sc.broadcast(numberOfHashFunctions)
a = sc.broadcast(a)
b = sc.broadcast(b)
productsDataset = sc.textFile('/home/keshav/Stony Brook/Big Data/MinHashingRecommendation/styles.csv')
productsAndItsDiscription = productsDataset.map(lambda row: formatRow(row))
wordAndProductAssociation = productsAndItsDiscription.flatMap(lambda value: [(x, value[0]) for x in value[1]])
productIdHashNameAndHashValue = wordAndProductAssociation.flatMap(lambda value: hash_(value[0],value[1], a.value, b.value, numberOfHashFunctions.value))
signatureMatrix = productIdHashNameAndHashValue.reduceByKey(lambda a,b: min(a,b))
productIdAndHashNameMinHashValue = signatureMatrix.map(lambda x: (x[0][0],(x[0][1],x[1])))
productIdAndHashNameMinHashValueList = productIdAndHashNameMinHashValue.groupByKey().map(lambda x: (x[0], list(x[1])))
productIdAndHashNameMinHashValueSorted = productIdAndHashNameMinHashValueList.map(lambda x: (x[0], str(sorted(x[1], key=lambda val: val[0]))))
df = sqlContext.createDataFrame(productIdAndHashNameMinHashValueSorted)
df.toPandas().to_csv("/home/keshav/Stony Brook/Big Data/MinHashingRecommendation/signatureMatrix.csv", header=False)

# productIdAndHashNameMinHashValue = signatureMatrix.map(lambda x: (x[0][0],(x[0][1],x[1])))
# a = [i+1 for i in range(0, numberOfHashFunctions)]
# print (hashName, " = ", uniqueWordNumber, "+", b)
# createRow = signatureMatrix.map(lambda x: (x[0][0], x[0][1], x[1]))
# df.write.csv('/home/keshav/Stony Brook/Big Data/MinHashingRecommendation/signatureMatrix.csv')
# df.coalesce(1).write.csv('/home/keshav/Stony Brook/Big Data/MinHashingRecommendation/signatureMatrix1.csv')
