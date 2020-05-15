# Find ProductIds Similar to ProductId : 9204
# 9204,Men,Footwear,Shoes,Casual Shoes,Black,Summer,2011,Casual,Puma Men Future Cat Remix SF Black Casual Shoes
# 53759,Men,Apparel,Topwear,Tshirts,Grey,Summer,2012,Casual,Puma Men Grey T-shirt
# 29114,Men,Accessories,Socks,Socks,Navy Blue,Summer,2012,Casual,Puma Men Pack of 3 Socks
# 1855,Men,Apparel,Topwear,Tshirts,Grey,Summer,2011,Casual,Inkfruit Mens Chain Reaction T-shirt
# 33846,Men,Footwear,Shoes,Casual Shoes,Black,Summer,2012,Casual,Puma Men Future Cat Black Shoes

import re
import ast
import hashlib
import numpy as np
from pyspark import SparkContext, SparkConf

def convertToList(s):
    return eval( "[%s]" % s )

def processLine(line):
    start = line.find(',')
    line = line[start+1:]
    end = line.find(',')
    productId = int(line[0: end])
    line = line[end+1:]
    correspondingHashInformation = ast.literal_eval(line)
    correspondingHashInformation = convertToList(correspondingHashInformation)[0]
    return (productId, correspondingHashInformation)

def findSimilarProducts(baseProduct, product):
    numberOfHashes = len(baseProduct[1])
    intersection = 0
    for i in range(0, numberOfHashes):
        if baseProduct[1][i][1] == product[1][i][1]:
            intersection += 1
    similarityScore = intersection / numberOfHashes
    return ((baseProduct[0], product[0]), similarityScore)

sc = SparkContext.getOrCreate()
path = '/home/keshav/Stony Brook/Big Data/MinHashingRecommendation/signatureMatrix.csv'
baseProductId = 9204
baseProductId = sc.broadcast(baseProductId)
signatureMatrixDataset = sc.textFile(path)
signatureMatrixDatasetSplit = signatureMatrixDataset.map(lambda line: processLine(line))
baseProduct = signatureMatrixDatasetSplit.filter(lambda x: x[0] == baseProductId.value).collect()[0]
baseProduct = sc.broadcast(baseProduct)
similarityScoreForEachProductWrtBaseProduct = signatureMatrixDatasetSplit.map(lambda product: findSimilarProducts(baseProduct.value, product))
similarityScoreForEachProductWrtBaseProductFiltered = similarityScoreForEachProductWrtBaseProduct.filter(lambda x: x[0][0] != x[0][1]) 
sortedBySimilarityScore = similarityScoreForEachProductWrtBaseProductFiltered.top(20, key=lambda x: x[1])
print (sortedBySimilarityScore)

# Results
# Base Product ID : 9204
# Top 20 Recommendations for ProductId : 9204
# [((9204, 33846), 0.696), ((9204, 33843), 0.696), ((9204, 26635), 0.696), ((9204, 6380), 0.69), ((9204, 22912), 0.674), ((9204, 13989), 0.664), ((9204, 26637), 0.642), ((9204, 18473), 0.638), ((9204, 8914), 0.622), ((9204, 9137), 0.59), ((9204, 5896), 0.576), ((9204, 22162), 0.576), ((9204, 22199), 0.576), ((9204, 26639), 0.572), ((9204, 38534), 0.566), ((9204, 12892), 0.564), ((9204, 26636), 0.564), ((9204, 33845), 0.564), ((9204, 38533), 0.564), ((9204, 33847), 0.564)]