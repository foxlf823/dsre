
import logging
import numpy as np
import torch
import random
import struct
import math
from collections import OrderedDict

UNKNOWN_WORD = 'UNK'
NIL_RELATION = 'NA'

def listArguments(args):

    args_dict = vars(args)
    for key,value in args_dict.items():
        logging.info('{} : {}'.format(key, str(value)))
        
def setFixedSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    random.seed(seed)
    
def loadWordEmbFromBinFile(filePath):
    '''
    filePath: the file path
    
    return
    wordVec: tensor, one row one word vector
    wordMapping: dict, key-word, value- word index
    wordList: list, index-word index, value-word
    '''

    
    wordList = []
    wordMapping = {}
    
    with open(filePath, 'rb') as f:
        wordTotal = int(_readString(f))
        dimension = int(_readString(f))
        logging.info('pretrained word number {}'.format(wordTotal))
        logging.info('pretrained word dimension {}'.format(dimension))
        
        wordVec = torch.zeros(wordTotal+1, dimension)
        
        wordList.append(UNKNOWN_WORD)
        wordMapping[UNKNOWN_WORD] = 0
        wordVec[0] = torch.from_numpy(np.random.uniform(-0.01, 0.01, size=(dimension)))
        word_index = 1 # 0 is for UNK, so other words start at 1
        
        for i in range(wordTotal):
            word = _readString(f)
            
            norm = 0.0
            word_vector = []
            for j in range(dimension):
                temp = _readFloat(f)
                word_vector.append(temp)
                norm = norm + temp*temp;
                
            norm = math.sqrt(norm)
            for j in range(dimension):
                word_vector[j] = word_vector[j] / norm
                
            wordMapping[word] = word_index
            wordList.append(word)
            wordVec[word_index] = torch.Tensor(word_vector)
            word_index += 1
                
            f.read(1) # a line break
            
    wordTotal += 1 # plus UNK
            
    return wordVec, wordList, wordMapping, dimension, wordTotal
    
def _readString(f):
    
    s = str()
    c = f.read(1).decode('iso-8859-1')
    while c != '\n' and c!=' ': 
        s = s+c
        c = f.read(1).decode('iso-8859-1')
    
    return s


def _readFloat(f):
    
    bytes4 = f.read(4)
    f_num = struct.unpack('f',bytes4)[0]
    return f_num

def loadIDMappingFile(filePath):
    '''
    filePath: the file path
    
    return
    relationMapping: dict, key-name, value- index
    nam: list, index-index, value-name
    '''

    relationMapping = {}
    nam = []
    relationTotal = 0
    
    with open(filePath, 'r') as f:
        for line in f.readlines():
            
            line = line.strip().split()
            relationMapping[line[0]] = int(line[1])
            relationTotal += 1
            nam.append(line[0])
            
    return relationMapping, nam, relationTotal

def loadData(filePath, testFilePath, wordMapping, relationMapping, limit):
    '''
    filePath: training data
    testFilePath: test data
    
    return
    bags_train: dict, key-e1+e2+relation, value-instance id
    headList: list, e1 id of the instance
    tailList: list, e2 id of the instance
    relationList: list, relation id of the instance
    sentenceLength: list, sentence length of the instance
    trainLists: list list, word id sequence of the instance
    trainPositionE1: e1 position id sequence of the instance
    trainPositionE2: e2 position id sequence of the instance
    bags_test, testheadList, testtailList, testrelationList, test_sentenceLength, testtrainLists, testPositionE1, testPositionE2: similar as above
    PositionMinE1, PositionMaxE1, PositionTotalE1: min, max, total positions of e1
    PositionMinE2, PositionMaxE2, PositionTotalE2: min, max, total positions of e2
    '''
    
    PositionMinE1 = 0
    PositionMaxE1 = 0
    PositionMinE2 = 0
    PositionMaxE2 = 0
    
    bags_train = OrderedDict()
    headList = []
    tailList = []
    relationList = []
    sentenceLength = []
    trainLists = []
    trainPositionE1 = []
    trainPositionE2 = []
    trainPieceWise = []
    
    with open(filePath, 'r') as f:
        for line in f.readlines():        
            line = line.strip().split('\t')
            
            e1 = line[0]
            e2 = line[1]
            
            head_s = line[2]
            head = wordMapping.get(head_s, wordMapping[UNKNOWN_WORD])
            
            tail_s = line[3]
            tail = wordMapping.get(tail_s, wordMapping[UNKNOWN_WORD])
            
            bag_list = bags_train.get(e1+"\t"+e2+"\t"+line[4])
            if bag_list is None:
                bag_list = []
                bag_list.append(len(headList))
                bags_train[e1+"\t"+e2+"\t"+line[4]] = bag_list
            else:
                bag_list.append(len(headList))
            
            num = relationMapping.get(line[4], relationMapping[NIL_RELATION])
            
            length = 0
            lefnum = 0
            rignum = 0
            tmpp = []
            tokens = line[5].split()
            for con in tokens:
                if con=="###END###":
                    break;
                gg = wordMapping.get(con, wordMapping[UNKNOWN_WORD])
                if con == head_s: 
                    lefnum = length
                if con == tail_s: 
                    rignum = length
                length += 1
                tmpp.append(gg)
            
            headList.append(head)
            tailList.append(tail)
            relationList.append(num)
            sentenceLength.append(length)
            
            con = []
            conl = []
            conr = []
            
            for i in range(length):
                con.append(tmpp[i])
                conl.append(lefnum-i)
                conr.append(rignum-i)
                if conl[i] >= limit: conl[i] = limit
                if conr[i] >= limit: conr[i] = limit
                if conl[i] <= -limit: conl[i] = -limit
                if conr[i] <= -limit: conr[i] = -limit
                if conl[i] > PositionMaxE1: PositionMaxE1 = conl[i]
                if conr[i] > PositionMaxE2: PositionMaxE2 = conr[i]
                if conl[i] < PositionMinE1: PositionMinE1 = conl[i]
                if conr[i] < PositionMinE2: PositionMinE2 = conr[i]
            
            trainLists.append(con)
            trainPositionE1.append(conl)
            trainPositionE2.append(conr)
            if lefnum>rignum:
                former = rignum
                latter = lefnum
            else:
                former = lefnum
                latter = rignum
            trainPieceWise.append([former+1, latter-former, length-1-latter])
            
    bags_test = OrderedDict()
    testheadList = []
    testtailList = []
    testrelationList = []
    test_sentenceLength = []
    testtrainLists = []
    testPositionE1 = []
    testPositionE2 = []
    testPieceWise = []
    
    with open(testFilePath, 'r') as f:
        for line in f.readlines():        
            line = line.strip().split('\t')
            
            e1 = line[0]
            e2 = line[1]
            
            head_s = line[2]
            head = wordMapping.get(head_s, wordMapping[UNKNOWN_WORD])
            
            tail_s = line[3]
            tail = wordMapping.get(tail_s, wordMapping[UNKNOWN_WORD])
            
            bag_list = bags_test.get(e1+"\t"+e2+"\t"+line[4])
            if bag_list is None:
                bag_list = []
                bag_list.append(len(testheadList))
                bags_test[e1+"\t"+e2+"\t"+line[4]] = bag_list
            else:
                bag_list.append(len(testheadList))
            
            num = relationMapping.get(line[4], relationMapping[NIL_RELATION])
            
            length = 0
            lefnum = 0
            rignum = 0
            tmpp = []
            tokens = line[5].split()
            for con in tokens:
                if con=="###END###":
                    break;
                gg = wordMapping.get(con, wordMapping[UNKNOWN_WORD])
                if con == head_s: 
                    lefnum = length
                if con == tail_s: 
                    rignum = length
                length += 1
                tmpp.append(gg)
            
            testheadList.append(head)
            testtailList.append(tail)
            testrelationList.append(num)
            test_sentenceLength.append(length)
            
            con = []
            conl = []
            conr = []
            
            for i in range(length):
                con.append(tmpp[i])
                conl.append(lefnum-i)
                conr.append(rignum-i)
                if conl[i] >= limit: conl[i] = limit
                if conr[i] >= limit: conr[i] = limit
                if conl[i] <= -limit: conl[i] = -limit
                if conr[i] <= -limit: conr[i] = -limit
                if conl[i] > PositionMaxE1: PositionMaxE1 = conl[i]
                if conr[i] > PositionMaxE2: PositionMaxE2 = conr[i]
                if conl[i] < PositionMinE1: PositionMinE1 = conl[i]
                if conr[i] < PositionMinE2: PositionMinE2 = conr[i]
            
            testtrainLists.append(con)
            testPositionE1.append(conl)
            testPositionE2.append(conr)
            if lefnum>rignum:
                former = rignum
                latter = lefnum
            else:
                former = lefnum
                latter = rignum
            testPieceWise.append([former+1, latter-former, length-1-latter])
    
    logging.info('PositionMinE1 {}, PositionMaxE1 {}, PositionMinE2 {}, PositionMaxE2 {}'
                 .format(PositionMinE1, PositionMaxE1, PositionMinE2, PositionMaxE2))

    # norm the positions from [PositionMinE1,PositionMaxE1] to [0,PositionTotalE1]
    for i in range(len(trainPositionE1)):
        length = sentenceLength[i]
        work1 = trainPositionE1[i]
        for j in range(length):
            work1[j] = work1[j] - PositionMinE1
        work2 = trainPositionE2[i]
        for j in range(length):
            work2[j] = work2[j] - PositionMinE2
            
    for i in range(len(testPositionE1)):
        length = test_sentenceLength[i]
        work1 = testPositionE1[i]
        for j in range(length):
            work1[j] = work1[j] - PositionMinE1
        work2 = testPositionE2[i]
        for j in range(length):
            work2[j] = work2[j] - PositionMinE2
    
    PositionTotalE1 = PositionMaxE1 - PositionMinE1 + 1;
    PositionTotalE2 = PositionMaxE2 - PositionMinE2 + 1;  
    
    return bags_train, headList, tailList, relationList, sentenceLength, trainLists, trainPositionE1, trainPositionE2, trainPieceWise, bags_test, testheadList, testtailList, testrelationList, test_sentenceLength, testtrainLists, testPositionE1, testPositionE2, testPieceWise, PositionMinE1, PositionMaxE1, PositionTotalE1, PositionMinE2, PositionMaxE2, PositionTotalE2
    
def myCuda(input):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        return input.cuda()
    else:
        return input
    
def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.
    
    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    
    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")
    
    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(tensor.narrow(int(dim), int(start), int(length)) 
        for start, length in zip(splits, split_sizes))