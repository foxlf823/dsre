import utils
import model
import torch
import logging
import torch.nn as nn
import torch.utils.data as D
from torch.autograd import Variable
import traceback
from collections import OrderedDict
from utils import NIL_RELATION
import os
import matplotlib.pyplot as plt

def showPRcurve(args):
        
    # collect all *.curve from the output directory
    allCurveFilePath = []
    for x in os.listdir(args.output):
        path = os.path.join(args.output,x)
        name, extension = os.path.splitext(x)
        if os.path.isfile(path) and extension =='.curve':
            logging.info('find {}'.format(x))
            allCurveFilePath.append((path, name))
    
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.title('Precision-Recall curve')
     
    fmts = ['x-','+-','*-','o-']
    for i, (path, name) in enumerate(allCurveFilePath):
        precision = []
        recall = []
        with open(path, 'r') as f:
            for line in f.readlines():
                _, _, _, p, _, r, _, _ = line.strip().split()
                precision.append(float(p))
                recall.append(float(r))
                
#         plt.step(recall, precision, color='b', where='pre') # follow http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
        
        fmt = fmts[i % len(fmts)]
        plt.plot(recall, precision, fmt, label=name)   
        

    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.) 
    plt.show()

def convertToSorted(listDict):
    # sorted by the confidence of the predicted relation
    # each triple: position in listDict, relation id, its confidence
    listTriple = [] 
    for i, d in enumerate(listDict):
        for l,s in d.items():
            if len(listTriple) == 0:
                listTriple.append((i, l, s))
            else:
                insertPosition = len(listTriple)
                for j, t in enumerate(listTriple):
                    if t[2] < s:
                        insertPosition = j
                        break
                listTriple.insert(insertPosition, (i, l, s))
                
    return listTriple

def generatePRCurve(path, goldLabels, predictedLabels):
    # generate PR curve
    prCurvePoints = []
    preds = convertToSorted(predictedLabels)
    prevP = -1
    prevR = -1
    START_OFFSET = 1
    for i in range(START_OFFSET, len(preds)+1):
        filteredLabels = preds[0:i]
        p, r, f1 = score1(goldLabels, filteredLabels)
        if p != prevP or r != prevR:
            ratio = i / len(preds)
            prCurvePoints.append((ratio, p, r, f1))
            prevP = p
            prevR = r
    
    # save them into a file
    with open(path, 'w') as f:
        for t in prCurvePoints:
            f.write("ratio {} P {} R {} F1 {}\n".format(t[0], t[1], t[2], t[3]))


def prepareInput(bool_requires_grad, bool_volatile, instance_index, headList, tailList, 
                 relationList, sentenceLength,
                 trainLists, trainPositionE1, trainPositionE2, trainPieceWise):
#     head = [headList[instance_index]]
#     tail = [tailList[instance_index]]
    relation = [relationList[instance_index]]
#     sentence_len = [sentenceLength[instance_index]]
    word_sequence = trainLists[instance_index]
    e1_position_sequence = trainPositionE1[instance_index]
    e2_position_sequence = trainPositionE2[instance_index]
    piece_wise = trainPieceWise[instance_index]
    
    relation = utils.myCuda(Variable(torch.LongTensor(relation), requires_grad=bool_requires_grad, volatile=bool_volatile))
    word_sequence = utils.myCuda(Variable(torch.LongTensor(word_sequence), requires_grad=bool_requires_grad, volatile=bool_volatile))
    e1_position_sequence = utils.myCuda(Variable(torch.LongTensor(e1_position_sequence), requires_grad=bool_requires_grad, volatile=bool_volatile))
    e2_position_sequence = utils.myCuda(Variable(torch.LongTensor(e2_position_sequence), requires_grad=bool_requires_grad, volatile=bool_volatile))
    # make a fake batch size 1
    return relation, word_sequence.unsqueeze(0), e1_position_sequence.unsqueeze(0), e2_position_sequence.unsqueeze(0), piece_wise

class BagDataset(D.Dataset):

    def __init__(self, bag_index_list):

        self.bag_index_list = bag_index_list
        self.size = len(bag_index_list)

    def __getitem__(self, index):
        return self.bag_index_list[index]

    def __len__(self):

        return self.size

def score1(listGoldLabels, listPredictedLabels):
    total = 0
    predicted = 0
    correct = 0
    
    for gold in listGoldLabels:
        total += len(gold)
        
    for t in listPredictedLabels:
        predicted += 1
        if t[1] in listGoldLabels[t[0]]:
            correct += 1
    
    p = correct / predicted
    r = correct / total
    if p != 0 and r != 0:
        f1 = 2*p*r/(p+r)
    else:
        f1 = 0
        
    return p, r, f1
    
def score(listGoldLabels, listPredictedLabels):
    total = 0
    predicted = 0
    correct = 0
    
    for i, _ in enumerate(listGoldLabels):
        goldLabels = listGoldLabels[i]
        predictedLabels = listPredictedLabels[i]
        
        total += len(goldLabels)
        predicted += len(predictedLabels)
        for label in predictedLabels.keys():
            if label in goldLabels:
                correct += 1
    
    p = correct / predicted
    r = correct / total
    if p != 0 and r != 0:
        f1 = 2*p*r/(p+r)
    else:
        f1 = 0
        
    return p, r, f1
    
def evaluate(m, bags_test, testheadList, testtailList, testrelationList, 
             test_sentenceLength, testtrainLists, testPositionE1, testPositionE2, testPieceWise, 
             relationMapping):
    m.eval()

    testBagGoldLabel = []
    testBagPredictedLabel = []
    for _,bag_instance_list in bags_test.items():
        
        bagGoldLabel = set()
        bagPredictLabel = dict()
        
        for instance_index in bag_instance_list:
            
            relation, word_sequence, e1_position_sequence, e2_position_sequence, piece_wise = prepareInput(False, True, instance_index, 
                                                                                                           testheadList, testtailList, testrelationList, 
                                                                                                           test_sentenceLength, testtrainLists, 
                                                                                                           testPositionE1, testPositionE2, testPieceWise)

            prediction = m.forward(word_sequence, e1_position_sequence, e2_position_sequence, piece_wise, True)
            
            max_value, max_id = prediction.max(1)
            predict_relation_id = max_id.cpu().data[0]
            predict_relation_score = max_value.cpu().data[0]
            relation_id = relation.cpu().data[0]
            
            # follow Surdeanu et.al., mimlre, EMNLP, 2012
            if relation_id != relationMapping[NIL_RELATION]:
                bagGoldLabel.add(relation_id)
            if predict_relation_id != relationMapping[NIL_RELATION]:
                # follow the at-least-one rule
                key = bagPredictLabel.get(predict_relation_id)
                if key is None:
                    bagPredictLabel[predict_relation_id] = predict_relation_score
                else:
                    if predict_relation_score > bagPredictLabel[predict_relation_id]:
                        bagPredictLabel[predict_relation_id] = predict_relation_score
        

                
        testBagGoldLabel.append(bagGoldLabel)
        testBagPredictedLabel.append(bagPredictLabel)
        
    p, r, f1 = score(testBagGoldLabel, testBagPredictedLabel)
    
    return p, r, f1, testBagGoldLabel, testBagPredictedLabel

def train(args, dimension, relationTotal, wordTotal, PositionTotalE1, PositionTotalE2, wordVec,
          bags_train, headList, tailList, relationList, sentenceLength, trainLists, trainPositionE1, trainPositionE2, trainPieceWise, 
          bags_test, testheadList, testtailList, testrelationList, test_sentenceLength, testtrainLists, testPositionE1, testPositionE2, testPieceWise, 
          relationMapping):
    dimensionC = args.conv
    dimensionWPE = args.wpe
    alpha = args.lr
    window = args.window
    dropout = args.dropout
    
    m = utils.myCuda(model.PCNN(dimensionC, relationTotal, dimensionWPE, dimension, 
                 window, wordTotal, PositionTotalE1, PositionTotalE2, wordVec, dropout))
    optimizer = torch.optim.Adadelta(m.parameters(), lr=alpha)
    loss_func = nn.CrossEntropyLoss()
    
    b_train = [] # bag name list
    c_train = [] # bag index list
    for k,_ in bags_train.items():
        c_train.append(len(b_train))
        b_train.append(k)
        
    train_datasets = BagDataset(c_train)
    if args.seed != 0:
        train_dataloader = D.DataLoader(train_datasets, args.batch, False, num_workers=1)
    else:
        train_dataloader = D.DataLoader(train_datasets, args.batch, True, num_workers=1)
        
#     test_b_train = []
#     test_c_train = []
#     for k,_ in bags_test.items():
#         test_c_train.append(len(test_b_train))
#         test_b_train.append(k)
#         
#     test_datasets = BagDataset(test_c_train)
#     if args.seed !=0:
#         test_dataloader = D.DataLoader(test_datasets, args.batch, False, num_workers=1)
#     else:
#         test_dataloader = D.DataLoader(test_datasets, args.batch, True, num_workers=1)

    logging.info('training...')
    best = 0
    
    for i in range(args.epochs):
        m.train()

        epoch_loss = 0
        batch_bag_number = 0
        
        for batch_bag_index_list in train_dataloader:
            
            batch_loss = 0
            
            for bag_index in batch_bag_index_list:
                bag = bags_train[b_train[bag_index]]
                
                tmp1 = 100000000
                tmp2 = -1
                
                for instance_index in bag:
                    
                    relation, word_sequence, e1_position_sequence, e2_position_sequence, piece_wise = prepareInput(False, False, instance_index, headList, tailList, 
                                                                                                       relationList, sentenceLength, trainLists, trainPositionE1, 
                                                                                                       trainPositionE2, trainPieceWise)

                    prediction = m.forward(word_sequence, e1_position_sequence, e2_position_sequence, piece_wise, False)
                    l = loss_func(prediction, relation).cpu().data[0]
                    
                    # find the max probability of a instance in this bag, namely the min loss
                    if l < tmp1:
                        tmp1 = l
                        tmp2 = instance_index
                        

                relation, word_sequence, e1_position_sequence, e2_position_sequence, piece_wise = prepareInput(False, False, tmp2, headList, tailList, 
                                                                                                   relationList, sentenceLength, trainLists, trainPositionE1, 
                                                                                                   trainPositionE2, trainPieceWise)

                prediction = m.forward(word_sequence, e1_position_sequence, e2_position_sequence, piece_wise, True)
                l = loss_func(prediction, relation)
                
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                
                batch_loss += l.cpu().data[0]
            
            batch_avg_loss = batch_loss/len(batch_bag_index_list)
            logging.debug('batch {} avg loss {}'.format(batch_bag_number, batch_avg_loss))     
            batch_bag_number += 1 
            epoch_loss += batch_avg_loss
            
        logging.debug('epoch {} avg loss {}'.format(i, epoch_loss/batch_bag_number))
        
        p, r, f1, _, _ = evaluate(m, bags_test, testheadList, testtailList, testrelationList, 
                 test_sentenceLength, testtrainLists, testPositionE1, testPositionE2, testPieceWise, 
                 relationMapping)
            
        logging.info('epoch {} p {} r {} f1 {}'.format(i, p, r, f1))
        
        if f1 > best:
            logging.info('epoch {} exceed the best, saving model...'.format(i))
            best = f1
            torch.save(m.state_dict(), args.output+'/'+args.signature+'.model')
        

def test(args, dimension, relationTotal, wordTotal, PositionTotalE1, PositionTotalE2, wordVec,
        bags_test, testheadList, testtailList, testrelationList, test_sentenceLength, testtrainLists, testPositionE1, testPositionE2, testPieceWise, 
        relationMapping):
    try:
        bestModelPath = args.output+'/'+args.signature+'.model'
        logging.info('loading the existing best model from {}'.format(bestModelPath))
        model_state_dict = torch.load(bestModelPath)
    except BaseException as e:
        traceback.print_exc()  
        return 
    
    dimensionC = args.conv
    dimensionWPE = args.wpe
    window = args.window
    dropout = args.dropout
    
    m = utils.myCuda(model.PCNN(dimensionC, relationTotal, dimensionWPE, dimension, 
                 window, wordTotal, PositionTotalE1, PositionTotalE2, wordVec, dropout))
    m.load_state_dict(model_state_dict)
    p, r, f1, testBagGoldLabel, testBagPredictedLabel = evaluate(m, bags_test, testheadList, testtailList, testrelationList, 
                 test_sentenceLength, testtrainLists, testPositionE1, testPositionE2, testPieceWise, 
                 relationMapping)
         
    logging.info('p {} r {} f1 {}'.format(p, r, f1))    
    
    prCurvePath = args.output+'/'+args.signature+'.curve'
    generatePRCurve(prCurvePath, testBagGoldLabel, testBagPredictedLabel)
    
    