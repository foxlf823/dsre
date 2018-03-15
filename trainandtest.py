import utils
import model
import torch
import logging
import torch.nn as nn
import torch.utils.data as D
from torch.autograd import Variable

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
    

def train(args, dimension, relationTotal, wordTotal, PositionTotalE1, PositionTotalE2, wordVec,
          bags_train, headList, tailList, relationList, sentenceLength, trainLists, 
          trainPositionE1, trainPositionE2, trainPieceWise):
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
    for k,v in bags_train.items():
        c_train.append(len(b_train))
        b_train.append(k)
        
    train_datasets = BagDataset(c_train)
    if args.seed != 0:
        train_dataloader = D.DataLoader(train_datasets, args.batch, False, num_workers=1)
    else:
        train_dataloader = D.DataLoader(train_datasets, args.batch, True, num_workers=1)

    logging.info('training...')
    
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
            
        logging.info('epoch {} avg loss {}'.format(i, epoch_loss/batch_bag_number))

    logging.info('training complete')

def test():
    pass