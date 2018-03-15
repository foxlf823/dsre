
import argparse
import logging
import utils
import trainandtest

logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-train', action='store_true', help='denote training')
parser.add_argument('-seed', default=0, type=int, help='if 0, randomly; else, fix the seed with the given value')
parser.add_argument('-emb', default='./data/vec.bin', help='the path of bin file of pretrained word embedding')
parser.add_argument('-rel', default='./data/relation2id.txt', help='the file path of relations and their ids')
parser.add_argument('-traindata', default='./data/train.txt', help='the file path of training data')
parser.add_argument('-testdata', default='./data/test.txt', help='the file path of test data')
parser.add_argument('-limit', default=30, type=int, help='maximum position, e.g., [-30, 30]')
parser.add_argument('-conv', default=230, type=int, help='number of convolutional filters')
parser.add_argument('-wpe', default=5, type=int, help='dimension of position embeddings')
parser.add_argument('-window', default=3, type=int, help='word window size')
parser.add_argument('-lr', default=0.02, type=float, help='initial learning rate')
parser.add_argument('-batch', default=16, type=int, help='Batch size for training.')
parser.add_argument('-epochs', default=15, type=int, help='Number of epochs to train for.')
parser.add_argument('-dropout', default=0.5, type=float, help='dropout probability')


args = parser.parse_args()
logging.info('### print all arguments ###')
utils.listArguments(args)
logging.info('')

if args.seed != 0:
    utils.setFixedSeed(args.seed)
    
logging.info('loading pretrained word embedding from {}'.format(args.emb))
wordVec, wordList, wordMapping, dimension, wordTotal = utils.loadWordEmbFromBinFile(args.emb)
 
relationMapping, nam, relationTotal = utils.loadIDMappingFile(args.rel)
logging.info('relation number {}'.format(len(nam)))

logging.info('loading data...')
bags_train, headList, tailList, relationList, sentenceLength, trainLists, trainPositionE1, trainPositionE2, trainPieceWise, bags_test, testheadList, testtailList, testrelationList, test_sentenceLength, testtrainLists, testPositionE1, testPositionE2, testPieceWise, PositionMinE1, PositionMaxE1, PositionTotalE1, PositionMinE2, PositionMaxE2, PositionTotalE2 = utils.loadData(args.traindata, args.testdata, wordMapping, relationMapping, args.limit) 
logging.info('bags in the training data: {}'.format(len(bags_train)))
logging.info('bags in the test data: {}'.format(len(bags_test)))

if args.train:
    trainandtest.train(args, dimension, relationTotal, wordTotal, PositionTotalE1, PositionTotalE2, wordVec,
                       bags_train, headList, tailList, relationList, sentenceLength, trainLists, trainPositionE1, trainPositionE2, trainPieceWise)
else:
    trainandtest.test()  
