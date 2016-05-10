import os
import sys
import time
import copy
import cPickle

import numpy as np
import scipy
import scipy.sparse
import theano
import theano.sparse as S
import theano.tensor as T
from collections import OrderedDict

from Classes_and_functions import *

datapath='data/'
dataset='FB15k'
Nent=16296
rhoE=1
rhoL=5
Nsyn=14951
Nrel=1345
loadmodel=False
loademb=False
op='Unstructured'
simfn='Dot'
ndim=50
nhid=50
marge=1.
lremb=0.1
lrparam=1.
nbatches=100
totepochs=2000
test_all=1
neval=50
seed=123
savepath='.'
loadmodelBi=False
loadmodelTri=False

op='TransE'
simfn='L2'
ndim=50
nhid=50
marge=0.5
lremb=0.01
lrparam=0.01
nbatches=100
totepochs=500
test_all=10
neval=1000
savepath='FB15k_TransE'
datapath='../data/'

#SET ARGUMENTS
state = DD()
state.datapath = datapath
state.dataset = dataset
state.Nent = Nent
state.Nsyn = Nsyn
state.Nrel = Nrel
state.loadmodel = loadmodel
state.loadmodelBi = loadmodelBi
state.loadmodelTri = loadmodelTri
state.loademb = loademb
state.op = op
state.simfn = simfn
state.ndim = ndim
state.nhid = nhid
state.marge = marge
state.rhoE = rhoE
state.rhoL = rhoL
state.lremb = lremb
state.lrparam = lrparam
state.nbatches = nbatches
state.totepochs = totepochs
state.test_all = test_all
state.neval = neval
state.seed = seed
state.savepath = savepath

if not os.path.isdir(state.savepath):
    os.mkdir(state.savepath)

channel = Channel(state)

print >> sys.stderr, state
np.random.seed(state.seed)

# Experiment folder
if hasattr(channel, 'remote_path'):
    state.savepath = channel.remote_path + '/'
elif hasattr(channel, 'path'):
    state.savepath = channel.path + '/'
else:
    if not os.path.isdir(state.savepath):
        os.mkdir(state.savepath)

# Positives
trainl = load_file(state.datapath + state.dataset + '-train-lhs.pkl')
trainr = load_file(state.datapath + state.dataset + '-train-rhs.pkl')
traino = load_file(state.datapath + state.dataset + '-train-rel.pkl')

traino = traino[-state.Nrel:, :]

# Valid set
validl = load_file(state.datapath + state.dataset + '-valid-lhs.pkl')
validr = load_file(state.datapath + state.dataset + '-valid-rhs.pkl')
valido = load_file(state.datapath + state.dataset + '-valid-rel.pkl')

valido = valido[-state.Nrel:, :]

# Test set
testl = load_file(state.datapath + state.dataset + '-test-lhs.pkl')
testr = load_file(state.datapath + state.dataset + '-test-rhs.pkl')
testo = load_file(state.datapath + state.dataset + '-test-rel.pkl')

testo = testo[-state.Nrel:, :]

# Index conversion
trainlidx = convert2idx(trainl)[:state.neval]
trainridx = convert2idx(trainr)[:state.neval]
trainoidx = convert2idx(traino)[:state.neval]

validlidx = convert2idx(validl)[:state.neval]
validridx = convert2idx(validr)[:state.neval]
validoidx = convert2idx(valido)[:state.neval]

testlidx = convert2idx(testl)[:state.neval]
testridx = convert2idx(testr)[:state.neval]
testoidx = convert2idx(testo)[:state.neval]

idxl = convert2idx(trainl)
idxr = convert2idx(trainr)
idxo = convert2idx(traino)

idxtl = convert2idx(testl)
idxtr = convert2idx(testr)
idxto = convert2idx(testo)

idxvl = convert2idx(validl)
idxvr = convert2idx(validr)
idxvo = convert2idx(valido)

true_triples=np.concatenate([idxtl,idxvl,idxl,idxto,idxvo,idxo,idxtr,idxvr,idxr]).reshape(3,idxtl.shape[0]+idxvl.shape[0]+idxl.shape[0]).T

# Model declarationpp state
leftop  = LayerTrans()
rightop = Unstructured()

# embeddings
embeddings = Embeddings(np.random, state.Nent, state.ndim, 'emb')

relationVec = Embeddings(np.random, state.Nrel, state.ndim, 'relvec')
embeddings = [embeddings, relationVec, relationVec]

simfn = eval(state.simfn + 'sim')

# Function compilation
trainfunc = TrainFn1Member(simfn, embeddings, leftop, rightop, marge=state.marge, rel=False)
ranklfunc = RankLeftFnIdx(simfn, embeddings, leftop, rightop, subtensorspec=state.Nsyn)
rankrfunc = RankRightFnIdx(simfn, embeddings, leftop, rightop, subtensorspec=state.Nsyn)

out = []
outb = []
state.bestvalid = -1

batchsize = trainl.shape[1] / state.nbatches
# ----------------------------------------------------------------------------
print >> sys.stderr, "BEGIN TRAINING"
timeref = time.time()
for epoch_count in xrange(1, state.totepochs + 1):
    # Shuffling
    order = np.random.permutation(trainl.shape[1])
    trainl = trainl[:, order]
    trainr = trainr[:, order]
    traino = traino[:, order]

    # Negatives
    trainln = create_random_mat(trainl.shape, np.arange(state.Nsyn))
    trainrn = create_random_mat(trainr.shape, np.arange(state.Nsyn))

    for i in range(state.nbatches):
        tmpl = trainl[:, i * batchsize:(i + 1) * batchsize]
        tmpr = trainr[:, i * batchsize:(i + 1) * batchsize]
        tmpo = traino[:, i * batchsize:(i + 1) * batchsize]
        tmpnl = trainln[:, i * batchsize:(i + 1) * batchsize]
        tmpnr = trainrn[:, i * batchsize:(i + 1) * batchsize]
        # training iteration
        outtmp = trainfunc(state.lremb, state.lrparam,
                tmpl, tmpr, tmpo, tmpnl, tmpnr)
        out += [outtmp[0] / float(batchsize)]
        outb += [outtmp[1]]

        # embeddings normalization
        embeddings[0].normalize()

    if (epoch_count % state.test_all) == 0:
        # model evaluation
        print >> sys.stderr, "-- EPOCH %s (%s seconds per epoch):" % (
                epoch_count,
                round(time.time() - timeref, 3) / float(state.test_all))
        timeref = time.time()
        print >> sys.stderr, "COST >> %s +/- %s, %% updates: %s%%" % (
                round(np.mean(out), 4), round(np.std(out), 4),
                round(np.mean(outb) * 100, 3))
        out = []
        outb = []
        resvalid = FilteredRankingScoreIdx(ranklfunc, rankrfunc, validlidx, validridx, validoidx, true_triples)
        state.valid = np.mean(resvalid[0] + resvalid[1])
        restrain = FilteredRankingScoreIdx(ranklfunc, rankrfunc, trainlidx, trainridx, trainoidx, true_triples)
        state.train = np.mean(restrain[0] + restrain[1])
        print >> sys.stderr, "\tMEAN RANK >> valid: %s, train: %s" % (state.valid, state.train)
        if state.bestvalid == -1 or state.valid < state.bestvalid:
            restest = FilteredRankingScoreIdx(ranklfunc, rankrfunc,testlidx, testridx, testoidx, true_triples)
            state.bestvalid = state.valid
            state.besttrain = state.train
            state.besttest = np.mean(restest[0] + restest[1])
            state.bestepoch = epoch_count
            # Save model best valid model
            f = open(state.savepath + '/best_valid_model.pkl', 'w')

            cPickle.dump(embeddings, f, -1)
            cPickle.dump(leftop, f, -1)
            cPickle.dump(rightop, f, -1)
            cPickle.dump(simfn, f, -1)

            f.close()
            print >> sys.stderr, "\t\t##### NEW BEST VALID >> test: %s" % (state.besttest)

        # Save current model
        f = open(state.savepath + '/current_model.pkl', 'w')

        cPickle.dump(embeddings, f, -1)
        cPickle.dump(leftop, f, -1)
        cPickle.dump(rightop, f, -1)
        cPickle.dump(simfn, f, -1)

        f.close()
        state.nbepochs = epoch_count
        print >> sys.stderr, "\t(the evaluation took %s seconds)" % (round(time.time() - timeref, 3))
        timeref = time.time()
        channel.save()
channel.COMPLETE
