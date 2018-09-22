# -*- coding:utf-8 -*-
# author: xiang578
# email: i@xiang578
# blog: www.xiang578.com
from numpy import *
import re
import feedparser


def load_data_set():
    postinglist=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classvec = [0, 1, 0, 1, 0, 1]    # 1 is abusive, 0 not
    return postinglist, classvec


def creat_vocablist(dataset):
    vocabset = set([])
    for document in dataset:
        vocabset = vocabset | set(document)
    return list(vocabset)


def set_of_words_to_vec(vocablist, inputset):
    returnvec = [0]*len(vocablist)
    for word in inputset:
        if word in vocablist:
            returnvec[vocablist.index(word)] = 1
        else:
            print "the word: %s is not in my vocabulary!" % word
    return returnvec


def bag_of_words_to_vec(vocablist, inputset):
    returnvec = [0]*len(vocablist)
    for word in inputset:
        if word in vocablist:
            returnvec[vocablist.index(word)] += 1
    return returnvec


def t_data_set():
    postinglist, classvec = load_data_set()
    vocablist = creat_vocablist(postinglist)
    print vocablist
    print set_of_words_to_vec(vocablist, postinglist[0])


def train_nb0(train_matrix, train_category):
    numtraindocs = len(train_matrix)
    numwords = len(train_matrix[0])
    pabusive = sum(train_category)/float(numtraindocs)
    p0num = ones(numwords)
    p1num = ones(numwords)
    p0denom = 2.0
    p1denom = 2.0
    for i in range(numtraindocs):
        if train_category[i] == 1:
            p1num += train_matrix[i]
            p1denom += sum(train_matrix[i])
        else:
            p0num += train_matrix[i]
            p0denom += sum(train_matrix[i])
    p1vect = log(p1num/p1denom)
    p0vect = log(p0num/p0denom)
    return pabusive, p1vect, p0vect


def t_train_nb0():
    postinglist, classvec = load_data_set()
    vocablist = creat_vocablist(postinglist)
    trainmat = []
    for doc in postinglist:
        trainmat.append(set_of_words_to_vec(vocablist, doc))
    pab, p1v, p0v = train_nb0(trainmat, classvec)
    print pab
    print p1v
    print p0v


def classify_nb(vec2calssify, p0vec, p1vec, pclass1):
    p1 = sum(vec2calssify * p1vec) + log(pclass1)
    p0 = sum(vec2calssify * p0vec) + log(1.0 - pclass1)
    if p1 > p0:
        return 1
    else:
        return 0


def t_nb():
    postinglist, classvec = load_data_set()
    vocablist = creat_vocablist(postinglist)
    trainmat = []
    for doc in postinglist:
        trainmat.append(set_of_words_to_vec(vocablist, doc))
    pab, p1v, p0v = train_nb0(trainmat, classvec)
    testentry = ['love', 'my', 'dalmation']
    thisdoc = array(set_of_words_to_vec(vocablist, testentry))
    print classify_nb(thisdoc, p0v, p1v, pab)
    testentry = ['stupid', 'garbage']
    thisdoc = array(set_of_words_to_vec(vocablist, testentry))
    print classify_nb(thisdoc, p0v, p1v, pab)


def text_parse(bigstring):
    listoftokens = re.split(r'\W*',bigstring)
    return [tok.lower() for tok in listoftokens if len(tok) > 2]


def spam_test():
    doclist = []
    classlist = []
    fulltext = []
    for i in range(1, 26):
        wordlist = text_parse(open('email/spam/%d.txt' % i).read())
        doclist.append(wordlist)
        fulltext.extend(wordlist)
        classlist.append(1)
        wordlist = text_parse(open('email/ham/%d.txt' % i).read())
        doclist.append(wordlist)
        fulltext.extend(wordlist)
        classlist.append(0)
    vocablist = creat_vocablist(doclist)
    trainingset = range(50)
    testset = []
    for i in range(10):
        randindex = int(random.uniform(0, len(trainingset)))
        testset.append(trainingset[randindex])
        del(trainingset[randindex])
    trainmat = []
    trainclass = []
    for i in trainingset:
        trainmat.append(set_of_words_to_vec(vocablist, doclist[i]))
        trainclass.append(classlist[i])
    pspam, p1, p0 = train_nb0(array(trainmat), array(trainclass))
    errorcount = 0
    for i in testset:
        if classify_nb(array(set_of_words_to_vec(vocablist, doclist[i])), p0, p1, pspam) != classlist[i]:
            errorcount += 1
            print doclist[i]
    print 'the error rate is: ', float(errorcount)/len(testset)


def t_feedparser():
    ny = feedparser.parse('https://newyork.craigslist.org/search/mar?format=rss')
    sf = feedparser.parse('https://sfbay.craigslist.org/search/mar?format=rss')
    print min(len(ny['entries']), len(sf['entries']))


def cal_most_freq(vocablist, fulltext):
    import operator
    freqdict = {}
    for i in vocablist:
        freqdict[i] = fulltext.count(i)
    sortedfreq = sorted(freqdict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedfreq[:30]


def local_words(ny, sf):
    minlen = min(len(ny['entries']), len(sf['entries']))
    doclist = []
    classlist = []
    fulltext = []
    for i in range(minlen):
        wordlist = text_parse(sf['entries'][i]['summary'])
        doclist.append(wordlist)
        fulltext.extend(wordlist)
        classlist.append(1)
        wordlist = text_parse(ny['entries'][i]['summary'])
        doclist.append(wordlist)
        fulltext.extend(wordlist)
        classlist.append(0)
    vocablist = creat_vocablist(doclist)
    trainingset = range(2 * minlen)
    testset = []
    for i in range(10):
        randindex = int(random.uniform(0, len(trainingset)))
        testset.append(trainingset[randindex])
        del (trainingset[randindex])
    top30words = cal_most_freq(vocablist, fulltext)
    for i in top30words:
        if i[0] in vocablist:
            vocablist.remove(i[0])
    print 'vocalist len', len(vocablist)
    trainmat = []
    trainclass = []
    for i in trainingset:
        trainmat.append(bag_of_words_to_vec(vocablist, doclist[i]))
        trainclass.append(classlist[i])
    pspam, p1, p0 = train_nb0(array(trainmat), array(trainclass))
    errorcount = 0
    for i in testset:
        if classify_nb(array(bag_of_words_to_vec(vocablist, doclist[i])), p0, p1, pspam) != classlist[i]:
            errorcount += 1
            print doclist[i]
    print 'the error rate is: ', float(errorcount) / len(testset)
    return vocablist, p1, p0


def get_top_words(ny, sf):
    import operator
    vocablist, psf, pny = local_words(ny, sf)
    topny = []
    topsf = []
    for i in range(len(psf)):
        if psf[i] > -6.0:
            topsf.append((vocablist[i], psf[i]))
        if pny[i] > -6.0:
            topny.append((vocablist[i], pny[i]))
    sortedsf = sorted(topsf, key=lambda pair: pair[1], reverse=True)
    print 'SF********'
    for i in sortedsf[:5]:
        print i[0]
    sortedny = sorted(topny, key=lambda pair: pair[1], reverse=True)
    print 'Ny********'
    for i in sortedny[:5]:
        print i[0]


def t_localwords():
    ny = feedparser.parse('https://newyork.craigslist.org/search/mar?format=rss')
    sf = feedparser.parse('https://sfbay.craigslist.org/search/mar?format=rss')
    # vocablist, psf, pny = local_words(ny, sf)
    # vocablist, psf, pny = local_words(ny, sf)
    get_top_words(ny, sf)


if __name__ == '__main__':
    # t_data_set()
    # t_train_nb0()
    # t_nb()
    #  spam_test()
    # t_feedparser()
    t_localwords()
