# ---------------------------------------------------------------------- #
# Simulator functions for probabilistic analysis of IRV elections, implementing the 
# algorithm from:
# Ahead of the Count: An Algorithm for Probabilistic Prediction of Instant Runoff (IRV) Elections
# by Kapoor & Staecker
#
# This code is written by Chris Staecker.
# Please contact us with any questions! cstaecker@fairfield.edu
# ---------------------------------------------------------------------- #

import sys,os
import random

import matplotlib.pyplot as plt
try:
    import mpltern
    MPLTERN_AVAILABLE = True
except ImportError:
    MPLTERN_AVAILABLE = False
    # mpltern only needed for ternary plots, not core functionality

from irvprob import *

from scipy.stats import norm

# ------------------------------------------------------------ #
# FOR RECOUNT SIMULATIONS

# takes a PreferenceSchedule object,
# gives probability distributions for recount results
# ------------------------------------------------------------ #
def recount_probtable(ps):
    # expected % increase in vote count after recount, from fairvote data
    meanadjust = 0.00077
    stdev = 0.00146

    minval = min(ps.dict.values())
    maxval = max(ps.dict.values())
    buckets = range(int(.9*minval), int(1.1*maxval))

    normalsdict = {r:(ps.dict[r]*(1+meanadjust),ps.dict[r]*stdev) for r in ps.dict.keys()}
    pt = probtable_from_normals(normalsdict, buckets)

    return pt

# ------------------------------------------------------------ #
# FOR ELECTION-NIGHT GRADUAL TALLYING SIMULATIONS
# ------------------------------------------------------------ #

# Takes a PreferenceSchedule representing "bound" (already tallied) votes,
# and creates a ProbabilityTable based on predicting normal distributions
# in proportion to those already tallied
def ptfrombound(boundps, unboundnumber, bucketsize, sigmafactor=1):
    # sigmafactor: higher values mean probabilities will guess wider,
    # which means the model makes its predictions with less confidence.

    # high sigma factor makes wide guesses- more conservative, less confident
    # low sigma factor makes narrow guesses - more aggressive, more confident
    
    boundnumber = boundps.votecount
    rankings = boundps.srankings

    bucketmax = unboundnumber # this is the biggest bucket (smallest is 0)
    buckets = range(0,bucketmax+1,bucketsize) 
    
    distribution = "normal"

    # by default, means in proportion to bound vote totals
    means = {}
    if boundnumber == 0:
        # all means equal if no bound votes yet
        means  = { r:(unboundnumber / len(rankings)) for r in rankings }
    else:
        means = { r:(unboundnumber * boundps.dict[r] / boundnumber) for r in rankings}

    # dict of proportion of votes that each ranking is getting
    share = {}
    for r in rankings:
        share[r] = boundps.dict[r] / boundps.votecount
   
    # standard deviations in proportion to the vote share they've already received
    stdevs = { r:(sigmafactor  * unboundnumber * share[r]) for r in rankings }
    
    # this is the probtable for the unbound votes
    unboundpt = {}
    
    # build the dict for the probtable
    d = {}

    for r in rankings:
        if stdevs[r] == 0:
            # this can happen for rankings which recieved no votes
            # in that case, predict again no votes
            d[r] = [1] + [0 for k in buckets[1:]]
        else:
            pdf = norm(loc=means[r],scale=stdevs[r]).pdf
            
            #normlist = [pdf(k)*bucketsize for k in buckets]
            
            # for efficiency, trim these lists, which are mostly 0
            # find upper and lower buckets index which contain middle +/-3 sigmas
            loweri = int(max(0,(means[r] - 3*stdevs[r]) // bucketsize))
            upperi = max(1,int((means[r] + 3*stdevs[r]) / bucketsize))
            
            normlist = ([0] * loweri) + [pdf(k)*bucketsize for k in buckets[loweri:upperi]] #+ ([0] * (len(buckets )-upperi))
            
            # sum won't be 1- it should be close to 1/bucketsize: rescale to make it 1
            y = sum(normlist)
            
            # Safety check: if y is too small (empty range or numerical underflow), use full range
            if y < 1e-10:
                normlist = [pdf(k)*bucketsize for k in buckets]
                y = sum(normlist)
            
            d[r] = [ x/y for x in normlist ]
    
    unboundpt = ProbabilityTable(d,bucketsize)
    
    # add in the bound votes to create final answer probtable
    # this involves boosting each column by the appropriate amount
    maxboost = bucketround(max(boundps.dict.values()),bucketsize)
    newbuckets = range(0,bucketmax+maxboost+1,bucketsize)
    
    newdict = {}
    for r in rankings:
        boundbuckets = bucketround(boundps.dict[r],bucketsize) // bucketsize
    
        leftpad = [0 for k in range(boundbuckets)]
        rightpad = [0 for k in range(maxboost//bucketsize - boundbuckets)]
        newdict[r] = leftpad + unboundpt.dict[r] + rightpad
        
    
    pt = ProbabilityTable(newdict, bucketsize)
    return pt
    
    
# takes a giant list of votes and tallies them into a PreferenceSchedule object
def ps_from_votelist(votelist):
    d = {}

    for v in votelist:
        if v in d.keys():
            d[v] += 1
        else:
            d[v] = 1
    return(PreferenceSchedule(d))



# Takes a PreferenceSchedule object and gradually counts up the votes, 
# making predictions as it tallies.
# returns a dict with keys the candidates, values big lists of winning probabilities in each stage

# counts votes in increments of stepsize (stepsize is a percentage)
# default is 1% at a time
def irvsimulator(ps, bucketsize,stepsize=.01,sigmafactor=1, votelist=[],bucketcap=float('inf'),silent=False):
    
    if votelist==[]: 
        # by default, use all the votes, in random order
        for r in ps.srankings:
            votelist += [r for k in range(ps.dict[r])]
        random.shuffle(votelist)
    
    wps = {}
    p = stepsize
    while p <= 1:
        numvotes = int(len(votelist) * p)

        if not silent:
            print("Calculating wps using " + str(round(100*p,1)) + " percent of the votes. (" + str(int(len(votelist)*p)) + " votes)")
                
        votesportion = votelist[:int(len(votelist)*p)]
        ps = ps_from_votelist(votesportion)
        
        pt = ptfrombound(ps, len(votelist) - len(votesportion), bucketsize)
        pt.bucketcap = bucketcap
        pt = pt.bucket_consolidate(bucketcap)
    
        wps[p] = pt.winning_probabilities()
        
        p = round(p+stepsize, 4)
    
    # return value is a dict of lists of winning probabilities
    # for each candidate after each simulation
    d = {}
    for c in ps.candidates:
        d[c] = [wps[p][c] for p in wps.keys()]
    return d
            
# rounds n to the nearest multiple of bs
def bucketround(n, bs):
    return(round(n / bs) * bs)

# gives a probability table of normal distributions. 
# input is a dict with stringlists as keys, and pairs as values. 
# pairs are (mean, stdev) for a normal distribution
def probtable_from_normals(d,buckets):
    pd = {}
    for r in d.keys():
        (mean,stdev) = d[r]
        
        # build l, which is the normal distribution we want
        
        if mean==0:
            # this can happen if there were zero votes for this ranking
            l = [0] * len(buckets)
            #i = max([i for i in range(len(buckets)) if buckets[i] <= mean])
            #l[i] = 1
            l[0] = 1
        else:
                
            l = [norm.pdf(b,mean,stdev) for b in buckets]
            
            # if stdev is too tight, the mean point may be more than 1
            # in this case, make it 1 (this should only happen for 1 value, and the rest
            # should be very close to 0)
            l = [min(1,x) for x in l]
            
            # if stdev is REALLY too tight, the whole list may be 0
            # in this case make the mean 1, all others 0
            if sum(l) == 0:
                l[int(mean)] = 1
            
            # rescale to sum to 1 (it should be close already)
            l = [x / sum(l) for x in l]
            
        pd[r] = l
    
    return ProbabilityTable(pd,buckets[1]-buckets[0])


# makes a matplotlib plot from irvsimulator output
def plotwps(wps,filename,candidatenames={},title=""):
    cands = list(wps.keys())
    n = len(wps[cands[0]])
    
    x = [ 100*k/n for k in range(n)]
    
    for c in cands:
        l = ""
        if int(c) in candidatenames.keys():
            l = candidatenames[int(c)]
        else:
            print("candidate " + str(c) + " not found")
            l = c
        plt.plot(x,wps[c], label=l)

    plt.legend(numpoints=1)
    
    plt.xlabel("percentage of vote bound")
    plt.ylabel("predicted win probability")
    plt.title(title)
    plt.ylim(-0.1, 1.1)
    plt.yticks([0,.25,.5,.75,1])
    
    plt.savefig(filename)
    plt.close()

# makes a matplotlib ternary plot from irvsimulator output
def triangleplotwps(wps,filename,candidatenames={},title=""):
    
    cands = list(wps.keys())

    ax = plt.subplot(projection="ternary")
    
    ks = list(wps.keys())
    print(ks)
    t, l, r = wps[ks[0]], wps[ks[1]], wps[ks[2]]

    ax.plot(t, l, r)

    ax.set_tlabel(candidatenames[int(cands[0])])
    ax.set_llabel(candidatenames[int(cands[1])])
    ax.set_rlabel(candidatenames[int(cands[2])])

    ax.grid(axis='t', which='both', linestyle=':')
    ax.grid(axis='l', which='both', linestyle=':')
    ax.grid(axis='r', which='both', linestyle=':')
    
    plt.savefig(filename)
    plt.close()
    