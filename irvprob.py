# ---------------------------------------------------------------------- #
# Main functions for probabilistic analysis of IRV elections, implementing the 
# algorithm from:
# Ahead of the Count: An Algorithm for Probabilistic Prediction of Instant Runoff (IRV) Elections
# by Kapoor & Staecker
#
# This code is written by Chris Staecker.
# Please contact us with any questions! cstaecker@fairfield.edu
# ---------------------------------------------------------------------- #

import csv
from numpy import cumsum
from scipy.signal import fftconvolve
from scipy.stats import norm

# ---------------------------------------------------------------------- #
# PREFERENCE SCHEDULES (PS)
# Data structure for a preference schedule
#
# Use it like:
# ps = PreferenceSchedule(
#        {'[C]': 2872, '[C,H]': 417, '[C,G]': 2229,
#         '[H]': 2679, '[H,G]': 2614, '[H,C]': 239,
#         '[G]': 1995, '[G,C]': 1444, '[G,H]': 2213 })
#
# (candidate names can't have commas or brackets in them!)

# PS object properties:
# ps.candidates = ['C','H','G']
# ps.rankings = [['C', 'G', 'H'], ['C', 'H', 'G'], ..., []]
# ps.srankings = ['[C,G,H]', '[C,H,G]', ..., '[]']
# ps.votecount = 16702

# PS object methods:
# ps.consolidated - gives an equivalent PS object using only undervotes (according to                       paper Lemma)
# ps.irv_results - computes the IRV winner, prints rounds information
#
# ---------------------------------------------------------------------- #


class PreferenceSchedule:
    def __init__(self,votesdict):
        cs = list(set([c for k in votesdict.keys() for c in stringlist_to_list(k) ]))
        cs.sort()
        cs.sort(key=len)
        self.candidates = cs
        
        rs = subsetperms(self.candidates)
        rs.sort()
        rs.sort(key=len)
        self.rankings  = rs
        self.srankings = [list_to_stringlist(r) for r in self.rankings]

        # votesdict may be "incomplete"- i.e. missing some possible rankings
        # the dict property always contains every ranking, with some values of 0
        d = {}
        for r in self.rankings:
            sr = list_to_stringlist(r)
            if sr in votesdict.keys():
                d[sr] = votesdict[sr]
            else:
                d[sr] = 0
                
        self.dict = d
        
        self.votecount = sum(self.dict.values())
    
    # consolidates votes according to lemma about "always undervotes". Returns another PS object
    def consolidated(self):
        newd = {r:0 for r in self.srankings}
        for r in self.rankings:
            if len(r) == len(self.candidates):
                q = r[:-1] # remove last ranked
                newd[list_to_stringlist(q)] += self.dict[list_to_stringlist(r)]
            else:
                newd[list_to_stringlist(r)] += self.dict[list_to_stringlist(r)]
        
        return PreferenceSchedule(newd)
    
    # runs the rounds and gives the winner
    # assumes no ties. If there are ties, it will choose some single-elimination, and the outcome is unpredictable.
    def irv_results(self):
        print("Votes this round:")
        print(self.dict)
        if len(self.candidates) == 1:
            c = self.candidates[0]
            print(str(c) + " is the winner.")
            return c
        else:
            # add up the 1st-position votes in this round
            vs = {c:0 for c in self.candidates}
            for r in self.rankings:
                if r != []:
                    c = r[0]
                    vs[c] += self.dict[list_to_stringlist(r)]
                    
            print("Round totals:")
            print(vs)
            
            # find candidate with the least amount
            loser = self.candidates[0]
            for c in self.candidates:
                if vs[c] < vs[loser]:
                    loser = c
            
            print(str(loser) + " is eliminated.")
            
            # build the dict for the next round, eliminating the loser
            newd = {}
            for r in self.rankings:
                newr = r.copy()
                if loser in r: 
                    newr.remove(loser)
                snewr = list_to_stringlist(newr)
                sr = list_to_stringlist(r)
                if snewr in newd.keys():
                    newd[snewr] += self.dict[sr]
                else:
                    newd[snewr] = self.dict[sr]
            
            newps = PreferenceSchedule(newd)
            return newps.irv_results()
    
# sums two PreferenceSchedules, returns a PreferenceSchedule
def prefScheduleSum(p,q):
    d = {}
    for r in set(list(p.dict.keys()) + list(q.dict.keys())):
        if r not in q.dict.keys():
            d[r] = p.dict[r]
        elif r not in p.dict.keys():
            d[r] = q.dict[r]
        else:
            d[r] = p.dict[r] + q.dict[r]

    return PreferenceSchedule(d)


# ------------------------------------------------------------ #
# PROBABILITY TABLES (PT)
# the main datastructure, representing a set of probability distributions,
# like the big chart in the paper introduction.
#
# initialize using a dict and stated bucketsize, like:
# d = {"[A,B]":[.01,.17,.34,.25,.13,.1], "[A,C]":[.5,.4,.07,.03,], 
#     "[B,A]":[0,0,.45,.31,.2,.04], "[B,C]":[0,.1,.3,.27,.19,.14],
#     "[C,A]":[.09,.17,.37,.21,.1,.06], "[C,B]":[.17,.75,.08],
#     "[A]":[.5,.5], "[B]":[.1,.3,.3,.2,.1],
#     "[C]":[.02,.33,.21,.2,.15,.09]}
# pt = ProbabilityTable(d, 100, bucketcap=1500)
#
# Note the various lists are not required to have the same length.
# Again candidate names can't have commas or brackets in them!

# Important Properties:
# pt.candidates, pt.rankings as for PS object
# pt.buckets = [0,100,200,300,400,500]
# pt.numbuckets = 6

# Important Methods:
# pt.winning_probabilities() - gives a dict of candidates and their prob of winning
# pt.latex - gives a big string for displaying pt as a latex tabular

# a PT carries with it a "bucket cap", which is a desired number of buckets which
# it will try to maintain during computations. (Every convolution operation will 
# essentially double the number of buckets, which will slow things down.) The bucket
# cap is maintained by consolidating buckets when appropriate. Default cap is infinity,
# so you need to specify a cap if you want to benefit from the efficiency.


class ProbabilityTable:
    def __init__(self, probsdict, bucketsize, bucketcap=float('inf')):
        self.bucketsize = bucketsize
        self.bucketcap = bucketcap
                
        self.dict = probsdict # avoid accessing the dict directly- use lookup method
        
        
        rs = sorted(list(probsdict.keys())) # list of all possible vote ranking strings, like
                                               # ["[A,C,D,B]","[A,D,B]",...,"[D]","[]"]
        rs.sort()
        rs.sort(key=len)
        self.rankings = rs
                                               
        cs = list(set([c for r in self.rankings for c in stringlist_to_list(r)]))
        cs.sort()
        cs.sort(key=len)
        self.candidates = cs

        
        self.numbuckets = max([len(probsdict[k]) for k in probsdict.keys()])
        self.buckets = range(0,self.numbuckets*bucketsize,bucketsize)
    
    # probability for ranking stringlist r in position i
    # if the appropriate dictionary list is too short, give 0
    def lookup(self,r,i):
        if len(self.dict[r]) <= i:
            return 0
        else:
            return self.dict[r][i]
        
    # returns resulting probability table after eliminating the candidate c
    def eliminate(self,c):
        d = {}
        #ans[voteslabel] = pt[voteslabel]

        s = self.rankings.copy()
    
        # classes of rankings which are equivalent after we remove c
        elimclasses = []
        while s != []:
            x = stringlist_to_list(s[0])
            ec = [q for q in s if rem(stringlist_to_list(q),c) == rem(x,c)]

            elimclasses.append(ec)
            for x in ec:
                s.remove(x)
        
        # convolve the appropriate columns, according to paper formula (6)
        for q in elimclasses:
            key = list_to_stringlist(rem(stringlist_to_list(q[0]),c)) # the new ranking, for this class, without c
            d[key] = convolve_list([self.dict[x] for x in q])
        
        newpt = ProbabilityTable(d,self.bucketsize)
        
        # enforce bucketcap
        if self.bucketcap < float('inf'):
            newpt = newpt.bucket_consolidate(self.bucketcap)

        return newpt
    
    # attempts to create a new PreferenceTable with the given number of buckets
    # by summing neighboring buckets
    # only sums full buckets, so the number of buckets in the result is:
    #  newnumbuckets + (numbuckets % newnumbuckets) 
    def bucket_consolidate(self,newnumbuckets):
        factor = self.numbuckets // newnumbuckets
        
        if factor == 0:
            return self
            
        newd = {}
        for k in self.dict.keys():
            newd[k] = []
            for i in range(0,self.numbuckets,factor):
                newd[k].append(sum(self.dict[k][i:i+factor]))
        
        return ProbabilityTable(newd, self.bucketsize * factor, bucketcap=self.bucketcap)
        
    # string to display this PT as a latex tabular
    # names is a dict taking internal candidate names to printable names
    # when omitcols=True, omits columns of full rankings (non-undervotes)
    # when omitexhausted=True, omits the column of exhausted votes. (ranking [])
    def latex(self,names,omitcols=True, omitexhausted=True):
        def numformat(x):
            if x<.005:
                return "."
            return f'{x:.2f}'
        
        ranks = self.rankings
        if omitcols:
            ranks = [r for r in ranks if len(stringlist_to_list(r)) < len(self.candidates)]
        
        if omitexhausted:
            ranks = [r for r in ranks if r != "[]"]
            
        
        s = r"\begin{tabular}{c|" + ("c" * (len(ranks))) + "}\n"
        s += "Votes & "
        for sr in ranks:
            lr = stringlist_to_list(sr)
            ln = [names[r] for r in lr]
            sn = list_to_stringlist(ln)
            sn = sn.replace(",","").replace("[","").replace("]","")
            s += r"$f_{" + sn + r"}$ & "
        s = s[:-2] + r"\\" + "\n" # kill extra &, put \\
        s += r"\hline" + " \n"
        
        for b in range(len(self.buckets)):
            if max([round(self.lookup(r,b),2) for r in ranks]) > 0:
            
                s += str(self.buckets[b]) + " & "
                for r in ranks:
                    s += numformat(self.lookup(r,b)) + " & "
    
                s = s[:-2] + r"\\" + "\n" # kill extra &, put \\
            
        s += r"\end{tabular}" + "\n"
        
        return s
    
    # list of probabilities of elimination of various rankings,
    # returns a dict like: "[A]": 0.023, "[B]": 0.421, "[A,C]": .000012
    # values for more than one candidate are bucket-tie probabilities
    def elimination_probabilities(self):
    
        # first a dict which gives probabilities for each candidate 
        # getting this many 1st-place rankings
        tau = {}
        for c in self.candidates:
            # convolve all columns starting with c
            # this is formula (8)
            tau[c] = convolve_list([self.dict[o] for o in self.rankings if startswith(stringlist_to_list(o),c)])
            
        newnumbuckets = max([len(tau[c]) for c in self.candidates])

        # now a cumulative version of the dict above
        kappa = {}
        for c in self.candidates:
            kappa[c] = cumsum(tau[c])
        
        candsets = nonemptysubsets(self.candidates)
        # sort by length, then alphabetical
        candsets = list(sorted(candsets, key=lambda x: (len(x),x[0]))) 
        
        ans = {}
        for s in candsets:
            # calculate equation (10)
            sum10 = 0
            for k in range(newnumbuckets):
                p = 1
                for c in self.candidates:
                    if c in s:
                        if k < len(tau[c]):
                            p *= tau[c][k]
                        else:
                            p *= 0
                    else:
                        if k < len(kappa[c]):
                            p *= 1-kappa[c][k]
                        else:
                            p *= 0
                sum10 += p
                
            key = "".join(s)
            key = list_to_stringlist(s)
    
            ans[key] = sum10
    
        return ans
        
    # gives an elimination probability for each candidate, with ties weighted
    # this is paper formula (11)
    def elimination_probabilities_tieweighted(self):
    
        ans = {}
        for c in self.candidates:
            ans[c] = 0
        
        ep = self.elimination_probabilities()

        for s in list(ep.keys()):
            l = stringlist_to_list(s)
            for c in l:
                ans[c] += ep[s] / len(l)
    
        return ans
    
    
    # returns a dict of dicts, giving all possibile elimination probability dicts for all possible rounds
    # Like {"ABC":{"A":.4, "B":.2, "C":.4}, "AB":{"A":0,"B":1}, ... }
    # a round labeled "ABC" means the round with only ABC in it (all others having been eliminated)
    def alleps(self):
        cands = self.candidates  # ["A","B","C","D"]
        thisround = list_to_stringlist(cands)      # "[A,B,C,D]"

        if len(cands)==1:
            return {cands[0]:{cands[0]:1}}
        else:
            ans = {}
            ans[thisround] = self.elimination_probabilities_tieweighted()

            for c in cands:
                #.update() is a dictionary concat
                ans.update(self.eliminate(c).alleps())
            return ans
            
    def winning_probabilities(self):
        eps = self.alleps()
        
        cands = self.candidates  # ["A","B","C","D"]
        thisround = list_to_stringlist(cands)      # "[A,B,C,D]"
        
        def cand_winning_prob(cand,round):    
            # if this is a round with only 1 candidate, they win with prob 1
            if len(round)==1:
                return 1
            
            else:
                # if this round has several candidates, the winning prob is a weighted sum across eliminations of other candidates
                p = 0
                for c in rem(round,cand):
                    newround = rem(round,c)
                    p += eps[list_to_stringlist(round)][c] * cand_winning_prob(cand,newround)
            
                return p
        
        return {c:cand_winning_prob(c,cands) for c in cands}

        
# ------------------------------------------------------------ #
# FUNCTIONS FOR IMPORT/EXPORT TO CSV FILES
# ------------------------------------------------------------ #

# applied when reading csv number data- blank becomes zero
def format(s):
    if s == "":
        return float(0)
    else:
        return float(s)
        
voteslabel = "Votes"

# read datafile into a ProbabilityTable 
# The file is CSV, first column is "Votes", with buckets 
# Then columns of probabilities with things like "[Bill,Steve,Anne]", etc
def probtable_from_file(f):
    d = {}
    with open(f) as csvfile:
        rows = list(csv.DictReader(csvfile, delimiter=',', quotechar='"'))
        
        rankings =  [ stringlist_to_list(k) for k in list(rows[0].keys())[1:] ]
        
        buckets = [int(row[voteslabel]) for row in rows]
        bucketsize = buckets[1]-buckets[0]
        
        for rk in rankings:
            srk = list_to_stringlist(rk)
            d[srk] = [format(row[srk]) for row in rows]
            
    pt = ProbabilityTable(d,bucketsize)
    return pt


# sanitizes a number for storage in the file
# the number is scipy data- we should round it and make positive
def sanitize(x):
    return abs(round(x,12))

# write a ProbabilityTable object into a file
# The column tops are "Votes", together with things like "ABC","ACB", etc
# The "Votes" column is the list of vote buckets
# the "ABC" column is a list of probabilities
def probtable_to_file(pt,f):
    with open(f,'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"')
        writer.writerow([voteslabel] + pt.rankings)

        length = pt.numbuckets
        for i in range(length):
            writer.writerow([str(pt.buckets[i])] + [str(sanitize(pt.lookup(r,i))) for r in pt.rankings])
    print("Wrote " + f)


# --------------------------------------------------------------------- #
# VARIOUS HELPER FUNCTIONS 
# --------------------------------------------------------------------- #

# takes a string like "[Anne||Bill||Joe]" and makes ["Anne","Bill","Joe"]
# Uses || as delimiter to avoid conflicts with candidate names containing commas
def stringlist_to_list(s):
    s = s[1:-1] # strip off brackets
    if s == "":
        return []
    else:
        return s.split('||')

# takes a list like ["Anne","Bill","Joe"] and makes "[Anne||Bill||Joe]"
# Uses || as delimiter to avoid conflicts with candidate names containing commas
def list_to_stringlist(s):
    if s == []:
        return "[]"
    elif len(s) == 1:
        return "[" + s[0] + "]"
    else:
        return "[" + "||".join(s) + "]"
        
# all possible shuffles of a nonempty list- assumes list has no repeats
def shuffles(l):
    if len(l)==1:
        return [l]
    else:
        return [[a] + xs for a in l for xs in shuffles([b for b in l if b!=a])]
    
# convolve a list of lists, 
def convolve_list(l):
    if len(l)==1:
        return l[0]
    else:
        k = convolve_list(l[1:]) #recursion
        ans = list(fftconvolve(l[0],k))
        return ans

# product of a list of numbers- apparently this is the most efficient way to do this
import operator
import functools
def prod(l):
    return functools.reduce(operator.mul, l, 1)
    
# set of subsets (sublists) (including l itself)
def subsets(l):
    if l==[]:
        return [[]]
    else:
        k = subsets(l[1:])
        return (k + [ [l[0]] + m for m in k])
        
# set of subset permutations of l (including l itself)
def subsetperms(l):
    ssps = []
    for s in subsets(l):
        ssps = shuffles(s) + ssps
    return ssps + [[]]

def nonemptysubsets(l):
    return ([s for s in subsets(l) if s != []])
    
def startswith(s,x):
    if s == []:
        return False
    else:
        return s[0]==x
           
## returns a list with the value x removed
def rem(l,x):
    return [a for a in l if a != x]


# ---------------------------------------------------------------------- #
# OBJECT FOR MAKING WEIGHTED ELIMINATION TREES
# ---------------------------------------------------------------------- #
class EliminationTree:
    def __init__(self, id, nodelabel, nodeweight, children):
        self.id = id
        self.label = nodelabel
        self.weight = nodeweight
        self.children = children

    def __str__(self):
        return f"Tree: {self.id} {self.label} {str(self.weight)}\n" + "".join(["\t" + str(c) for c in self.children])

    def numroots(self):
        if self.children == []:
            return 1
        else:
            return sum([l.numroots() for l in self.children])

    # string for a tikz standalone document showing the tree
    def tikzstring(self,wps,candnames={},vertical=False):
        # Default settings for graph output
        hspace_for_vertical = 1.5 # horz space between two nodes (center to center) at the bottom
        vspace_for_vertical = 7 # vert space between two rows of the tree
    
        hspace_for_horizontal = 7
        vspace_for_horizontal = .75
    
        scale = .5
        
        output = r'\documentclass{standalone}' + "\n" + r'\usepackage{tikz}' + "\n" + r'\begin{document}' + "\n\n\n"
        
        output += r'\tikzset{vert/.style={draw=black,circle}}' + "\n"
        output += r'\tikzset{weight/.style={fill=white,line width=.1,draw=black,rectangle,pos=.7}}' + "\n"
        output += r'\begin{tikzpicture}'
        output += r'[scale=' + str(scale) + r',every node/.style={scale=' + str(scale) + r'}]' + "\n"
        
        # offsets children
        def draw_shift(t,xoffset,yoffset):
            def sanitize_id(i):
                # kill commas and brackets
                return "".join([c for c in i if c not in ",[]"])
            
            def texsanitize(x): # fix up numbers
                x = abs(round(x*100,1))
                s = str(x)
                if x == 0:
                    s = "0"
                elif x == 100:
                    s = "100"
                
                return str(s) + r'\%'
            
            # root node
            output = ""
            output += r'\node[vert] (' + sanitize_id(t.id) + r') at (' + str(xoffset) + ',' + str(yoffset) + ') {' + t.label +'};' + "\n"
            
            # child nodes
            rootcount = 0
            
            bigkids = [c for c in t.children]
            for c in bigkids:
                if vertical:
                    output += draw_shift(c,xoffset + hspace_for_vertical * (-t.numroots()/2 + (rootcount + c.numroots()/2)) ,yoffset - vspace_for_vertical)
                else:
                    output += draw_shift(c,xoffset + hspace_for_horizontal, yoffset - vspace_for_horizontal * (-t.numroots()/2 + (rootcount + c.numroots()/2)))
                    
                rootcount += c.numroots()
            
                # edge to child node
                output += r'\draw[line width=' + str(c.weight*5) + '] ('
                output += sanitize_id(t.id)
                if vertical:
                    output += r') to[out=270,in=90] node[weight] {'
                else:
                    output += r') to[out=0,in=180] node[weight] {'
                output += texsanitize(c.weight) + r'} (' + sanitize_id(c.id) + ');\n'
        
            return output
            
                
        output += draw_shift(self,0,0)
                
        output += r'\end{tikzpicture}' + "\n\n\n" + r'\end{document}'
    
        return output    
    
# takes a ProbabilityTable object and makes an EliminationTree object out of it
def elimination_tree(pt,candnames={}):

    thisround = pt.candidates
    
    ept = pt.alleps()
    
    def eptree_root_path(round,path,w):
        r = list_to_stringlist(round)
        
        kids = []
        if len(round) > 1:
            kids = [ eptree_root_path(rem(round,c), path+[c], w*ept[r][c]) for c in round]

        label = ",".join([candnames[c] for c in round])
        
        return EliminationTree(r + "x" + list_to_stringlist(path), label, w, kids)
    
    return eptree_root_path(thisround,[],1)

# generates a tikz standalone file, saves it in outfilename
def export_elimination_tree(pt,outfilename,candnames={}):
    if candnames == {}:
        for d in pt.candidates:
            candnames[d] = d
            
    wps = pt.winning_probabilities()
    
    t = elimination_tree(pt,candnames)
    tex = t.tikzstring(wps,candnames)
    
    with open(outfilename, "w") as latexfile:
        latexfile.write(tex)