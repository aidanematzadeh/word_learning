import math

"""
wmmapping.py

Data-structures for mapping the words to meanings and word-feature alignments.

"""

class Meaning:
    """
    A Meaning object contains features and their probabilities of being part 
    of the meaning of a specific word. A Meaning object is associated with a
    specific word through a Lexicon.
    
    Members:
        meaning_probs -- dictionary to map features to probabilities
        unseen -- probability associated to all unseen features
    
    """
    
    def __init__(self, beta):
        """
        Create a meaning object with each feature being equally likely to be part
        of this meaning with probability 1.0/beta 
        
        """
        self._meaning_probs = {}
        self._unseen = 1.0/float(beta)

    #BM getValue
    def prob(self, feature):
        """ 
        Return the probability of this word having feature as part of its
        meaning.
        
        """
        if feature in self._meaning_probs:
            return self._meaning_probs[feature]
        else:
            return self._unseen

    #BM getSeenPrims
    def seen_features(self):
        """ Return a set of all features seen so far with this word. """
        return set(self._meaning_probs.keys())

    #BM getSortedPrims
    def sorted_features(self):
        """ 
        Return a list, sorted by probability, of each (prob, feature) seen so 
        far. 
        
        """
        items = self._meaning_probs.items()
        ranked = [ [v[1],v[0]] for v in items ]
        ranked.sort(reverse=True)
        return ranked    
    
    def unseen_prob(self):
        """ Return the probability of an unseen feature. """
        return self._unseen
    
    def copy(self, meaning):
        """ Copy the information from Meaning meaning into this meaning. """
        for feature in meaning._meaning_probs.keys():
            self._meaning_probs[feature] = meaning._meaning_probs[feature]
        self._unseen = meaning._unseen 
    
    def __str__(self):
        """ Format this meaning to print intelligibly."""
        result = str(self.sorted_features()) + " "
        result += "< " + str(self._unseen) + " >\n"
        return result


class Lexicon:
    """
    A Lexicon object maps words to Meaning objects.
    
    Members:
        word_meanings -- dictionary mapping words to Meaning objects
        beta -- inverse probability of unseen features being part of the meaning
        
    """

    def __init__(self, beta, words):
        """
        Create an empty Lexicon of words, such that each word in words has a
        Meaning with unseen probability 1.0/beta. See Meaning docstring.
            
        """
        self._word_meanings = {}
        self._beta = beta
        for w in words:
            self._word_meanings[w] = Meaning(beta)


    #BM getWords
    def words(self):
        """ Return a list of all words in this lexicon. """
        return self._word_meanings.keys()
    
    #BM getMeaning
    def meaning(self, word):
        """ Return a copy of the Meaning object corresponding to word. """
        meaning = Meaning(self._beta)
        if self._word_meanings.has_key(word):
            meaning.copy(self._word_meanings[word])
        return meaning
        

    #BM getSeenPrims
    def seen_features(self, word):
        """ Return a list of features encountered - so far - with word. """
        if word in self._word_meanings:
            return self._word_meanings[word].seen_features()
        return []
        
    #BM setValue
    def set_prob(self, word, feature, prob):
        """ 
        Set the probability of feature being part of the meaning of word to prob
        
        """
        if word not in self._word_meanings:
            self._word_meanings[word] = Meaning(self._beta)
        self._word_meanings[word]._meaning_probs[feature] = prob

    #BM getValue
    def prob(self, word, feature):
        """ Return the probability of feature being part of the meaning of word. """
        if word in self._word_meanings:
            if feature in self._word_meanings[word]._meaning_probs:
                return self._word_meanings[word]._meaning_probs[feature]
            return self._word_meanings[word]._unseen
        return 0.0

    #BM getUnseen
    def unseen(self, word):
        """ 
        Return the probability of an unseen feature being part of the meaning of
         word. 
         
        """
        if word in self._word_meanings:
            return self._word_meanings[word]._unseen
        return 0.0

    #BM setUnseen
    def set_unseen(self, word, unseen_value):
        """ 
        Set the probability of an unseen feature being part of the meaning of 
        word to unseen_value. 
        
        """
        if word in self._word_meanings:
            self._word_meanings[word]._unseen = unseen_value
        else:
            meaning = Meaning(self._beta)
            meaning._unseen = unseen_value
            self._word_meanings[word] = meaning


class Alignments:
    """
    Store calculated alignment probabilities for each word-feature pair at each 
    time step.
        
    Members:
        probs -- Dictionary where keys are "word feature" and values are lists of
            the form [sum_aligns, {time:alignment_value, time:alignment_value ...}] 
            where sum_aligns is the sum alignment value for this word-feature 
            pair of all alignment_value's so far.
            
            sum_aligns is the association score: 
                assoc_t(w,f) = assoc_(t-1)(w,f) + alignment_t(w|f) 
            where t represents the time step.
            
            alignment_t(w|f) is exactly the entry probs[w + " " + f][1][t]
            
        unseen -- #BM write
        
    """
    
    def __init__(self, alpha):
        """ Create an empty Alignments object with smoothing value 1.0/alpha """
        self._probs = {}
        # Allow no smoothing
        if alpha <= 0:
            self._unseen = 0.0
        else:
            self._unseen = 1.0 / alpha
        
    def sum_alignments(self, word, feature):
        """ 
        Return the sum alignment probability for the pair "word feature". If the
        entry does not exist, create one.
        
        """
        wf = word + " " + feature
        if wf not in self._probs:
            self.create_entry(word, feature)
        return self._probs[wf][0]
    
    def alignments(self, word, feature):
        """ 
        Return a dictionary of time : alignment_value pairs for the pair 
        "word feature". If the entry does not exist, create one.
        
         """
        wf = word + " " + feature
        if wf not in self._probs:
            self.create_entry(word, feature)
        return self._probs[wf][1]
    
    def add_alignment(self, word, feature, time, alignment):
        """ 
        Add (time, alignment) to the probabilities for word-feature pair and 
        update the sum alignment probability with alignment.
        
        """
        wf = word + " " + feature
        if wf not in self._probs:
            self.create_entry(word, feature)
        alignments = self._probs[wf]
        alignments[1][time] = alignment
        alignments[0] += alignment
    
    def add_decay_sum(self, word, feature, time, alignment, decay):
        """
        Calculate the association, using an alternative forgetting sum, for the
        pair word-feature, having alignment alignment at time time. decay is a
        constant decay factor.
        This is calculated as:
        
        assoc_{time}(word,feature) =
        
            assoc_{time'}(word,feature)
           ---------------------------------------------------  + alignment
            (time - time')**(decay/assoc_{time'}(word,feature))
            
        Where time' is the last time that the association between word and 
        feature was calculated.
        
        """
        wf = word + " " + feature
        if wf not in self._probs:
            self.create_entry(word, feature)
        alignments = self._probs[wf]
        
        # Association at time t'
        time_pr_assoc = alignments[0]
        # Last time word-feature association was calculated
        last_time = max(alignments.keys()) 
        
        assoc_decay = decay / time_pr_assoc
        alignments[0] = time_pr_assoc / math.pow(time - last_time + 1, assoc_decay) 
        alignments[0] += alignment
        alignments[1][time] = alignment # To keep track of time t'
    
    def create_entry(self, word, feature):
        """ 
        Create an empty alignment entry in this data structure for 
        "word feature". 
        
        """
        wf = word + " " + feature
        self._probs[wf] = []
        self._probs[wf].append(0.0)
        self._probs[wf].append({})
