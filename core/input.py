import re
import sys
from wmmapping import Lexicon

"""
input.py

Helper class and methods for processing the input data - Corpus and the gold 
standard lexicon.

"""

class Corpus:
    """
    File container object to wrap a corpus file. 
    The file is required to be in a format of (repeated):
    
    {delimiter-line}
    SENTENCE: {word}:{tag} {word}:{tag} {word}:{tag} 
    SEM_REP:  ,{meaning}#{meaning-number},{meaning}#{meaning-number}
    
    Note the leading comma in the line for the semantic representation.
    
    Members:
        name -- name of this corpus, also path to the file
        handle -- reference to the corpus' file object
    
    """
    
    def __init__(self, name):
        """ Create a Corpus for the corpus file located at name. """
        self._name = name
        self._handle = open(self._name)

    def next_pair(self):
        """
        Return a tuple of the next input utterance - scene pairs in this corpus.
        The first value of the tuple is the set of unique words appearing in the 
        utterance. The second value of the tuple is the set of unique features 
        appearing in the scene.
        
        """
        line = self._handle.readline() # Remove delimiter line
        if line == "":
            return([],[])
        else:
            sentence = self._handle.readline().strip('\n') + " "
            meaning = self._handle.readline().strip('\n')
            
            words = re.findall("([^ ]+)\s", sentence) 
            del words[0] # Remove leading "SENTENCE: " identifier
            words = list(set(words)) # Remove duplicates
            
            # if no referent(i.e. feature set) found, 
            # reads in each feature seperately and independently
            if not (';' in meaning):
                
                features = re.findall(",([^,]+)", meaning)
                features = list(set(features))
                return (words, features)             
            
            # otherwise switch to referent reading mode, split the 
            # features in the scene representation according to each referent
            else:
                # get rid of the leading "SEM_REP"
                feature_set = meaning[10:]
                feature_set =re.findall("([^;]+);", feature_set) 
                
                # create a list containing all individual faetures
                # for each referent
                list_of_referent = []
                for referent in feature_set:
                    features = re.findall("([^,]+)", referent) 
                    if features:
                        list_of_referent.append(features)
                
                set_of_referent = []       
                # remove duplicates
                for referent in list_of_referent:
                    if referent not in set_of_referent:
                        set_of_referent.append(referent)   
                return (words, set_of_referent) 

    def close(self):
        """ Close the file backing this Corpus. """
        self._handle.close()



"""
    Reading the gold lexicon (lexicon to compare learned model against)

"""

#BM readAll
def read_gold_lexicon(gold_lex_path, beta):
    """
    Return a Lexicon corresponding to the file at gold_lex_path.
    The file is required to be in a format of (repeated):
    
    {word}:{tag} {other-word}#{meaning-number}:{probability},
    {delimiter-line}
    
    Note the trailing comma. Many word#meaning-number:probability pairs are 
    expected.
    
    """
    f = open(gold_lex_path) 
    # Start the lexicon empty and popular as the file is parsed
    gold_lex = Lexicon(beta, [])               

    while 1:
        (word, features) = next_lexeme(f)
        if word == "": 
            break

        for feat in features:
            feat = feat + ":"
            (feature, prob) = re.findall("([^:]+):", feat)
            gold_lex.set_prob(word, feature, eval(prob))

        gold_lex.set_unseen(word, 0.0)

    f.close()
    return gold_lex

def next_lexeme(f):
    """ 
    Return the next (word,features) tuple of file f.
    See docstring of read_gold_lexicon for expected file format.
    
    """
    line = f.readline()
    if line == "":
        return ("", [])
    else: 
        line = re.sub(" ", ",", line, count=1)
        List = re.findall("([^,]+),", line)
        f.readline()            
        return (List[0], List[1:])
    
def all_features(path):
    """
    Return the set of all features found in the lexicon file located at path.
    See docstring of read_gold_lexicon for the expected file format. 
    
    """
    f = open(path, 'r')
    features = set()
    line = f.readline()
    while line != "":
        line = re.sub(" ", ",", line, count=1)
        List = re.findall("([^,]+),", line)
        for feature in List[1:]:
            features.add(feature[:feature.find(':')])
        line = f.readline()            
    
    f.close()
    return features
