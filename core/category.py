import re
import sys
import os
import numpy
import math
import time
try:
    from nltk.corpus import wordnet as wn
except ImportError:
    print "[Import Warning:  nltk.corpus -> wordnet] Module not found, wordnet labeling not possible"

try:
    from sklearn.cluster import Ward, DBSCAN
except ImportError:
    print "[Import Warning:  sklearn.cluster] Module not found, semantic clustering of categories not possible"

#from hcluster import linkage, fcluster, pdist
#from scipy.spatial.distance import pdist


#try:
#    from mlabwrap import mlab
#except ImportError:
#    print "[Import Warning:  mlabwrap] Module not found, semantic clustering of categories not possible"

from wmmapping import Meaning
import constants as CONST
import evaluate


#BM Half done method of CategoryLearner? ::
"""
# AFSANEH: This method has to be modified, so that the meaning of the word is 
    # an average of the meanings of all of its senses, weighted by their frequencies, 
    # collected in self.wordlist. A uniformed probability distribution should be 
    # returned for an unseen word.
    def getWordMeaning(self, word):
        meaning = Meaning(self.M)
        return(meaning)
        # meaning.update(dict([ (f,v) for (v,f) in self.learner.original_lex.getSortedPrims(word)]), 
        # self.learner.original_lex.getUnseen(word))
        # return dict([ (f,v) for (v,f) in self.learner.original_lex.getSortedPrims(word) ])

"""


class Category:
    """
    A Category object contains a list of words for this category, its label and
    the precision of that label, as well as Meaning to encapsulate the general
    semantic meaning of the category this represents.
    
    Members:
        meaning_normalization -- a normalization factor for the Meaning
        words -- a dictionary of words to their frequency of appearance in this
            category
        id -- the numeric ID of this category
        label -- the label given to this category
        precision -- the precision of the label of this category (typically how
            many words in the category have the wordnet label of this label over
            the total number of words in the category)
        meaning - the Meaning object for this category
    
    """
    def __init__(self, meaning_normalization, id, label, precision):
        # TODO meaning normalization is the same as beta?
        self._meaning_normalization = meaning_normalization
        self._id = id
        self._label = label
        self._precision = precision
        self._words = {}
        self._meaning = Meaning(meaning_normalization)
        
    #BM addNewWord   
    def add_word(self, word):
        """ 
        Add the word word to this category if it is not already present, 
        otherwise increase the frequency count of this word.
        
        """
        if word not in self._words:
            self._words[word] = 1
        else:
            self._words[word] += 1
            
        #BM this was commented out, important?    
        #self.updateMeaning(meaning)   
    
    #BM updateBatchMeaning    
    def update_batch_meaning(self, lexicon, features):
        """
        Perform a batch update of this categories' meaning for all words of this
        category based on the Lexicon lexicon for all in list features.
        
        """
        #BM Calculated differently than the update method...
        num_word_tokens = sum(self._words.values())
        
        # Get the weighted probably of each word(/feature) of this category
        for feature in features:
            weighted_feature_prob = 0.0
            for word in self._words.keys():
                occurrences = self._words[word]
                weighted_feature_prob += lexicon.prob(word, feature) * occurrences
            weighted_feature_prob = weighted_feature_prob / num_word_tokens
            self._meaning._meaning_probs[feature] = weighted_feature_prob
        
        # Do the same for the unseen probability
        unseen = 0.0
        for word in self._words:
            occurrences = self._words[word]
            unseen += lexicon.unseen(word) * occurrences
        unseen = unseen / num_word_tokens
        self._meaning._unseen = unseen


    #BM updateMeaning 
    def update(self, word, old_meaning, lexicon, features):
        """
        Update this category meaning by averaging over the old Meaning 
        old_meaning for all seen features so far with the current meaning, 
        with respect to word. The probabilities of each word-feature pair are 
        extracted from the Lexicon lexicon and the list features
        
        """
        if word not in self._words:
            self.add_word(word)

        #BM ask Afsaneh about wfreq which was here        
        num_word_tokens = len(self._words.keys()) 

        # Update this categories meaning probabilities by averaging over the
        # difference between the old mean for this word and the meaning
        for feature in features:
            new = lexicon.prob(word, feature)
            old = old_meaning.prob(feature)
            updated = float(new - old) / num_word_tokens
            self._meaning._meaning_probs[feature] += updated
          
        # Do the same for the unseen probability  
        new_unseen = lexicon.unseen(word)
        old_unseen = old_wmeaning.unseen_prob()
        updated = float(new_unseen - old_unseen) / num_word_tokens
        self._meaning._unseen += updated
        

    #BM checkProbabilities
    def check_probs(self):
        """ Print information about this category. """
        meaning = self._meaning
        features = meaning._meaning_probs.keys()
        total = 0.0
        for feature in features:
            total += meaning.prob(feature)
        
        print "Category: ", self._label
        print "  Seen probabilities: ", total
        unseen = meaning._unseen * (self._meaning_normalization - len(features))
        print "  Unseen probability: ", unseen
        total += unseen 
        print "  Total: ", total

        
class CategoryLearner:
    """
    A CategoryLearner wraps Category objects and maintains word to Category
    relations.
    
    Members:
        meaning_normalization -- a normalization factor for Category Meaning
            objects
        words_to_categories -- a dictionary of words to category IDs
        categories -- a dictionary of category IDs to Category objects
        
    
    """
    def __init__(self, meaning_normalization, categories_dictionary, lexicon, features):
        """
        Create a CategoryLearner object from the dictionary categories_dictionary
        and print off information for each Category object created. Lexicon
        lexicon and the set features are used to update each category.
        
        Prerequisite: categories is a diction object built by one of the data
        pre-processing functions in category.py.
        
        """
        #TODO change the name of the lexicon 
        self._meaning_normalization = meaning_normalization
        self._words_to_categories = {}
        self._categories = {}

 #       self._wordnet_categories = {}
        
        self.make_categories(categories_dictionary, lexicon, features)
        #for id in self._categories.keys(): 
        #     self.print_category(id)
        #     self.print_label(id)
        
        
    #BM makeCategories   
    def make_categories(self, categories, lexicon, features):
        """
        Populate this CategoryLearner object with categories (labeled with 
        precisions) and words based on the dictionary object categories. Lexicon
        lexicon and the set features are used to update each category.
        
        Prerequisite: categories is a dictionary object built by one of the data
        pre-processing functions in category.py.
        
        """
        for id in categories.keys():
            category = categories[id]
            self.add_category(id, category["label"], category["precision"])
            
            for word in category["words"]:
                self.add_word(id, word)
                                   
            self._categories[id].update_batch_meaning(lexicon, features)
        
    #BM addCategory
    def add_category(self, id, label, precision):
        """
        Create a new category with ID id, label label, and precision precision.
        
        """
        if id in self._categories:
            raise KeyError("Category with ID "+str(id)+" already exists")
        self._categories[id] = Category(self._meaning_normalization, id, label, 
                                           precision)

    #BM addWordToCategory
    def add_word(self, id, word):
        """ Add word word to the category with ID id. """
        if id not in self._categories:
            raise KeyError("Category with ID "+str(id)+" does not exists")
        self._categories[id].add_word(word)
        
        if word not in self._words_to_categories:
            self._words_to_categories[word] = {}
        if id in self._words_to_categories[word]:
            self._words_to_categories[word][id] += 1
        else:
            self._words_to_categories[word][id] = 1
    
    #BM getWordCategory
    def word_category_label(self, word):
        """
        Return the label of the category that word appears in.
        If word is not categorized, return the empty string.
        
        """
        if word in self._words_to_categories:
            category_id = (self._words_to_categories[word].keys())[-1]
            return self._categories[category_id]._label
        return ""
    
    def word_category(self, word):
        """
        Return the category ID of the category that word word appears in.
        If word is not categorized, return -1.
        
        """
        if word in self._words_to_categories:
            return (self._words_to_categories[word].keys())[-1]
        return -1

    #BM categorizeWord
    def categorize(self, word, sim_type, beta, scene_features=None, wn_category=None, meaning=None):
        """
        Categorize word.
        If a wordnet category wn_category is passed then the categorization 
        process is performed based on the category that matches this word's 
        wordnet category that has the highest precision. The precision is returned.
        If a Meaning meaning is passed then the categoriziation is calculated        
        using a sim_type similarity calculation with smoothing factor beta
        against each category's meaning. The similarity score is returned.
        If scene_features are passed, the categorization is done by finding
        the category that matches the features in the scene.
        
        """
        score = [0, -1, ""] # Precision, ID, Label, last two for debugging

        for category in self._categories.values():
            #BM Again, is this needed
            #if catname == "0":continue
            if wn_category is not None:
                if category._label == wn_category and category._precision > score[0]:
                    score[0] = category._precision
                    score[1] = category._id
                    score[2] = category._label
            
            elif meaning is not None:
                sim = evalulate.calculate_similarity(beta, meaning, 
                                                     category._meaning, sim_type)
                if sim > score[0]:
                    score[0] = sim
                    score[1] = category._id
                    score[2] = category._label
            
            elif scene_features is not None:
                cat_sorted_features =  category._meaning.sorted_features()[:10] #TODO
                cat_features = []
                for item in cat_sorted_features:
                    cat_features.append(item[1])
                intersect = len(set(cat_features) & set(scene_features))
                
                print category._label, intersect
                
                if intersect > score[0]:
                    score[0] = intersect
                    score[1] = category._id
                    score[2] = category._label


            
        return score[1]
 
    #BM printCategoryLabel    
    def print_label(self, id):
        """ 
        Print the label and precision of that label for category with ID id.
        
        """
        if id not in self._categories:
            print "No category with ID: ", id
            return
        category = self._categories[id]

        category_labels = {}
        # Find all the labels of words in this category, the true label is 
        # the most frequent label
        for word in category._words.keys(): 
            label = wordnet_category(word)
            if label == CONST.NONE: 
                continue
            
            if label not in category_labels:
                category_labels[label] = 0
            category_labels[label] += 1

        label = ""
        max = 0
        for l in category_labels.keys():
            if category_labels[l] > max:
                max = category_labels[l]
                label = l
        if label == "":
            print "No label found for category with ID: ", id
            return
        
        print "Label: ", label
        print "Precision: ", float(category_labels[label]/sum(category_labels.values()))

    #BM printCategory
    def print_category(self, id):
        """ Print detailed category information. """
        category = self._categories[id]
        print "%s (%d): { " % (id, len(category._words.keys()))
        for word in category._words.keys(): 
            print word, ":", category._words[word], ", ",
        print " }\n"
        print category._meaning
        print "\n"

    #BM checkProbabilities
    def check_probs(self):
        """ Print category probability information for all categories. """
        for category in self._categories.values():
            category.check_probs()

    #BM getMeaning
    def prob(self, id, feature):
        """
        Return the probability of feature being part of the meaning of the 
        category corresponding to ID id. If no such category exists, return 0. 
        
        """
        if id in self._categories:
            return self._categories[id]._meaning.prob(feature)
        return 0
        
    #BM updateMeaning
    def update(self, id, word, old_meaning, lexicon, features):
        """
        Update category with category id ID, if it exists. The category meaning 
        is updated by averaging over the old Meaning old_meaning for all seen 
        features so far with the categories current meanings, with respect to 
        word. The probabilities of each word-feature pair are extracted from the
        Lexicon lexicon and the list of features features.
        
        """
        if id in self._categories:
            self._categories[id].update(word, old_meaning, lexicon, features)
            
        
#===============================================================================
#    Pre-process Data - Semantic Clustering
#===============================================================================

AVOID_CLUSTERS = ["noun.process", "noun.Tops", "noun.relation", "noun.motive"]

#BM cluster
def semantic_clustering_categories(beta, words, lexicon, features, stopwords, pos=CONST.ALL, 
                                   n_clusters=30, linkage_method="weighted", sim="cosine"):
    """
    Perform semantic clustering on the lists words and features, using the 
    Lexicon lexicon for calculating meanings of words. stopwords is a list of 
    words to be disregarded. The clustering is performed using the matlab
    pdist, linkage, and cluster operations.
    
    Return a dictionary of category IDs where each ID corresponds to a dictionary
    with keys "words" (a list of words in that category) "label" (the wordnet
    label of that category) and "precision" (how precise the label is).
    
    """
    filtered_words = []
    meanings = []
    
    print "Words so far: ", len(words)
    
    features = list(features)
    features.sort()
    sorted_words = words.keys()
    sorted_words.sort()

    for word in sorted_words:
        if pos != CONST.ALL:
            if word in stopwords or not word.endswith(pos):
                continue
    
        label = wordnet_category(word)
        # Do not use words with no label or are not in meaningful clusters
        if label == CONST.NONE or label in AVOID_CLUSTERS:
            continue
        
        meaning_vec = numpy.zeros(len(features) + 1)  #Last index to store the unseen prob
        meaning = lexicon.meaning(word)

        for ind in range(len(features)):
#            ind = features.index(feature)
            meaning_vec[ind] = meaning.prob(features[ind])
        
        meaning_vec[len(features)] = meaning.unseen_prob()

        filtered_words.append(word)
        meanings.append(meaning_vec)
        

    print "Number of words in Clustering: ", len(filtered_words) 
    print "Compute structured hierarchical clustering..."
    
    def __distance(u, v):
        u_unseen = u[-1]
        u = u[:-1]
        
        v_unseen = v[-1]
        v = v[:-1]
        
        cos = numpy.dot(u, v)
        
        seen_count = len(u)
        cos += (beta - seen_count) * u_unseen * v_unseen
    
        x = math.sqrt(numpy.dot(u, u) + (pow(u_unseen, 2) * (beta - seen_count)))
        y = math.sqrt(numpy.dot(v, v) + (pow(v_unseen, 2) * (beta - seen_count)))
    
        if x*y < cos: print "arr", cos, x * y
        return max(1 - cos/(x * y), 0)
 
    # Compute clustering
    st = time.time()   
    #clusters = Ward(n_clusters).fit(meanings)
 
    distance = pdist(meanings, __distance)
    z = linkage(distance, linkage_method)

  #  clusters = fcluster(z, 0.9 * distance.max(), 'distance')
    clusters = fcluster(z, n_clusters, 'maxclust')

    print "Elaspsed time: ", time.time() - st
    return form_categories_dict(clusters, filtered_words), clusters, filtered_words 

#BM form_clusters
def form_categories_dict(clusters, words):
    """
    Form a category dictionary object based on the 2D array clusters, formed from
    the cluster operation, for the words in list words.
    The category dictionary consists of category IDs where each ID corresponds 
    to a dictionary with keys "words" (a list of words in that category) "label" 
    (the wordnet label of that category) and "precision" (how precise the label is).
    
    """
    num_clusters = numpy.max(clusters)

    # See docstring for dictionary categories' structure
    #BM This has the +1 to not use categories[0], need to ask about this
    
    categories = {} 
    #Note that cluster labels start from 1
    for j in range(1, num_clusters + 1):
        categories[j] = {}
        categories[j]["words"] = []
   
    # Add words to their respective categories
    for i in range(len(words)):
        categories[clusters[i]]["words"].append(words[i])

    # Label each category using the wordnet labels
    return label_categories(categories)

        

class WordnetLabels:
    def __init__(self):
        self._wordnet_categories = {}

    def wordnet_label(self, word):
        """ 
        Return the wordnet category of word, a word of the form "word#{pos_tag}".
        
        """
        pos = word[word.find(":") + 1:]
        w = word[:word.find(":")]
        
        if word in self._wordnet_categories:
            return self._wordnet_categories[word]


        if pos not in [CONST.N, CONST.V, CONST.ADJ, CONST.ADV]:
            self._wordnet_categories[word] = CONST.NONE
            return CONST.NONE

        right_sense = 0 
        if pos == CONST.N:
            senses = wn.synsets(w, wn.NOUN) 
            if len(senses) >=1:
                # There are special cases for nouns where the correct sense is not
                # always the first sense.
                right_sense = self._correct_sense(senses, w)
        elif pos == CONST.V:
            senses = wn.synsets(w, wn.VERB)  
        elif pos == CONST.ADJ:
            senses = wn.synsets(w, wn.ADJ) 
        elif pos == CONST.ADV:
            senses = wn.synsets(w, wn.ADV) 

        if len(senses) < 1: 
            self._wordnet_categories[word] = CONST.NONE
            return CONST.NONE
        else:
            self._wordnet_categories[word] =  senses[right_sense].lexname
            return senses[right_sense].lexname


    def _correct_sense(self, senses, noun):
        """ 
        For the list of wordnet senses senses corresponding to the noun noun, return
        the index of the correct sense in senses based on special cases 
        (see cases below).
        
        """
        # Special Case 1: For alphabetic nouns, return the "letter" sense index
        if len(noun) == 1 and noun != 'i':
            for ind in range(len(senses)):
                if "letter of the Roman alphabet" in senses[ind].definition:
                    return ind

        # Special Case 2: For plant-food nouns, pick "food" sense over "plant" sense
        food_sense = -1
        plant_sense = -1
        for ind in range(len(senses)):
            if food_sense < 0 and "noun.food" in senses[ind].lexname:
                food_sense = ind

            if plant_sense < 0 and "noun.plant" in senses[ind].lexname:
                plant_sense = ind

        if food_sense >= 0 and plant_sense >= 0:
            return food_sense
        else :
            return 0 # The default correct sense













#===============================================================================
#    Helper Functions
#===============================================================================

def label_categories(categories):
    """
    categories is a dictionary of IDs to lists of words in that specific category.
    Using the words each category will be given a label - the most frequent 
    wordnet category of words in the given category - and precision - the number
    of words in the category with the most frequent wordnet label normalized by
    the total in the category. These are added as "label" and "precision" 
    dictionary entries at the same level as "words".
    
    """
    #Count for each label in the whole corpus
    labels_count = {}

    for category in categories.keys():
        labels = {}
        category_words_count = 0
        
        # Count the frequency of each label type of words in this category and
        # the number of words in this category
        for word in categories[category]["words"]:
            label = wordnet_category(word)
            if label == CONST.NONE: 
                continue
            
            if label not in labels:
                labels[label] = 0
            labels[label] += 1
            #BM only labeled words are counted? But they're left in the category
            category_words_count += 1

        if len(labels) < 1:
            continue
        
        # This category's label is the most frequent of all the words' labels
        true_label = ""
        freq = 0
        for label in labels:
            if labels[label] > freq:
                freq = labels[label]
                true_label = label
            
            if not labels_count.has_key(label):
                labels_count[label] = 0
            labels_count[label] += labels[label]


        categories[category]["label"] = true_label
        categories[category]["precision"] = float(freq) / category_words_count
        categories[category]["freq"] = float(freq)

    for category in categories:
        categories[category]["recall"] = categories[category]["freq"] / labels_count[categories[category]["label"]]

    return categories
    

def form_wordnet_categories(words):
    """
    Return a list of wordnet categories (lexname) for the given words.
    """
    wordnet_lexnames = {}
    lexname_counts = 1
    errors = 0
    wordnet_categories_list = numpy.zeros(len(words))
    wordnet_categories_dic = {}

    for i in range(len(words)):
        label = wordnet_category(words[i])

        if label == CONST.NONE: 
            errors += 1

        if not label in wordnet_lexnames.keys():
            wordnet_lexnames[label] = lexname_counts
            wordnet_categories_dic[label] = []
            lexname_counts += 1
        
        wordnet_categories_dic[label].append(words[i])
        wordnet_categories_list[i] = wordnet_lexnames[label]

    print "number of words without WN categories", errors
    return wordnet_categories_list, wordnet_categories_dic





#===============================================================================
#    Pre-process Data - POS Tag Categorization
#===============================================================================

def tagged_corpus_categories(corpus_path):
    """
    Construct a category dictionary object using the corpus at corpus_path such
    that each category is a part of speech tag.
    
    The category dictionary consists of category IDs where each ID corresponds 
    to a dictionary with keys "words" (a list of words in that category) "label" 
    (part of speech tab) and "precision" (for this the precision is always 1).
    
    """
    categories = {} 
    # Categories must be index by an ID number, but we're using pos tag labels
    pos_to_cat_id = {}
    next_id = 1 #BM again, skip id=0
    corpus_file = open(corpus_path)
    line = corpus_file.readline()
    
    # Add words to their respective categories
    while line != "":
        tagged_words = re.findall("([^ ]+)\s", line.strip('\n')) 
        for tagged in tagged_words:
            (word,category) = tagged.split(':')
            
            if category not in pos_to_cat_id:
                categories[next_id] = {}
                categories[next_id]["words"] = set()
                categories[next_id]["label"] = category
                categories[next_id]["precision"] = 1
                pos_to_cat_id[category] = next_id
                next_id += 1
                
            categories[pos_to_cat_id[category]]["words"].add(word)
    
        line = indata.readline()   
    
    return categories

