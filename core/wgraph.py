from graph_tool.all import *
import matplotlib.pyplot as plt
from operator import itemgetter

from category import WordnetLabels
import constants as CONST
import evaluate


import numpy as np
import scipy.stats
#from scipy.misc import logsumexp
import math, random

import os, sys, copy
#import powerlaw

from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic


#---------------------------------------------------#
def process_norms(norms_dir, words_filter):
    print "process_norms ---", "words in graph", len(words_filter)
    #CUE, TARGET, NORMED?, #G, #P, FSG, BSG,i

    norms = {}
    for filename in os.listdir(norms_dir):
        norm_file = open(norms_dir + "/" + filename, 'r')
        for line in norm_file:
            if line.startswith("<"):
                continue

            nodes = line.strip().split(',')
            cue = nodes[0].strip().lower() + ":N"
            target = nodes[1].strip().lower() + ":N"

            if cue in words_filter: #TODO
               if target in words_filter: #TODO
                    if not norms.has_key(cue):
                        norms[cue] = []

                    norms[cue].append([target, float(nodes[5])])

    return norms

def wordnet_similarity(words, sim_measure, wnlabels):
    sims = {}
    brown_ic = wordnet_ic.ic('ic-brown.dat')

    for word in words:
        w = word[:word.find(":")]
        senses = wn.synsets(w, wn.NOUN)
        if len(senses) < 1: continue #TODO
        right_sense = wnlabels._correct_sense(senses, w)

        targets = []

        for othword in words:
            if word == othword: continue

            othw = othword[:othword.find(":")]
            othw_senses = wn.synsets(othw, wn.NOUN)
            if len(othw_senses) < 1: continue
            othw_right_sense = wnlabels._correct_sense(othw_senses, othw)

            #print w, othw, senses[right_sense], othw_senses[othw_right_sense]

            if sim_measure == "jcn":
                sim = senses[right_sense].jcn_similarity(othw_senses[othw_right_sense], brown_ic)
            elif sim_measure ==  "wup":
                sim = senses[right_sense].wup_similarity(othw_senses[othw_right_sense])
            elif sim_measure == "path":
                sim = senses[right_sense].path_similarity(othw_senses[othw_right_sense])


            targets.append([othword, sim])

        targets = sorted(targets, reverse=True, key=itemgetter(1))
        #print word, targets
        sims[word] = targets
    return sims


def process_rg_norms(path, words_filter):
    ''' reading Rubenstein-Goodenough or
    The WordSimilarity-353 Test Collection norms'''

    rg_norms = {}
    infile = open(path, 'r')
    for line in infile:
        nodes = line.strip().split()
        cue = nodes[0].strip().lower() + ":N"
        target = nodes[1].strip().lower() + ":N"
        if cue == target: continue

        if cue in words_filter and target in words_filter: #TODO
            if not rg_norms.has_key(cue):
                rg_norms[cue] = []
            rg_norms[cue].append([target, float(nodes[2])])

    print "---------------------------------------------------------"
    for cue in rg_norms:
        rg_norms[cue] = sorted(rg_norms[cue], reverse=True, key=itemgetter(1))
        print cue, rg_norms[cue]


    return rg_norms

#--------------------------------------------------------------------------------#

class Category:


    def __init__(self, cat_id, coupling, lambda0, a0, miu0, sigma0):
        self._id = cat_id

        self._words = {}

        self._feature_values = {}

        #the values representing posterior prob. of f|k
        self._features_dist = {}

        #The most dominant wordnet label among the words in the category
        self._label = ""
        self._precision = 0
        self._recall = 0
        self._freq = 0

        #The coupling parameter used in prior
        self._coupling = coupling

        #confidence in features prior mean
        self._lambda_0 = lambda0 #
        #confidence in features prior vairance
        self._a_0 = a0
        # the prior mean
        self._miu_0 = miu0

        #the prior variance
        self.sigma_sq_0 = sigma0 *sigma0


    def add_word(self, word, word_features):
        ''' adding a word to the category '''

        if not self._words.has_key(word):
            self._words[word] = 0
        self._words[word] += 1

        #keeping the value/score of each feature
        for feature in word_features:
            if not self._feature_values.has_key(feature):
                self._feature_values[feature] = []
                self._features_dist[feature] = {}

            self._feature_values[feature].append(word_features[feature])

        self._update_features_post_pred()

    def _update_features_post_pred(self):
        '''update features posterior predictive '''

        #number of objects in K -- token freq of words
        n_k = np.sum(self._words.values())

        for feature in self._feature_values:
            lambda_i = self._lambda_0 + n_k

            a_i =  self._a_0 + n_k

            mean_fi =  np.mean(self._feature_values[feature])
            var_fi = np.var(self._feature_values[feature])

            miu_0 = mean_fi
            #miu_0 = self._miu_0

            #mean of the generalized t distribution
            miu_i = (self._lambda_0 * miu_0 + n_k * mean_fi)/(self._lambda_0 + n_k)

            sigma_sq_i = self._a_0 * self.sigma_sq_0 + (n_k-1) * var_fi + ((self._lambda_0 * n_k)/(self._lambda_0 + n_k)) * (miu_0 - mean_fi)**2
            sigma_sq_i /= (self._a_0 + n_k)

            #variance of the generalized t distribution
            var_i = sigma_sq_i * (1 + 1.0/lambda_i)

            self._features_dist[feature]["miu"] = miu_i
            self._features_dist[feature]["var"] = var_i
            self._features_dist[feature]["a"] = a_i #freedom parameter

    def _prior(self, n):
        ''' n is the number of objects seen so far'''

        #number of objects in K -- token freq of words
        n_k = np.sum(self._words.values())
        alpha = (1.0 - self._coupling)/ self._coupling

        if n_k == 0: # empty/new category
            return math.log(alpha) - math.log(alpha + n)

        return math.log(n_k) - math.log(alpha + n)

    def _likelihood_vectorized(self, word, word_features):
        ''' word_features maps f:v in for a word '''
        def eval_feature_value(feature):
            mu  = self._features_dist[feature]["miu"] if feature in self._features_dist else self._miu_0
            var = self._features_dist[feature]["var"] if feature in self._features_dist else self.sigma_sq_0 * (1 + 1.0/self._lambda_0)
            return (word_features[feature] - mu) /np.sqrt(var)

        a_array     = [ self._features_dist[f]["a"] if f in self._features_dist else self._a_0 for f in word_features ]
        fval_array  = [ eval_feature_value(f) for f in word_features ]

        ll_list = scipy.stats.t.logpdf(fval_array, a_array)

        return sum(ll_list)

    def _likelihood(self, word, word_features, vectorized = True):
        ''' word_features maps f:v in for a word '''
        if vectorized:
            return self._likelihood_vectorized(word, word_features)

        ll = 0.0 # P(F|K) = pie_i P(f_i|K), likelihood

        for feature in word_features:
            #calculating p(f_i|k)
            feature_value = word_features[feature]

            var_i = self.sigma_sq_0 * (1 + 1.0/self._lambda_0)
            miu_i = self._miu_0
            a_i =  self._a_0

            if feature in self._features_dist:
                miu_i = self._features_dist[feature]["miu"]
                var_i = self._features_dist[feature]["var"]
                a_i = self._features_dist[feature]["a"]  #freedom parameter


            feature_value = (feature_value - miu_i) /np.sqrt(var_i)

            try:
                ll += (scipy.stats.t.logpdf(feature_value, a_i))
            except:
                print "error in t", miu_i, var_i, a_i, feature_value


        return ll

    def posterior(self, word, word_features, n):
        _prior = self._prior(n)
        _ll = self._likelihood(word, word_features)

        if math.exp(_prior+_ll)<0 or math.exp(_ll+_prior) > 1:
            print "error"
#        print word, "cat:", self._id, "prior",math.exp(_prior), "LL:", math.exp(_ll) , "post",math.exp(_prior + _ll)
        return _prior + _ll


# ===================================================================================

class WordsGraph():

    def __init__(self, hubs_num, sim_threshold, hub_type, \
        coupling, lambda0, a0, miu0, sigma0, sampling_method):

        self._graph = Graph(directed=False)
        self._graph.vertex_properties["label"] = self._graph.new_vertex_property("string")
        self._graph.vertex_properties["acqscore"] = self._graph.new_vertex_property("double")
        self._graph.vertex_properties["freq"] = self._graph.new_vertex_property("int")

        self._graph.edge_properties["distance"] = self._graph.new_edge_property("double")
        self._graph.edge_properties["similarity"] = self._graph.new_edge_property("double")

        # Maps each word to the vertex object
        self._words_vertex_map = {}

        # Information/parameters about categories
        self._categories = {}
        self._words_token_freq = 0.0
        self._coupling = coupling
        self._lambda0  = lambda0
        self._a0 = a0
        self._miu0 = miu0
        self._sigma0 = sigma0
        self._sampling_method = sampling_method
        
        self._wnlabels = WordnetLabels()

        # Parameters
        # Number of hubs that are compared with each word
        self._hubs_num = hubs_num #75
        # The similarity threshold for connecting two words
        self._sim_threshold = sim_threshold #0.6
        self._hub_type = hub_type


        self.max_computations = []
        self.computations = []
        #The number of computations used in updating current edges
        self.update_computations = []
        #the number of new computations done
        self.new_computations = []


        # List to keep top nodes
        self._most_frequent_nodes = []
        
        self._highest_degree_nodes = []
    
    
    def _select_features(self, word_features, N=15):
        '''Select N number of features that have high prob. from a word'''
        #should move to meaning
        sorted_features = []
        for feature in word_features:
            sorted_features.append([feature, word_features[feature]])
        sorted_features = sorted(sorted_features, key=itemgetter(1))

        selected_features = {}
        for feature, value in sorted_features[:N]:
            selected_features[feature] = value


        return selected_features


    def _calculate_similarity(self, word_features, other_word_features):
        ''' calculate simiarity between two words, given the specific features '''
        #should move to evaluate
        features = set(word_features.keys()) | set(other_word_features.keys())

        meaning1_vec = np.zeros(len(features))
        meaning2_vec = np.zeros(len(features))

        i = 0
        for feature in features:
            if word_features.has_key(feature):
                meaning1_vec[i] = word_features[feature]
            if other_word_features.has_key(feature):
                meaning2_vec[i] = other_word_features[feature]
            i += 1

        cos = np.dot(meaning1_vec, meaning2_vec)

        x = math.sqrt(np.dot(meaning1_vec, meaning1_vec))

        y = math.sqrt(np.dot(meaning2_vec, meaning2_vec))

        return  cos / (x * y)
    
    
    def calc_vertex_score(self, rel_freq, rel_degree, recency):
        if self._hub_type == "hub-freq":
            return rel_freq

        if self._hub_type == "hub-degree":
            return rel_degree

        if self._hub_type == "hub-recency":
            return recency

        if self._hub_type == "hub-freq-degree-recency":
            return rel_freq * rel_degree * recency

        if self._hub_type == "hub-freq-degree":
            return 0.5 * (rel_freq + rel_degree)
        
        '''
        sum_freq = 0.0
        for v in self._graph.vertices():
            sum_freq +=  self._graph.vertex_properties["freq"][v]

        sum_degree = float(sum(self._graph.degree_property_map("total").a))

        for v in self._graph.vertices():
            rel_freq = self._graph.vertex_properties["freq"][v] / sum_freq

            rel_degree = 0
            if sum_degree != 0:
                rel_degree = v.out_degree() / sum_degree

            word = self._graph.vertex_properties["label"][v]
            recency = 1.0 / (ctime - last_time[word] +   1)

            score = self.calc_vertex_score(rel_freq, rel_degree, recency)
            vertices.append([v, score])

        #print word, rel_freq,  recency, rel_degree, score

        vertices = sorted(vertices, key=itemgetter(1), reverse=True)

        return vertices[:hubs_num]
        '''

    def update_most_list(self, vertex, vertex_value, maximum_list, list_desired_size, t="deg"):
        '''  '''
        list_desired_size =  150 #TODO CHANGE
        
        #TODO sorting the list is not the most effient way to this -- change?
        #print vertex, vertex_value
        for i in range(len(maximum_list)):
            v = maximum_list[i][0]
            if  self._graph.vertex_properties["label"][v]  ==  self._graph.vertex_properties["label"][vertex] :
                if vertex_value < maximum_list[i][1] and t=="freq":
                    print "ERROR", self._graph.vertex_properties["label"][v], v, maximum_list[i][1], vertex_value
                
                maximum_list[i][1]= vertex_value
                maximum_list.sort(key=itemgetter(1)) 

                return 
        
        if len(maximum_list) < list_desired_size:
            maximum_list.append([vertex, vertex_value])
            maximum_list.sort(key=itemgetter(1))
        else:
            if vertex_value > maximum_list[0][1]:
                maximum_list[0][0]= vertex
                maximum_list[0][1]= vertex_value
                maximum_list.sort(key=itemgetter(1)) 

       # print maximum_list

    '''    
    if self._hub_type == "hub-degree-freq-context":
        for w in context:
            vertices.append([self._words_vertex_map[w], 1.])

        return self._highest_degree_nodes + self._most_frequent_nodes + vertices

    if self._hub_type == "hub-freq-context":
        for w in context:
            vertices.append([self._words_vertex_map[w], 1.])

        return self._most_frequent_nodes + vertices
    '''

    def select_hubs(self, context):
        vertices = []
        vert_num = len(self._words_vertex_map.keys())
        hubs_num = int(round(self._hubs_num * vert_num))


        if self._hub_type in ["hub-freq", "hub-freq-random"]:
            vertices = self._most_frequent_nodes[-1 * hubs_num:][:]

        if self._hub_type in ["hub-degree", "hub-degree-random"]:
            vertices = self._highest_degree_nodes[-1 * hubs_num:][:]
        
        if self._hub_type in ["hub-context", "hub-context-random", \
                "hub-categories-context", "hub-categories-prob-context"]:
            #hubs_num = self._hubs_num 
           
            selected_context = context
            if hubs_num  < len(context):
                selected_context = context[-1 * hubs_num:]
                
            for w in selected_context:
                vertices.append([self._words_vertex_map[w], 1.])

        if self._hub_type in ["hub-random", "hub-context-random", "hub-freq-random", "hub-degree-random",\
                "hub-categories-random", "hub-categories-prob-random"]:
            #hubs_num = self._hubs_num 

            indices = range(0, vert_num)
            if vert_num > hubs_num:
                indices = self.random_selection(hubs_num, 0, vert_num - 1)
            
            for index in indices:
                vertices.append([self._graph.vertex(index), 1.]) 

        return vertices
 


    def random_selection(self, num, min_range, max_range):
        selected = []
        used = set([])
        
        while len(selected) < num:
            rand_index = random.randint(min_range, max_range)
            if rand_index in used: continue
            used.add(rand_index)
            selected.append(rand_index)
        
        return selected

    def add_edge(self, word, other_word, word_m, other_word_m, beta, simtype):
        ''' Add an edge between the two given words, if their similarity is
        higher than a threshold'''

        if word == other_word:
            return False

        #sim = self.calculate_similarity(word_features, other_word_features)
        sim = evaluate.calculate_similarity(beta, word_m, other_word_m, simtype)


        # if the words are similar enough, connect them.
        # TODO this can be done probabilistically -- that is connect based on similarity
        if sim >= self._sim_threshold:

            vert = self._words_vertex_map[word]
            other_vert =  self._words_vertex_map[other_word]

            new_edge = self._graph.add_edge(vert, other_vert)
            self._graph.edge_properties["distance"][new_edge] = max(0, 1 - sim)
            self._graph.edge_properties["similarity"][new_edge] = sim

            #update the list of nodes with most degree
            
            self.update_most_list(vert, vert.out_degree(), self._highest_degree_nodes, self._hubs_num)
            self.update_most_list(other_vert, other_vert.out_degree(), self._highest_degree_nodes ,self._hubs_num)
            return True

        return False

    def evaluate_categories(self, filename):
        ''' This function use wordnet labels to calcualte precision & recall for created categories'''

        #Count for each label in all the categories
        labels_count = {}

        for category_id in self._categories.keys():
            category = self._categories[category_id]
            category_labels = {}
            category_words_count = 0

            # Count the frequency of each label type of words in this category and
            # the number of words in this category
            for word in category._words.keys():
                label = self._wnlabels.wordnet_label(word)

                #TODO We are considering words that do not have a wn-label as a single label
                #if label == CONST.NONE:
                #    continue

                if label not in category_labels:
                    category_labels[label] = 0

                # Add the frequecy of the word in the category
                category_labels[label] += category._words[word]
                category_words_count += category._words[word]

            #if len(labels) < 1:
            #    continue

            # This category's label is the most frequent of all the words' labels
            most_frequent_label = ""
            freq = 0

            print "category", category._id
            for label in category_labels:
                print "wn-label", label, category_labels[label], category_words_count

                if category_labels[label] > freq:
                    freq = category_labels[label]
                    most_frequent_label = label

                if not labels_count.has_key(label):
                    labels_count[label] = 0
                labels_count[label] += category_labels[label]

            category._label = most_frequent_label
            category._precision = float(freq) / category_words_count
            category._freq =  float(freq)
            print"----"

        statfile = open(filename + "categories.txt", 'a')
        #print
        all_precisions = []
        all_recalls = []
        for category_id in self._categories:
            category = self._categories[category_id]
            category._recall = category._freq / labels_count[category._label]
            print category._id, category._label,"freq", category._freq, np.sum(category._words.values()) ,'---precision', category._precision, "recall", category._recall
            all_precisions.append(category._precision)
            all_recalls.append(category._recall)
            statfile.write("id %s label %s freq %d precision %.2f recall %.2f \n" % \
                    (category._id, category._label, np.sum(category._words.values()), category._precision, category._recall))

        statfile.write("avg_precision %.2f avg_recall %.2f \n" % (np.mean(all_precisions), np.mean(all_recalls)))
        statfile.close()

    def pick_category(self, post_prob_k):

        # Find the category with max post prob
        if self._sampling_method =='map': #local MAP
            max_category_id = 1
            for category_id in post_prob_k:
                if post_prob_k[category_id] > post_prob_k[max_category_id]:
                    max_category_id = category_id
            return max_category_id

        elif self._sampling_method == 'spf': #single-particle particle filter
#            print self._sampling_method
            rand = random.random()
            min_range = 0
            denom = logsumexp(post_prob_k.values())

            for category_id in post_prob_k:
                ppk = math.exp(post_prob_k[category_id] - denom)
                if min_range <= rand < ppk + min_range:
                    return category_id
                min_range += ppk

    def select_words_from_categ(self, post_prob_k):
            denom = logsumexp(post_prob_k.values())
            selected_words = set([])

            vert_num = len(self._words_vertex_map.keys())
            #hubs_num = round(self._hubs_num * vert_num)
            
            # TODO changed from round to ceil June 5
            hubs_num = np.ceil(self._hubs_num * vert_num)


            for category_id in self._categories:
                ppk = math.exp(post_prob_k[category_id] - denom)
                select_words_num = round(hubs_num * ppk)

                categ_words = self._categories[category_id]._words.keys()[:]
                indices = range(0, len(categ_words))
                if len(categ_words) > select_words_num:
                    indices = self.random_selection(select_words_num, 0, len(categ_words) - 1)
                
                for index in indices:
                    selected_words.add(categ_words[index])
            
            #print len(selected_words)
            return selected_words

    def _add_word_to_category(self, word, word_m):#, lexicon, marked, beta, simtype):
        ''' n is token size not type size'''

        post_prob_k = {} #P(K|W), where K is the category

        for category_id in self._categories:
            category = self._categories[category_id]
            post_prob_k[category_id] = category.posterior(word, word_m._meaning_probs, self._words_token_freq)
            #post_prob_k[category_id] = category.posterior(word, word_top_features, self._words_token_freq)


        new_category = Category(len(self._categories) + 1, self._coupling, self._lambda0, self._a0, \
        self._miu0, self._sigma0)
        post_prob_k[new_category._id] = new_category.posterior(word, word_m._meaning_probs, self._words_token_freq)
#        post_prob_k[new_category._id] = new_category.posterior(word, word_top_features, self._words_token_freq)


        selected_category_id = self.pick_category(post_prob_k)

        # Add the new category
        if selected_category_id == len(self._categories) + 1:
            self._categories[len(self._categories) + 1] = new_category

        # Add the word to the chosen category
        self._categories[selected_category_id].add_word(word, word_m._meaning_probs)
        self._words_token_freq += 1
        #print word, selected_category_id

        selected_words = []
        # Pick x number of words from each category proportional to p(k|f)
        if self._hub_type.startswith("hub-categories-prob"):
            selected_words = self.select_words_from_categ(post_prob_k)
            
        else:
            categ_words = self._categories[selected_category_id]._words.keys()[:]
            categ_words_num = len(categ_words)

            # when hub-type == hub-categories
            selected_words =  categ_words[:]
            
            if self._hub_type in ["hub-categories-context", "hub-categories-random", "hub-categories-partial"]: 
                vert_num = len(self._words_vertex_map.keys())
                hubs_num = round(self._hubs_num * vert_num)
                #hubs_num = self._hubs_num 

                if categ_words_num > hubs_num:
                    indices = self.random_selection(hubs_num, 0, categ_words_num -1)
                    selected_words = []
                    for index in indices:
                        selected_words.append(categ_words[index])


        categ_hubs = []
        for oth_word in selected_words:
            oth_node = self._words_vertex_map[oth_word]
            categ_hubs.append([oth_node, 1])
            
        return categ_hubs



    def add_word(self, context, word, acq_score, lexicon, last_time, ctime, beta, simtype):
        ''' add a new word to the graph or update its connections '''

        marked = set([]) # Mark vertices that already visited
        word_m = lexicon.meaning(word)
#        word_top_features =self.select_features(word_m._meaning_probs)

        # add the word to the graph
        if not word in self._words_vertex_map:
            self._words_vertex_map[word] = self._graph.add_vertex()
            self._graph.vertex_properties["label"][self._words_vertex_map[word]] = word
            self._graph.vertex_properties["freq"][self._words_vertex_map[word]] = 0
            
            if len(self._highest_degree_nodes) < self._hubs_num:
                self._highest_degree_nodes.append([self._words_vertex_map[word], 0])


        # if the words was in the graph, update its connections 
        #TODO Investigate if we need to do this.
        else:
            vertex = self._words_vertex_map[word]
            edges = list(vertex.out_edges())
            for edge in edges:
                target_w =  self._graph.vertex_properties["label"][edge.target()]

                if target_w == word:
                    target_w = self._graph.vertex_properties["label"][edge.source()]
                    print "ERROR"

                target_w_m = lexicon.meaning(target_w)
                #target_w_top_features = self.select_features(target_w_m._meaning_probs)

                marked.add(self._words_vertex_map[target_w])

                self.add_edge(word, target_w, word_m, target_w_m, beta, simtype)
                self._graph.remove_edge(edge)

            self.update_computations.append(len(marked))

        vert = self._words_vertex_map[word]
        self._graph.vertex_properties["acqscore"][vert] = acq_score

        self._graph.vertex_properties["freq"][vert] = \
        self._graph.vertex_properties["freq"][vert] + 1
        
        self.update_most_list(vert, self._graph.vertex_properties["freq"][vert], self._most_frequent_nodes, self._hubs_num, "freq")

        categ_hubs = []
        hubs = []
        '''
        vert_num = len(self._words_vertex_map.keys())
        #number of comparisons
        hubs_num = int(round(self._hubs_num * vert_num))
        #deduct the number of used comparisons, ie, # of current edges that are updated.
        hubs_num -= len(marked) 
        if not (hubs_num in ["hub-categories", "hub-categories-prob", \
                "hub-categories-partial", "hub-context", "hub-random",\
                "hub-freq", "hub-degree"]):
            hubs_num  = hubs_num // 2
        '''

        if self._hub_type.startswith("hub-categories"):
            categ_hubs = self._add_word_to_category(word, word_m)#, lexicon, marked, beta, simtype) #TODO
       
        if not  (self._hub_type in ["hub-categories", "hub-categories-partial", "hub-categories-prob"]):
            hubs = self.select_hubs(context)
        
#        print word 

        if self._hub_type in ["hub-random", "hub-context", "hub-context-random",\
                "hub-degree", "hub-degree-random", "hub-freq", "hub-freq-random"] \
                or self._hub_type.startswith("hub-categories"):
            # "hub-categories", "hub-categories-context", "hub-categories-random", "hub-categories-partial"]:
            update_num = 0
            # calculate similarity of the word and hub
            for hub, score in (hubs + categ_hubs):
                if hub in marked: continue
                marked.add(hub)

                hword = self._graph.vertex_properties["label"][hub]
                hword_m = lexicon.meaning(hword)
                edge_added = self.add_edge(word, hword, word_m, hword_m, beta, simtype)
                update_num +=1
            
            self.new_computations.append(update_num)
        '''               
        #TODO WE ARE NOT USING THIS
        else:
            # calculate similarity of the word and hub
            for hub, score in hubs:
                if hub in marked: continue
                marked.add(hub)

                hword = self._graph.vertex_properties["label"][hub]
                hword_m = lexicon.meaning(hword)

                edge_added = self.add_edge(word, hword, word_m, hword_m, beta, simtype)
                if not edge_added: continue

                for neighbor in hub.all_neighbours():
                    if neighbor in marked: continue
                    marked.add(neighbor)
                    neighbor_word = self._graph.vertex_properties["label"][neighbor]
                    nword_m = lexicon.meaning(word)
                    
                    self.add_edge(word, neighbor_word, word_m, nword_m, beta, simtype)
        '''
        # calculate the number of computations
        self.max_computations.append(self._graph.num_vertices())
        
        #print "number of computations" ,  len(marked)
        
        self.computations.append(len(marked))


    def plot(self, graph, filename):
        """ Plot a graph """
#        ebet = betweenness(graph)[1]
        name = graph.vertex_properties["label"]
#        acq_scores = graph.vertex_properties["acqscore"]
        distances = graph.edge_properties["distance"]
        deg = graph.degree_property_map("total")
        pos = sfdp_layout(graph)
        #arf_layout(graph)
        graph_draw(graph, pos= pos, vertex_text=name, vertex_font_size=12, vertex_fill_color= deg, vorder=deg,\
        edge_pen_width=distances, output=filename + "graph.png", output_size=(3000,3000), nodesfirst=False)

    def print_hubs(self, filename, last_time, ctime):
        """ Print hubs of the graph """

        hubs = self.select_hubs([])
        stat_file = open(filename, 'a')
        stat_file.write("\nThe final hubs of the semantic network:\n")
        st = ""
        if hubs != None:
            for hub,score in hubs:
                st +=  self._graph.vertex_properties["label"][hub] + ","
            stat_file.write(st + "\n")
        stat_file.close()

    def print_computations(self, filename):

        stat_file = open(filename, 'a')
        (avg, std) = np.mean(self.max_computations), np.std(self.max_computations)
        stat_file.write("\navg maximum computations over words:" + "%.2f +/- %.2f" % (avg, std) + "\n")

        (avg, std) = np.mean(self.computations), np.std(self.computations)
        stat_file.write("avg actual computations over words:" + "%.2f +/- %.2f" % (avg, std) + "\n")

        (avg, std) = np.mean(self.update_computations), np.std(self.update_computations)
        stat_file.write("avg update computations over words:" + "%.2f +/- %.2f" % (avg, std) + "\n")

        (avg, std) = np.mean(self.new_computations), np.std(self.new_computations)
        stat_file.write("avg new computations over words:" + "%.2f +/- %.2f" % (avg, std) + "\n")

        stat_file.close()


    def calc_distances(self, graph):
        distance = {}
        for v in graph.vertices():
            w = graph.vertex_properties["label"][v]
            if not w in distance.keys():
                distance[w] = {}

            distmap = shortest_distance(graph, v, weights=graph.edge_properties["distance"]) #TODO
            for othv in graph.vertices():
                othw = graph.vertex_properties["label"][othv]
                if othw == w:
                    continue
                distance[w][othw] = distmap[othv]
        return distance

    def calc_graph_ranks(self, graph_distances ):
        # Rank the targets for each cue in the graph
        graph_ranks = {}
        for cue in graph_distances:
            graph_ranks[cue] = {}

            sorted_targets = []
            for target in graph_distances[cue]:
                sorted_targets.append([target, graph_distances[cue][target]])
            sorted_targets = sorted(sorted_targets, key=itemgetter(1))

            max_rank = 100000
            for ind in range(len(sorted_targets)):
                if sorted_targets[ind][1] == sys.float_info.max or sorted_targets[ind][1] == 2147483647:
                    rank = max_rank
                else:
                    rank = ind + 1

                graph_ranks[cue][sorted_targets[ind][0]] = rank

        return graph_ranks

    def calc_correlations(self, gold_sim, distances, consider_notconnected):
        print "calc_correlations"
        graph_pairs, gold_pairs = [], []
        not_connected = 0
        all_pairs = 0.0


        for cue in gold_sim:
            for target, score in gold_sim[cue]:
                all_pairs += 1

                if distances[cue][target] ==  sys.float_info.max or \
                    distances[cue][target]== 2147483647:
                    not_connected += 1

                    #print cue, target, score, distances[cue][target]

                    if not consider_notconnected:
                        continue

                gold_pairs.append(score)  #TODO sim vs distance
                graph_pairs.append(distances[cue][target])
        print "--------------------"
        if len(graph_pairs) == 0:
            print "nothing matched"
            return (0.0,0.0), (0.0, 0.0), 0.0

        #pearson_r, pearson_pvalue
        pearson = scipy.stats.pearsonr(gold_pairs, graph_pairs)
        #spearman_t, spearman_pvalue
        spearman = scipy.stats.spearmanr(gold_pairs, graph_pairs)

        print "not connected", not_connected, all_pairs
        return pearson, spearman, not_connected/all_pairs


    def calc_median_rank(self, gold_sims, graph_ranks):
        """ calculate the median rank of the first five associates """

        ranks = {}
        for r in range(5):
            ranks[r] = []

        for cue in gold_sims:
            for index in range(min(len(gold_sims[cue]), 5)):
                target = gold_sims[cue][index][0]
                target_rank = graph_ranks[cue][target]
                ranks[index].append(target_rank)

        return ranks



    def evaluate_semantics(self, graph_distances, graph_ranks,  gold_sim, filename, gold_name):


        ranks = self.calc_median_rank(gold_sim, graph_ranks)
        for index in ranks:
            print ranks[index]
        
        stat_file = open(filename, 'a')
        stat_file.write("evaluation using " + gold_name + "\n")
        stat_file.write("median rank of first five associates for " + str(len(gold_sim.keys())) + " cues\n")
        for i in range(len(ranks.keys())):
            #print ranks[i], numpy.median(ranks[i])
            stat_file.write(str(i+1) + " associate. number of cue-target pairs: %d" % len(ranks[i]) +\
            " median rank: %.2f" %  np.median(ranks[i])+"\n")

        # Calc correlations
        pearson, spearman, not_connected = self.calc_correlations(gold_sim, graph_distances, False)
        stat_file.write("\n Not considering pairs that are not connected in the graph\n")
        stat_file.write("pearson  correlation %.2f p-value %.2f" % pearson + "\n")
        stat_file.write("spearman correlation %.2f p-value %.2f" % spearman + "\n")
        stat_file.write("cue-target pairs that are not_connected in the graph %.2f" % not_connected + "\n\n")

        pearson, spearman, not_connected = self.calc_correlations(gold_sim, graph_distances, True)
        stat_file.write("Considering pairs that are not connected in the graph\n")
        stat_file.write("pearson  correlation %.2f p-value %.2f" % pearson + "\n")
        stat_file.write("spearman correlation %.2f p-value %.2f" % spearman + "\n")
        stat_file.write("cue-target pairs that are not_connected in the graph %.2f" % not_connected + "\n\n")

    def evaluate(self, last_time, current_time, gold_lexicon, learned_lexicon, beta, simtype, data_path, filename):
        words = self._words_vertex_map.keys()

        #gold_graph = self.create_final_graph(words, gold_lexicon, beta, simtype)
        #learned_graph = self.create_final_graph(words, learned_lexicon, beta, simtype)
        
        grown_graph = self._graph

        if self._hub_type != "hub-categories":
            self.print_hubs(filename + "_grown.txt", last_time, current_time) #CHECK

        self.print_computations(filename + "_grown.txt") #CHECK

        
        #nelson_norms = process_norms(data_path +"/norms/", words)
        #wn_jcn_sims = wordnet_similarity(words, "jcn", self._wnlabels)
        wn_wup_sims = wordnet_similarity(words, "wup", self._wnlabels)
        #wn_path_sims = wordnet_similarity(words, "path",self._wnlabels)

        #rg_norms = process_rg_norms(data_path+"/Rubenstein-Goodenough.txt", words)
        #wordsims353_norms = process_rg_norms(data_path + "/wordsim353/combined.tab",words)
        
        for g, tag in [[grown_graph, "_grown"]]:#, [gold_graph, "_gold"], [learned_graph, "_learned"]]:
        #    self.plot(g, filename + tag + "_")
            self.calc_small_worldness(g, filename + tag)

            
            distances = self.calc_distances(g)
            graph_ranks = self.calc_graph_ranks(distances)

            self.evaluate_semantics(distances, graph_ranks, wn_wup_sims, filename + tag + ".txt", "wordnet using WUP sim measure")
        
        #    self.evaluate_semantics(distances, graph_ranks, nelson_norms, filename + tag + ".txt", "Nelson norms")
        #    self.evaluate_semantics(distances, graph_ranks, wn_jcn_sims, filename + tag + ".txt", "wordnet using JCN sim measure")
        #    self.evaluate_semantics(distances, graph_ranks, wn_path_sims, filename + tag + ".txt", "wordnet using Path sim measure")
        #    self.evaluate_semantics(distances, graph_ranks, rg_norms, filename + tag + ".txt", "Rubenstein-Goodenough norms")

        #    self.evaluate_semantics(distances, graph_ranks, wordsims353_norms, filename + tag + ".txt", "Wordsim353 norms")
            

    def calc_small_worldness(self, graph, filename):
        avg_c, median_sp = self.calc_graph_stats(graph, filename)

        rand_graph = Graph(graph)
        rejection_count = random_rewire(rand_graph, "erdos")
        print "rejection count", rejection_count
        rand_avg_c, rand_median_sp = self.calc_graph_stats(rand_graph, filename)

        stat_file = open(filename + ".txt", 'a')
        stat_file.write("small-worldness %.3f" % ((avg_c / rand_avg_c)/(float(median_sp)/rand_median_sp)) + "\n\n")
        stat_file.close()

    def calc_graph_stats(self, graph, filename):
        """ calc graph stats """

        """Average Local Clustering Coefficient"""
        local_clust_co = local_clustering(graph)
        avg_local_clust = vertex_average(graph, local_clust_co)

        """Average Degree (sparsity)"""
        avg_total_degree = vertex_average(graph, "total")

        nodes_num = graph.num_vertices()
        edges_num = graph.num_edges()

        """ Largest Component of the Graph"""
        lc_labels = label_largest_component(graph)

        lc_graph = Graph(graph)
        lc_graph.set_vertex_filter(lc_labels)
        lc_graph.purge_vertices()

        """Average Shortest Path in LCC"""
        lc_distances = lc_graph.edge_properties["distance"]
        dist = shortest_distance(lc_graph)#, weights=lc_distances) #TODO
        dist_list = []
        for v in lc_graph.vertices():
            dist_list += list(dist[v].a)


        """ Median Shortest Path """
        distances = graph.edge_properties["distance"] #TODO
        gdist = shortest_distance(graph)#, weights=distances)
        graph_dists = []
        counter = 0
        for v in graph.vertices():
            for othv in gdist[v].a:
                if othv != 0.0: # not to include the distance to the node
                    graph_dists.append(othv)
                else:
                    counter +=1
      #  print "num v", graph.num_vertices(), counter
        median_sp = np.median(graph_dists)
      #  print "median", median_sp#, graph_dists


        stat_file = open(filename + ".txt", 'a')
        stat_file.write("number of nodes:"+ str(nodes_num) + "\nnumber of edges:" + str(edges_num) + "\n")
        stat_file.write("avg total degree:" + "%.2f +/- %.2f" % avg_total_degree  + "\n")
        stat_file.write("sparsity:" + "%.2f" % (avg_total_degree[0] / float(nodes_num))  + "\n")

        stat_file.write("number of nodes in LLC:"+  str(lc_graph.num_vertices()) + "\nnumber of edges in LLC:" + str(lc_graph.num_edges()) + "\n")
        stat_file.write("connectedness:" + "%.2f" % (lc_graph.num_vertices()/float(nodes_num)) + "\n")
        stat_file.write("avg distance in LCC:" + "%.2f +/- %.2f" % (np.mean(dist_list), np.std(dist_list)) + "\n\n")

        stat_file.write("avg local clustering coefficient:" + "%.2f +/- %.2f" % avg_local_clust + "\n")
        stat_file.write("median distnace in graph:" + "%.2f" % median_sp + "\n\n")

       # Plotting the degree distribution
        ''' 
        plt.clf()
        hist = vertex_hist(graph, "total")
        prob_hist = []
        sum_hist = sum(hist[0])
        for h in hist[0]:
            prob_hist.append(h/float(sum_hist))

        plt.plot(hist[1][1:], prob_hist, 'ro')#, normed=False, facecolor='green', alpha=0.5)
        plt.xlabel('K')
        plt.gca().set_yscale("log")
        plt.gca().set_xscale("log")
        plt.ylabel('P(K)')
        #plt.title(r'Histogram of degrees of the graph')
        #data_1 = graph.degree_property_map("total").a#, graph.degree_property_map, len( graph.degree_property_map("total").a)
        #fit = powerlaw.Fit(data_1) TODO
        #stat_file.write("alpha of powerlaw " + str(fit.power_law.alpha) + "\n\n")
        #print fit.power_law.xmin
        #fit.plot_pdf(ax=plt.gca(),  linewidth=3, color='b')
        #fit.power_law.plot_pdf(ax=plt.gca(), color='g', linestyle='--')
        plt.savefig(filename + "_loglog_degree_histogram.png")
        
        plt.clf()
        plt.plot(hist[1][1:], prob_hist, 'ro')#, normed=False, facecolor='green', alpha=0.5)
        plt.xlabel('K')
        plt.ylabel('P(K)')
        #
        plt.savefig(filename + "_degree_histogram.png")
        '''
        
        stat_file.close()

        return avg_local_clust[0], median_sp

    def create_final_graph(self, words, lexicon, beta, simtype):
        """ create a graph, given a set of words and their meanings """

        graph = Graph(directed=False)
        graph.vertex_properties["label"] = graph.new_vertex_property("string")
        graph.edge_properties["distance"]  = graph.new_edge_property("double")
        graph.vertex_properties["acqscore"] = graph.new_vertex_property("double")


        word_vertex_map = {}

        for word in words:
            word_vertex_map[word] = graph.add_vertex()
            graph.vertex_properties["label"][word_vertex_map[word]] = word


        for word in words:
            for otherword in words:
                if word == otherword:
                    continue

                vert = word_vertex_map[word]
                othervert =  word_vertex_map[otherword]

                if graph.edge(vert, othervert) != None or graph.edge(othervert, vert)!= None:
                    continue

                word_m = lexicon.meaning(word)
#                word_m_top_features = self.select_features(word_m._meaning_probs)

                otherword_m = lexicon.meaning(otherword)
#                otherword_m_top_features = self.select_features(otherword_m._meaning_probs)


                #sim = self.calculate_similarity(word_m_top_features, otherword_m_top_features)
                sim = evaluate.calculate_similarity(beta, word_m, otherword_m, simtype)

                if sim >= self._sim_threshold:
                    new_edge = graph.add_edge(vert, othervert)
                    graph.edge_properties["distance"][new_edge] = max(0, 1 - sim ) #distance #TODO

        return graph



