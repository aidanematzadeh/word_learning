import constants as CONST
import math
import numpy 
from numpy.linalg import norm

"""
evaluate.py

A collection of methods of evaluating a similarity score. See config.ini section
Similarity and constants.py Section 4.

"""

def sim_average_precision(true, learned):
    """
    Calculate and return the similarity score using Average Precision. true is a 
    dictionary mapping features to their true probability scores and learned
    is a ranked list where each entry is a list [feature, probability].
    
    """
    i = 0
    num_features = 0

    precisions = {}
    while i < len(learned) and num_features < len(true.keys()):
        j = i
        while j < len(learned) and learned[j][0] == learned[i][0]:
            feature = learned[j][1]
            if true.has_key(feature):
                num_features += 1
            j += 1
        if not num_features in precisions:
            precisions[num_features] = float(num_features) / float(j)
        i = j

    sum_precision = 0.0
    for key in precisions.keys():
        sum_precision += precisions[key]
    avg_precision = sum_precision / len(precisions.keys())

    return avg_precision


def sim_euclidean_length(beta, meaning):
    """
    Calculate and return the similarity score using Euclidean length of the 
    meaning Meaning object, using the list of probabilities for seen features
    as the vector. beta is used for smoothing the result.
    
    """
    seen_count = 0
    length = 0.0
    for feature in meaning.seen_features():
        v = meaning.prob(feature)
        length += v * v
        seen_count += 1

    v = meaning.unseen_prob()
    length += (beta - seen_count) * v * v
    return math.sqrt(length)


def sim_jensen_shannon_divergence(beta, learned, true):
    """
    Calculate and return the similarity score by the using Jensen-Shannon 
    Divergence, measuring the difference between two probability distributions.
    The distributions are the meaning probabilities of the learned Meaning 
    learned and the true Meaning true. beta is used as a smoothing factor.
    
    """
    features = learned.seen_features() | true.seen_features()

    learned_sum = 0.0
    true_sum = 0.0
    for feature in features: 
        p = learned.prob(feature)
        q = true.prob(feature)
        m = .5 * (p + q)
        if m > 0.0:
            if p > 0.0:
                learned_sum += p * math.log((p / m),2) 
            if q > 0.0:
                true_sum += q * math.log((q / m),2) 

    # Account for the unseen probability to consider the entire distributions
    seen_count = len(features)
    unseen_p = learned.unseen_prob()
    unseen_q = true.unseen_prob()
    m = .5 * (unseen_p + unseen_q)
    if m > 0.0:
        if unseen_p > 0.0:
            learned_sum += (beta - seen_count) * unseen_p * math.log((unseen_p/m),2)
        if unseen_q > 0.0:
            true_sum += (beta - seen_count) * unseen_q * math.log((unseen_q/m),2)

    return 0.5 * (learned_sum + true_sum)


def sim_cosine(beta, meaning1, meaning2):
    """
    Calculate and return the similarity score using the Cosine method, comparing
    the probabilities within Meaning of first word and Meaning of second word as the vectors.
    beta is used as a smoothing factor.
    
    features contain all the features of the gold lexicon.
    """
    features = meaning1.seen_features() | meaning2.seen_features()
    
    meaning1_vec = numpy.zeros(len(features))
    meaning2_vec = numpy.zeros(len(features))

    i = 0
    for feature in features:
        meaning1_vec[i] = meaning1.prob(feature)
        meaning2_vec[i] = meaning2.prob(feature)
        i += 1
    
    cos = numpy.dot(meaning1_vec, meaning2_vec)
    
    seen_count = len(features)
    cos += (beta - seen_count) * meaning1.unseen_prob() * meaning2.unseen_prob()
    
    x = math.sqrt(numpy.dot(meaning1_vec, meaning1_vec) \
    + (pow(meaning1.unseen_prob(), 2) * (beta - seen_count)))
    
    y = math.sqrt(numpy.dot(meaning2_vec, meaning2_vec) \
    + (pow(meaning2.unseen_prob(), 2) * (beta - seen_count)))
   
    return  cos / (x * y)


    # Slower old code    
    #learned_norm = pow(norm(learned_vec),2)
    #learned_norm += (pow(learned.unseen_prob(), 2) * (beta - seen_count))
    #true_norm = pow(norm(true_vec),2) 
    #true_norm += (pow(true.unseen_prob(), 2) * (beta - seen_count))
    #return cos / (math.sqrt(learned_norm *  true_norm))


def calculate_similarity(beta, meaning1, meaning2, simtype):
    """
    Calculate and return the similarity score of Meaning 1 to Meaning 2
    using the similarity measure corresponding to simtype. beta is used as a 
    smoothing factor.
    """
    
    if simtype == CONST.COS:
        return sim_cosine(beta, meaning1, meaning2)
    elif simtype == CONST.JSD:
        return (1.0 - sim_jensen_shannon_divergence(beta, meaning1, meaning2))
 

