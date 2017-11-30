import sys
import os
import re
import getopt
import learn
import evaluate
import wmmapping
import learnconfig
import numpy
import pickle


"""
main.py

Main file for beginning the learning procedure.

"""


def usage():
    print "usage:"
    print "  main.py -c (--corpus) -l (--lexicon) -i (--inputdir) -o (--output) -C (--config) -s (--stop)"
    print ""
    print "  --corpus:   input corpus"
    print "  --lexicon:  original lexicon"
    print "  --inputdir:  input directory (used if no corpus is given)"
    print "  --output:   output directory"
    print "  --config:   configuration file"
    print "  --stop:     stopwords file"
    print "  --help:     prints this usage"
    print ""


def main():
    try:
        options_list = ["help", "corpus=", "lexicon=", "inputdir=", "output=", "config=", "stop="]
        opts, args = getopt.getopt(sys.argv[1:], "hc:l:i:o:C:s:", options_list) 
    except getopt.error, msg:
        print msg
        usage()
        sys.exit(2)

    if len(opts) < 4:
        usage()
        sys.exit(0)

    corpus_path = ""
    stop = ""
    for o, a in opts: 
        if o in ("-h", "--help"):
            usage()
            sys.exit(0)
        if o in ("-c", "--corpus"):
            corpus_path = a
        if o in ("-l", "--lexicon"):
            lexname = a
        if o in ("-i", "--inputdir"):
            indir = a
        if o in ("-o", "--output"):
            outdir = a
        if o in ("-C", "--config"):
            config_path = a
        if o in ("-s", "--stop"):
            stop = a
            
    # Configure the learner
    learner_config = learnconfig.LearnerConfig(config_path)
    print "ww"
    stopwords = []
    if len(stop) > 2: # At least a.b as name 
        stopwords_file = open(stopwords_path, 'r')
        for line in stopwords_file:
            stopwords.append(line.strip()+ ":N") 
                
    learner = learn.Learner(lexname, learner_config, stopwords)
    
    # Make directories if neccessary
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # If no corpus_path is given, then all of the files in the input directory
    # are corpus' to use.    
    if corpus_path == "":
        for dirpath, dirnames, filenames in os.walk(indir):
            for fname in filenames:
                if re.search("#[0-9]*\.txt$", fname): 
                #--AF:test--if re.search("#1\.txt$", fname): 
                    # Write the output data to a directory structure that matches
                    # the input directory, except the root is outdir.
                    path = dirpath
                    outdirpath = path.replace(indir, outdir)
                    #--AF:test--print "dirpath:", dirpath
                    #--AF:test--print "outdirpath:", outdirpath
                    if not os.path.exists(outdirpath):
                        os.makedirs(outdirpath)

                    corpus = dirpath + "/" + fname
                    print "Processing Corpus: ", corpus
                    learner.process_corpus(corpus, outdirpath)
                    print "Done"
                    learner.reset()
    else:
        learner.process_corpus(corpus_path, outdir)


    if learner._stats_flag:
        print "output files ..... "
        # write_learned_lex(learner, outdir)
        # write_alignments(learner, outdir)
        write_acq_score_timestamp(learner, outdir)
        print("alignment", learner.alignment_method, corpus_path)        

#===============================================================================
#        Sample methods for viewing the data
#===============================================================================

def write_alignments(learner, outdir):
    """ 
    Write the contents of the learner to a file in directory outdir.
    The file is named as: aligns_lm_{lambda}_a{alpha}_ep{epsilon} where lambda, 
    alpha, and epsilon come from the Learner learner.
    
    The file is written as :
        {word}--{feature} [ ({time}, {alignment}), ({time},{alignment}), ...]
        
    for each word-feature pair that occurred at least once together.
    
    """
    lm = learner._lambda
    a = learner._alpha
    ep = learner._epsilon
    filename = "%s%s%s%s%s%s%s" % (outdir, "/aligns_lm_", lm, "_a", a, "_ep", ep)
    output = open(filename, 'w')
    
    for word in learner._wordsp.all_words(0):
        for feature in learner._features:
            # Get the list of all {time:alignment_score} entries for this
            # word-feature pair
            
            alignments = learner._aligns.alignments(word,feature)
            if len(alignments.items()) == 0:
                continue
            
            line = "%s--%s [ " % (word, feature)
            for t,val in alignments.items():
                line += "(%d, %6.5f), " % (t, val)
            line += " ]\n\n"
            output.write(line)
    output.close()
    

def write_learned_lex(learner, outdir):
    """
    Write the contents of the learner to a file in directory outdir.
    The file is named as: lex_lm_{lambda}_a{alpha}_ep{epsilon} where lambda, 
    alpha, and epsilon come from the Learner learner.
    
    The file is written as :
        {word}:{frequency} [{feature}:({ture_probability}, {learned_probability}),
            ...] 
        <<{word's_aqcuisition_score}>>>
    for each word.
    
    """
    lm = learner._lambda
    a = learner._alpha
    ep = learner._epsilon
    filename = "%s%s%s%s%s%s%s" % (outdir, "/lex_lm_", lm, "_a", a, "_ep", ep)
    output = open(filename, 'w')
    min_prob = 0.0001
    # Print statistics on each word
    for word in learner._wordsp.all_words(0):
        # Get the list of all (probability, feature) pairs from the true lexicon
        prob_feature_pairs = learner._gold_lexicon.meaning(word).sorted_features()
        learned_meaning = learner._learned_lexicon.meaning(word)
        line = "%s:%d [" % (word, learner._wordsp.frequency(word))
        for true_prob,feature in prob_feature_pairs:
            if true_prob > min_prob:
                learned_prob = learned_meaning.prob(feature)
                # True prob compared to the learned prob for this feature
                line += "%s:(%6.5f, %6.5f), " % (feature, true_prob, learned_prob)
        line += " ]\n\n"
        output.write(line)
        # Add in the computed acquisition score for this word
        comp = "   << %f >>\n\n" % (learner.acquisition_score(word))
        output.write(comp)
    output.close()
    
def write_acq_score_timestamp(learner, outdir):
    '''
    added by Shanshan Huang
    Record the acquisition score in each time step in pickle format,
    with the option to save the whole learner object into a single pickle
    (warning: very large!!!!)
    These pickle files are later used to plot learning graphs used for 
    analysis in the paper
    '''
    
#    with open(outdir+'/learned_meaning.pkl', "wb") as handle:
#        pickle.dump(learner._learned_lexicon, handle) 

#    with open(outdir+'/acq_scores.pkl', "wb") as handle:
#        pickle.dump(learner._acquisition_scores, handle) 
    
    with open(outdir+'/acq_score_timestamp.pkl', "wb") as handle:
        pickle.dump(learner._acq_score_list, handle)  

#    with open(outdir+'/learner.pkl', "wb") as handle:
#        pickle.dump(learner, handle)



#===============================================================================
#        Main method execution point
#===============================================================================

if __name__ == "__main__":
    main()
    
