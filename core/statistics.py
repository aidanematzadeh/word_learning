import re
import numpy
import constants as CONST
from evaluate import *

"""
statistics.py

Data-structures for gathering statistics while the model learns.

"""

#===============================================================================
#    Word Statistics
#===============================================================================

class WordProps:
    """
    Properties of a word.
    
    Members:
    
    word -- the word that these properties apply to
    wfreq -- word frequency (initially 1)
    first_time -- time of first occurrence
    first_vsize -- vocabulary size at the time of the first occurrence
    lrnd_time -- time at word the word has been learned (acquisition passes
        the similarity threshold set in config file) (initially -1)
    lrnd_freq -- occurrences needed to learn the word (initially -1)
    lrnd_vsize -- vocabulary size at the time that this word is learned 
        (initially -1)
    last_time -- last time that this word has been observed (initially -1)
    
    """
    
    def __init__(self, word, time, vsize):
        """ 
        Create a word property object for word, first encountered at time time and
         a vocabulary of size vsize. 
         
        """
        self._word = word
        self._wfreq = 1              
        self._first_time = time      
        self._first_vsize = vsize    
        self._lrnd_time = -1         
        self._lrnd_freq = -1         
        self._lrnd_vsize = -1        
        self._last_time = -1         

    def write(self, handle):
        """ Write these word properties to the file object handle. """
        props = str(self._wfreq)+","
        props += str(self._first_time)+","
        props += str(self._first_vsize)+","
        props += str(self._lrnd_time)+","
        props += str(self._lrnd_freq)+","
        props += str(self._lrnd_vsize)+","
        props += str(self._last_time)+","
        props += "\n"
        
        handle.write(props)

    def inc_wfreq(self):
        """ Increment the frequency of this word occurring. """
        self._wfreq = self._wfreq + 1


class WordPropsTable:
    """
    Map words to WordProps objects.
    
    Members:
    
    words -- dictionary of word - WordProps pairs
    file_prefix -- the prefix of the file name that this object writes to
    
    """
    
    def __init__(self, file_prefix):
        """ 
        Create an empty WordPropsTable object which writes to a file prefixed by
         file_prefix. 
         
        """
        self._words = {}
        self._words_token_freq = 0

        self._file_prefix = file_prefix

    def reset(self):
        """ Reset this object to be empty. """ 
        self._words = {}

    def write(self, corpus_name, outdir, itr):
        """ 
        Write this object to a new file called {outdir}/{self._file_prefix}_itr,
        with one time and its properties per line, and close the file.
        If corps_name matches the standard of ending in "#{num}.txt" then the
        file will be {outdir}/#{num}_{self._file_prefix}_itr 
        
        """
        if re.search("#[0-9]*\.txt$", corpus_name):
            # Name standard for corpora, used to distinguish multiple in one 
            # directory
            num = corpus_name[corpus_name.index('#'):-4] # -4 to remove ".txt"
            handle = open(outdir + "/" + num + "_" + self._file_prefix + itr + ".csv", "w")
        else:
            handle = open(outdir + "/" + self._file_prefix + itr + ".csv", "w")
            
        # Write header
        handle.write("word, freq, first_time, first_vsize, lrnd_time, lrnd_freq, "
                     + "flrnd_vsize, last_time\n")
        
        words = self._words.keys()
        words.sort()
        for word in words:
            handle.write(word+",")
            self._words[word].write(handle)
        handle.write("\n")
        
        handle.close()

    def read(self, handle):
        """ 
        Read from the file handle to populate this object.
        Prerequisite: handle was written to using the write() method of this 
            class.
        
        """
        handle.readline() # Remove header
        while 1:
            line = handle.readline().strip("\n")
            if line=="":
                break
            w, l = line.split("=")
            props = l.split(",")
            
            self.add_word(w, int(props[1]), int(props[2]))
            wordprops = self._words[w]
            wordprops._wfreq = int(props[0])
            wordprops._lrnd_time = int(props[3])
            wordprops._lrnd_freq = int(props[4])
            wordprops._lrnd_vsiz = int(props[5])
            wordprops._last_time = int(props[6])

    #BM hasWord
    def has_word(self, word):
        """ Return whether word has been encountered yet or not. """
        return word in self._words

    #BM addWord
    def add_word(self, word, time, vsize):
        """ 
        Add a new entry in this table for word, encountered at time time with
        vocabulary size vsize.
        
        """
        self._words[word] = WordProps(word, time, vsize)
        self._words_token_freq += 1

    #BM inreaseWFreq
    def inc_frequency(self, word):
        """ Increment the frequency of word. """
        try:
            self._words[word]._wfreq += 1
            self._words_token_freq += 1

        except KeyError:
            raise KeyError("Word '"+word+"' not in WordPropsFile, frequency cannot be incremented") 
        

    #BM updateLastTime
    def update_last_time(self, word, time):
        """ Update the last time that word has been encountered to time. """
        try:
            self._words[word]._last_time = time
        except KeyError:
            raise KeyError("Word '"+word+"' not in WordPropsFile, time cannot be updated")
        

    #BM updateLrndProps
    def update_lrnd_props(self, word, lrnd_time, lrnd_freq, lrnd_vsize):
        """ 
        Update the learned time, learned frequency, and learned vocabulary size
         of word as long as word has not been learned. 
         
        """
        try:
            wprops = self._words[word]
            # Only update if word has not been learned
            if wprops._lrnd_time == -1:
                wprops._lrnd_time = lrnd_time
            if wprops._lrnd_freq == -1:
                wprops._lrnd_freq = lrnd_freq
            if wprops._lrnd_vsize == -1:
                wprops._lrnd_vsize = lrnd_vsize
        except KeyError:
            raise KeyError("Word '"+word+"' not in WordPropsFile, properties cannot be updated")
        
     
    #BM getWFreq
    def frequency(self, word):
        """ Return the frequency of word or 0 if word has not been encountered. """
        if word in self._words:
            return self._words[word]._wfreq
        return 0 # Aida, sep 2013 was -1

    def prob(self, word):
        """ Return the probability of word, f(w)/sum(f(w')) """
        if word in self._words:
            return self._words[words]._wfreq / float(self._words_token_freq)

        
        return 0.0

    def lrnd_frequency(self, word):
        """ 
        Return the frequency needed to learn word or -1 if word has not 
        been encountered. 
        
        """
        if word in self._words:
            return self._words[word]._lrnd_freq
        return -1

    def first_time(self, word):
        """ Return the first time word occurred or -1 if it has not been encountered. """
        if word in self._words:
            return self._words[word]._first_time
        return -1

    #BM getAllWords
    def all_words(self, minfreq):
        """ Return all words whose frequencies are greater than minfreq. """
        if minfreq <= 0:
            return self._words.keys()
        else:
            allwords = []
            for w in self._words.keys():
                if self._words[w]._wfreq > minfreq:
                    allwords.append(w)
            return allwords

    #BM getWCount
    def count(self, minfreq):
        """ Return the number of words whose frequency is greater than minfreq. """
        if minfreq <= 0:
            return len(self._words.keys())
        else:
            count = 0
            for w in self._words.keys():
                if self._words[w]._wfreq > minfreq:
                    count += 1
            return count

  
#===============================================================================
#    Context Statistics
#===============================================================================

class ContextProps:
    """
    Store context information for a specific word type
    
    Members:
    
    word -- the word that this context information applies to
    context_times -- a set of integer times where the sentence processed at each
        of these times contained this word
    unique_words -- a set of the unique words that have appeared in sentences
        alongside of this word
    context_familiarities -- context familiarity values of this word for each
        context that it appeared in (ordered for earliest to latest appearance)
    
    """
    
    def __init__(self, word):
        """ Create an empty ContextProps object for word. """
        self._word = word
        self._context_times = set()
        self._unique_words = set()
        self._context_familiarities = []
        self._cd_kys09_list = []
        self._cd_list = []
        self._similarity_list = []
        self._comprehension_list = []
        
    def add_context(self, context, time):
        """
        Add time as a time that this word has appeared in. For each word that
        appeared in the context at time time, record the unique words.
        
        """
        self._context_times.add(time)
        for word in context:
            self._unique_words.add(word)
        
 
























#===============================================================================
#    Context Statistics
#===============================================================================

class ContextProps:
    """
    Store context information for a specific word type
    
    Members:
    
    word -- the word that this context information applies to
    context_times -- a set of integer times where the sentence processed at each
        of these times contained this word
    unique_words -- a set of the unique words that have appeared in sentences
        alongside of this word
    context_familiarities -- context familiarity values of this word for each
        context that it appeared in (ordered for earliest to latest appearance)
    
    """
    
    def __init__(self, word):
        """ Create an empty ContextProps object for word. """
        self._word = word
        self._context_times = set()
        self._unique_words = set()
        self._context_familiarities = []
        self._cd_kys09_list = []
        self._cd_list = []
        self._similarity_list = []
        self._comprehension_list = []
        
    def add_context(self, context, time):
        """
        Add time as a time that this word has appeared in. For each word that
        appeared in the context at time time, record the unique words.
        
        """
        self._context_times.add(time)
        for word in context:
            self._unique_words.add(word)
        
            
class ContextPropsTable:
    """
    Store statistics about each word and the contexts they appear in.
    
    In the docstrings below C_j is a context (set of the unique words in the 
    utterance processed at time j) that the word in question has appeared in.
    
    Members:
    
    file_name -- the name of the file that these statistics will be written to
    familiarity_smoothing -- factor to smooth unfamiliar words (0 frequency) by
    aoe_normalization -- normalization term for the age of exposure 
    words -- a dictionary of words to ContextProps objects
    contexts -- a dictionary of times to sets of words corresponding to a 
        context.
    fam_measure -- familiarity measure to use, currently accepted values are 
        CONST.FREQ, CONST.LOG_FREQ, and CONST.NOVEL_FREQ.
    """
    
    def __init__(self, file_name, smoothing, fam_measure):
        """
        Create a ContextPropsTable object using file_prefix as the prefix of the
        file that this information will be saved to. The familiarity smoothing
        is defaulted to smoothing. Familiarity is calculated using the fam_measure
        type of calculation.
        
        """
        self._file_name = file_name
        self._familiarity_smoothing = smoothing
        self._fam_measure= fam_measure
        self._aoe_normalization = 1
        self._words = {}
        self._contexts = {}
        
    def reset(self):
        """ Reset this object by clearing the internal data structures."""
        self._words = {}
        self._contexts = {}    
        
    def add_context(self, context, time, similarities, comprehensions):
        """ 
        Add context (a set of words from the sentence processed at time time) to
        this object at time time. For each word in context, record the context
        familiarity, cd, cd_KYS09, similarity, and comprehension. The similarity
        of each word comes from the dictionary similarities, which maps words to 
        similarity values. The same is true for the dictionary comprehensions.
        
        """
        self._contexts[time] = context
        
        for word in context:
            if word not in self._words:
                self._words[word] = ContextProps(word)
        
        # Record Context Familiarities - BEFORE adding the words current context
        # as familiarity should be calculated up to but not including time time.
        self.context_familiarities(context)
        
        # Now the context can be recorded for each word
        for word in context:
            self._words[word].add_context(context, time)
            
        # Record cd_kys09, cd, similarities, and comprehensions for each word 
        # token of context
        for word in context:
            context_props = self._words[word]
            context_props._cd_list.append(self.cd_count(word))
            context_props._cd_kys09_list.append(self.cd_KYS09(word))
            context_props._similarity_list.append(similarities[word])
            context_props._comprehension_list.append(comprehensions[word])
            
    def add_similarity(self, word, sim):
        """ 
        Add sim to the list of similarities calculated at each context time that
        word has appeared at. 
        
        """
        try:
            self._words[word]._similarity_list.append(sim)
        except KeyError:
            raise KeyError("Word '"+word+"' not in ContextPropsTable, " +
                           "similarities cannot be updated")
        
        
    def add_comprehension(self, word, comp):
        """ 
        Add comp to the list of comprehension scores calculated at each context
        time that word has appeared at. 
        
        """
        try:
            self._words[word]._comprehension_list.append(comp)
        except KeyError:
            raise KeyError("Word '"+word+"' not in ContextPropsTable, " +
                           "comprehension scores cannot be updated")
            
    def write(self, corpus_name, outdir):
        """ 
        Write this object to two new files, one for statistics at the word type
        level and another for at the word token level.
        The files are written to the directory named outdir and as named as
        "types_{self._file_name}.csv" and "tokens_{self._file_name}.csv" 
        respectively. If corpus_name matches the standard of ending in #{num}.txt" 
        then the files will be prefixed with "#{num}_".
        
        """
        if re.search("#[0-9]*\.txt$", corpus_name):
            num = corpus_name[corpus_name.index('#'):-4] # -4 to remove ".txt"
            types_handle = open(outdir + "/" + num + "_types_" + self._file_name + ".csv", "w")
            tokens_handle = open(outdir + "/" + num + "_tokens_" + self._file_name + ".csv", "w")
        else:
            types_handle = open(outdir + "/types_" + self._file_name + ".csv", "w")
            tokens_handle = open(outdir + "/tokens_" + self._file_name + ".csv", "w")
        
        words = self._words.keys()
        words.sort()
        
        self.write_type_stats(words, types_handle)
        self.write_token_stats(words, tokens_handle)
        
        
    def write_type_stats(self, words, handle):
        """
        Write to the file object handle the word type statistics for each word
        in the list words.
        
        """
        # Write header
        handle.write("word, freq, cd_KYS09, cd_count, cdc_norm, cd_fam, cdf_norm, "
                     + "first_cf, cf, cf_norm, ae, first_lu, mlu, similarity, "
                     + "comprehension\n")
        # Write data
        for word in words:
            props = str(word) + ","
            props += str(self.freq(word)) + ","
            props += str(self.cd_KYS09(word)) + ","
            props += str(self.cd_count(word)) + ","
            if len(self._words[word]._context_times) > 0:
                props += str(self.cd_count(word)/len(self._words[word]._context_times)) + ","
            else:
                # No cdc_norm
                props += "0,"
#AF--<
            props += str(self.cd_fam(word)) + ", "
            if len(self._words[word]._context_times) > 0:
                props += str(self.cd_fam(word)/len(self._words[word]._context_times)) + ","
            else:
                # No cdf_norm
                props += "0,"
#AF--> 
            if len(self._words[word]._context_familiarities) > 0:
                props += str(self._words[word]._context_familiarities[0]) + ","
                props += str(numpy.sum(self._words[word]._context_familiarities)) + ","
                props += str(numpy.mean(self._words[word]._context_familiarities)) + ","
            else:
                # No first_cf, cf, or cf_norm
                props += "0,0,0,"
            ae = min(self._words[word]._context_times)/self._aoe_normalization
            props += str(ae) + ","
            props += str(self.first_lu(word)) + ","
            props += str(self.mlu(word)) + ","
            props += str(self.similarity(word)) + ","
            props += str(self.comprehension(word)) + "\n"
            handle.write(props)
        
        handle.close()
        
    def write_token_stats(self, words, handle):
        """
        Write to the file object handle the word token statistics for each word
        in the list words.
        
        """
        # Write header
        handle.write("token, frequency_thus_far, cd_KYS09, cd, time, cf, "
                     + "similarity, comprehension\n")
        # Write data
        for word in words:
            context_props = self._words[word]
            times = list(context_props._context_times)
            times.sort()
            ind = 0
            for time in times:
                props = str(word) + "-" + str(time) + ","
                props += str(ind+1) + ","
                props += str(context_props._cd_kys09_list[ind]) + ","
                props += str(context_props._cd_list[ind]) + ","
                props += str(time) + ","
                props += str(context_props._context_familiarities[ind]) + ","
                props += str(context_props._similarity_list[ind]) + ","
                props += str(context_props._comprehension_list[ind]) + "\n"
                handle.write(props)
                ind += 1
                
        handle.close()
    
    def first_lu(self, word):
        """ Return the length of the first utterance that contains word. """
        try:
            context_prop = self._words[word]
            first_context = min(context_prop._context_times)
            return len(self._contexts[first_context])
        except KeyError:
            if word not in self._words:
                raise KeyError("Word '"+word+"' not in ContextPropsTable")
            else:
                raise KeyError("Context #" + str(first_context) + " not in " +
                               "set of contexts: "+ str(self._contexts))
        except ValueError:
            raise ValueError("Context times for word '" + word + "' empty")
        
    
    def mlu(self, word):
        """ Return the mean length of utterances that contain word. """
        try:
            context_prop = self._words[word]
            lens = [len(self._contexts[t]) for t in context_prop._context_times]
            if len(lens) == 0:
                return 0.0
            return float(sum(lens)) / len(lens)
        except KeyError:
            if word not in self._words:
                raise KeyError("Word '"+word+"' not in ContextPropsTable")
            else:
                raise KeyError("Context #" + str(t) + " not in " +
                               "set of contexts: "+ str(self._contexts))
    
    def similarity(self, word):
        """ Return the most recent similarity measure of word. """
        try:
            return self._words[word]._similarity_list[-1]
        except KeyError:
            raise KeyError("Word '"+word+"' not in ContextPropsTable")
        except IndexError:
            raise IndexError("Similarity list for word '"+word+"' empty")
    
    def comprehension(self, word):
        """ Return the most recent comprehension score of word. """
        try:
            return self._words[word]._comprehension_list[-1]
        except KeyError:
            raise KeyError("Word '"+word+"' not in ContextPropsTable")
        except IndexError:
            raise IndexError("Comprehension list for word '"+word+"' empty")
    
    def freq(self, word):
        """ Return the number of occurrences of word. """
        try:
            return len(self._words[word]._context_times)
        except KeyError:
            raise KeyError("Word '"+word+"' not in ContextPropsTable")
    
    def cd_KYS09(self, word):
        """ Return the number of unique words co-occurring with word. """
        try:
            return len(self._words[word]._unique_words) - 1 # -1 to not include word
        except KeyError:
            raise KeyError("Word '"+word+"' not in ContextPropsTable")
    
#AF--<
    def cd_fam(self, word):
        """
        Return the total context novelty degree for word. 
        This is calculated as:
            
                        Sum of familiarity of words in C_j that are not in union(C_1,...C_j-1)
            SUM {C_j}:  -----------------------------------------------------
                        Num of words in C_j
                
        """
        total = 0.0
        # All times (j's)
        context_times = []
        try:
             context_times = list(self._words[word]._context_times)
        except KeyError:
            raise KeyError("Word '"+word+"' not in ContextPropsTable")
       
        context_times.sort()
        # union(C_1,...,C_j-1)
        words_seen = set()
        last_time_ind = 0
        for t in context_times:
             #print word, t, " -> "
             # Words in C_j that are not in union(C_1,...,C_j-1)
             novel_words = self._contexts[t] - words_seen
             #print "   ", words_seen, self._contexts[t], novel_words
             sumfam = 0.0
             for w in novel_words:
                 if not w == word:
                     sumfam += self.familiarity(w, t)
                 #print "   ", w, self.familiarity(w,t), total
             if len(self._contexts[t]) > 1:
                 total += float(sumfam) / (len(self._contexts[t])-1)
             words_seen = words_seen | self._contexts[context_times[last_time_ind]]
             last_time_ind += 1
        #print "total: ", total
        return total
#AF-->
 
    def cd_count(self, word):
        """
        Return the total context novelty degree for word. 
        This is calculated as:
            
                        Num of words in C_j that are not in union(C_1,...C_j-1)
            SUM {C_j}:  -----------------------------------------------------
                        Num of words in C_j
                
        """
        total = 0.0
        # All times (j's)
        context_times = []
        try:
             context_times = list(self._words[word]._context_times)
        except KeyError:
            raise KeyError("Word '"+word+"' not in ContextPropsTable")
        
        context_times.sort()
        # union(C_1,...,C_j-1)
        words_seen = set()
        last_time_ind = 0
        for t in context_times:
            #print word, t, " -> "
            if t == min(context_times):
                total += 1
            else:
                words_seen = words_seen | self._contexts[context_times[last_time_ind]]
                last_time_ind += 1
                # Words in C_j that are not in union(C_1,...,C_j-1)
                novel_words = self._contexts[t] - words_seen
                #print "   ", words_seen, self._contexts[t], novel_words
                if len(self._contexts[t]) > 1:
                    #print float(len(novel_words)) / (len(self._contexts[t])-1)
                    total += float(len(novel_words)) / (len(self._contexts[t])-1)
        #print "total: ", total
        return total
    
    def context_familiarities(self, context):
        """
        Calculate the context familiarities for the words in context.
        
        """
        denom = float(len(context) - 1)
        if denom < 1:
            # Only one word
            word = list(context)[0]
            self._words[word]._context_familiarities.append(self.familiarity(word))
            return
        
        # Calculate familiarity for each word in the context
        familiarities = {}
        for word in context:
            familiarities[word] = self.familiarity(word)
        
        # Calculate context familiarity w.r.t each word
        for w1 in context:
            cf = 0.0
            for w2 in context:
                if w2 != w1:
                     f = familiarities[w2]
                     cf += f/denom
            #AF--July 31/12--<
            #AF: cf is the total number of utterances that contain at least one novel word
            #AF if self._fam_measure == CONST.NOVEL_COUNT and cf > 0: cf = 1
            #AF-->
            self._words[w1]._context_familiarities.append(cf)
                
#AF    def familiarity(self, word):
    def familiarity(self, word, time=-1):
        """ 
        Calculate the familiarity of word, based on self._fam_measure:
        
        CONST.FREQ -- Return the frequency of word 
        CONST.LOG_FREQ -- Return the log frequency of word, or log of 
            self._familiarity_smoothing if the word is new
        CONST.COUNT -- Return 0 if frequency is low (here 0), return 1 otherwise
        CONST.NOVEL_COUNT -- Return 1 if the context contains one or more novel words, return 0 otherwise
        CONST.FREQ_GRP -- Return a value that reflects the frequency group of word 
        
        """
#AF--<
        if time <= 0:
            freq = len(self._words[word]._context_times)
        else:
            # need to find frequency of word up to time time
            context_times = list(self._words[word]._context_times)
            context_times.sort()
            for j in range(0,len(context_times)):
                if context_times[j] >= time: break
            freq = j

        if self._fam_measure == CONST.FREQ:
            if word in self._words and freq > 0:
                return freq 
            return self._familiarity_smoothing
        
        elif self._fam_measure == CONST.LOG_FREQ:
            if word in self._words and freq > 0:
                return math.log(freq)
            return math.log(self._familiarity_smoothing)
        
        elif self._fam_measure == CONST.COUNT:
            if word not in self._words:
                return self._familiarity_smoothing  #AF 0
            if word in self._words and freq == 0:
                return self._familiarity_smoothing  #AF 0
            return 1

#AF----old code
#AF       if self._fam_measure == CONST.FREQ:
#AF         if word in self._words:
#AF                return len(self._words[word]._context_times)
#AF            return self._familiarity_smoothing
#AF        
#AF        elif self._fam_measure == CONST.LOG_FREQ:
#AF            if word in self._words and len(self._words[word]._context_times) > 0:
#AF                return math.log(len(self._words[word]._context_times))
#AF            return math.log(self._familiarity_smoothing)
#AF        
#AF        elif self._fam_measure == CONST.COUNT:
#AF            if word not in self._words:
#AF                return 0
#AF            if word in self._words and len(self._words[word]._context_times) == 0:
#AF                return 0
#AF            return 1

#AF--<
        elif self._fam_measure == CONST.NOVEL_COUNT:
            if word not in self._words:
                return 1
            if word in self._words and freq == 0:
                return 1
            return self._familiarity_smoothing

        elif self._fam_measure == CONST.FREQ_GRP:
            if word in self._words:
                if freq == 0:
                    return 1
                elif freq == 1:
                    return 2
                elif freq > 2 and freq < 5:
                    return 5
                elif freq >= 5:
                    return 10
            return 0
#-->
        
#===============================================================================
#    Time Statistics
#===============================================================================
class TimePropsTable:
    """
    Store statistics for each time step.
    
    Members:
    
    times -- dictionary of time - statistics string pairs
    file_prefix -- the prefix of the file name that this object writes to
    
    """
    
    def __init__(self, file_prefix):
        """ Create an empty TimePropsTable object to write to file named name. """
        self._file_prefix = file_prefix
        self._times = {}

    def reset(self):
        """ Reset this object by clearing the internal data structures. """
        self._times = {}

    #BM addTime
    def add_time(self, time, heard, learned, avg_acq):
        """ 
        Add a new entry for properties for time time. The properties are:
        
        - Number of word types heard at some time point before t
        - Number of word types learned at some time point before t
        - Average acquisition score of novel nouns encountered
            at least once at some time point before t
        - Average acquisition score of novel nouns encountered
            at least twice at some time point before t
        - Average acquisition score of all words that encountered
             at some time point before t
        - Average acquisition score of words learned at some time 
            point before t
        - Average acquisition score of all nouns encountered at some
            time point before t
        - Average acquisition score of all verbs encountered at some 
            time point before t
        - Average acquisition score of all non-nouns and non-verbs
            encountered at some time point before t
        
        """
        
        props = []
        props.append(str(time))
        props.append(str(heard))
        props.append(str(learned))
        
        if CONST.NOV_N_MIN1 in avg_acq:
            props.append(str(avg_acq[CONST.NOV_N_MIN1]))
        else:
            props.append("-1")
            
        if CONST.NOV_N_MIN2 in avg_acq:
            props.append(str(avg_acq[CONST.NOV_N_MIN2]))
        else:
            props.append("-1")
                        
        if CONST.ALL in avg_acq:
            props.append(str(avg_acq[CONST.ALL]))
        else:
            props.append("-1")
        
        if CONST.LRN in avg_acq:
            props.append(str(avg_acq[CONST.LRN]))
        else:
            props.append("-1")
        
        if CONST.N in avg_acq:
            props.append(str(avg_acq[CONST.N]))
        else:
            props.append("-1")
            
        if CONST.V in avg_acq:
            props.append(str(avg_acq[CONST.V]))
        else:
            props.append("-1")
        
        if CONST.OTH in avg_acq:
            props.append(str(avg_acq[CONST.OTH]) + "\n")
        else:
            props.append("-1\n")

        self._times[time] = ",".join(props) 

    def write(self, corpus_name, outdir, itr):
        """ 
        Write this object to a new file called {outdir}/{self._file_prefix}_itr,
        with one time and its properties per line, and close the file.
        If corps_name matches the standard of ending in "#{num}.txt" then the
        file will be {outdir}/#{num}_{self._file_prefix}_itr 
        
        """
        if re.search("#[0-9]*\.txt$", corpus_name):
            # Name standard for corpora, used to distinguish multiple in one 
            # directory
            num = corpus_name[corpus_name.index('#'):-4] # -4 to remove ".txt"
            handle = open(outdir + "/" + num + "_" + self._file_prefix + itr + ".csv", "w")
        else:
            handle = open(outdir + "/" + self._file_prefix + itr + ".csv", "w")
            
        # Write header
        handle.write("time, heard, learned, nov_n_min1, nov_n_min2, all, lrn, "
                     + "n, v, oth\n")
        
        times = self._times.keys()
        timeprops = self._times
        times.sort()
        for time in times:
            handle.write(timeprops[time])
        handle.write("\n")
        
        handle.close()
        
    def read(self, handle):
        """ 
        Read from the file handle to populate this object.
        Prerequisite: handle was written to using the write() method of this 
            class.
        
        """
        handle.readline() # Remove the header line
        while 1:
            line = handle.readline().strip("\n")
            if line=="":
                break
            props = line.split(",")
            
            avg_acq = {}
            avg_acq[CONST.NOV_N_MIN1] = float(props[3])
            avg_acq[CONST.NOV_N_MIN2] = float(props[4])
            avg_acq[CONST.ALL] = float(props[5])
            avg_acq[CONST.LRN] = float(props[6])
            avg_acq[CONST.N] = float(props[7])
            avg_acq[CONST.V] = float(props[8])
            avg_acq[CONST.OTH] = float(props[9])
                        
            self.add_time(int(props[0]), int(props[1]), int(props[2]), avg_acq)
            
        
    @classmethod
    def as_properties_dicitionary(cls, handle):
        """
        Dump the time properties file contents in handle to a dictionary.
        Note -- handle is required to be a file that was been written to by
        a TimePropsFile object.
        
        """
        handle.readline() # Remove the header line
        properties = {}
        while 1:
            line = handle.readline().strip("\n")
            if line=="":
                break
            props = line.split(",")
            
            t = {}
            t[CONST.HEARD] = int(props[1])
            t[CONST.LEARNED] = int(props[2])
            t[CONST.NOV_N_MIN1] = float(props[3])
            t[CONST.NOV_N_MIN2] = float(props[4])
            t[CONST.ALL] = float(props[5])
            t[CONST.LRN] = float(props[6])
            t[CONST.N] = float(props[7])
            t[CONST.V] = float(props[8])
            t[CONST.OTH] = float(props[9])
            
            time = int(props[0])
            properties[time] = t
        
        return properties
    
