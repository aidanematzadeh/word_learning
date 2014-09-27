import matplotlib.pyplot as plt
from scipy import stats
import constants as CONST
from statistics import *

"""
plot.py

Helper methods for plotting statistics and data of the learned model.

"""
#===============================================================================

"""
Private lists and dictionaries for plotting multiple learner statistics on a
single plot.

"""
_colors = ['k', 'b', 'r', 'g', 'c', 'm', 'r']
_lines = ['-', '--', '-.', ':']
_bars = ['/', '-', "\\", ""]
    

#===============================================================================
#        Meaning Plotting
#===============================================================================
def plot_meaning_probs(word, time, learned_m, true_m, outdir):
    """
    Plot the learned Meaning learned_m of and true Meaning from the gold lexicon,
    true_m, corresponding to the word word. The probability of each feature
    being part of word's meaning (both learned and true) is plotted.
    The resulting plot is saved in the directory outdir and named as:
        {word}_{time}.png
    
    """
    
    features = list(learned_m.seen_features()) | true_m.seen_features()
    features.sort()

    true_probs = []
    learned_probs = []
    labels = []

    # Sort the data for plotting
    for feature in features:
        labels.append(feature)
        true_probs.append(true_m.prob(feature))
        learned_probs.append(learned_m.prob(feature))

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(features, true_probs, 'o', markersize=10, label='True')
    ax.plot(features, learned_probs, '^', markersize=10,label='Learned')
    
    xtickNames = plt.setp(ax, xticklabels=[""] + labels + [""])
    plt.setp(xtickNames, rotation=25, fontsize=7)

    plt.xlabel('Features')
    plt.ylabel('p(f|w)')
    plt.legend()
    ax.set_xlim([0, len(features)])
    ax.set_ylim([0,.6])
    filename = outdir + '/'+ word + '_' + str(time) + '.png'
    print "Plot: ", filename
    plt.savefig(filename)

#===============================================================================
#        Words Properties related
#===============================================================================

#BM plotProps, combined getProps in here
def plot_fast_mapping(wordprops_dir, outdir, max_sentences, Lambda):
    """
    Plot the fast-mapping effect of the words learned in the WordPropsTable 
    file in directory wordprops_dir. The plot will be saved to the directory 
    located at outdir with name formatted as 
    {outdir}fm_{Lambda}_{max_sentences}.pdf .
    
    """
    # Populate a WordPropsTable object
    wordprops = WordPropsTable("")
    wordprops_file = open(wordprops_dir, "r")
    wordprops.read(wordprops_file)
    
    max_freq = 0
    times = []
    exposures = []
    
    # Extract relevant properties
    for word in wordprops.all_words(0):
        lrnd_freq = wordprops.lrnd_frequency(word)
        if not lrnd_freq == -1:
            # Only consider learned words
            times.append(wordprops.first_time(word))
            exposures.append(lrnd_freq)
            if lrnd_freq > max_freq:
                max_freq = lrnd_freq

    # Create plot
    plt.clf()
    plt.axis([0,max_sentences,0,max_freq])
    plt.plot(times, exposures, 'bo')
    plt.xlabel("Time of first exposure")
    plt.ylabel("Number of usages needed to learn")
    plt.title("Overall Pattern of Learning Novel Words")
    filename = "%sfm_%d_%d.pdf" % (outdir, Lambda, max_sentences)
    print "Plot: ", filename
    plt.savefig(filename)
 
    
#===============================================================================
#        Time Properties related
#===============================================================================

def plot_multiple_learning_curves(timeprops_list, labels, outdir, max_time):
    """
    See plot_learning_curve docstring. 
    
    Plot the learning curve of each TimePropsFile in the list of file names
    timeprops_list where timeprops_list[i] corresponds to a Learner that has
    label labels[i].    
    """
    plt.clf()
    plt.axis([0, max_time, 0, 1])
    
    i = 0
    for l in labels:
        line = _colors[i] + _lines[i]
        (times, ratios) = _learning_curve_data(timeprops_list[i])
        plt.plot(times, ratios, line, label=l)
        i +=1
    
    plt.xlabel("Time")
    plt.ylabel("Proportion of words learned")
    title = "Learning Curve"
    plt.title(title)
    filename = "%slcurves_%d.png" % (outdir, max_time)
    print "Plot: ", filename
    plt.savefig(filename)  
    
    
def plot_learning_curve(timeprops_dir, outdir, max_time, Lambda):
    """
    Plot the learning curve based on the contents of the TimePropsFile saved
    at directory timeprops_dir.
    
    The plot will be saved to the directory located at outdir with name 
    formatted as {outdir}lcurve_{Lambda}_{max_time}.png.
    
    """
    (times, ratios) = _learning_curve_data(timeprops_dir)

    # Plot
    plt.clf()
    plt.axis([0,max_time,0,1])
    plt.plot(times, ratios, 'b-')
    plt.xlabel("Time")
    plt.ylabel("Proportion of words learned")
    title = "Learning Curve"
    plt.title(title)
    filename = "%slcurve_%d_%d.png" % (outdir, Lambda, max_time)
    print "Plot: ", filename
    plt.savefig(filename)
    

def plot_multiple_vocab_growths(timeprops_list, labels, outdir, max_time):
    """
    See plot_vocab_growth docstring. 
    
    Plot the vocabulary growth of each TimePropsFile in the list of file names 
    timeprops_list where timeprops_list[i] corresponds to a Learner that has 
    label labels[i].
    
    """
    plt.clf()
    #plt.axis([0, max_time, 0, 1])
    
    i = 0
    for l in labels:
        line = _colors[i % len(_colors)] + _lines[i % len(_lines)]
        (heards, ratios) = _vocab_growth_data(timeprops_list[i])
        plt.plot(heards, ratios, line, label=l)
        i +=1
    
    plt.legend(loc="upper left")#lower right")
    plt.xlabel("Word types received")
    plt.ylabel("Proportion of words learned")
    plt.title("Vocabulary Growth")
    filename = "%svgrowth_%d.png" % (outdir, max_time)
    print "Plot: ", filename
    plt.savefig(filename)
    

def plot_vocab_growth(timeprops_dir, outdir, max_time, Lambda):
    """
    Plot the vocabulary growth based on the contents of the TimePropsFile saved
    at directory timeprops_dir.
    
    The plot will be saved to the directory located at outdir with name 
    formatted as {outdir}vgrowth_{Lambda}_{max_time}.png .
    
    """
    (heards, ratios) = _vocab_growth_data(timeprops_dir)

    # Plot
    plt.clf()
    plt.axis([0,max_time,0,1])
    plt.plot(heards, ratios, 'b-')
    plt.xlabel("Word types received")
    plt.ylabel("Proportion of words learned")
    plt.title("Vocabulary Growth")
    filename = "%svgrowth_%d_%d.png" % (outdir, Lambda, max_time)
    print "Plot: ", filename
    plt.savefig(filename)


def plot_vocab_spurt(timeprops_dir, outdir, max_sentences, Lambda):
    """
    Plot vocabulary spurt based on the contents of the TimePropsFile saved
    at directory timeprops_dir.
    
    The plot will be saved to the directory located at outdir with name 
    formatted as {outdir}vspurt_{Lambda}_{max_sentences}.png .
    
    """
    timeprops_file = open(timeprops_dir)
    # Extract a dictionary of time properties from the TimePropsFile
    timeprops_dictionary = TimePropsTable.as_properties_dicitionary(timeprops_file)
    
    times = timeprops_dictionary.keys()
    heard_units = []
    learned_differences = []
    prev_num_learned = 0
    times.sort()
    
    # Extract relevant properties
    for time in times:
        properties = timeprops_dictionary[time]
        num_learned = properties["learned"]
        num_heard   = properties["heard"]
        maxheard = num_heard # maxheard will be num_heard of last time
        # Only plot every 50 words heard
        if num_heard % 50 == 0:
            if num_learned - prev_num_learned < 0:
                diff = 0
            else:
                # Difference used to establish the strength of the "spurt"
                diff = num_learned - prev_num_learned
            heard_units.append(num_heard)
            learned_differences.append(diff)
            prev_num_learned = num_learned

    # Plot
    plt.clf()
    plt.axis([0,maxheard,0,1])
    plt.vlines(heard_units, [0]*len(heard_units), learned_differences)
    plt.xlabel("Word types received")
    plt.ylabel("Number of words learned per 50 words received")
    plt.title("Vocabulary Spurt")
    filename = "%svspurt_%d_%d.png" % (outdir, Lambda, max_sentences)
    print "Plot: ", filename
    plt.savefig(filename)


def plot_avg_acquisition_score(timeprops_dir, outdir, max_time, Lambda):
    """
    Plot average acquisition scores based on the contents of the TimePropsFile
    saved at directory timeprops_dir.
    
    The plot will be saved to the directory located at outdir with name 
    formatted as {outdir}acqscores_{Lambda}_{max_time}.png .
    
    """
    timeprops_file = open(timeprops_dir)
    # Extract a dictionary of time properties from the TimePropsFile
    timeprops_dictionary = TimePropsTable.as_properties_dicitionary(timeprops_file)
    
    times = timeprops_dictionary.keys()
    all_tags = []
    learned = []
    nouns = []
    verbs = []
    others = []
    times.sort()
    
    # Extract relevant properties
    for time in times:
        properties = timeprops_dictionary[time]
        if properties[CONST.ALL] >= 0:
            all_tags.append(properties[CONST.ALL])
        if properties[CONST.LRN] >= 0:
            learned.append(properties[CONST.LRN])
        if properties[CONST.N] >= 0:
            nouns.append(properties[CONST.N])
        if properties[CONST.V] >= 0:
            verbs.append(properties[CONST.V])
        if properties[CONST.OTH] >= 0:
            others.append(properties[CONST.OTH])

    # Plot    
    plt.clf()
    plt.axis([0,max_time,0,1])
    plt.plot(times, all_tags, 'k-', label="ALL")
    plt.plot(times, learned, 'r-', label="LRND")
    plt.plot(times, nouns, 'g--', label="N")
    plt.plot(times, verbs, 'g-', label="V")
    plt.plot(times, others, 'g:', label="OTH")
    plt.legend('upper left')
    plt.xlabel("Time")
    plt.ylabel("Avg Acquisition for All, Lrnd, N, V, & OTH")
    plt.title("Acquisition Score")
    filename = "%sacqscores_%d_%d.png" % (outdir, Lambda, max_time)
    print "Plot: ", filename
    plt.savefig(filename)

def _map_pos_tag(pos):
    if pos == CONST.N:
        return "nouns"
    
    if pos == CONST.V:
        return "verbs"

    if pos == CONST.ALL:
        return "words"



def plot_multiple_avg_acq_scores(timeprops_list, labels, outdir, max_time, pos_tag):
    """
    Plot the average acquisition scores for the given pos_tag of each of the
    files corresponding to the list of paths to TimePropsFile(s)
    timeprops_list. Each file is the time properties file of a unique Learner
    where timeprops_list[i] belongs to the Learner labeled labels[i].
    
    The plot will be saved to the directory located at outdir with name 
    formatted as {outdir}avg_acq_{pos_tag}_{max_time}.png .
    
    """
    plt.clf()
    plt.axis([0, max_time, 0, 1])
    
    i = 0
    for l in labels:

        print "l", l
        line = _colors[i % len(_colors)] + _lines[i % len(_lines)]

        timeprops_file = open(timeprops_list[i])
        # Extract a dictionary of time properties from the TimePropsFile
        timeprops_dictionary = TimePropsTable.as_properties_dicitionary(timeprops_file)
        
        times = timeprops_dictionary.keys()
        words = []
        times.sort()
        
        for time in times:
            properties = timeprops_dictionary[time]
            
            if properties[pos_tag] >= 0:
                words.append(properties[pos_tag])
    
        plt.plot(times, words, line, label=l)
        i +=1
    
    plt.legend(loc="upper left")
    plt.xlabel('Time')
    plt.ylabel("Average Acq score of all the" + _map_pos_tag(pos_tag) )
    filename = "%savg_acq_%s_%d.png" % (outdir, pos_tag.lower(), max_time)
    
    print "Plot: ", filename
    plt.savefig(filename)


def plot_bar_multiple_novel_nouns(timeprops_list, labels, steps, outdir):
    """
    Plot average acquisition scores of novel nouns for each of the TimePropsFiles
    in the list of file names timeprops_list. The scores as plotted as bar graphs
    separated into steps (ie: 2000, 8000, 15000) where the average score of each
    learner is plotted at with the information from time 0 to the given time step.
    labels is a list of labels for each of the learners that produced each file 
    in timeprops_list. steps is the list of steps to plot.
    
    
    The plot will be saved to the directory located at outdir with name 
    formatted as {outdir}avg_acq_score_novel_nouns.png .
    
    """
    steps.sort()
    
    N = int(len(steps))
    ind = numpy.arange(N)
    width = 0.1 # width of the bars
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Rectangles to be graphed on this bar graph
    rects = []
    
    i = 0
    for timeprops in timeprops_list:
        print "-----------", timeprops
        timeprops_file = open(timeprops)
        # Extract a dictionary of time properties from the TimePropsFile
        timeprops_dictionary = TimePropsTable.as_properties_dicitionary(timeprops_file)
        
        times = timeprops_dictionary.keys()
        times.sort()
        
        scores = {}
        for step in steps:
            scores[step] = []
            
        # Sort the data for novel nouns into groups corresponding to the steps
        for time in times:
            properties = timeprops_dictionary[time]
            print time,  properties[CONST.NOV_N_MIN1] 
            if properties[CONST.NOV_N_MIN1] >= 0:
                for step in steps:
                    if time <= step:
                        scores[step].append(properties[CONST.NOV_N_MIN1])
         
        print "scores", scores
        means = []
        serrs = []
        for step in steps:
            means.append(numpy.mean(scores[step]))
            serrs.append(stats.sem(scores[step]))
        
        print "means", means
        rect = ax.bar(ind + i * width, means, width, color=_colors[i % len(_colors)], yerr=serrs)
        _label_bar(ax, rect, _bars[i % len(_bars)])
        rects.append(rect)

        i +=1
    
    
    ax.set_ylim([0.0, 0.7])
    ax.set_ylabel('Average Acq score of Novel Nouns')
    ax.set_xlabel('Time')
    ax.set_xticks(ind + width)
    
    steps_str = [str(x) for x in steps]
    ax.set_xticklabels(steps_str)

    ax.legend(rects, labels, "upper right",prop={'size':4} )

    filename = "%savg_acq_score_novel_nouns.png" % (outdir)
    print "Plot: ", filename
    plt.savefig(filename)
    

def plot_bar_multiple_acq_scores_freq(wordprops_list, learners, labels, steps, outdir):
    """
    Plot average acquisition scores of
    """
    steps.sort()
    
    N = int(len(steps))
    ind = numpy.arange(N)
    width = 0.2 # width of the bars
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Rectangles to be graphed on this bar graph
    rects = []
    
    i = 0
    for index in range(len(wordprops_list)):
        
        wordprops = wordprops_list[index]
        lrnr = learners[index]

#        times.sort()
        words = wordprops.all_words(0)    
        scores = {}
        for step in steps:
            scores[step] = []
            
        # Sort the data for novel nouns into groups corresponding to the steps
        for w in words:
            wfreq = wordprops.frequency(w)

            tsteps = [0] + steps+ [1000000]
            for j in range(len(steps)):
                if wfreq > tsteps[j] and wfreq <= tsteps[j+1]:
                    if math.isnan(lrnr.acquisition_score(w)):
                        print "---", w, wfreq, steps[j], (lrnr.acquisition_score(w))
                    #print wfreq, tsteps[j], tsteps[j+1], steps[j]
                    scores[steps[j]].append(lrnr.acquisition_score(w))
        
        means = []
        serrs = []
        for step in steps:
            means.append(numpy.mean(scores[step]))
            serrs.append(stats.sem(scores[step]))
                
        rect = ax.bar(ind + i * width, means, width, color=_colors[i], yerr=serrs)
        _label_bar(ax, rect, _bars[i])
        rects.append(rect)

        i +=1
    
    
  #  ax.set_ylim([0.0, 0.7])
    ax.set_ylabel('Average Acq score of all words')
    ax.set_xlabel('Freq')
    ax.set_xticks(ind + width)
             
    tsteps = [0] + steps+ [1000000]
    steps_str = []
    for x in range(len(steps)):
        steps_str.append('('+str(tsteps[x])+","+str(tsteps[x+1])+']') 

    ax.set_xticklabels(steps_str)

    ax.legend(rects, labels, "upper right" )

    filename = "%savg_acq_score_freq_all.png" % (outdir)
    print "Plot: ", filename
    plt.savefig(filename)

#===============================================================================
#    Helper Functions
#===============================================================================

def _vocab_growth_data(timeprops_dir):
    """
    Return the vocabulary growth data on the contents of the TimePropsFile saved
    at directory timeprops_dir. 
    The vocabulary growth data is returned as (heards, ratios) where heards is a 
    list of the number of word types heard at each time point and ratios is a list
    of the ration of heard word types vs learned word types at each time point.
    
    """
    timeprops_file = open(timeprops_dir)
    # Extract a dictionary of time properties from the TimePropsFile
    timeprops_dictionary = TimePropsTable.as_properties_dicitionary(timeprops_file)
    
    times = timeprops_dictionary.keys()
    ratios = []
    heards = []
    times.sort()
    
    # Extract relevant properties
    for time in times:
        properties = timeprops_dictionary[time]
        num_learned = properties[CONST.LEARNED]
        num_heard   = properties[CONST.HEARD]
        if num_learned == 0:
            ratio = 0.0
        else:
            ratio = float(num_learned) / float(num_heard)
        ratios.append(ratio)
        heards.append(num_heard)

    return (heards, ratios)


def _learning_curve_data(timeprops_dir):
    """
    Return the learning curve data on the contents of the TimePropsFile saved
    at directory timeprops_dir. 
    The learning curve data is returned as (times, ratios) where times is a list
    of all times that a sentence-utterance was processed at and ratios is a list
    of the ration of heard word types vs learned word types at each time point.
    
    """
    timeprops_file = open(timeprops_dir)
    # Extract a dictionary of time properties from the TimePropsFile
    timeprops_dictionary = TimePropsTable.as_properties_dicitionary(timeprops_file)
    
    times = timeprops_dictionary.keys()
    ratios = []
    times.sort()
    
    # Extract relevant properties
    for time in times:
        properties = timeprops_dictionary[time]
        num_learned = properties[CONST.LEARNED]
        num_heard   = properties[CONST.HEARD]
        if num_learned == 0:
            ratio = 0.0
        else:
            ratio = float(num_learned) / float(num_heard)
        ratios.append(ratio)

    return (times, ratios)


def _label_bar(ax, rects, hatch_type):
    """
    Label the bar, rects, of a bar graph and set its background pattern to be
    hatch_type.
    
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+0.01*rect.get_width()/2.0, 1.02*height, \
                '%.2f' % float(height), ha='left', va='bottom', fontsize=9)
        
        if hatch_type != "":
            rect.set_hatch(hatch_type)
    
