"""
constants.py

A collection of constants used for various aspects of the word-meaning learning
procedure.

"""


"""
Section 1: Parts of Speech Tags

"""
ALL = 'ALL'     # All tags
N = 'N'         # Nouns
V = 'V'         # Verbs
OTH = 'OTH'     # Other, non-nouns and non-verbs
ADV = 'ADV'     # Adverbs
ADJ = 'ADJ'     # Adjectives
NONE = 'NONE'

ALL_TAGS = ['ALL', 'N', 'V', 'OTH', 'ADV', 'ADJ', 'NONE']

"""
Section 2: Average Acquisition Statistics Constants

"""
NOV_N_MIN1 = 'NOV-N-MIN1'   # Novel nouns that occurred at least once key
NOV_N_MIN2 = 'NOV-N-MIN2'   # Novel nouns that occurred at least twice key
LRN = 'LRN'                 # Learned words average acquisition score key


HEARD='HEARD'               # Num of heard words key
LEARNED='LEARNED'           # Num of learned words key

"""
Section 3: Association Calculation Types

"""
SUM = 'SUM'     # Use alignment sums
ACT = 'ACT'     # Calculate using an activation function with decaying memory
DEC_SUM = 'DEC_SUM' # Calculate using a mix of both SUM and ACT

ALL_ASSOC_TYPES = ['SUM', 'ACT', 'DEC_SUM']

"""
Section 4: Similarity Calculation Types

"""
COS = 'COS'     # Use Cosine method 
JSD = 'JSD'     # Use Jensen-Shannan Divergence
EUC = 'EUC'     # Use Euclidean Distance
AP = 'AP'       # Use Average Precision

ALL_SIM_TYPES = ['COS', 'JSD', 'EUC', 'AP']

"""
Section 5: Other

"""
NWL = 'NWL'     # Novel Word Learning categories task

#AF CD_COUNT = 'CD_COUNT' # Contextual Diversity calculation - count # of new words in context
#AF CD_FAMILIARITY = 'CD_FAMILIARITY' # Contextual Diversity calculation - sum over familiarity of new words in context

FREQ = 'FREQ'   # Context Familiarity calculation - Frequency count
LOG_FREQ = 'LOG_FREQ' # Context Familiarity calculation - Log of Frequency count
COUNT = 'COUNT' # Context Familiarity calculation - proportion of familiar words 
#AF--<
NOVEL_COUNT = 'NOVEL_COUNT' # Context Familiarity calculation - total number of novel words
FREQ_GRP = 'FREQ_GRP' # Context Familiarity calculation - Frequency groups/bins
#AF-->

ALL_FAM_MEASURES = ['NWL', 'FREQ', 'LOG_FREQ', 'COUNT', 'NOVEL_COUNT', 'FREQ_GRP']



