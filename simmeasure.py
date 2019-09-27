"""Similarity Measure Class."""
import hashlib
from config import (
    QGRAM_LEN,
    QGRAM_PADDING,
    PADDING_START_CHAR,
    PADDING_END_CHAR
)


def editdist(str1, str2, min_threshold = None):
  """Return approximate string comparator measure (between 0.0 and 1.0)
     using the edit (or Levenshtein) distance.
  """
  # Quick check if the strings are empty or the same - - - - - - - - - - - - -
  #
  if (str1 == '') or (str2 == ''):
    return 0.0
  elif (str1 == str2):
    return 1.0

  n = len(str1)
  m = len(str2)
  max_len = max(n,m)

  if (min_threshold != None):
    if (isinstance(min_threshold, float)) and (min_threshold > 0.0) and \
       (min_threshold > 0.0):

      len_diff = abs(n-m)
      w = 1.0 - float(len_diff) / float(max_len)

      if (w  < min_threshold):
        return 0.0  # Similariy is smaller than minimum threshold

      else: # Calculate the maximum distance possible with this threshold
        max_dist = (1.0-min_threshold)*max_len

    else:
      logging.exception('Illegal value for minimum threshold (not between' + \
                      ' 0 and 1): %f' % (min_threshold))
      raise Exception

  if (n > m):  # Make sure n <= m, to use O(min(n,m)) space
    str1, str2 = str2, str1
    n, m =       m, n

  current = list(range(n+1))

  for i in range(1, m+1):

    previous = current
    current =  [i]+n*[0]
    str2char = str2[i-1]

    for j in range(1,n+1):
      substitute = previous[j-1]
      if (str1[j-1] != str2char):
        substitute += 1

      # Get minimum of insert, delete and substitute
      #
      current[j] = min(previous[j]+1, current[j-1]+1, substitute)

    if (min_threshold != None) and (min(current) > max_dist):
      return 1.0 - float(max_dist+1) / float(max_len)

  w = 1.0 - float(current[n]) / float(max_len)

  assert (w >= 0.0) and (w <= 1.0), 'Similarity weight outside 0-1: %f' % (w)

  #print w
  return w

# ============================================================================

class SimMeasure:
  """General class that implements similarity measures.
  """

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  def __init__(self):
    """Initialisation, noting to do.
    """

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  def sim(self, s1, s2, do_cache=False):
    """Calculate and return the similarity between two strings (a value
       between 0 and 1).

       If this similarity should be cached set the argument do_cache to True.
    """

# ----------------------------------------------------------------------------

class DiceSim(SimMeasure):
  """Class that implements the Dice coefficient for the two input strings.
     This methods uses the constants: QGRAM_LEN and QGRAM_PADDING (and if this
     constant is set to True also PADDING_START_CHAR and PADDING_END_CHAR).

     If the argument do_cache is set to True then the generated q-gram lists
     will be stored in a dictionary to prevent their repeated computation.
  """

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  def __init__(self):
    """Initialise the cache.
    """

    self.q_gram_cache = {}  # Store strings converted into q-grams. Keys in
                            # this will be strings and their values their
                            # q-gram list
    self.sim_cache = {}     # Store the string pair and its similarity in a
                            # cache as well

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  def sim(self, s1, s2, do_cache=False):
    """Calculate the similarity between the given two strings. The method
       returns a value between 0.0 and 1.0.

       If this similarity should be cached set the argument do_cache to True.
    """

    assert do_cache in [True, False]

    if (s1 == s2):  # Quick check for equality
      return 1.0

    # Check if the string pair has been compared before
    #
    if ((s1,s2) in self.sim_cache):
      return self.sim_cache[(s1,s2)]

    q_minus_1 = QGRAM_LEN - 1

    # Convert input strings into q-gram lists
    #
    if (do_cache == True) and (s1 in self.q_gram_cache):
      l1 = self.q_gram_cache[s1]

    else:  # Need to calculate q-gram list for the first string
      if (QGRAM_PADDING == True):
        ps1 = PADDING_START_CHAR*q_minus_1 + s1 + PADDING_END_CHAR*q_minus_1
      else:
        ps1 = s1

      l1 = [ps1[i:i+QGRAM_LEN] for i in range(len(ps1) - q_minus_1)]

      if (do_cache == True):
        self.q_gram_cache[s1] = l1

    if (do_cache == True) and (s2 in self.q_gram_cache):
      l2 = self.q_gram_cache[s2]

    else:  # Need to calculate q-gram list for the second string
      if (QGRAM_PADDING == True):
        ps2 = PADDING_START_CHAR*q_minus_1 + s2 + PADDING_END_CHAR*q_minus_1
      else:
        ps2 = s2

      l2 = [ps2[i:i+QGRAM_LEN] for i in range(len(ps2) - q_minus_1)]

      if (do_cache == True):
        self.q_gram_cache[s2] = l2

    common = len(set(l1).intersection(set(l2)))

    sim = 2.0*common / (len(l1)+len(l2))

    if (do_cache == True):
      self.sim_cache[(s1,s2)] = sim

    return sim

# ----------------------------------------------------------------------------

class BloomFilterSim(SimMeasure):
  """Class that implements a Bloom filter similarity measure, where strings
     are converted into Bloom filters, and then their Dice coefficient is
     calculated.

     The values of QGRAM_LEN and QGRAM_PADDING (and if this constant is set
     to True also PADDING_START_CHAR and PADDING_END_CHAR) are used to
     convert strings into q-gram lists, which are then converted into Bloom
     filters.
  """

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  def __init__(self):
    """Initialise a Bloom filter.
    """

    self.bf_cache = {}   # A cache for strings (keys) and their BF (values)
    self.sim_cache = {}  # Store the string pair and its similarity in a
                         # cache as well.

    # The two hash functions to be used for the Bloom filter
    #
    self.h1 = hashlib.sha1
    self.h2 = hashlib.md5

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  def str2bf(self, s, num_hash_funct, bf_len, do_cache=False):
    """Convert a single string into a Bloom filter (a set with the index
       values of the bits set to 1 according to the given parameters (Bloom
       filter length and number of hash functions).

       This method returns the generated Bloom filter (as a set).

       If do_cache is set to True then the Bloom filter for this string will
       be stored.
    """

    assert num_hash_funct > 0
    assert bf_len > 0

    if s in self.bf_cache:
      return self.bf_cache[s]

    h1 = self.h1
    h2 = self.h2

    q_minus_1 = QGRAM_LEN

    if (QGRAM_PADDING == True):
      ps = PADDING_START_CHAR*q_minus_1 + s + PADDING_END_CHAR*q_minus_1
    else:
      ps = s

    q_gram_list = [ps[i:i+QGRAM_LEN] for i in range(len(ps) - q_minus_1)]

    bloom_set = set()

    for q in q_gram_list:

      hex_str1 = h1(q).hexdigest()
      int1 =     int(hex_str1, 16)

      hex_str2 = h2(q).hexdigest()
      int2 =     int(hex_str2, 16)

      for i in range(num_hash_funct):
        gi = int1 + i*int2
        gi = int(gi % bf_len)

        bloom_set.add(gi)

    if (do_cache == True):  # Store in cache
     self.bf_cache[s] = bloom_set

    return bloom_set

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  def sim(self, s1, s2, do_cache=False):
    """Calculate the similarity between the given two strings. The method
       returns a value between 0.0 and 1.0.

       If this similarity should be cached set the argument do_cache to True.
    """

    assert do_cache in [True, False]

    if (s1 == s2):  # Quick check for equality
      return 1.0

    # Check if the string pair has been compared before
    #
    if ((s1,s2) in self.sim_cache):
      return self.sim_cache[(s1,s2)]

    bf1 = self.str2bf(s1, do_cache)
    bf2 = self.str2bf(s2, do_cache)

    num_bit1 = len(bf1)
    num_bit2 = len(bf2)
    num_bit_common = len(bf1.intersection(bf2))

    sim = 2.0*num_bit_common / (num_bit1 + num_bit2)

    if (do_cache == True):
      self.sim_cache[(s1,s2)] = sim

    return sim
