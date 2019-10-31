import os
import math
import random
import hashlib

from pprlindex import PPRLIndex
from config import QGRAM_LEN, QGRAM_PADDING, PADDING_END_CHAR, PADDING_START_CHAR


class PPRLIndexBloomFilterHLSH(PPRLIndex):
    """Class that implements a Bloom filter and Hamming Locality Sensitive
     hashing (LSH) approach for PPRL indexing.

     Based on the approach described in:

     A Framework for Accurate, Efficient Private Record Linkage
     by Elizabeth Durham
     PhD thesis, Faculty of the Graduate School of Vanderbilt University, 2012

     Attribute values are converted into Bloom filters, bits from these Bloom
     filters are randomly extracted and combined into a record Bloom filter,
     and records that have the same bits set to 1 in common are inserted into
     the same blocks.
  """

    # --------------------------------------------------------------------------

    def __init__(self, num_hash_funct, one_bit_set_perc=50, random_seed=42):
        """Initialise the class.

       Arguments:
       - num_hash_funct    The number of hash functions used when generating
                           Bloom filter values.
       - one_bit_set_perc  A percentage value between 1 and 99 (default is 50
                           percent) which gives the number of bits that should
                           be set to 1 in a Bloom filter, based on the number
                           of hash functions to be used and the average number
                           of q-grams to be expected in attribute values.
       - random_seed       An integer number. This argument is required to
                           make sure both database owners will generate the
                           same sequence of random values.
    """

        assert num_hash_funct > 0
        assert (one_bit_set_perc > 0) and (one_bit_set_perc < 100)
        assert random_seed != None

        self.num_hash_funct = num_hash_funct
        self.one_bit_set_perc = one_bit_set_perc
        self.random_seed = random_seed

        # The two databases by Alice and Bob
        #
        self.rec_dict_alice = None
        self.rec_dict_bob = None

        self.attr_bf_len_list = None  # A list with the attribute (field) Bloom
        # filter lengths.

        self.attr_bf_sel_list = None  # List which for each attribute will contain
        # the bits (their index numbers) that are to
        # be sampled from attribute Bloom filters
        # into the record Bloom filter.

        self.shuffle_bit_list = None  # To be set to a list containing a random
        # order of bit indexes to be used to shuffle
        # the record Bloom filters.

        self.hlsh_sample_bits_list = None  # List which will contain several sets
        # for the random bit positions that are
        # extracted from record Bloom filters
        # to form the blocking key values
        # (these correspond to the Hamming
        # locality sensitive hashing values).

        self.bf_cache = {}  # A cache for Bloom filters (keys are strings and
        # values their Bloom filters as sets)

    # --------------------------------------------------------------------------

    def __calc_attr_bf_len__(self, attr_select_list, num_rec=10000):
        """Use records from the two databases and calculate the length (in bits)
       of the attribute (field) Bloom filters using the approach described by
       Durham (page 71).

       For each of the selected attributes (listed in attr_select_list), the
       average lengths (in characters) will be estimated from a sample
       (num_rec) of records from the two databases.

       The method returns a list (with same length as the attr_select_list)
       containing the Bloom filter lengths for each selected attribute.
    """

        assert num_rec > 0

        # Make sure the databases have been loaded
        #
        assert self.rec_dict_alice != None
        assert self.rec_dict_bob != None

        num_attr = len(attr_select_list)
        avrg_len_list = [0.0] * num_attr

        alice_rec_id_list = list(self.rec_dict_alice.keys())
        bob_rec_id_list = list(self.rec_dict_bob.keys())

        check_alice_rec_id_list = alice_rec_id_list[:num_rec]
        check_bob_rec_id_list = bob_rec_id_list[:num_rec]

        num_rec_checked = 0

        for rec_id in check_alice_rec_id_list:
            check_rec = self.rec_dict_alice[rec_id]

            for i in range(num_attr):
                attr_val = check_rec[attr_select_list[i]]
                attr_val_len = len(attr_val)
                avrg_len_list[i] += attr_val_len

            num_rec_checked += 1

        for rec_id in check_bob_rec_id_list:
            check_rec = self.rec_dict_bob[rec_id]

            for i in range(num_attr):
                attr_val = check_rec[attr_select_list[i]]
                attr_val_len = len(attr_val)
                avrg_len_list[i] += attr_val_len

            num_rec_checked += 1

        # Normalise counts in to average lengths
        #
        for i in range(num_attr):
            tmp_avrg = float(avrg_len_list[i]) / float(num_rec_checked)
            avrg_len_list[i] = int(tmp_avrg)

        print('Average number of characters in attributes:', avrg_len_list)

        p = float(self.one_bit_set_perc) / 100.0  # Percentage a bit is set to 1

        # Calculate number of q-grams and then Bloom filter bits
        #
        attr_bf_len = []
        for avrg_len_char in avrg_len_list:
            avrg_num_q_gram = avrg_len_char - QGRAM_LEN + 1
            if (QGRAM_PADDING == True):
                avrg_num_q_gram += 2 * (QGRAM_LEN - 1)

            k_g = self.num_hash_funct * avrg_num_q_gram

            m = 1.0 / (1.0 - math.pow(p, 1. / (k_g)))

            attr_bf_len.append(int(math.ceil(m)))

        print('Field Bloom filter length:', attr_bf_len)

        return attr_bf_len

    # --------------------------------------------------------------------------

    def __str2bf__(self, s, bf_len, do_cache=False):
        """Convert a single string into a Bloom filter (a set with the index
       values of the bits set to 1 according to the given Bloom filter length.

       This method returns the generated Bloom filter as a set.

       If do_cache is set to True then the Bloom filter for this string will
       be stored.
    """

        if (s in self.bf_cache):
            return self.bf_cache[s]

        h1 = hashlib.sha1
        h2 = hashlib.md5
        num_hash_funct = self.num_hash_funct

        q_minus_1 = QGRAM_LEN - 1

        if (QGRAM_PADDING == True):
            ps = PADDING_START_CHAR * q_minus_1 + s + PADDING_END_CHAR * q_minus_1
        else:
            ps = s

        q_gram_list = [ps[i:i + QGRAM_LEN] for i in range(len(ps) - q_minus_1)]

        bloom_set = set()

        for q in q_gram_list:

            hex_str1 = h1(q.encode('utf-8')).hexdigest()
            int1 = int(hex_str1, 16)

            hex_str2 = h2(q.encode('utf-8')).hexdigest()
            int2 = int(hex_str2, 16)

            for i in range(num_hash_funct):
                gi = int1 + i * int2
                gi = int(gi % bf_len)

                bloom_set.add(gi)

        if (do_cache == True):  # Store in cache
            self.bf_cache[s] = bloom_set

        return bloom_set

    # --------------------------------------------------------------------------

    def build_index_alice(self, attr_select_list, attr_bf_sample_list,
                          num_bits_hlsh, num_iter_hlsh):
        """Method which builds the index for the first database owner.

       The generated index needs to be stored in the variable self.index_alice

       Arguments:
       - attr_select_list     A list of column numbers that will be used to
                              extract attribute values from the given records,
                              and convert them into Bloom filters.
       - attr_bf_sample_list  A list with percentage numbers (one per selected
                              attribute) which specifies the percentage of
                              bits to sample from an attribute Bloom filter.
       - num_bits_hlsh        Number of bits to sample from record Bloom
                              filters to generate the locality sensitive
                              hashing (LSH) values.
       - num_iter_hlsh        Number of times a record Bloom filter is
                              sampled.
    """

        # Set random seed so both database owners have the same start
        #
        random.seed(self.random_seed)

        assert self.rec_dict_alice != None

        assert len(attr_select_list) == len(attr_bf_sample_list)
        self.attr_select_list = attr_select_list

        assert num_bits_hlsh > 0
        assert num_iter_hlsh > 0

        if (self.shuffle_bit_list == None):
            self.attr_bf_len_list = self.__calc_attr_bf_len__(attr_select_list)

            # Generate list which specifies which bits to sample from each attribute
            # Bloom filter
            #
            self.attr_bf_sel_list = []
            i = 0
            for i in range(len(attr_select_list)):
                attr_bf_len = self.attr_bf_len_list[i]  # Attribute BF length

                # Number of bits to sample for this attribute
                #
                attr_bf_sample = int(math.ceil(float(attr_bf_sample_list[i]) / \
                                               100.0 * attr_bf_len))
                self.attr_bf_sel_list.append(set(random.sample(list(range(attr_bf_len)),
                                                               attr_bf_sample)))
                # print i, attr_bf_len, attr_bf_sample, len(self.attr_bf_sel_list[-1])

            self.rec_bf_len = sum(self.attr_bf_len_list)
            # print self.rec_bf_len

            # Generate a random permutation of bit positions
            #
            self.shuffle_bit_list = list(range(self.rec_bf_len))
            random.shuffle(self.shuffle_bit_list)

        if (self.hlsh_sample_bits_list == None):

            rec_bits = list(range(self.rec_bf_len))

            # Generate num_iter_hlsh lists each containing num_bits_hlsh bit
            # positions
            #
            self.hlsh_sample_bits_list = []
            for i in range(num_iter_hlsh):
                rand_bit_list = random.sample(rec_bits, num_bits_hlsh)
                self.hlsh_sample_bits_list.append(rand_bit_list)
                # print self.hlsh_sample_bits_list[-1]
                assert min(self.hlsh_sample_bits_list[-1]) >= 0
                assert max(self.hlsh_sample_bits_list[-1]) < self.rec_bf_len

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.index_alice = self.__generate_data_set_blocks__(self.rec_dict_alice,
                                                             attr_select_list)
        print('Index for Alice contains %d blocks' % (len(self.index_alice)))

        stat = self.block_stats(self.index_alice)
        min_block_size, med_blk_size, max_block_size, avr_block_size, std_dev, blk_len_list = stat

        wr_file_name = './logs/HLSH_data_alice.csv'
        wr_file = open(wr_file_name, 'a')

        tot = 0.0
        for i in blk_len_list:
            tot = tot + (i - avr_block_size) * (i - avr_block_size)
            wr_file.write(str(i) + ',')
        wr_file.write(os.linesep)
        wr_file.close()

        return min_block_size, med_blk_size, max_block_size, avr_block_size, std_dev

        # del self.rec_dict_alice

    # --------------------------------------------------------------------------

    def build_index_bob(self, attr_select_list, attr_bf_sample_list,
                        num_bits_hlsh, num_iter_hlsh):
        """Method which builds the index for the second database owner.

       The generated index needs to be stored in the variable self.index_bob

       Arguments:
       - attr_select_list     A list of column numbers that will be used to
                              extract attribute values from the given records,
                              and convert them into Bloom filters.
       - attr_bf_sample_list  A list with percentage numbers (one per selected
                              attribute) which specifies the percentage of
                              bits to sample from an attribute Bloom filter.
       - num_bits_hlsh        Number of bits to sample from record Bloom
                              filters to generate the locality sensitive
                              hashing (LSH) values.
       - num_iter_hlsh        Number of times a record Bloom filter is
                              sampled.
    """

        # Set random seed so both database owners have the same start
        #
        random.seed(self.random_seed)

        assert self.rec_dict_bob != None

        assert len(attr_select_list) == len(attr_bf_sample_list)
        self.attr_select_list = attr_select_list

        assert num_bits_hlsh > 0
        assert num_iter_hlsh > 0

        if (self.shuffle_bit_list == None):
            self.attr_bf_len_list = self.__calc_attr_bf_len__(attr_select_list)

            # Generate list which specifies which bits to sample from each attribute
            # Bloom filter
            #
            self.attr_bf_sel_list = []
            i = 0
            for i in range(len(attr_select_list)):
                attr_bf_len = self.attr_bf_len_list[i]  # Attribute BF length

                # Number of bits to sample for this attribute
                #
                attr_bf_sample = int(math.ceil(float(attr_bf_sample_list[i]) / \
                                               100.0 * attr_bf_len))
                self.attr_bf_sel_list.append(random.sample(list(range(attr_bf_len)),
                                                           attr_bf_sample))
                # print i, attr_bf_len, attr_bf_sample, len(self.attr_bf_sel_list[-1])

            self.rec_bf_len = sum(self.attr_bf_len_list)
            # print self.rec_bf_len

            # Generate a random permutation of bit positions
            #
            self.shuffle_bit_list = list(range(self.rec_bf_len))
            random.shuffle(self.shuffle_bit_list)

        if (self.hlsh_sample_bits_list == None):

            rec_bits = list(range(self.rec_bf_len))

            # Generate num_iter_hlsh lists each containing num_bits_hlsh bit
            # positions
            #
            self.hlsh_sample_bits_list = []
            for i in range(num_iter_hlsh):
                rand_bit_list = random.sample(rec_bits, num_bits_hlsh)
                self.hlsh_sample_bits_list.append(rand_bit_list)
                # print self.hlsh_sample_bits_list[-1]
                assert min(self.hlsh_sample_bits_list[-1]) >= 0
                assert max(self.hlsh_sample_bits_list[-1]) < self.rec_bf_len

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self.index_bob = self.__generate_data_set_blocks__(self.rec_dict_bob,
                                                           attr_select_list)
        print('Index for Bob contains %d blocks' % (len(self.index_bob)))

        stat = self.block_stats(self.index_bob)
        min_block_size, med_blk_size, max_block_size, avr_block_size, std_dev, blk_len_list = stat

        wr_file_name = './logs/HLSH_data_bob.csv'
        wr_file = open(wr_file_name, 'a')

        tot = 0.0
        for i in blk_len_list:
            tot = tot + (i - avr_block_size) * (i - avr_block_size)
            wr_file.write(str(i) + ',')
        wr_file.write(os.linesep)
        wr_file.close()

        return min_block_size, med_blk_size, max_block_size, avr_block_size, std_dev

        # del self.rec_dict_bob

    # --------------------------------------------------------------------------

    def __generate_data_set_blocks__(self, rec_dict, attr_select_list):
        """Generate the blocks for the given record dictionary. Each record (its
       record identifier) is inserted into one or several blocks according to
       the bit pattern of its Bloom filter.

       Arguments:
       - rec_dict          A dictionary containing records, with the the keys
                           being record identifiers and values being a list of
                           attribute values for each record.
       - attr_select_list  A list of column numbers that will be used to
                           extract attribute values from the given records and
                           generate Bloom filters for these attributes.

       The method returns a dictionary which contains the generated blocks.
    """

        print()
        print('Generate basic blocks for data set:')

        assert rec_dict != None

        str2bf = self.__str2bf__

        block_dict = {}  # Resulting blocks generated

        num_rec_done = 0

        attr_bf_len_list = self.attr_bf_len_list
        attr_bf_sel_list = self.attr_bf_sel_list
        shuffle_bit_list = self.shuffle_bit_list
        hlsh_sample_bits_list = self.hlsh_sample_bits_list

        # Calculate the start 'indexes' for the different attribute Bloom filters
        #
        bf_start_index_list = [0]

        for attr_bf_len in attr_bf_len_list:
            bf_start_index_list.append(bf_start_index_list[-1] + attr_bf_len)
        bf_start_index_list = bf_start_index_list[:-1]  # Last element no needed

        # Main loop over all records in database
        #
        for (rec_id, rec_list) in rec_dict.items():
            num_rec_done += 1
            if (num_rec_done % 10000 == 0):
                print('  Processed %d of %d records' % (num_rec_done, len(rec_dict)))

            # Convert the selected attributes into Bloom filters, then sample bits
            # from them, and combine into record Bloom filter
            #
            i = 0
            for col_num in attr_select_list:
                attr_val = rec_list[col_num]
                attr_bf = str2bf(attr_val, attr_bf_len_list[i])

                # Only keep the bits that are to be sampled
                #
                attr_sel_bf = attr_bf_sel_list[i].intersection(attr_bf)

                # Modify bit positions for record Bloom filter
                #
                if (i == 0):  # No offset needed
                    rec_bf = attr_sel_bf
                else:
                    attr_offset = bf_start_index_list[i]
                    for bit_pos in attr_sel_bf:
                        rec_bf.add(bit_pos + attr_offset)

                i += 1

            # Shuffle bits in record Bloom filter
            #
            rec_bf_shuffle = set()
            for bit_pos in rec_bf:
                rec_bf_shuffle.add(shuffle_bit_list[bit_pos])

            # Generate the desired number of blocking key values by extracting
            # selected bits from the shuffled Bloom filter
            #
            for sample_bit_list in hlsh_sample_bits_list:

                block_str = ''
                for bit_pos in sample_bit_list:
                    if (bit_pos in rec_bf_shuffle):
                        block_str += '1'
                    else:
                        block_str += '0'
                assert len(block_str) == len(sample_bit_list), (len(block_str), len(sample_bit_list))

                block_rec_id_list = block_dict.get(block_str, [])
                block_rec_id_list.append(rec_id)
                block_dict[block_str] = block_rec_id_list

                # print block_str, block_rec_id_list

        return block_dict

    # --------------------------------------------------------------------------

    def generate_blocks(self):
        """Method which generates the blocks based on the built two index data
      structures.

      Because a candidate record pair can occur in several blocks we need to
      record all pairs that have been generated, and only keep a pair in a
      single block.
   """

        block_dict = {}

        num_cand_rec_pairs = 0

        block_num = 0  # Each block get a unique number

        blocks_alice = self.index_alice  # Keys are cluster identifiers, values the
        blocks_bob = self.index_bob  # corresponding lists of record
        # identifiers

        for block_bit_str in blocks_alice:

            if (block_bit_str in blocks_bob):
                # Get list of record identifiers in this block
                #
                block_rec_list_alice = blocks_alice[block_bit_str]
                block_rec_list_bob = blocks_bob[block_bit_str]

                block_dict[block_num] = (block_rec_list_alice, block_rec_list_bob)
                num_cand_rec_pairs += len(block_rec_list_alice) * \
                                      len(block_rec_list_bob)
                block_num += 1

        self.block_dict = block_dict

        print('Final indexing contains %d blocks' % (len(block_dict)))
        # print block_dict
        return len(block_dict)
