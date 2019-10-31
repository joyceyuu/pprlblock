import gzip
import math
import random
from tqdm import tqdm
from itertools import product
from memory_profiler import profile
import time


class PPRLIndex:
    """General class that implements an indexing technique for PPRL.
  """

    # --------------------------------------------------------------------------

    def __init__(self):
        """Initialise base class.
    """

        # The two databases by Alice and Bob
        #
        self.rec_dict_alice = None
        self.rec_dict_bob = None

    # --------------------------------------------------------------------------

    def __read_csv_file__(self, file_name, header_line, rec_id_col=None):
        """This method reads a comma separated file and returns a dictionary where
       the keys are the unique record identifiers (either taken from the file
       or assigned by the function) and the values are lists that contain the
       actual records.

       Arguments:
       - file_name    The name of the CSV file to read. If the file ends with
                      a '.gz' extension it is assumed it is GZipped.
       - header_line  A flag, True or False, if True then the first line is
                      assumed to contain the column (attribute) names and it
                      is skipped.
       - rec_id_col   The number (starting from 0) of the column that contains
                      unique record identifiers. If no such are available in
                      the file then this value must be set to None (default).
                      In this case each record is given a unique integer
                      number as identifier.
    """

        assert header_line in [True, False]

        rec_dict = {}  # Dictionary to contain the read records

        if (file_name.lower().endswith('.gz')):
            in_file = gzip.open(file_name)  # Open gzipped file
        else:
            in_file = open(file_name)  # Open normal file

        # Skip header line if necessary
        #
        if (header_line == True):
            header_line = in_file.readline()  # Skip over header line

        rec_count = 0

        for rec in in_file:
            rec = rec.lower().strip()
            if type(rec) == bytes:
                rec = rec.decode()
            rec = rec.split(',')
            clean_rec = list(map(lambda x: x.strip(), rec))  # Remove all surrounding whitespaces

            if (rec_id_col == None):
                rec_id = str(rec_count)  # Assign unique number as record identifier
            else:
                rec_id = clean_rec[rec_id_col]  # Get record identifier from file

            assert rec_id not in rec_dict, ('Record ID not unique:', rec_id)

            rec_dict[rec_id] = clean_rec

            rec_count += 1
        # print rec_dict
        return rec_dict

    # --------------------------------------------------------------------------

    # @profile
    def load_database_alice(self, file_name, header_line=True, rec_id_col=None,
                            ent_id_col=None):
        """Load the file which contains the data of the first database owner.

       If given, the rec_id_col is the index of where unique record
       identifiers are available, while the ent_id_col (if given) is the
       column that contains unique entity identifiers.
    """
        rec_id_col_alice = rec_id_col  # None - NC
        self.ent_id_col_alice = ent_id_col

        self.rec_dict_alice = self.__read_csv_file__(file_name, header_line,
                                                     rec_id_col_alice)
        print('Loaded Alice database: %d records' % (len(self.rec_dict_alice)))

    # --------------------------------------------------------------------------

    # @profile
    def load_database_bob(self, file_name, header_line=True, rec_id_col=None,
                          ent_id_col=None):
        """Load the file which contains the data of the second database owner.

       If given, the rec_id_col is the index of where unique record
       identifiers are available, while the ent_id_col (if given) is the
       column that contains unique entity identifiers.
    """

        rec_id_col_bob = rec_id_col  # None - NC
        self.ent_id_col_bob = ent_id_col

        self.rec_dict_bob = self.__read_csv_file__(file_name, header_line,

                                                   rec_id_col_bob)
        print('Loaded Bob database:   %d records' % (len(self.rec_dict_bob)))

    # --------------------------------------------------------------------------

    def load_and_select_ref_values(self, file_name, header_line,
                                   attr_select_list, num_vals, random_seed=0):
        """This method randomly selects a certain number of values from the given
       list of records by extracting and concatenating values from the
       attributes in the given attribute selection list. Each value will be
       unique.

       The resulting list of values will be stored in a variable
       self.ref_val_list

       Arguments:
       - file_name         The name of the CSV file to read. If the file ends
                           with a '.gz' extension it is assumed it is GZipped.
       - header_line       A flag, True or False, if True then the first line
                           is assumed to contain the column (attribute) names
                           and it is skipped.
       - attr_select_list  A list of column numbers that will be used to
                           extract attribute values from the given records,
                           and concatenate them into one string value which is
                           then used as reference value (and added to the
                           result list if it differs from all other reference
                           values).
      - num_vals           The number of unique reference values that are to
                           be generated and that will be returned.
      - random_seed        An integer value which will be used to initialise
                           the random seed generator at the beginning of the
                           function.

       It is assumed that the file contains more records than the number of
       reference values to be selected, and that the file also contains the
       required number of unique values in the selected attributes.
    """

        random.seed(random_seed)

        rec_dict = self.__read_csv_file__(file_name, header_line)

        print('Loaded reference values database: %d records' % (len(rec_dict)))

        # Extract attribute values from record dictionary
        #
        rec_attr_val_list = []

        for rec in rec_dict.values():

            # Generate reference value by combining selected attribute values
            #
            new_ref_val = ''
            for col_num in attr_select_list:
                new_ref_val += rec[col_num]
            rec_attr_val_list.append(new_ref_val)

        assert len(rec_attr_val_list) == len(rec_dict)

        ref_val_list = []

        while (len(ref_val_list) < num_vals):
            rand_ref_val = random.choice(rec_attr_val_list)  # Select one value

            if (rand_ref_val not in ref_val_list):  # Only add if different
                ref_val_list.append(rand_ref_val)

        self.ref_val_list = ref_val_list

        print('  Selected %d random reference values' % (len(self.ref_val_list)))

    # --------------------------------------------------------------------------

    def load_and_select_ref_values_alice(self, file_name, header_line,
                                         attr_select_list, num_vals, random_seed=0):
        """This method randomly selects a certain number of values from the given
       list of records by extracting and concatenating values from the
       attributes in the given attribute selection list. Each value will be
       unique.

       The resulting list of values will be stored in a variable
       self.ref_val_list_alice

       Arguments:
       - file_name         The name of the CSV file to read. If the file ends
                           with a '.gz' extension it is assumed it is GZipped.
       - header_line       A flag, True or False, if True then the first line
                           is assumed to contain the column (attribute) names
                           and it is skipped.
       - attr_select_list  A list of column numbers that will be used to
                           extract attribute values from the given records,
                           and concatenate them into one string value which is
                           then used as reference value (and added to the
                           result list if it differs from all other reference
                           values).
      - num_vals           The number of unique reference values that are to
                           be generated and that will be returned.


       It is assumed that the file contains more records than the number of
       reference values to be selected, and that the file also contains the
       required number of unique values in the selected attributes.
    """

        random.seed(random_seed)

        rec_dict = self.__read_csv_file__(file_name, header_line)

        print('Loaded reference values database: %d records' % (len(rec_dict)))

        # Extract attribute values from record dictionary
        #
        rec_attr_val_list = []

        for rec in rec_dict.values():

            # Generate reference value by combining selected attribute values
            #
            new_ref_val = ''
            for col_num in attr_select_list:
                new_ref_val += rec[col_num]
            rec_attr_val_list.append(new_ref_val)

        assert len(rec_attr_val_list) == len(rec_dict)

        ref_val_list = []

        while (len(ref_val_list) < num_vals):
            rand_ref_val = random.choice(rec_attr_val_list)  # Select one value

            if (rand_ref_val not in ref_val_list):  # Only add if different
                ref_val_list.append(rand_ref_val)

        self.ref_val_list_alice = ref_val_list

        print('  Selected %d random reference values' % (len(self.ref_val_list_alice)))

    # --------------------------------------------------------------------------

    def load_and_select_ref_values_bob(self, file_name, header_line,
                                       attr_select_list, num_vals, random_seed=1):
        """This method randomly selects a certain number of values from the given
       list of records by extracting and concatenating values from the
       attributes in the given attribute selection list. Each value will be
       unique.

       The resulting list of values will be stored in a variable
       self.ref_val_list_bob

       Arguments:
       - file_name         The name of the CSV file to read. If the file ends
                           with a '.gz' extension it is assumed it is GZipped.
       - header_line       A flag, True or False, if True then the first line
                           is assumed to contain the column (attribute) names
                           and it is skipped.
       - attr_select_list  A list of column numbers that will be used to
                           extract attribute values from the given records,
                           and concatenate them into one string value which is
                           then used as reference value (and added to the
                           result list if it differs from all other reference
                           values).
      - num_vals           The number of unique reference values that are to
                           be generated and that will be returned.


       It is assumed that the file contains more records than the number of
       reference values to be selected, and that the file also contains the
       required number of unique values in the selected attributes.
    """

        random.seed(random_seed)

        rec_dict = self.__read_csv_file__(file_name, header_line)

        print('Loaded reference values database: %d records' % (len(rec_dict)))

        # Extract attribute values from record dictionary
        #

        rec_attr_val_list = []

        for rec in rec_dict.values():

            # Generate reference value by combining selected attribute values
            #
            new_ref_val = ''
            for col_num in attr_select_list:
                new_ref_val += rec[col_num]
            rec_attr_val_list.append(new_ref_val)

        assert len(rec_attr_val_list) == len(rec_dict)

        ref_val_list = []

        while (len(ref_val_list) < num_vals):
            rand_ref_val = random.choice(rec_attr_val_list)  # Select one value

            if (rand_ref_val not in ref_val_list):  # Only add if different
                ref_val_list.append(rand_ref_val)

        self.ref_val_list_bob = ref_val_list

        print('  Selected %d random reference values' % (len(self.ref_val_list_bob)))

    # --------------------------------------------------------------------------

    def build_index_alice(self, attr_select_list):
        """Method which builds the index for the first database owner.

       The generated index needs to be stored in the variable self.index_alice

       Argument:
       - attr_select_list  A list of column numbers that will be used to
                           extract attribute values from the given records,
                           and concatenate them into one string value which is
                           then used as reference value (and added to the
                           result list if it differs from all other reference
                           values).

       See derived classes for actual implementations.
    """

        self.index_alice = None
        self.attr_select_list_alice = attr_select_list

    # --------------------------------------------------------------------------

    def build_index_bob(self, attr_select_list):
        """Method which builds the index for the second database owner.

       The generated index needs to be stored in the variable self.index_bob

      Argument:
       - attr_select_list  A list of column numbers that will be used to
                           extract attribute values from the given records,
                           and concatenate them into one string value which is
                           then used as reference value (and added to the
                           result list if it differs from all other reference
                           values).

       See derived classes for actual implementations.
   """

        self.index_bob = None
        self.attr_select_list_bob = attr_select_list

    # --------------------------------------------------------------------------

    def generate_blocks(self):
        """Method which generates blocks based on the built two index data
       structures.

       The implementations of method must store generated blocks in a
       variable self.block_dict. This dictionary must consist of block
       identifiers as keys, with values being pairs of record identifier lists,
       the first from Alice and the second from Bob.
    """

        assert self.index_alice != None
        assert self.index_bob != None

        self.block_dict = {}  # To hold the generated blocks

    # --------------------------------------------------------------------------

    @profile
    def assess_blocks(self):
        """Method which calculates the measures RR, PC and PQ for the generated
       blocks.

       |M| is the number of true matches in the candidate record pairs and |N|
       the number of non-matches in the candidate record pairs, |Mt| and |Nt|
       the total number of true matches and non-matches, respectively, and
       |A| and |B| the number of records in the two databases (with
       |Mt| + |Nt| = |A| * |B|).

       Reduction ratio is calculated as:

         rr = 1 - (|M| + |N|) / (|A| * |B|)

       Pairs completeness is calculated as:

         pc = |M| / |Mt|

       Pairs quality is calculated as:

         pq = |M| / (|M| + |N|)

       The method returns the three values rr, pc, pq

       If a record pair is a true match or not is decided based on the values
       in their ent_id_col's.
    """

        assert self.block_dict != None
        block_dict = self.block_dict

        rec_dict_alice = self.rec_dict_alice
        rec_dict_bob = self.rec_dict_bob

        num_rec_alice = len(rec_dict_alice)
        num_rec_bob = len(rec_dict_bob)

        print()
        print("Number of records in Alice's database: %d" % (num_rec_alice))
        print("Number of records in Bob's database:   %d" % (num_rec_bob))

        num_all_true_matches = 0

        # We can only calculate PC and PQ if both data sets have entity
        # identifiers
        #
        if (self.ent_id_col_alice != None) and (self.ent_id_col_bob != None):
            ent_id_col_alice = self.ent_id_col_alice
            ent_id_col_bob = self.ent_id_col_bob
            print(ent_id_col_alice, ent_id_col_bob)
            alice_ent_id_dict = {}  # Will contain counts of how often an ID occurs
            bob_ent_id_dict = {}

            for rec in self.rec_dict_alice.values():
                rec_ent_id = rec[ent_id_col_alice]
                rec_ent_id_count = alice_ent_id_dict.get(rec_ent_id, 0) + 1
                alice_ent_id_dict[rec_ent_id] = rec_ent_id_count

            for rec in self.rec_dict_bob.values():
                rec_ent_id = rec[ent_id_col_bob]
                rec_ent_id_count = bob_ent_id_dict.get(rec_ent_id, 0) + 1
                bob_ent_id_dict[rec_ent_id] = rec_ent_id_count

            # Count how often an entity identifier occurs in both data sets
            #
            for (ent_id, ent_id_count) in alice_ent_id_dict.items():
                num_all_true_matches += ent_id_count * bob_ent_id_dict.get(ent_id, 0)


            # clean memory
            del self.rec_dict_alice
            del self.rec_dict_bob
            del self.ref_val_list
            del rec_dict_alice
            del rec_dict_bob

            # Make sure each candidate pair is only counted once
            #
            # num_cand_rec_pairs = 0
            # cand_pairs_list
            # for (block_key, block_data) in block_dict.items():
            #   # if i % 100 == 0:
            #   #     print('Processing block %d of %d' % (block_num, num_blocks))
            #   alice_rec_id_list = block_data[0]
            #   bob_rec_id_list =   block_data[1]
            #   cand_pairs_list = len(alice_rec_id_list) * len(bob_rec_id_list)
            #   num_cand_rec_pairs += cand_pairs_list
            #
            # print("Total number of record pairs:          %d" % \
            #     (num_rec_alice*num_rec_bob))
            # print("Number of candidate record pairs:      %d" % (num_cand_rec_pairs))
            #

            # Calculate number of true and false matches in candidate record pairs
            #
            block_num = 0
            num_blocks = len(block_dict)
            cand_rec_pair_cache_dict = {}
            num_block_true_matches = 0
            num_block_false_matches = 0
            block_dict_size = len(block_dict)

            print('Finding number of candidate pairs...')
            for (block_key, block_data) in tqdm(block_dict.items()):
                # if block_num % int(block_dict_size / 5) == 0:
                    # print('Processing block %d of %d' % (block_num, num_blocks))
                alice_rec_id_list = block_data[0]
                bob_rec_id_list = block_data[1]

                for alice_rec_id in alice_rec_id_list:

                    # Set of Bob's entity IDs for this one from Alice
                    #
                    alice_cache_set = cand_rec_pair_cache_dict.get(alice_rec_id, set())

                    ent_id_alice = rec_dict_alice[alice_rec_id][ent_id_col_alice]

                    for bob_rec_id in bob_rec_id_list:
                        if bob_rec_id not in alice_cache_set:  # New record pair
                            alice_cache_set.add(bob_rec_id)

                            ent_id_bob = rec_dict_bob[bob_rec_id][ent_id_col_bob]
                            if ent_id_alice == ent_id_bob:  # A true match
                                num_block_true_matches += 1
                            else:
                                num_block_false_matches += 1

                    cand_rec_pair_cache_dict[alice_rec_id] = alice_cache_set

                block_num += 1
                del alice_rec_id_list

            num_cand_rec_pairs = num_block_true_matches + num_block_false_matches
            # for i, (alice_rids, bob_rids) in block_dict.items():
            #
            #     # remove duplicates if any
            #     alice_rids = set(alice_rids)
            #     bob_rids = set(bob_rids)
            #
            #     n = len(alice_rids) * len(bob_rids)
            #     num_cand_rec_pairs += n
            #     # print('Processing block={} number of pairs={:,}'.format(i, n))
            #     for a, b in product(alice_rids, bob_rids):
            #         if a == b:
            #             num_block_true_matches += 1
            #         else:
            #             num_block_false_matches += 1
            # print('Total time for calculate candidate pairs={}'.format(delta_cand_pairs))

            print("Number of all true matches:            %d" % num_all_true_matches)

        else:  # No entity identifiers, just count candidate record pairs
            num_cand_rec_pairs = 0
            for block_rec_dict in block_dict.values():
                for bob_rec_id_list in block_rec_dict.values():
                    num_cand_rec_pairs += len(bob_rec_id_list)

        total_rec = num_rec_alice * num_rec_bob

        print("Total number of record pairs:          %d" % (total_rec))
        print("Number of candidate record pairs:      %d" % (num_cand_rec_pairs))
        print()
        print(num_all_true_matches, num_block_true_matches, num_block_false_matches)

        rr = 1.0 - float(num_cand_rec_pairs) / float(num_rec_alice * num_rec_bob)

        assert (0.0 <= rr) and (rr <= 1.0), rr

        if (num_all_true_matches > 0):
            pc = float(num_block_true_matches) / float(num_all_true_matches)
            assert (0.0 <= pc) and (pc <= 1.0), pc

        else:
            pc = -1.0  # Set to an illegal value as true matches are not known

        if (num_cand_rec_pairs > 0):
            pq = float(num_block_true_matches) / float(num_cand_rec_pairs)
            assert (0.0 <= pq) and (pq <= 1.0), pq

        else:
            pq = -1.0  # Set to an illegal value as true matches are not known

        return rr, pc, pq, num_cand_rec_pairs

    def block_stats(self, blocks):
        """Calculate few statistics for blocks."""
        min_block_size = math.inf
        max_block_size = 1
        avr_block_size = 0
        blk_len_list = []
        for block in blocks.values():
            block_len = len(block)
            min_block_size = min(min_block_size, block_len)
            max_block_size = max(max_block_size, block_len)
            avr_block_size += block_len
            blk_len_list.append(block_len)
        avr_block_size /= len(blocks.values())
        sorted(blk_len_list)
        halfn = int(len(blk_len_list) / 2)
        if not len(blk_len_list) % 2:
            med_blk_size = (blk_len_list[halfn - 1] + blk_len_list[halfn]) / 2.0
        else:
            med_blk_size = blk_len_list[halfn]
        sum = 0.0
        for i in blk_len_list:
            sum = sum + (i - avr_block_size) * (i - avr_block_size)
        std_dev = math.sqrt(sum / float(len(blk_len_list)))
        print('  Smallest block: %d' % (min_block_size))
        print('  Largest block:  %d' % (max_block_size))
        print('  Average block:  %d' % (avr_block_size))
        print('  Median block:   %d' % (med_blk_size))
        print('  std dev:        %d' % (std_dev))

        return min_block_size, med_blk_size, max_block_size, avr_block_size, std_dev, blk_len_list

    def disclosure_risk(self):
        """Find disclosure risk sorted array back."""
        # construct a dictionary of record and block they are in
        alice = {}
        bob = {}

        def get_record_block_dict(dct, records, blk_id):
            """Add records and block id to dct."""
            for r in records:
                if r in dct:
                    dct[r].append(blk_id)
                else:
                    dct[r] = [blk_id]

        for blk_id, (a_vals, b_vals) in self.block_dict.items():
            # alice
            get_record_block_dict(alice, a_vals, blk_id)
            # bob
            get_record_block_dict(bob, b_vals, blk_id)

        # compute the block size
        alice_blk_size = {k: len(v) for k, (v, _) in self.block_dict.items()}
        bob_blk_size = {k: len(v) for k, (_, v) in self.block_dict.items()}

        # for record that only belongs to 1 block, risk=1/block size
        alice_risk = {}
        bob_risk = {}

        def get_block_intersection(blk_ids, index):
            """Find intersection of blocks given block ids."""
            block_dict = self.block_dict
            intersection = block_dict[blk_ids[0]][index]
            for bid in blk_ids[1:]:
                intersection = list(set(intersection) & set(block_dict[bid][index]))
            return intersection

        def create_risk(rec_blk_dict, blk_size, risk_dict, index):
            """Construct disclosure risk dictionary."""
            for k, blk in rec_blk_dict.items():
                if len(blk) == 1:
                    risk_dict[k] = 1. / blk_size[blk[0]]
                else:
                    # find the intersection of blocks
                    risk_dict[k] = 1. / len(get_block_intersection(blk, index))

        create_risk(alice, alice_blk_size, alice_risk, 0)
        create_risk(bob, bob_blk_size, bob_risk, 1)

        self.alice_risk = alice_risk
        self.bob_risk = bob_risk

        return alice_risk, bob_risk
