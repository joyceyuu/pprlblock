import os
import time
import math
from collections import defaultdict

from pprlindex import PPRLIndex

class PPRLIndexPSignature(PPRLIndex):
    """Class that implements the PPRL indexing technique:

        Reference scalability entity resolution using probability signatures
        on parallel databases.

        This class includes an implmentation of p-sig algorithm.
    """

    def __init__(self, sigfunc, maximum_block_size, eta, tao):
        """Initialize the class and set the required parameters.

        Arguments:
        - sigfunc              function that takes one of those values and returns an
                               iterable of potential signatures
        - maximum_block_size   expected recurrence frequency of a signature
        - tao                  the threshold for us to adopt a link if the probability
                               for two records to share a signature exceeds tao
        - rou                  the threshold we consider a subrecord if its probability
                               of being a signature exceeds rou

        """
        self.sigfunc = sigfunc
        self.maximum_block_size = maximum_block_size
        self.eta = eta
        self.tao = tao


    def create_bloom_filter_alice(self):
        """Create Bloom filter on Alice's records."""



    def __microblocks__(self, records, sigfunc, sigfunc_args):
        """Construct micro blocks.

        Arguments
        -----------
        records: dict
                 key=record ID, value=list of fields
        sigfunc: function
                 return signature computed by sigfunc
        sigfunc_args: dict
                      the arguments needed for sigfunc

        """
        potential_blocks = defaultdict(set)
        for idx, value in records.items():
            signature = sigfunc(value, **sigfunc_args)
            potential_blocks[signature].add(idx)
        microblocks = {key: idxs for key, idxs in potential_blocks.items() if
                       1 < len(idxs) < self.maximum_block_size}
        return microblocks

    def build_index_alice(self, sigfunc, sigfunc_args):
        """Build revert index for alice data."""
        start_time = time.time()
        assert self.rec_dict_alice != None
        revert_index = self.__microblocks__(self.rec_dict_alice,
                                            sigfunc, sigfunc_args)
        self.index_alice = revert_index
        alice_time = time.time() - start_time
        stat = self.block_stats(revert_index)
        min_block_size,med_blk_size,max_block_size,avr_block_size,std_dev,blk_len_list = stat
        return min_block_size,med_blk_size,max_block_size,avr_block_size,std_dev

    def build_index_bob(self, sigfunc, sigfunc_args):
        """Build revert index for alice data."""
        start_time = time.time()
        assert self.rec_dict_bob != None
        revert_index = self.__microblocks__(self.rec_dict_bob,
                                            sigfunc, sigfunc_args)
        self.index_bob = revert_index
        bob_time = time.time() - start_time
        stat = self.block_stats(revert_index)
        min_block_size,med_blk_size,max_block_size,avr_block_size,std_dev,blk_len_list = stat
        return min_block_size,med_blk_size,max_block_size,avr_block_size,std_dev

    def generate_blocks(self):
        """Generates blocks based on built two index."""
        block_dict = {}

        index_alice = self.index_alice
        index_bob = self.index_bob

        cand_blk_key = 0
        for (block_id, block_vals) in index_alice.items():
            bob_block_vals = index_bob.get(block_id, None)
            if bob_block_vals != None:
                block_dict[cand_blk_key] = (block_vals, bob_block_vals)
                cand_blk_key += 1
        self.block_dict = block_dict
        print('Final indexing contains %d blocks' % (len(block_dict)))
        return len(block_dict)
