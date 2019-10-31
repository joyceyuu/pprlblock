import os
import time
import numpy

from pprlindex import PPRLIndex


class hclustering(PPRLIndex):

    def __init__(self, dist, nb, wn, ep):
        """Initialise the class and set the required parameters.

       Arguments:
       - nb                 Number of blocks to be generated.
       - dist               A function which takes two strings as input and
                            returns a similarity value between 0 and 1.
    """

        self.nb = nb
        self.dist = dist
        self.wn = wn
        self.ep = ep

        self.ref_val_list = []
        self.alice_clusters = {}
        self.bob_clusters = {}

    # --------------------------------------------------------------------------

    def hcluster(self):

        assert self.ref_val_list != None
        # print sorted(self.ref_val_list)

        clust = {}
        id = 0
        # clusters are initially just the individual rows
        for ref_val in self.ref_val_list:
            clust[id] = [ref_val]
            id += 1

        distance = self.dist
        distances = {}
        currentclustid = len(clust)

        nb = self.nb  ## number of blocks

        while len(clust) > nb:
            # print len(clust)
            lowestpair = [0, 1]
            closest = 0.0  # distance(clust[0][0],clust[1][0])
            # print closest

            clust_ids = list(clust.keys())
            # loop through every pair looking for the smallest distance
            for i in range(len(clust_ids)):
                for j in range(i + 1, len(clust_ids)):

                    cluster1 = clust_ids[i]
                    cluster2 = clust_ids[j]

                    if (cluster1, cluster2) not in distances:
                        # distances is the cache of distance calculations
                        alli = clust[cluster1]
                        allj = clust[cluster2]
                        # print alli,allj
                        sim_val = 0.0
                        for ai in alli:
                            for aj in allj:
                                sim_val += distance(ai, aj)
                        sim_val = sim_val / (len(alli) * len(allj))

                        distances[(cluster1, cluster2)] = sim_val

                    d = distances[(cluster1, cluster2)]

                    if d > closest:
                        closest = d
                        lowestpair = [cluster1, cluster2]

            # Merge the two clusters -strings
            mergevec = clust[lowestpair[0]] + clust[lowestpair[1]]
            # print 'm', mergevec, closest, lowestpair

            # create the new cluster
            clust[currentclustid] = mergevec

            # cluster ids that weren't in the original set
            currentclustid += 1
            del clust[lowestpair[0]]
            del clust[lowestpair[1]]

        # print clust
        return clust

    # --------------------------------------------------------------------------

    def __insert_records__(self, clust, rec_dict, attr_select_list):

        dist = self.dist
        clusters = {}
        for cid in clust:
            clusters[cid] = []
        print('assign records into clusters')
        # print rec_dict

        num_rec_done = 0

        # Insert the records into the clusters
        #
        for (rec_id, rec_list) in rec_dict.items():
            num_rec_done += 1
            if (num_rec_done % 10000 == 0):
                print(num_rec_done, len(rec_dict))

            # Generate the BKV for this record
            #
            bk_val = ''
            for col_num in attr_select_list:
                bk_val += rec_list[col_num]
            # Calculate sim between this BKV and all ref values
            # and assign the record to the closest
            #
            max_sim = 0.0
            closest = 0
            for i in clust:
                this_clust = clust[i]
                this_clust_id = i

                for ref in this_clust:
                    sim_val = dist(bk_val, ref)
                    # print sim_val,ref,rec,max_sim
                    if sim_val > max_sim:
                        closest = this_clust_id
                        max_sim = sim_val

            clusters[closest] += [rec_id]

        avr_block_size = 0
        for c in list(clusters.values()):
            avr_block_size += len(c)
        avr_block_size /= len(clusters)

        # print clusters, avr_block_size
        return clusters, avr_block_size

    # --------------------------------------------------------------------------

    def __add_noise__(self, clusters, avr_blk_size):

        wn = self.wn
        ep = self.ep

        b = 2 / ep
        Ey = float(avr_blk_size)
        mu = -b * numpy.log(2 * Ey / (Ey + wn))
        u_list = []

        for c in clusters:
            u = int(numpy.random.laplace(mu, b))
            # print u
            u_list.append(u)
            for i in range(u):
                clusters[c].append('fake')

        return clusters, u_list

    # --------------------------------------------------------------------------

    def build_index_alice(self, attr_select_list, clust):
        """Build the index for Alice assuming the reference values have
       been generated.

       Argument:
       - attr_select_list  A list of column numbers that will be used to
                           extract attribute values from the given records,
                           and concatenate them into one string value which is
                           then used as reference value (and added to the
                           result list if it differs from all other reference
                           values).
    """

        start_time = time.time()

        self.attr_select_list_alice = attr_select_list

        assert self.rec_dict_alice != None

        self.index_alice, avr_blk_size = self.__insert_records__(clust, self.rec_dict_alice, \
                                                                 self.attr_select_list_alice)

        self.index_alice, u_list = self.__add_noise__(self.index_alice, avr_blk_size)

        alice_time = time.time() - start_time

        stat = self.block_stats(self.index_alice)
        min_block_size, med_blk_size, max_block_size, avr_block_size, std_dev, blk_len_list = stat

        wr_file_name = './logs/hclust_alice.csv'
        wr_file = open(wr_file_name, 'a')

        sum = 0.0
        for i in blk_len_list:
            sum = sum + (i - avr_block_size) * (i - avr_block_size)
            wr_file.write(str(i) + ',')
        wr_file.write(os.linesep)
        wr_file.close()

        return min_block_size, med_blk_size, max_block_size, avr_block_size, std_dev, alice_time, u_list

    # --------------------------------------------------------------------------

    def build_index_bob(self, attr_select_list, clust):
        """Build the index for Bob assuming the reference values have
       been generated.

       Argument:
       - attr_select_list  A list of column numbers that will be used to
                           extract attribute values from the given records,
                           and concatenate them into one string value which is
                           then used as reference value (and added to the
                           result list if it differs from all other reference
                           values).
    """

        start_time = time.time()

        self.attr_select_list_bob = attr_select_list

        assert self.rec_dict_bob != None

        self.index_bob, avr_blk_size = self.__insert_records__(clust, self.rec_dict_bob, \
                                                               self.attr_select_list_bob)

        self.index_bob, u_list = self.__add_noise__(self.index_bob, avr_blk_size)

        bob_time = time.time() - start_time

        stat = self.block_stats(self.index_bob)
        min_block_size, med_blk_size, max_block_size, avr_block_size, std_dev, blk_len_list = stat

        wr_file_name = './logs/hclust_bob.csv'
        wr_file = open(wr_file_name, 'a')

        sum = 0.0
        for i in blk_len_list:
            sum = sum + (i - avr_block_size) * (i - avr_block_size)
            wr_file.write(str(i) + ',')
        wr_file.write(os.linesep)
        wr_file.close()

        return min_block_size, med_blk_size, max_block_size, avr_block_size, std_dev, bob_time, u_list

    # --------------------------------------------------------------------------

    def generate_blocks(self):
        """Method which generates the blocks based on the built two index data
       structures.
    """

        block_dict = {}  # contains final candidate record pairs

        index_alice = self.index_alice
        index_bob = self.index_bob

        # print index_alice.keys()
        # print index_bob.keys()
        #
        cand_blk_key = 0
        for (block_id, block_vals) in index_alice.items():
            # assert len(block_vals) >= k, (block_id, len(block_vals))

            bob_block_vals = index_bob[block_id]
            bob_block_vals = [i for i in bob_block_vals if i != 'fake']

            alice_block_vals = [i for i in block_vals if i != 'fake']

            block_dict[cand_blk_key] = (alice_block_vals, bob_block_vals)

            cand_blk_key += 1

        self.block_dict = block_dict
        # print block_dict
        print('Final indexing contains %d blocks' % (len(block_dict)))

        return len(block_dict)
