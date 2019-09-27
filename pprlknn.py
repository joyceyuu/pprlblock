import os
import math

from pprlindex import PPRLIndex


class PPRLIndexKAnonymousNearestNeighbourClustering(PPRLIndex):
  """Class that implements the PPRL indexing technique:

     Reference table based k-anonymous private blocking
     A Karakasidis, VS Verykios
     27th Annual ACM Symposium on Applied Computing, 859-864, 2012.

     Each generated block contains at least k records.

     The class includes an implementation of a nearest neighbour clustering
     algorithm.
  """

  # --------------------------------------------------------------------------

  def __init__(self, k, sim_measure, min_sim_threshold, use_medoids=False):
    """Initialise the class and set the required parameters.

       Arguments:
       - k                  The minimum block size (in number of records that
                            need to be in a block).
       - sim_measure        A function which takes two strings as input and
                            returns a similarity value between 0 and 1.
       - min_sim_threshold  A similarity threshold between 0 and 1 which is
                            used to decide if a value is to be added to an
                            existing cluster (if a similarity value is equal
                            to or larger than this minimum threshold, or if a
                            new cluster is to be generated (if the similarity
                            is below this threshold).
       - use_medoids        A variation of this approach, rather than using
                            all the clustered reference values in the
                            clustering of attribute values only use the
                            medoids from the reference clusters (reference
                            values in the cluster centers).
                            The default is not to use the medoids but the
                            complete clusters.
    """

    self.k =                 k
    self.sim_measure =       sim_measure
    self.min_sim_threshold = min_sim_threshold

    assert use_medoids in [True, False]
    self.use_medoids =     use_medoids
    self.cluster_medoids = None

    self.ref_val_list = None
    self.clusters =     None

  # --------------------------------------------------------------------------

  def __nn_clustering__(self):
    """A method which applies nearest neighbour clustering (Durham 2003,
       Algorithm 5.7, page 142) to the list of reference values using a given
       similarity measure and similarity threshold.

       Values that have a similarity equal to or larger than the given
       threshold will be inserted into the same cluster.

       The method stores a list self.clusters where each element of the list
       is a list that contains the values in a cluster.
    """

    assert self.ref_val_list != None

    val_list =           self.ref_val_list
    sim_measure =        self.sim_measure
    min_sim_threshold =  self.min_sim_threshold

    # Initialise first cluster as the first value
    #
    clusters = [[val_list[0]]]

    # Main loop over all other values in the given list
    #
    for val in val_list[1:]:

      max_sim_cluster_id = -1   # Cluster number with highest similarity
      max_sim =            0.0  # Highest similarity value

      for c in range(len(clusters)):  # Loop over all existing clusters
        this_cluster_value_list = clusters[c]

        for cluster_val in this_cluster_value_list:
          s = sim_measure(val, cluster_val, True)  # Cache q-grams and result
          if (s > max_sim):
            max_sim =            s  # New highest similarity found
            max_sim_cluster_id = c  # Assign new best cluster number

      if (s >= min_sim_threshold):  # Add value into an exisiting cluster
        best_cluster_val_list = clusters[max_sim_cluster_id]
        best_cluster_val_list.append(val)
        clusters[max_sim_cluster_id] = best_cluster_val_list

      else:  # Form new cluster
        clusters.append([val])

    print('Found %d clusters' % (len(clusters)))

    self.clusters = clusters
    del self.ref_val_list

  # --------------------------------------------------------------------------

  def __get_cluster_centers__(self):
    """A method which for each cluster finds the most central element (string)
       according to the similarity measure that was used to generate the
       clusters.

       A cluster center is defined as the string that is furthest away from
       all strings in other clusters.

       The method returns a dictionary which for each cluster contains its
       central string.
    """

    assert self.clusters != None

    clusters =    self.clusters
    sim_measure = self.sim_measure

    cluster_centers = []

    for cluster_elem_list in clusters:

      # A dictionary which for each element in this cluster will contain the
      # maximum similarity to any element in an other cluster
      #
      cluster_elem_max_sim_dict = {}

      for cluster_val in cluster_elem_list:  # All values in this cluster

        cluster_val_max_sim = 0.0  # Maximum similarity for this element
                                   # with any element in another cluster

        # Check all other clusters, not the current cluster itself
        #
        for other_cluster_elem_list in clusters:
          if (other_cluster_elem_list != cluster_elem_list):

            for other_cluster_val in other_cluster_elem_list:
              s = sim_measure(cluster_val, other_cluster_val, True)
              cluster_val_max_sim = max(cluster_val_max_sim, s)

        cluster_elem_max_sim_dict[cluster_val] = cluster_val_max_sim

      #print 'cluster:'
      #print cluster_elem_list
      #print cluster_elem_max_sim_dict

      min_sim =  1.0
      min_elem = ''
      for (cluster_elem, s) in cluster_elem_max_sim_dict.items():
        if (s < min_sim):
          min_sim =  s
          min_elem = cluster_elem

      assert min_elem in cluster_elem_list

      #print '  center:', min_elem
      #print

      cluster_centers.append(min_elem)

    return cluster_centers
    del self.clusters

  # --------------------------------------------------------------------------

  def __get_cluster_medoids__(self):
    """A method which for each cluster finds the most central element
       (string), called the medoid, according to the similarity measure that
       was used to generate the clusters.

       A medoid is the element that has the highest average similarity to all
       other elements in a cluster.

       The method returns a dictionary which for each cluster contains its
       medoid (string).
    """

    assert self.clusters != None

    clusters =    self.clusters
    sim_measure = self.sim_measure

    cluster_medoids = []

    # Process each cluster individually
    #
    for cluster_elem_list in clusters:

      # A list which for each element in this cluster will contain the sum of
      # similarities to all other elements in the cluster
      #
      cluster_elem_sim_sum_list = []

      for cluster_val in cluster_elem_list:  # All values in this cluster

        cluster_val_sim_sum = -1.0  # Take similarity of element with it self
                                    # into account

        for other_cluster_val in cluster_elem_list:
          s = sim_measure(cluster_val, other_cluster_val, False)
          cluster_val_sim_sum += s

        cluster_elem_sim_sum_list.append([cluster_val_sim_sum,cluster_val])

      cluster_medoid_data = max(cluster_elem_sim_sum_list)
      #print
      #print cluster_elem_sim_sum_list
      #print cluster_medoid_data

      cluster_medoids.append(cluster_medoid_data[1])  # Keep medoid element

    return cluster_medoids
    del self.clusters

  # --------------------------------------------------------------------------

  def __generate_data_set_blocks__(self, rec_dict, attr_select_list):
    """Generate the blocks for the given record dictionary. Each record (its
       record identifier) is inserted into its closest cluster.

       Arguments:
       - rec_dict          A dictionary containing records, with the the keys
                           being record identifiers and values being a list of
                           attribute values for each record.
       - attr_select_list  A list of column numbers that will be used to
                           extract attribute values from the given records,
                           and concatenate them into one string value which is
                           then used as reference value (and added to the
                           result list if it differs from all other reference
                           values).

       The method returns a dictionary which contains the generated blocks.
    """

    print()
    print('Generate basic blocks for data set:')

    assert rec_dict != None

    sim_measure =       self.sim_measure
    min_sim_threshold = self.min_sim_threshold
    clusters =          self.clusters

    if (self.use_medoids == True):
      cluster_medoids = self.cluster_medoids
      assert len(clusters) == len(cluster_medoids)

    block_dict = {}  # Resulting blocks generated

    num_rec_done = 0

    for (rec_id, rec_list) in rec_dict.items():
      num_rec_done += 1
      if (num_rec_done % 10000 == 0):
        print('  Processed %d of %d records' % (num_rec_done, len(rec_dict)))

      # Generate the blocking key value for this record
      #
      bk_val = ''
      for col_num in attr_select_list:
        bk_val += rec_list[col_num]

      max_sim_cluster_id = -1   # Cluster number with highest similarity
      max_sim =            0.0  # Highest similarity value

      # Compare value with all medoids
      #
      if (self.use_medoids == True):
        for c in range(len(clusters)):
          medoid_val = cluster_medoids[c]
          s = sim_measure(bk_val, medoid_val)  # Don't cache anything
          if (s > max_sim):
            max_sim =            s  # New highest similarity found
            max_sim_cluster_id = c  # Assign new best cluster number

      else:  # Compare with all reference values in all clusters

        for c in range(len(clusters)):  # Loop over all existing clusters
          this_cluster_value_list = clusters[c]

          for cluster_val in this_cluster_value_list:
            s = sim_measure(bk_val, cluster_val)  # Don't cache anything
            if (s > max_sim):
              max_sim =            s  # New highest similarity found
              max_sim_cluster_id = c  # Assign new best cluster number

      # Add record into cluster with highest similarity
      #
      best_block_rec_id_list = block_dict.get(max_sim_cluster_id, [])
      best_block_rec_id_list.append(rec_id)
      block_dict[max_sim_cluster_id] = best_block_rec_id_list

    return block_dict

  # --------------------------------------------------------------------------

  def build_index_alice(self, attr_select_list):
    """Build the index for Alice assuming clusters of reference values have
       been generated.

       Argument:
       - attr_select_list  A list of column numbers that will be used to
                           extract attribute values from the given records,
                           and concatenate them into one string value which is
                           then used as reference value (and added to the
                           result list if it differs from all other reference
                           values).
    """

    self.attr_select_list_alice = attr_select_list

    if (self.clusters == None):  # Only needs to be done once (same clusters
      self.__nn_clustering__()   # for both database owners)

      if (self.use_medoids == True):
        self.cluster_medoids = self.__get_cluster_medoids__()

    assert self.rec_dict_alice != None

    self.index_alice = self.__generate_data_set_blocks__(self.rec_dict_alice,
                                                         attr_select_list)
    print('Index for Alice contains %d blocks' % (len(self.index_alice)))


    stat = self.block_stats(self.index_alice)
    min_block_size,med_blk_size,max_block_size,avr_block_size,std_dev,blk_len_list = stat

    wr_file_name = './logs/kNN_data_alice.csv'
    wr_file = open(wr_file_name, 'a')

    sum = 0.0
    for i in blk_len_list:
      sum = sum + (i - avr_block_size)*(i - avr_block_size)
      wr_file.write(str(i)+',')
    wr_file.write(os.linesep)
    wr_file.close()

    return min_block_size,med_blk_size,max_block_size,avr_block_size,std_dev

  # --------------------------------------------------------------------------

  def build_index_bob(self, attr_select_list):
    """Build the index for Bob assuming clusters of reference values have
       been generated.

       Argument:
       - attr_select_list  A list of column numbers that will be used to
                           extract attribute values from the given records,
                           and concatenate them into one string value which is
                           then used as reference value (and added to the
                           result list if it differs from all other reference
                           values).
    """

    self.attr_select_list_bob = attr_select_list

    if (self.clusters == None):  # Only needs to be done once (same clusters
      self.__nn_clustering__()   # for both database owners)

    assert self.rec_dict_bob != None

    self.index_bob = self.__generate_data_set_blocks__(self.rec_dict_bob,
                                                       attr_select_list)
    print('Index for Bob contains %d blocks' % (len(self.index_bob)))

    stat = self.block_stats(self.index_bob)
    min_block_size,med_blk_size,max_block_size,avr_block_size,std_dev,blk_len_list = stat

    wr_file_name = './logs/kNN_data_bob.csv'
    wr_file = open(wr_file_name, 'a')

    sum = 0.0
    for i in blk_len_list:
      sum = sum + (i - avr_block_size)*(i - avr_block_size)
      wr_file.write(str(i)+',')
    wr_file.write(os.linesep)
    wr_file.close()
    
    return min_block_size,med_blk_size,max_block_size,avr_block_size,std_dev

  # --------------------------------------------------------------------------

  def generate_blocks(self):
   """Method which generates the blocks based on the built two index data
      structures.
   """

   block_dict = {}
   num_cand_rec_pairs = 0

   blocks_alice = self.index_alice  # Keys are cluster identifiers, values the
   blocks_bob =   self.index_bob    # corresponding lists of record
                                    # identifiers

   blocks_alice_list = list(blocks_alice.keys())

   block_num = 0  # Each block get a unique number

   for cluster_id in blocks_alice_list:
     if (cluster_id in blocks_bob):

       # Get list of record identifiers in this block
       #
       block_rec_list_alice = blocks_alice[cluster_id]
       block_rec_list_bob =   blocks_bob[cluster_id]

       if len(block_rec_list_alice) >= self.k and \
          len(block_rec_list_bob)>= self.k:

         block_dict[block_num] = (block_rec_list_alice, block_rec_list_bob)

         num_cand_rec_pairs += len(block_rec_list_alice) * \
                               len(block_rec_list_bob)
         block_num += 1

   self.block_dict = block_dict

   print('Final indexing contains %d blocks' % (len(block_dict)))

   return len(block_dict)
