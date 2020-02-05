
#####################################################################

# The pairs of data sets to be used for testing (the first one will be loaded
# and processed by Alice, the second by Bob)
#

def experiment_data(mod_test_mode):
    """Return corresponding experiment dataset pairs."""
    if (mod_test_mode == 'ex'):
        data_sets_pairs = [['./datasets/example/10_25_overlap_no_mod_alice.csv',
                            './datasets/example/10_25_overlap_no_mod_bob.csv']]

    elif (mod_test_mode == 'no'):  # OZ No-mod
        data_sets_pairs = [
            # ['./datasets/4611_50_overlap_no_mod_alice.csv',
            #  './datasets/4611_50_overlap_no_mod_bob.csv'],

            ['./datasets/46116_50_overlap_no_mod_alice.csv',
             './datasets/46116_50_overlap_no_mod_bob.csv'],

            # ['./datasets/461167_50_overlap_no_mod_alice.csv',
            #  './datasets/461167_50_overlap_no_mod_bob.csv'],

            ['./datasets/4611676_50_overlap_no_mod_alice.csv',
             './datasets/4611676_50_overlap_no_mod_bob.csv'],

        ]

    elif (mod_test_mode == 'mod'):  # Data sets with modifications (for Bob) OZ Mod
        data_sets_pairs = [
            # ['./datasets/173_25_overlap_no_mod_alice.csv.gz',
            # './datasets/173_25_overlap_with_mod_bob_1.csv.gz'],
            # ['./datasets/173_50_overlap_no_mod_alice.csv.gz',
            # './datasets/173_50_overlap_with_mod_bob_1.csv.gz'],
            # ['./datasets/173_75_overlap_no_mod_alice.csv.gz',
            # './datasets/173_75_overlap_with_mod_bob_1.csv.gz'],

            # ['./datasets/1730_25_overlap_no_mod_alice.csv.gz',
            # './datasets/1730_25_overlap_with_mod_bob_1.csv.gz'],
            ['./datasets/1730_50_overlap_no_mod_alice.csv.gz',
             './datasets/1730_50_overlap_with_mod_bob_1.csv.gz']]
        # ['./datasets/1730_75_overlap_no_mod_alice.csv.gz',
        # './datasets/1730_75_overlap_with_mod_bob_1.csv.gz'],

        # ['./datasets/17294_25_overlap_no_mod_alice.csv.gz',
        # './datasets/17294_25_overlap_with_mod_bob_1.csv.gz'],
        # ['./datasets/17294_50_overlap_no_mod_alice.csv.gz',
        #  './datasets/17294_50_overlap_with_mod_bob_1.csv.gz']]#,
        # ['./datasets/17294_75_overlap_no_mod_alice.csv.gz',
        # './datasets/17294_75_overlap_with_mod_bob_1.csv.gz'],

        # ['./datasets/172938_25_overlap_no_mod_alice.csv.gz',
        # './datasets/172938_25_overlap_with_mod_bob_1.csv.gz'],
        # ['./datasets/172938_50_overlap_no_mod_alice.csv.gz',
        # './datasets/172938_50_overlap_with_mod_bob_1.csv.gz'],
        # ['./datasets/172938_75_overlap_no_mod_alice.csv.gz',
        # './datasets/172938_75_overlap_with_mod_bob_1.csv.gz']]


    elif (mod_test_mode == 'lno'):  # OZ largest dataset No-mod
        data_sets_pairs = [
            # ['./datasets/1729379_25_overlap_no_mod_alice.csv.gz',
            # './datasets/1729379_25_overlap_no_mod_bob.csv.gz'],
            ['./datasets/1729379_50_overlap_no_mod_alice.csv.gz',
             './datasets/1729379_50_overlap_no_mod_bob.csv.gz']]
        # ['./datasets/1729379_75_overlap_no_mod_alice.csv.gz',
        # './datasets/1729379_75_overlap_no_mod_bob.csv.gz']]

    elif (mod_test_mode == 'lmod'):  # OZ largest dataset mod
        data_sets_pairs = [
            # ['./datasets/1729379_25_overlap_no_mod_alice.csv.gz',
            # './datasets/1729379_25_overlap_with_mod_bob_1.csv.gz'],
            ['./datasets/1729379_50_overlap_no_mod_alice.csv.gz',
             './datasets/1729379_50_overlap_with_mod_bob_1.csv.gz']]
        # ['./datasets/1729379_75_overlap_no_mod_alice.csv.gz',
        # './datasets/1729379_75_overlap_with_mod_bob_1.csv.gz']]

    elif (mod_test_mode == 'nc'):  # NC dataset
        data_sets_pairs = [
            ['./datasets/ncvoter-temporal-1.csv',
             './datasets/ncvoter-temporal-2.csv']]

    elif (mod_test_mode == 'syn'):  # OZ Cor No-mod
        data_sets_pairs = [
            ['./datasets/4611_50_overlap_no_mod_alice.csv',
             './datasets/4611_50_overlap_no_mod_bob.csv'],
            ['./datasets/46116_50_overlap_no_mod_alice.csv',
             './datasets/46116_50_overlap_no_mod_bob.csv'],
            ['./datasets/461167_50_overlap_no_mod_alice.csv',
             './datasets/461167_50_overlap_no_mod_bob.csv']]  # ,
        # ['./datasets/4611676_50_overlap_no_mod_alice.csv',
        # './datasets/4611676_50_overlap_no_mod_bob.csv']]

    elif (mod_test_mode == 'syn_mod'):  # OZ Cor Light-mod, Med-mod, and Heavy-mod
        data_sets_pairs = [
            ['./datasets/4611_50_overlap_no_mod_alice.csv',
             './datasets/4611_50_overlap_with_mod_bob_1.csv'],
            ['./datasets/4611_50_overlap_no_mod_alice.csv',
             './datasets/4611_50_overlap_with_mod_bob_2.csv'],
            ['./datasets/4611_50_overlap_no_mod_alice.csv',
             './datasets/4611_50_overlap_with_mod_bob_4.csv'],
            ['./datasets/46116_50_overlap_no_mod_alice.csv',
             './datasets/46116_50_overlap_with_mod_bob_1.csv'],
            ['./datasets/46116_50_overlap_no_mod_alice.csv',
             './datasets/46116_50_overlap_with_mod_bob_2.csv'],
            ['./datasets/46116_50_overlap_no_mod_alice.csv',
             './datasets/46116_50_overlap_with_mod_bob_4.csv'],
            ['./datasets/461167_50_overlap_no_mod_alice.csv',
             './datasets/461167_50_overlap_with_mod_bob_1.csv'],
            ['./datasets/461167_50_overlap_no_mod_alice.csv',
             './datasets/461167_50_overlap_with_mod_bob_2.csv'],
            ['./datasets/461167_50_overlap_no_mod_alice.csv',
             './datasets/461167_50_overlap_with_mod_bob_4.csv']]  # ,
        # ['./datasets/4611676_50_overlap_no_mod_alice.csv',
        # './datasets/4611676_50_overlap_with_mod_bob_1.csv'],
        # ['./datasets/4611676_50_overlap_no_mod_alice.csv',
        # './datasets/4611676_50_overlap_with_mod_bob_2.csv'],
        # ['./datasets/4611676_50_overlap_no_mod_alice.csv',
        # './datasets/4611676_50_overlap_with_mod_bob_4.csv']]

    elif (mod_test_mode == 'nc_syn'):  # NC Cor No-mod
        data_sets_pairs = [
            ['./datasets/5488_50_overlap_no_mod_alice.csv',
             './datasets/5488_50_overlap_no_mod_bob.csv'],
            ['./datasets/54886_50_overlap_no_mod_alice.csv',
             './datasets/54886_50_overlap_no_mod_bob.csv'],
            ['./datasets/548860_50_overlap_no_mod_alice.csv',
             './datasets/548860_50_overlap_no_mod_bob.csv']]  # ,

    elif (mod_test_mode == 'nc_syn_mod'):  # NC Cor Light-mod, Med-mod, and Heavy-mod
        data_sets_pairs = [
            ['./datasets/5488_50_overlap_no_mod_alice.csv',
             './datasets/5488_50_overlap_with_mod_bob_1.csv'],
            ['./datasets/5488_50_overlap_no_mod_alice.csv',
             './datasets/5488_50_overlap_with_mod_bob_2.csv'],
            ['./datasets/5488_50_overlap_no_mod_alice.csv',
             './datasets/5488_50_overlap_with_mod_bob_4.csv'],
            ['./datasets/54886_50_overlap_no_mod_alice.csv',
             './datasets/54886_50_overlap_with_mod_bob_1.csv'],
            ['./datasets/54886_50_overlap_no_mod_alice.csv',
             './datasets/54886_50_overlap_with_mod_bob_2.csv'],
            ['./datasets/54886_50_overlap_no_mod_alice.csv',
             './datasets/54886_50_overlap_with_mod_bob_4.csv'],
            ['./datasets/548860_50_overlap_no_mod_alice.csv',
             './datasets/548860_50_overlap_with_mod_bob_1.csv'],
            ['./datasets/548860_50_overlap_no_mod_alice.csv',
             './datasets/548860_50_overlap_with_mod_bob_2.csv'],
            ['./datasets/548860_50_overlap_no_mod_alice.csv',
             './datasets/548860_50_overlap_with_mod_bob_4.csv']]  # ,

    return data_sets_pairs