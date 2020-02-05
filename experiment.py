import time
import numpy as np
import pandas as pd
from blocklib import generate_candidate_blocks, assess_blocks_2party, generate_blocks


def match(file_alice, file_bob, config):
    """Run record matching experiment with blocklib API."""
    num_recs_alice = int(file_alice.split('/')[-1].split('_')[0])
    num_recs_bob = int(file_bob.split('/')[-1].split('_')[0])

    print('Loading dataset Alice n={}'.format(num_recs_alice))
    alice_data = pd.read_csv(file_alice)
    alice_data = alice_data.replace(np.nan, '', regex=True)
    alice_data = alice_data.to_dict(orient='split')['data']

    print('Loading dataset Bob n={}'.format(num_recs_bob))
    bob_data = pd.read_csv(file_alice)
    bob_data = bob_data.replace(np.nan, '', regex=True)
    bob_data = bob_data.to_dict(orient='split')['data']
    print('Example data = {}'.format(alice_data[0]))

    # build candidate blocks
    start_time = time.time()
    print('Building reversed index of Alice')
    block_obj_alice = generate_candidate_blocks(alice_data, config)

    print('Building reversed index of Bob')
    block_obj_bob = generate_candidate_blocks(bob_data, config)
    dbo_time = time.time() - start_time

    # build final blocks
    start_time = time.time()
    print('Filtering reversed index - Generate final blocks')
    filtered_alice, filtered_bob = generate_blocks([block_obj_alice, block_obj_bob], K=2)
    lu_time = time.time() - start_time
    num_blocks = len(filtered_alice)

    # assess
    subdata1 = [x[0] for x in alice_data]
    subdata2 = [x[0] for x in bob_data]
    rr, pc = assess_blocks_2party([filtered_alice, filtered_bob], [subdata1, subdata2])

    return dbo_time, lu_time, num_blocks, rr, pc, num_recs_alice, block_obj_alice, block_obj_bob



def get_block_stats(block_obj):
    """Get block stats from block object."""
    # gather statistics
    a_min_blk = block_obj.state.stats['min_size']
    a_med_blk = block_obj.state.stats['med_size']
    a_max_blk = block_obj.state.stats['max_size']
    a_avg_blk = block_obj.state.stats['avg_size']
    a_std_dev = block_obj.state.stats['std_size']
    return a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev


def get_output(method_name, num_recs_alice, num_recs_bob, num_ref_val, K, rr, pc, pq, dbo_time, lu_time, assess_time,
                 block_alice_stats, block_bob_stats, num_blocks, num_cand_rec_pairs):
    """Write experiment output to dataframe."""
    a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev = block_alice_stats
    b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev = block_bob_stats
    result = dict(Method=method_name, alice_num_recs=num_recs_alice, bob_num_recs=num_recs_bob, num_ref_val=num_ref_val,
                  K=K, rr=rr, pc=pc, pq=pq, dbo_time=dbo_time, lu_time=lu_time, assess_time=assess_time,
                  tot_time=dbo_time + lu_time, a_min_blk=a_min_blk, a_med_blk=a_med_blk, a_max_blk=a_max_blk,
                  a_avg_blk=a_avg_blk, a_std_dev=a_std_dev, b_min_blk=b_min_blk, b_med_blk=b_med_blk, b_max_blk=b_max_blk,
                  b_avg_blk=b_avg_blk, b_std_dev=b_std_dev, num_blocks=num_blocks, num_cand_rec_pairs=num_cand_rec_pairs)

    result = {k: [v] for k, v in result.items()}
    df = pd.DataFrame.from_dict(result)
    df.columns = ['Method',
                  'alice_num_recs', 'bob_num_recs', 'num_ref_val', 'K',
                  'rr', 'pc', 'pq', 'dbo_time', 'lu_time', 'assess_time', 'tot_time',
                  'a_min_blk', 'a_med_blk', 'a_max_blk', 'a_avg_blk', 'a_std_dev',
                  'b_min_blk', 'b_med_blk', 'b_max_blk', 'b_avg_blk', 'b_std_dev',
                  'num_blocks', 'num_cand_rec_pairs']
    return df


def update_result(df, method_name, existing_file):
    """Update existing result dataframe."""
    ddf = pd.read_csv('{}_n={}.csv'.format(existing_file, num_recs_alice))
    ddf = ddf[ddf['Method'] != method_name]
    print(ddf)
    finaldf = pd.concat([ddf, df], axis=0)
    print(finaldf)
    finaldf.to_csv('result2_n={}.csv'.format(num_recs_alice), index=False)