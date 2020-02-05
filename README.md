# pprlblock

Python routines for private indexing/blocking for PPRL (Privacy Preserving Record Linkage)

### Depdencency

You can install the following dependencies with 

`pip install -r requirements.txt`

```text
bitarray==1.2.0
blocklib==0.0.3
cycler==0.10.0
decorator==4.4.0
Fuzzy==1.2.2
jedi==0.15.1
json5==0.8.5
jsonschema==3.1.1
matplotlib==3.1.1
memory-profiler==0.55.0
notebook==6.0.1
numpy==1.17.2
pandas==0.25.1
scipy==1.3.1
tqdm==4.36.1
```

### Main Script
The main comparison scrips is in `comp_pb.py`. Firstly specify your datasets in 
`get_experiment_data.py`. Then run

```python comp_pb.py no```
where no means no modification.

### Six private blocking/indexing methods:
1. Three-party k-nearest neighbour based clustering (Kar12 kNN)
  * Reference table based k-anonymous private blocking
     A Karakasidis, VS Verykios
     27th Annual ACM Symposium on Applied Computing, 859-864, 2012.
     
2. Three-party Sorted neighbourhood clustering - SIM based merging (Vat13PAKDD - SNC3PSim)
  *  Blocks are generated by sorting reference values, and inserting the
     values from the databases into an inverted index where reference values
     are the keys. Blocks are then formed that each contain at least k
     records.
   
3. Three-party Bloom filter Locality Sensitive hashing based blocking (Dur12 - HLSH)
  *  A Framework for Accurate, Efficient Private Record Linkage
     by Elizabeth Durham
     PhD thesis, Faculty of the Graduate School of Vanderbilt University, 2012
     
4. Two-party Sorted neighbourhood clustering (Vat13CIKM - SNC2P)
  *  Blocks are generated by sorting random reference values, and inserting the
     values from the databases into an inverted index where reference values
     are the keys. Blocks are then formed such that each contains at least k
     records. Database owners exchange the reference values at least one from
     each cluster which are sorted over which a slab of width w is moved to
     find the clusters of Alice and Bob to be merged to generate candidate
     record pairs.

5. Two-party hclustering based blocking (Kuz13 - HCLUST)
