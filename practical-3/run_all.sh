SKIPGRAM=models/europarl.pkl
EMBED_ALIGN=models/ull-practical3-embedalign
K_FOLD=10

for comp_method in avg min max concat
do
    python evaluate_models.py results_$comp_method $comp_method --skipgram $SKIPGRAM --embed-align $EMBED_ALIGN --k-fold $K_FOLD
done
