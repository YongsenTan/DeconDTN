# echo "'Gray':"
# python -u irm.py \
#   --l2_regularizer_weight=0.0001\
#   --lr=0.001 \
#   --penalty_anneal_iters=0 \
#   --penalty_weight=0.0 \
#   --steps=20 \
#   --grayscale=1

echo "ERM:"
python -u irm.py \
  --l2_regularizer_weight=0.0001 \
  --lr=0.001 \
  --penalty_anneal_iters=0 \
  --penalty_weight=0.0 \
  --steps=20 \
  --grayscale=1 \
  --n_restarts=5

echo "IRM:"
python -u irm.py \
  --l2_regularizer_weight=0.0001 \
  --lr=0.001 \
  --penalty_anneal_iters=20 \
  --penalty_weight=10000 \
  --steps=20 \
  --grayscale=1 \
  --n_restarts=5




