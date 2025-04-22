# echo "'Gray':"
# python -u irm.py \
#   --l2_regularizer_weight=0.0001\
#   --lr=0.001 \
#   --penalty_anneal_iters=0 \
#   --penalty_weight=0.0 \
#   --steps=20 \
#   --grayscale=1

echo "ERM + SentenceBERT:"
python -u irm.py \
  --l2_regularizer_weight=1e-5 \
  --lr=0.005 \
  --penalty_anneal_iters=0 \
  --penalty_weight=0.0 \
  --steps=20 \
  --grayscale=1 \
  --n_restarts=5

echo "IRM + SentenceBERT:"
python -u irm.py \
  --l2_regularizer_weight=1e-5 \
  --lr=0.005 \
  --penalty_anneal_iters=5 \
  --penalty_weight=10000 \
  --steps=20 \
  --grayscale=1 \
  --n_restarts=5 

echo "ERM + RoBERTa:"
python -u irm.py \
  --l2_regularizer_weight=1e-5 \
  --lr=0.00001 \
  --penalty_anneal_iters=0 \
  --penalty_weight=0.0 \
  --steps=10 \
  --grayscale=1 \
  --n_restarts=5 \
  --model='bert'

echo "IRM + RoBERTa:"
python -u irm.py \
  --l2_regularizer_weight=1e-5 \
  --lr=0.00001 \
  --penalty_anneal_iters=1 \
  --penalty_weight=10000 \
  --steps=10 \
  --grayscale=1 \
  --n_restarts=5 \
  --model='bert'



