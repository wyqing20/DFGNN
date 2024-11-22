# DFGNN
#### step1:  go to the dataset folder, and run the merge ipynb to generate the train.csv  now all dataset are available
#### step2: run the code

example: 
For recommendation task
python main.py --model_name HLGCNCLRec --emb_size 64 --n_layers 2  --lr 1e-3  --l2 1e-6   --dataset Kindle --weight 1.0 --temp 1.0 --gpu 6 --batch_size 512 --eval_batch_size 1024 
For Feedback Type Recognition Task

python main.py --model_name HLGCNCLReg --emb_size 64 --n_layers 2  --lr 1e-4  --l2 1e-6  --dataset Kindle --weight 1.0 --temp 0.03 --gpu 6 --batch_size 512 --eval_batch_size 1024

The hyper-parameter of all datasets is 
RS Task weight and  temp lr: 

Kindle 1.0 1.0 0.001
GFood 1.0 0.05  0.0001
Yelp  1.0  0.3 0.001
ML1M: 1.0 1.0 0.001
Arts: 1.0 1.0 0.001

FTR weight and  temp task: 
Kindle 1.0 0.03 0.0001
GFood 1.0 1.0 0.0001
Yelp  0.1 0.05 0.0001
ML1M: 0.5 1.0 0.0001
Arts: 3.0 0.05 0.0001
