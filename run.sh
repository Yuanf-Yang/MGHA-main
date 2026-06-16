python main_ml_dhd.py --method ml_hd --dataset cora  --seed 123 \
                      --rand_split_class --valid_num 500 --test_num 1000 --hidden_channels 128 --num_layers 1 --num_heads 4 \
                      --attn_dropout 0.8 --ff_dropout 0.8 --num_centroids 64 --num_centroids_list 32 8 --use_dynamic_gate  --gnum_layers 2 \
                      --lr 0.01 --weight_decay 0.0005 --dropout 0.8  --no_feat_norm --ghidden_channels 128

python main_ml_dhd.py --method ml_hd --dataset citeseer  --seed 123 --protocol semi \
                      --rand_split_class --valid_num 500 --test_num 1000 --hidden_channels 128 --num_layers 1 --num_heads 2 \
                      --attn_dropout 0.85 --ff_dropout 0.8 --num_centroids 64 --num_centroids_list 200 64 16 --use_dynamic_gate --gnum_layers 2 \
                      --lr 0.01 --weight_decay 0.01 --dropout 0.8  --no_feat_norm --ghidden_channels 128 

python main_ml_dhd.py  --method ml_hd --dataset pubmed  --seed 123  --protocol semi \
                       --rand_split_class  --valid_num 500 --test_num 1000  --hidden_channels 128 --num_layers 2 --num_heads 1 \
                       --attn_dropout 0.85 --ff_dropout 0.85 --num_centroids 64 --num_centroids_list 64 16 4 --use_dynamic_gate  --gnum_layers 2 \
                       --lr 0.01 --weight_decay 0.0005 --dropout 0.85  --no_feat_norm --ghidden_channels 128

python main_ml_dhd.py --method ml_hd --dataset film --seed 42 --epochs 1000\
                      --valid_num 500 --test_num 500 --hidden_channels 128 --num_layers 1 --num_heads 2 \
                      --attn_dropout 0.8 --ff_dropout 0.8 --num_centroids 64 --num_centroids_list 200 64 16 --use_dynamic_gate --gnum_layers 2 \
                      --lr 0.01 --weight_decay 0.008 --dropout 0.5 --ghidden_channels 128 --no_gnn  

python main_ml_dhd.py --method ml_hd --dataset squirrel --seed 42 \
                      --valid_num 500 --test_num 500 --hidden_channels 128 --num_layers 3 --num_heads 1 \
                      --attn_dropout 0.5 --ff_dropout 0.5 --num_centroids 64 --num_centroids_list 128 40 10 --use_dynamic_gate --gnum_layers 2 \
                      --lr 0.05 --weight_decay 0.0005 --dropout 0.5 --ghidden_channels 128 

python main_ml_dhd.py --method ml_hd --dataset chameleon --seed 42 \
                      --valid_num 500 --test_num 500 --hidden_channels 128 --num_layers 3 --num_heads 1 \
                      --attn_dropout 0.4 --ff_dropout 0.4 --num_centroids 64 --num_centroids_list 32 10  --use_dynamic_gate --gnum_layers 2 \
                      --lr 0.01 --weight_decay 0.001 --dropout 0.4 --ghidden_channels 128 




 