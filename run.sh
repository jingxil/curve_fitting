mkdir ./models
rm ./models/*

python preprocess.py -data_path data.txt
python trainer.py -train_data data.txt.train.pt -valid_data data.txt.valid.pt -batch_size 2
model=$(ls ./models | sort | head -n 1)
python estimator.py -model models/$model -data data.txt.test.pt
