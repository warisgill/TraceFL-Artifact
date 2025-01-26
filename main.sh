key_start='image_classification_exp'
num_clients=100
clients_per_round=10
dirichlet_alpha="0.1"


echo "      ****************** Image Classification Experiments ******************"
img_models="resnet18"
img_datasets="mnist"
num_rounds=10
python -m tracefl.main --multirun exp_key=$key_start model.name=$img_models dataset.name=$img_datasets num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=$dirichlet_alpha | tee -a logs/exp_$key_start.log
