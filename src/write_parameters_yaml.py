import yaml
import pprint

# set parameters and write to yaml file
parameters = {
    'Training': {
        'limit_train_batches': 0.10,
        'max_epochs': 10,
        'batch_size': 32,
        'early_stopping_patience': 2,
        'log_every_n_steps': 25,
    },
    'Optimizer': {
        'optimizer': 'adam',
        'weight_decay': 0.9,
        'lr': 6e-6,
    },
    'Loss_Function': {
        'loss_function': 'qwk',
    },
    'Model': {
        'backbone': 'resnet34',
        'features': 512,
        'outputs': 3,
   },
}

pprint.pprint(parameters)

with open("parameters.yaml", 'w') as yaml_file:
    data = yaml.dump(parameters, yaml_file)