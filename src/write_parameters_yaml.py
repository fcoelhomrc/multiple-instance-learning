import yaml
import pprint
parameters = {
    'Registry': {
        'project': "debug-runs",
        'name': "backbone",
        'wandb_root_dir': "src/wandb_outputs",
        'checkpoints_dir': "src/model_checkpoints",
    },
    'Data': {
        'mount_point': "/home/felipe/ExternalDrives"
        # '/nas-ctm01/partners/IMPDIAGNOSTICS/cadpath/CRC'
    },
    'Training': {
        'batch_size': 32,
        'max_epochs': 30,
        'epochs_before_unfreeze': 10,
        'gain_before_unfreeze': 0.1,
        'gain_after_unfreeze': 1.2,
        'limit_train_batches': 0.90,
        'early_stopping_patience': 5,
        'log_every_n_steps': 1,
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

# set parameters and write to yaml file

pprint.pprint(parameters)

with open("parameters.yaml", 'w') as yaml_file:
    data = yaml.dump(parameters, yaml_file)