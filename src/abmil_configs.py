import yaml
import pprint
parameters = {
    'Registry': {
        'project': "ABMIL_CRC",
        'name': "ABMIL_CRC",
        'wandb_root_dir': "src/wandb_outputs",
        'checkpoints_dir': "src/model_checkpoints",
    },
    'Data': {
        # 'mount_point': "/home/felipe/ExternalDrives"
        'mount_point': '/nas-ctm01/partners/IMPDIAGNOSTICS/cadpath/CRC',
        'encoder': 'Resnet34_ImageNet',
    },
    'Training': {
        'batch_size': 1,
        'max_epochs': 30,
        'limit_train_batches': 0.10,
        'early_stopping_patience': 5,
        'log_every_n_steps': 10,
    },
    'Optimizer': {
        'optimizer': 'adam',
        'weight_decay': 1e-5,
        'lr': 6e-6,
    },
    'Loss_Function': {
        'loss_function': 'cross_entropy',
    },
    'Model': {
        'embedding_dim': 512,
        'num_heads': 6,
        'dropout': 0.1,
   },
}

# set parameters and write to yaml file

pprint.pprint(parameters)

with open("abmil_parameters.yaml", 'w') as yaml_file:
    data = yaml.dump(parameters, yaml_file)