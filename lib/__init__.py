import os

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# base_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

config = {
    'logs': 'logs/'
}
datasets_path = {
    'air': 'datasets/air_quality',
    'la': 'datasets/metr_la',
    'bay': 'datasets/pems_bay',
    'synthetic': 'datasets/synthetic',
    'electrocity' : 'datasets/electrocity',
    'electrocity2' : 'datasets/electrocity2',
    'india' : 'datasets/india'
}
epsilon = 1e-8

for k, v in config.items():
    config[k] = os.path.join(base_dir, v)
for k, v in datasets_path.items():
    datasets_path[k] = os.path.join(base_dir, v)
