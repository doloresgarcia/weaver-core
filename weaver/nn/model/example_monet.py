import torch
from weaver.nn.model.mo_net import MoNet


class MoNetWrapper(torch.nn.Module):
    def __init__(self, dev, **kwargs) -> None:
        super().__init__()
        self.mod = MoNet(dev)

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(points, features, lorentz_vectors, mask)


def get_model(data_config, dev,  **kwargs):
    

    #pf_features_dims = len(data_config.input_dicts['pf_features'])
    #num_classes = len(data_config.label_value)
    model = MoNetWrapper(dev
    )

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()