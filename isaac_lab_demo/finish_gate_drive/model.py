import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space


# define the shared model
class CNNBackboneSharedModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction, role="policy")
        DeterministicMixin.__init__(self, clip_actions, role="value")

        # shared CNN backbone
        self.backbone = nn.Sequential(
            nn.LazyConv2d(16, kernel_size=8, stride=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        # separated layers ("policy")
        self.policy_layer = nn.Sequential(
            nn.LazyLinear(32),
            nn.ELU(),
            nn.LazyLinear(self.num_actions)
        )
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)

        # separated layer ("value")
        self.value_layer = nn.Sequential(
            nn.LazyLinear(32),
            nn.ELU(),
            nn.Linear(32, 1)
        )

    # override the .act(...) method to disambiguate its call
    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)
    
    # forward the input to compute model output according to the specified role
    def compute(self, inputs, role=""):
        # inputs: {'states': (num_envs, WHC)}
        
        if role == "policy":
            # save shared layers/network output to perform a single forward-pass
            states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
            # taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
            features = self.backbone(torch.permute(states, (0, 3, 1, 2)))
            self._shared_output = features
            output = self.policy_layer(features)
            return output, self.log_std_parameter, {}
        
        elif role == "value":
            # use saved shared layers/network output to perform a single forward-pass, if it was saved
            if self._shared_output is None:
                states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
                # taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
                features = self.backbone(torch.permute(states, (0, 3, 1, 2)))
                shared_output = features
            else:
                shared_output = self._shared_output
            self._shared_output = None
            output = self.value_layer(shared_output)
            return output, {}
