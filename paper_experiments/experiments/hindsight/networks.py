import torch
import torch.nn.functional as f


class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        hdim1 = 400
        hdim2 = 300
        self._h1 = torch.nn.Linear(state_dim, hdim1)
        self._h2 = torch.nn.Linear(hdim1, hdim2)

        self._mean_out = torch.nn.Linear(hdim2, action_dim)
        torch.nn.init.constant_(self._mean_out.bias, 0.)
        torch.nn.init.normal_(self._mean_out.weight, std=0.01)
        self._action_dim = action_dim

    def forward(self, states: torch.Tensor):
        x = f.relu(self._h1(states))
        x = f.relu(self._h2(x))
        return f.tanh(self._mean_out(x))


class QNetwork(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        hdim1 = 400
        hdim2 = 300
        self._h1 = torch.nn.Linear(state_dim + action_dim, hdim1)
        self._h2 = torch.nn.Linear(hdim1, hdim2)

        self._v_out = torch.nn.Linear(hdim2, 1)

    def forward(self, states: torch.Tensor, actions: torch.tensor):
        x = f.relu(self._h1(torch.cat((states, actions), dim=1)))
        x = f.relu(self._h2(x))
        return self._v_out(x)


