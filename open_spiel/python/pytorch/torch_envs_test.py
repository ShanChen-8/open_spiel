"""Tests for torch_envs wrappers."""

import pytest

torch = pytest.importorskip("torch")

from open_spiel.python.pytorch import torch_envs


def test_reset_returns_tensors():
  env = torch_envs.TorchEnvWrapper("tic_tac_toe")
  ts = env.reset()
  assert isinstance(ts.observations, torch.Tensor)
  assert ts.legal_actions_mask.shape[-1] == torch_envs.ACTION_SPACE_SIZE
  assert ts.team_names == ["jammer", "radar"]


def test_vectorized_stack_and_step():
  venv = torch_envs.TorchSyncVectorEnv("tic_tac_toe", num_envs=2)
  ts = venv.reset()
  assert ts.observations.shape[0] == 2
  actions = [[int(ts.legal_actions_mask[0, 0].nonzero()[0])],
             [int(ts.legal_actions_mask[1, 0].nonzero()[0])]]
  ts_next = venv.step(actions)
  assert ts_next.rewards.shape[0] == 2


def test_non_cooperative_validator_raises_on_shared_params():
  class Shared(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
      return self.linear(x)

  shared = Shared()
  jammer_policy = torch.nn.Linear(4, 2)
  radar_policy = torch.nn.Linear(4, 2)
  jammer_value = shared
  radar_value = shared

  try:
    torch_envs.validate_non_cooperative(
        {"jammer_policy": jammer_policy, "radar_policy": radar_policy},
        {"jammer_value": jammer_value, "radar_value": radar_value})
  except ValueError as exc:
    assert "Non-cooperative constraint" in str(exc)
  else:  # pragma: no cover
    raise AssertionError("Expected a ValueError for shared parameters")


def test_action_space_guard():
  env = torch_envs.TorchEnvWrapper("tic_tac_toe")
  ts = env.reset()
  illegal_action = [[torch_envs.ACTION_SPACE_SIZE]]
  try:
    env.step(illegal_action[0])
  except ValueError as exc:
    assert "exceeds enforced action space" in str(exc)
  else:  # pragma: no cover
    raise AssertionError("Expected a ValueError for action overflow")

