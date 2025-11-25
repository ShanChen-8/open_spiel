"""PyTorch-friendly environment wrappers with non-cooperative constraints.

These wrappers mirror :class:`open_spiel.python.rl_environment.Environment` but
add conveniences for PyTorch agents:

* Observations and legal action masks are returned as :class:`torch.Tensor`.
* The action space is padded/truncated to ``ACTION_SPACE_SIZE`` (50) to
  simplify model heads.
* Team identity for jammer/radar style roles is encoded in the outputs.
* A helper enforces non-cooperative training by checking that policy and value
  networks do not share parameters or gradients across teams.

The wrappers expose single-environment and batched synchronous APIs so they can
feed PPO/DQN style learners without additional glue code.
"""

import dataclasses
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from open_spiel.python import rl_environment


ACTION_SPACE_SIZE = 50


@dataclasses.dataclass
class TorchTimeStep:
  """Torch flavored container mirroring :class:`rl_environment.TimeStep`.

  Attributes:
    observations: ``(num_players, obs_dim)`` float tensor.
    legal_actions_mask: ``(num_players, ACTION_SPACE_SIZE)`` float tensor.
    rewards: ``(num_players,)`` float tensor.
    discounts: ``(num_players,)`` float tensor.
    step_type: :class:`rl_environment.StepType` value.
    team_ids: ``(num_players,)`` long tensor encoding jammer/radar style roles.
    team_names: human readable labels aligned with ``team_ids``.
  """

  observations: torch.Tensor
  legal_actions_mask: torch.Tensor
  rewards: torch.Tensor
  discounts: torch.Tensor
  step_type: rl_environment.StepType
  team_ids: torch.Tensor
  team_names: Sequence[str]

  def first(self) -> bool:
    return self.step_type == rl_environment.StepType.FIRST

  def mid(self) -> bool:
    return self.step_type == rl_environment.StepType.MID

  def last(self) -> bool:
    return self.step_type == rl_environment.StepType.LAST

  def current_player(self) -> int:
    return int(self.team_ids[0].item())


def _legal_mask_from_actions(actions: Iterable[int], device: torch.device) -> torch.Tensor:
  mask = torch.zeros(ACTION_SPACE_SIZE, device=device)
  for action in actions:
    if action >= ACTION_SPACE_SIZE:
      raise ValueError(
          f"Action id {action} exceeds enforced action space {ACTION_SPACE_SIZE}"
      )
    mask[action] = 1.0
  return mask


def _stack_legal_masks(legal_actions: Sequence[Sequence[int]], device: torch.device
                      ) -> torch.Tensor:
  masks = [_legal_mask_from_actions(actions, device) for actions in legal_actions]
  return torch.stack(masks, dim=0)


def validate_non_cooperative(
    policy_modules: Mapping[str, torch.nn.Module],
    value_modules: Mapping[str, torch.nn.Module]) -> None:
  """Checks that policy/value modules do not share parameters across teams.

  The helper raises ``ValueError`` when it detects reused parameter objects to
  guard against unintended gradient sharing between jammer and radar roles. This
  makes it safe to plug independent networks into PPO/DQN style learners while
  still keeping the enforcement close to the environment adapter.
  """

  def _collect_ids(modules: Mapping[str, torch.nn.Module]) -> Dict[int, str]:
    param_to_owner: Dict[int, str] = {}
    for owner, module in modules.items():
      for param in module.parameters():
        param_id = id(param)
        if param_id in param_to_owner:
          raise ValueError(
              f"Module '{owner}' reuses parameters already registered to "
              f"'{param_to_owner[param_id]}'. Non-cooperative constraint violated.")
        param_to_owner[param_id] = owner
    return param_to_owner

  policy_param_ids = _collect_ids(policy_modules)
  for owner, module in value_modules.items():
    for param in module.parameters():
      param_id = id(param)
      if param_id in policy_param_ids:
        raise ValueError(
            f"Value module '{owner}' shares parameters with policy module "
            f"'{policy_param_ids[param_id]}'. Non-cooperative constraint violated.")


class TorchEnvWrapper:
  """Single-environment wrapper returning PyTorch tensors and team metadata."""

  def __init__(self,
               game: str,
               team_names: Optional[Sequence[str]] = None,
               device: Optional[torch.device] = None,
               **kwargs):
    self._device = device or torch.device("cpu")
    self._env = rl_environment.Environment(game, **kwargs)
    self._team_names = list(team_names) if team_names else ["jammer", "radar"]
    if len(self._team_names) != self._env.num_players:
      raise ValueError(
          "Number of team labels must match the number of players in the game.")
    self._team_ids = torch.arange(self._env.num_players, device=self._device)

  @property
  def num_players(self) -> int:
    return self._env.num_players

  @property
  def observation_spec(self):
    return self._env.observation_spec()

  def reset(self) -> TorchTimeStep:
    return self._convert(self._env.reset())

  def step(self, actions: Sequence[int]) -> TorchTimeStep:
    return self._convert(self._env.step(actions))

  def _convert(self, time_step: rl_environment.TimeStep) -> TorchTimeStep:
    info_state = torch.tensor(time_step.observations["info_state"],
                              device=self._device,
                              dtype=torch.float32)
    legal_actions = time_step.observations["legal_actions"]
    legal_masks = _stack_legal_masks(legal_actions, self._device)
    rewards = torch.tensor(time_step.rewards or [0.0] * self.num_players,
                           device=self._device,
                           dtype=torch.float32)
    discounts = torch.tensor(time_step.discounts or [0.0] * self.num_players,
                             device=self._device,
                             dtype=torch.float32)
    return TorchTimeStep(
        observations=info_state,
        legal_actions_mask=legal_masks,
        rewards=rewards,
        discounts=discounts,
        step_type=time_step.step_type,
        team_ids=self._team_ids,
        team_names=self._team_names)


class TorchSyncVectorEnv:
  """Synchronous vectorized environment emitting stacked Torch tensors."""

  def __init__(self,
               game: str,
               num_envs: int,
               team_names: Optional[Sequence[str]] = None,
               device: Optional[torch.device] = None,
               **kwargs):
    self._device = device or torch.device("cpu")
    self._envs = [
        TorchEnvWrapper(game, team_names=team_names, device=self._device, **kwargs)
        for _ in range(num_envs)
    ]

  @property
  def num_envs(self) -> int:
    return len(self._envs)

  @property
  def num_players(self) -> int:
    return self._envs[0].num_players

  def reset(self) -> TorchTimeStep:
    steps = [env.reset() for env in self._envs]
    return self._stack(steps)

  def step(self, actions: Sequence[Sequence[int]]) -> TorchTimeStep:
    if len(actions) != self.num_envs:
      raise ValueError("Actions must be provided per environment instance.")
    steps = [self._envs[i].step(actions[i]) for i in range(self.num_envs)]
    return self._stack(steps)

  def _stack(self, steps: Sequence[TorchTimeStep]) -> TorchTimeStep:
    return TorchTimeStep(
        observations=torch.stack([s.observations for s in steps], dim=0),
        legal_actions_mask=torch.stack([s.legal_actions_mask for s in steps],
                                       dim=0),
        rewards=torch.stack([s.rewards for s in steps], dim=0),
        discounts=torch.stack([s.discounts for s in steps], dim=0),
        step_type=steps[0].step_type,
        team_ids=torch.stack([s.team_ids for s in steps], dim=0),
        team_names=steps[0].team_names)
