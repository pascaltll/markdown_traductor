# Введение
Пусть $\pi = \pi (s, a; w)$ - аппроксимация стратегии с параметрами $w$. Тогда градиент стратеции имеет вид (см. лекции **5. Градиент стратегии**, $d ^ {\pi}$ - траектории, сэмплируемые стратегией $\pi$):
$$ \nabla_w J(w) = \mathop{\mathbb{E}}_{s \sim d ^ {\pi}, \, a \sim \pi} \left[ Q ^ {\pi} (s, a) \nabla_w \log{\pi(a | s; w)}\right]$$

Наиболее распространённым подходом в on-policy алгоритмах семейства актор-критик является оценка градиента $J$ (или суррогатного функционала, напр. в PPO) по состояниям и действиям, сэмплированным из траектории, полученной выполнением текущей стратегии в среде (см. семинар Актор-критик).

В данной работе предлагается реализовать альтернативный вариант обучения на основе градиента стратегии, применимый в средах с конечным дискретным множеством действий. Обозначим $\mathcal{A} = \{a \}$ - дискретное множеством действий. Тогда мы можем переписать выражение следующим образом ($\hat{Q}(s, a; u)$ - аппроксиматор $Q$ с параметрами $u$):
$$\begin{align}
\nabla_w J(w) &= \mathop{\mathbb{E}}_{s \sim d ^ {\pi}, \, a \sim \pi} \left[ Q ^ {\pi} (s, a) \nabla_w \log{\pi(a | s; w)}\right] = \mathop{\mathbb{E}}_{s \sim d ^ {\pi}} \left[ \sum_{a \in \mathcal{A}} \pi(a | s; w) Q ^ {\pi} (s, a) \nabla_w \log{\pi(a | s; w)}\right] = \mathop{\mathbb{E}}_{s \sim d ^ {\pi}} \left[ \sum_{a \in \mathcal{A}} \nabla_w \pi(a | s; w) Q ^ {\pi} (s, a) \right] \approx \\ &\approx \mathop{\mathbb{E}}_{s \sim d ^ {\pi}} \left[ \nabla_w \sum_{a \in \mathcal{A}} \pi(a | s; w) \hat{Q} (s, a; u) \right]
\end{align} $$
Таким образом при обучении параметры $w$ подстраиваются так, чтобы максимизировать $\sum_{a \in \mathcal{A}} \pi(a | s; w) \hat{Q} (s, a; u) = \tilde{V}(s; w, u)$ - оценку полезности состояний, рассчитанную по текущей оценке полезности действий $\hat{Q}$. При этом действия из траекторий $d ^ {\pi}$ явным образом в $\nabla J$ не входят.

# Задание
1. **Реализовать предложенную схему on-policy обучения актора-критика.** За основу можно взять Advantage Actor Critic (A2C) (см. реализация ниже). Модель критика $\hat{Q}(s, a; u)$ аппроксимирует полезность действий (см. пример архитектуры Q-сети в тетрадке к семинару DQN) и обучается on-policy с помощью n-шаговых отдач. Оптимизационная задача для актора $\pi(s, a; w)$ : $\sum_{a \in \mathcal{A}} \pi(a | s; w) \text{ stop-gradient}[\hat{Q} (s, a; u)] \rightarrow \max_\limits{w}$. Сравнить результаты реализованного алгоритма с A2C.
2. **Исследовать влияние базового уровня на реализованный алгоритм.** Использовать функцию преимущества $A(s, a) = Q(s, a) - V(s)$ для расчёта градиента стратегии: $\sum_{a \in \mathcal{A}} \pi(a | s; w) \text{ stop-gradient}[\hat{A}(s, a)] \rightarrow \max_\limits{w}$, где $\hat{A}(s, a) = \hat{Q}(s, a; u) - \tilde{V}(s; w, u) = \hat{Q}(s, a; u) - \sum_{a \in \mathcal{A}} \pi(a | s; w) \hat{Q} (s, a; u)$

В качестве среды взять задачу PongNoFrameskip-v4 из набора сред [Atari](https://ale.farama.org/environments/).

В отчёте на одном графике должны быть представлены кривые эпмирических зависимостей отдачи, усреднённой по 60 eval-эпизодам и двум запускам обучения алгоритма (т.е. 30 эпизодов для одного запуска) для:
- A2C
- алгоритма, реализующего задание 1
- алгоритма, реализующего задание 2

В отчёте должны быть ссылки на проекты (public доступ) в WandB с логированием метрик экспериментов. По названию Run'а должно быть понятно, к какому эксперименту он относится, например: a2c_seed-0, a2c_seed-100, task1_seed-0, task1_seed-100, task2_seed-0, task2_seed-100. Также в отчёте должен быть параграф с анализом результатов.

#### Дополнительная информация
- Обучение можно остановить при достижении отдачи >18 (обычно хватает 15 млн шагов, среднее время обучения в google colab - 10 часов)
- Гиперпарамтры для A2C указаны в примере
- Гиперпараметры для алгоритмов, реализующих задания 1 и 2: `pg_coef=0.1`, `vf_coef=0.5`, `ent_coef=0.001`, `lr=0.0015` (другие гиперпараметры совпадают с A2C)
- Реализация stop-gradient в PyTorch: $\text{stop-gradient}[X]$ $\rightarrow$ `X.detach()`

# Реализация A2C и запуск обучения в google colab


```python
try:
    import google.colab
    COLAB = True
    if COLAB:
        ! pip install 'gymnasium[atari,accept-rom-license]==0.29.1' wandb==0.19.1 && pip install --no-deps stable-baselines3==2.3.2
except:
    COLAB = False

NEED_DRIVE = True
HAS_DRIVE = False
```

    The history saving thread hit an unexpected error (OperationalError('attempt to write a readonly database')).History will not be written to the database.


### При обучении в google colab чек-поинт необходимо сохранять в google drive, чтобы не потерять данные при отключении тетрадки


```python
if NEED_DRIVE and COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    HAS_DRIVE = True
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



```python
import json
import math
import random
import sys
import os
import time
from collections import deque
from typing import Dict, Type, Any, Tuple, NamedTuple, Optional, Union, Generator, List

import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import AtariPreprocessing, RecordEpisodeStatistics, FrameStack
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

import wandb
os.environ["WANDB_API_KEY"] = "your wandb api key"
```

#### Misc


```python
def set_seed_everywhere(seed: int, using_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if using_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

## Environment
Используем векторизованные среды: агент взаимодействует сразу с `n_envs` экземлярами среды. Таким образом в буфер собираются данные из `n_envs` эпизодов, повышая разнообразие данных, на которых обучается стратегия. 


```python
def make_env(n_envs, env_id):
    def _make():
        env = gym.make(env_id)
        env = RecordEpisodeStatistics(env)
        env = AtariPreprocessing(env)
        env = FrameStack(env, num_stack=4)

        return env

    vec_env = AsyncVectorEnv([_make for _ in range(n_envs)])

    return vec_env
```

## Policy


```python
class Encoder(nn.Module):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int = 512,
    ) -> None:
        super().__init__()
        self.features_dim = features_dim
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class Policy(nn.Module):
    def __init__(self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr: float,):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        self.action_net = nn.Linear(self.features_dim, self.action_space.n)
        self.value_net = nn.Linear(self.features_dim, 1)

        self.optimizer = RMSpropTFLike(self.parameters(), lr=self.lr, eps=1e-5)

    def make_features_extractor(self):
        return Encoder(self.observation_space)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        features = self.extract_features(obs)
        action_logits = self.action_net(features)
        distribution = Categorical(logits=action_logits)
        actions = distribution.mode() if deterministic else distribution.sample()
        policy_output = {'actions': actions, 'log_probs': distribution.log_prob(actions)}

        values = self.value_net(features)
        policy_output['values'] = values

        return policy_output

    def extract_features(self, obs: torch.Tensor):
        obs = obs.float() / 255.
        features = self.features_extractor(obs)

        return features

    def evaluate_actions(self, obs: torch.Tensor, actions_taken: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.extract_features(obs)
        action_logits = self.action_net(features)
        distribution = Categorical(logits=action_logits)
        log_probs = distribution.log_prob(actions_taken)
        entropy = distribution.entropy()
        values = self.value_net(features)

        return {'log_probs': log_probs, 'values': values, 'entropy': entropy}

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)['values']

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)
```

## Buffer
Версия буфера, где хранятся данные из `n_envs` сред.


```python
class BufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    values: torch.Tensor
    log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class Buffer:
    def __init__(
            self,
            device: Union[torch.device, str] = "auto",
            gamma: float = 0.99,
            n_envs: int = 1,
    ):
        self.device = torch.device(device)
        self.n_envs = n_envs
        self.gamma = gamma
        self.observations = None
        self.actions = None
        self.rewards = None
        self.returns = None
        self.values = None
        self.log_probs = None
        self.episode_starts = None
        self.advantages = None

    def reset(self) -> None:
        self.observations = [[] for _ in range(self.n_envs)]
        self.actions = [[] for _ in range(self.n_envs)]
        self.rewards = [[] for _ in range(self.n_envs)]
        self.returns = [None for _ in range(self.n_envs)]
        self.episode_starts = [[] for _ in range(self.n_envs)]
        self.values = [[] for _ in range(self.n_envs)]
        self.log_probs = [[] for _ in range(self.n_envs)]
        self.advantages = [None for _ in range(self.n_envs)]

    def size(self) -> int:
        return sum(len(env_actions) for env_actions in self.actions)

    @staticmethod
    def flatten(buffer: list, dtype=None) -> np.ndarray:
        data = []
        for env_data in buffer:
            data.extend(env_data)

        return np.stack(data).astype(dtype)

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: torch.Tensor,
            log_prob: torch.Tensor,
    ) -> None:
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        log_prob = log_prob.clone().cpu().numpy()
        value = value.clone().flatten().cpu().numpy()
        for env_id in range(self.n_envs):
            self.observations[env_id].append(obs[env_id])
            self.actions[env_id].append(action[env_id])
            self.rewards[env_id].append(reward[env_id])
            self.episode_starts[env_id].append(episode_start[env_id])
            self.values[env_id].append(value[env_id])
            self.log_probs[env_id].append(log_prob[env_id])

    def to_torch(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, device=self.device)

    def get(self) -> BufferSamples:
        self.observations = self.flatten(self.observations)
        self.actions = self.flatten(self.actions)
        self.advantages = self.flatten(self.advantages, dtype=np.float32)
        self.returns = self.flatten(self.returns, dtype=np.float32)
        self.log_probs = self.flatten(self.log_probs, dtype=np.float32)
        self.values = self.flatten(self.values, dtype=np.float32)

        data = (
            self.observations,
            self.actions,
            self.values,
            self.log_probs,
            self.advantages.reshape(-1),
            self.returns.reshape(-1),
        )

        return BufferSamples(*tuple(map(self.to_torch, data)))

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        last_values = last_values.flatten().cpu().numpy()
        for env_id in range(self.n_envs):
            values = self.values[env_id]
            actions = self.actions[env_id]
            rewards = self.rewards[env_id]
            episode_starts = self.episode_starts[env_id]
            assert len(values) == len(actions)
            assert len(rewards) == len(actions)
            assert len(episode_starts) == len(actions)

            returns = [None] * len(rewards)
            advantages = [None] * len(rewards)
            episode_starts.append(dones[env_id])
            reward_to_go = last_values[env_id]
            for step in reversed(range(len(rewards))):
                is_next_step_terminal = episode_starts[step + 1]
                reward_to_go = 0 if is_next_step_terminal else reward_to_go
                reward_to_go = rewards[step] + self.gamma * reward_to_go
                returns[step] = reward_to_go
                advantages[step] = reward_to_go - values[step]

            self.returns[env_id] = returns
            self.advantages[env_id] = advantages
```

## A2C


```python
class A2C:
    def __init__(
            self,
            env: gym.vector.VectorEnv,
            lr: float = 7e-4,
            n_steps: int = 5,
            gamma: float = 0.99,
            ent_coef: float = 0.01,
            vf_coef: float = 0.25,
            pg_coef: float = 1.0,
            max_grad_norm: float = 0.5,
            stats_window_size: int = 100,
            seed: Optional[int] = None,
            device: Union[torch.device, str] = "cuda",
            eval_env=None,
            n_eval_episodes=None,
            eval_interval=None,
    ):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        print(f"Using {self.device} device")

        self.num_timesteps = 0
        self._num_timesteps_at_start = 0
        self._episode_num = 0
        self.seed = seed
        self.start_time = None
        self.lr = lr
        self._last_obs = None
        self._last_episode_starts = None
        self._stats_window_size = stats_window_size
        self.ep_info_buffer = deque(maxlen=self._stats_window_size)
        self._n_updates = 0

        self.observation_space = env.single_observation_space
        self.action_space = env.single_action_space
        self.n_envs = env.num_envs
        self.env = env
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_interval = eval_interval or math.inf
        self.next_eval_step = self.eval_interval

        self.n_steps = n_steps
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.pg_coef = pg_coef
        self.max_grad_norm = max_grad_norm
        self.log_dict = {}
        self.evaluation_history = []
        self.checkpoint_path = None

        self.set_random_seed(self.seed)
        self.rollout_buffer = Buffer(
            device=self.device,
            gamma=self.gamma,
            n_envs=self.n_envs,
        )
        self.policy = Policy(
            self.observation_space, self.action_space, self.lr
        )
        self.policy = self.policy.to(self.device)

    def _update_info_buffer(self, infos: Dict[str, Any]) -> None:
        assert self.ep_info_buffer is not None

        if 'final_info' not in infos:
            return

        for idx, info in enumerate(infos['final_info']):
            if info is not None:
                maybe_ep_info = info.get("episode")
                if maybe_ep_info is not None:
                    self.ep_info_buffer.extend([maybe_ep_info])

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            return
        set_seed_everywhere(seed, using_cuda=self.device.type == torch.device("cuda").type)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def collect_rollouts(
            self,
            env,
            buffer: Buffer,
    ):
        self.policy.set_training_mode(False)
        assert self._last_obs is not None, "No previous observation was provided"

        step = 0
        buffer.reset()
        while step < self.n_steps:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(self._last_obs, device=self.device)
                policy_result = self.policy(obs_tensor)
                actions, values, log_probs = policy_result['actions'], policy_result['values'], policy_result[
                    'log_probs']
            actions = actions.cpu().numpy()
            new_obs, rewards, terminateds, truncateds, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            self._update_info_buffer(infos)
            dones = terminateds | truncateds
            self._episode_num += np.sum(dones).item()
            actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            for idx, truncated in enumerate(truncateds):
                if (
                        truncated
                        and "final_observation" in infos
                        and infos["final_observation"][idx] is not None
                ):
                    terminal_obs = torch.as_tensor(infos["final_observation"][idx], device=self.device)
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs.unsqueeze(0)).item()
                    rewards[idx] += self.gamma * terminal_value

            buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

            if self.num_timesteps >= self.next_eval_step:
                self.evaluate()
                self.next_eval_step += self.eval_interval

            step += 1

        with torch.no_grad():
            values = self.policy.predict_values(torch.as_tensor(np.array(new_obs), device=self.device))

        buffer.compute_returns_and_advantage(last_values=values, dones=dones)

    def train(self) -> None:
        self.policy.set_training_mode(True)

        rollout_data = self.rollout_buffer.get()
        actions = rollout_data.actions.long().flatten()
        policy_output = self.policy.evaluate_actions(rollout_data.observations, actions)
        log_prob = policy_output['log_probs']
        values = policy_output['values']
        entropy = policy_output['entropy']
        advantages = rollout_data.advantages

        # Policy gradient loss
        policy_loss = -self.pg_coef * (advantages * log_prob).mean()

        # Value loss
        value_loss = self.vf_coef * F.mse_loss(values, rollout_data.returns.reshape_as(values))

        # Entropy loss
        entropy = torch.mean(entropy)
        entropy_loss = -self.ent_coef * entropy

        loss = policy_loss + entropy_loss + value_loss

        # Optimization step
        self.policy.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
        self._n_updates += 1

        self.log_dict = dict([
            ("train/entropy_loss", entropy_loss.item()), ("train/policy_gradient_loss", policy_loss.item()),
            ("train/value_loss", value_loss.item()), ("train/loss", loss.item()),
            ("train/grad_norm", grad_norm.item()), ("train/n_updates", self._n_updates),
            ("train/entropy", entropy.item()),
        ])

    def _dump_logs(self, iteration: int) -> None:
        assert self.ep_info_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.log_dict["time/iterations"] = iteration
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.log_dict["rollout/ep_rew_mean"] = np.mean(
                np.asarray([ep_info["r"] for ep_info in self.ep_info_buffer])).item()
            self.log_dict["rollout/ep_len_mean"] = np.mean(
                np.asarray([ep_info["l"] for ep_info in self.ep_info_buffer])).item()

        self.log_dict["time/fps"] = fps
        self.log_dict["time/time_elapsed"] = int(time_elapsed)
        self.log_dict["time/total_timesteps"] = self.num_timesteps
        self.log_dict["time/total_episodes"] = self._episode_num
        print(json.dumps(self.log_dict, sort_keys=True, indent=4), flush=True)
        if wandb.run is not None:
            wandb.log(step=self.num_timesteps, data=self.log_dict)

    def evaluate(self):
        self.policy.set_training_mode(False)
        if self.eval_env is None or self.n_eval_episodes is None:
            return

        n_eval_episodes = self.n_eval_episodes
        eval_env = self.eval_env
        n_envs = eval_env.num_envs
        returns = []
        lengths = []
        episode_counts = np.zeros(n_envs, dtype=np.int32)
        episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype=np.int32)
        observations, _ = eval_env.reset()
        while (episode_counts < episode_count_targets).any():
            with torch.no_grad():
                obs_tensor = torch.as_tensor(np.array(observations), device=self.device)
                actions = self.policy(obs_tensor)['actions']
            actions = actions.cpu().numpy()
            new_observations, rewards, terminateds, truncateds, infos = eval_env.step(actions)
            dones = terminateds | truncateds

            for i in range(n_envs):
                if episode_counts[i] < episode_count_targets[i]:
                    if dones[i]:
                        assert 'final_info' in infos, 'Evaluation environments must be wrapped into RecordEpisodeStatistics'
                        info = infos['final_info'][i]
                        returns.append(info['episode']['r'].item())
                        lengths.append(info['episode']['l'].item())
                        episode_counts[i] += 1

            observations = new_observations

        assert len(returns) == n_eval_episodes
        assert len(lengths) == n_eval_episodes

        self.evaluation_history.append({'step': self.num_timesteps, 'returns': returns, 'lengths': lengths})
        self.save_checkpoint()

        log_eval_dict = {
            'time/total_timesteps': self.num_timesteps,
            'eval/return': np.mean(returns),
            'eval/length': np.mean(lengths),
        }
        if wandb.run is not None:
            wandb.log(step=self.num_timesteps, data=log_eval_dict)

        log_eval_dict_stdout = {
            'time/total_timesteps': self.num_timesteps,
            'eval/return': f'{np.mean(returns)} +/- {np.std(returns, ddof=1) / np.sqrt(len(returns))}',
            'eval/length': f'{np.mean(lengths)} +/- {np.std(lengths, ddof=1) / np.sqrt(len(lengths))}',
        }

        print(json.dumps(log_eval_dict_stdout, sort_keys=True, indent=4))

    def save_checkpoint(self):
        if self.checkpoint_path is None:
            print('Skip checkpoint saving')
            return

        checkpoint = {
            'policy': self.policy.state_dict(), 'optimizer': self.policy.optimizer.state_dict(),
            'num_timesteps': self.num_timesteps, '_episode_num': self._episode_num, '_n_updates': self._n_updates,
            'ep_info_buffer': self.ep_info_buffer, 'evaluation_history': self.evaluation_history,
        }
        os.makedirs(self.checkpoint_path, exist_ok=True)
        torch.save(checkpoint, os.path.join(self.checkpoint_path, 'checkpoint.pt'))
        print(f'Save checkpoint to {self.checkpoint_path}')

    def load_checkpoint(self, load_checkpoint_path):
        print(f'Loading checkpoint from folder: {load_checkpoint_path}')
        file_path = os.path.join(load_checkpoint_path, 'checkpoint.pt')
        checkpoint = torch.load(file_path, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy'])
        self.policy.optimizer.load_state_dict(checkpoint['optimizer'])
        self.num_timesteps = checkpoint['num_timesteps']
        self._num_timesteps_at_start = self.num_timesteps
        self._episode_num = checkpoint['_episode_num']
        self._n_updates = checkpoint['_n_updates']
        self.ep_info_buffer = checkpoint['ep_info_buffer']
        self.evaluation_history = checkpoint['evaluation_history']
        self.next_eval_step = self.evaluation_history[-1]['step'] + self.eval_interval
        print(f'Resume from step: {self.num_timesteps}')

    def learn(
            self,
            total_timesteps: int,
            log_interval: int = 1,
            checkpoint_path: str = None,
    ):
        self.checkpoint_path = checkpoint_path
        assert self.env is not None

        self.start_time = time.time_ns()

        if self._last_obs is None:
            self._last_obs, _ = self.env.reset(seed=self.seed)
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)

            if self.eval_env is not None:
                self.eval_env.reset(seed=self.seed + self.n_envs)

        iteration = 0
        while self.num_timesteps < total_timesteps:
            self.collect_rollouts(self.env, self.rollout_buffer)

            iteration += 1

            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()
```

# Run algorithm


```python
def run_algorithm(
    save_checkpoint_path, project, run_name, seed, eval_interval, total_timesteps,
    resume_checkpoint_path=None, resume_run_id=None):
    run = wandb.init(
        project=project,
        name=f'{run_name}',
        resume='never' if resume_run_id is None else 'must',
        id=resume_run_id,
    )

    env = make_env(env_id='PongNoFrameskip-v4', n_envs=16)
    eval_env = make_env(env_id='PongNoFrameskip-v4', n_envs=30)

    model = A2C(
        env=env, lr=0.0007, n_steps=5,
        vf_coef=0.25, ent_coef=0.01, pg_coef=1,
        seed=seed,
        eval_env=eval_env, n_eval_episodes=30, eval_interval=eval_interval,
    )

    if resume_checkpoint_path is not None:
        model.load_checkpoint(resume_checkpoint_path)

    model.learn(
        total_timesteps=total_timesteps,
        log_interval=100,
        checkpoint_path=save_checkpoint_path,
    )

    run.finish()
```

### Пример сохранения чек-поинта


```python
run_algorithm(
    save_checkpoint_path='/content/drive/My Drive/rl_homework/a2c/test/1',
    project='Homework_ActorCritic', run_name='test', seed=0,
    eval_interval=9000, total_timesteps=20000,
)
```

    /usr/local/lib/python3.11/dist-packages/notebook/utils.py:280: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
      return LooseVersion(v) >= LooseVersion(check)
    [34m[1mwandb[0m: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
    /usr/local/lib/python3.11/dist-packages/wandb/sdk/wandb_init.py:202: PydanticDeprecatedSince20: The `copy` method is deprecated; use `model_copy` instead. See the docstring of `BaseModel.copy` for details about how to handle `include` and `exclude`. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/
      settings = self._wl.settings.copy()
    [34m[1mwandb[0m: Currently logged in as: [33mula_elfray[0m. Use [1m`wandb login --relogin`[0m to force relogin



Tracking run with wandb version 0.19.1



Run data is saved locally in <code>/content/wandb/run-20250312_151100-nij2ao0p</code>



Syncing run <strong><a href='https://wandb.ai/ula_elfray/Homework_ActorCritic/runs/nij2ao0p' target="_blank">test</a></strong> to <a href='https://wandb.ai/ula_elfray/Homework_ActorCritic' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/ula_elfray/Homework_ActorCritic' target="_blank">https://wandb.ai/ula_elfray/Homework_ActorCritic</a>



View run at <a href='https://wandb.ai/ula_elfray/Homework_ActorCritic/runs/nij2ao0p' target="_blank">https://wandb.ai/ula_elfray/Homework_ActorCritic/runs/nij2ao0p</a>


    Using cuda device
    {
        "time/fps": 339,
        "time/iterations": 100,
        "time/time_elapsed": 23,
        "time/total_episodes": 0,
        "time/total_timesteps": 8000,
        "train/entropy": 1.791576623916626,
        "train/entropy_loss": -0.017915766686201096,
        "train/grad_norm": 0.02656707726418972,
        "train/loss": -0.05745035782456398,
        "train/n_updates": 99,
        "train/policy_gradient_loss": -0.045731063932180405,
        "train/value_loss": 0.0061964732594788074
    }
    Save checkpoint to /content/drive/My Drive/rl_homework/a2c/test/1
    {
        "eval/length": "3770.5333333333333 +/- 94.27174871277175",
        "eval/return": "-20.033333333333335 +/- 0.22223180055985828",
        "time/total_timesteps": 9008
    }
    {
        "rollout/ep_len_mean": 3351.75,
        "rollout/ep_rew_mean": -20.75,
        "time/fps": 171,
        "time/iterations": 200,
        "time/time_elapsed": 93,
        "time/total_episodes": 12,
        "time/total_timesteps": 16000,
        "train/entropy": 1.7915773391723633,
        "train/entropy_loss": -0.017915772274136543,
        "train/grad_norm": 0.05245092883706093,
        "train/loss": -0.113218292593956,
        "train/n_updates": 199,
        "train/policy_gradient_loss": -0.11032344400882721,
        "train/value_loss": 0.01502092182636261
    }
    Save checkpoint to /content/drive/My Drive/rl_homework/a2c/test/1
    {
        "eval/length": "3628.4333333333334 +/- 75.92900015892545",
        "eval/return": "-20.266666666666666 +/- 0.13504646568022544",
        "time/total_timesteps": 18000
    }







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>eval/length</td><td>█▁</td></tr><tr><td>eval/return</td><td>█▁</td></tr><tr><td>rollout/ep_len_mean</td><td>▁</td></tr><tr><td>rollout/ep_rew_mean</td><td>▁</td></tr><tr><td>time/fps</td><td>█▁</td></tr><tr><td>time/iterations</td><td>▁█</td></tr><tr><td>time/time_elapsed</td><td>▁█</td></tr><tr><td>time/total_episodes</td><td>▁█</td></tr><tr><td>time/total_timesteps</td><td>▁▂▇█</td></tr><tr><td>train/entropy</td><td>▁█</td></tr><tr><td>train/entropy_loss</td><td>█▁</td></tr><tr><td>train/grad_norm</td><td>▁█</td></tr><tr><td>train/loss</td><td>█▁</td></tr><tr><td>train/n_updates</td><td>▁█</td></tr><tr><td>train/policy_gradient_loss</td><td>█▁</td></tr><tr><td>train/value_loss</td><td>▁█</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>eval/length</td><td>3628.43333</td></tr><tr><td>eval/return</td><td>-20.26667</td></tr><tr><td>rollout/ep_len_mean</td><td>3351.75</td></tr><tr><td>rollout/ep_rew_mean</td><td>-20.75</td></tr><tr><td>time/fps</td><td>171</td></tr><tr><td>time/iterations</td><td>200</td></tr><tr><td>time/time_elapsed</td><td>93</td></tr><tr><td>time/total_episodes</td><td>12</td></tr><tr><td>time/total_timesteps</td><td>18000</td></tr><tr><td>train/entropy</td><td>1.79158</td></tr><tr><td>train/entropy_loss</td><td>-0.01792</td></tr><tr><td>train/grad_norm</td><td>0.05245</td></tr><tr><td>train/loss</td><td>-0.11322</td></tr><tr><td>train/n_updates</td><td>199</td></tr><tr><td>train/policy_gradient_loss</td><td>-0.11032</td></tr><tr><td>train/value_loss</td><td>0.01502</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">test</strong> at: <a href='https://wandb.ai/ula_elfray/Homework_ActorCritic/runs/nij2ao0p' target="_blank">https://wandb.ai/ula_elfray/Homework_ActorCritic/runs/nij2ao0p</a><br> View project at: <a href='https://wandb.ai/ula_elfray/Homework_ActorCritic' target="_blank">https://wandb.ai/ula_elfray/Homework_ActorCritic</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20250312_151100-nij2ao0p/logs</code>


### Возобновление обучения из чек-поинта
resume_run_id - последний сегмент в URL run'а, который хотим возобновить:

`https://wandb.ai/ula_elfray/Homework_ActorCritic/runs/`**nij2ao0p**


```python
run_algorithm(
    save_checkpoint_path='/content/drive/My Drive/rl_homework/a2c/test/2',
    project='Homework_ActorCritic', run_name='test', seed=1,
    eval_interval=9000, total_timesteps=40000,
    resume_checkpoint_path='/content/drive/My Drive/rl_homework/a2c/test/1',
    resume_run_id='nij2ao0p')
```

    /usr/local/lib/python3.11/dist-packages/wandb/sdk/wandb_init.py:202: PydanticDeprecatedSince20: The `copy` method is deprecated; use `model_copy` instead. See the docstring of `BaseModel.copy` for details about how to handle `include` and `exclude`. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/
      settings = self._wl.settings.copy()



Tracking run with wandb version 0.19.1



Run data is saved locally in <code>/content/wandb/run-20250312_151428-nij2ao0p</code>



Resuming run <strong><a href='https://wandb.ai/ula_elfray/Homework_ActorCritic/runs/nij2ao0p' target="_blank">test</a></strong> to <a href='https://wandb.ai/ula_elfray/Homework_ActorCritic' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/ula_elfray/Homework_ActorCritic' target="_blank">https://wandb.ai/ula_elfray/Homework_ActorCritic</a>



View run at <a href='https://wandb.ai/ula_elfray/Homework_ActorCritic/runs/nij2ao0p' target="_blank">https://wandb.ai/ula_elfray/Homework_ActorCritic/runs/nij2ao0p</a>


    Using cuda device
    Loading checkpoint from folder: /content/drive/My Drive/rl_homework/a2c/test/1
    Resume from step: 18000


    <ipython-input-8-4d861f853964>:270: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      checkpoint = torch.load(file_path)


    {
        "rollout/ep_len_mean": 3454.0714285714284,
        "rollout/ep_rew_mean": -20.571428298950195,
        "time/fps": 340,
        "time/iterations": 100,
        "time/time_elapsed": 23,
        "time/total_episodes": 14,
        "time/total_timesteps": 26000,
        "train/entropy": 1.7915782928466797,
        "train/entropy_loss": -0.01791578345000744,
        "train/grad_norm": 0.04131273552775383,
        "train/loss": 0.05740572512149811,
        "train/n_updates": 323,
        "train/policy_gradient_loss": 0.0661173090338707,
        "train/value_loss": 0.009204201400279999
    }
    Save checkpoint to /content/drive/My Drive/rl_homework/a2c/test/2
    {
        "eval/length": "3725.4333333333334 +/- 64.68572446986715",
        "eval/return": "-20.2 +/- 0.13896166675593022",
        "time/total_timesteps": 27008
    }
    {
        "rollout/ep_len_mean": 3512.4444444444443,
        "rollout/ep_rew_mean": -20.592592239379883,
        "time/fps": 179,
        "time/iterations": 200,
        "time/time_elapsed": 89,
        "time/total_episodes": 27,
        "time/total_timesteps": 34000,
        "train/entropy": 1.7915823459625244,
        "train/entropy_loss": -0.017915822565555573,
        "train/grad_norm": 0.06179024651646614,
        "train/loss": -0.10954311490058899,
        "train/n_updates": 423,
        "train/policy_gradient_loss": -0.10661293566226959,
        "train/value_loss": 0.014985642395913601
    }
    Save checkpoint to /content/drive/My Drive/rl_homework/a2c/test/2
    {
        "eval/length": "3778.8333333333335 +/- 99.92864216481784",
        "eval/return": "-20.233333333333334 +/- 0.21271822241076074",
        "time/total_timesteps": 36000
    }







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>eval/length</td><td>▁█</td></tr><tr><td>eval/return</td><td>█▁</td></tr><tr><td>rollout/ep_len_mean</td><td>▁█</td></tr><tr><td>rollout/ep_rew_mean</td><td>█▁</td></tr><tr><td>time/fps</td><td>█▁</td></tr><tr><td>time/iterations</td><td>▁█</td></tr><tr><td>time/time_elapsed</td><td>▁█</td></tr><tr><td>time/total_episodes</td><td>▁█</td></tr><tr><td>time/total_timesteps</td><td>▁▂▇█</td></tr><tr><td>train/entropy</td><td>▁█</td></tr><tr><td>train/entropy_loss</td><td>█▁</td></tr><tr><td>train/grad_norm</td><td>▁█</td></tr><tr><td>train/loss</td><td>█▁</td></tr><tr><td>train/n_updates</td><td>▁█</td></tr><tr><td>train/policy_gradient_loss</td><td>█▁</td></tr><tr><td>train/value_loss</td><td>▁█</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>eval/length</td><td>3778.83333</td></tr><tr><td>eval/return</td><td>-20.23333</td></tr><tr><td>rollout/ep_len_mean</td><td>3512.44444</td></tr><tr><td>rollout/ep_rew_mean</td><td>-20.59259</td></tr><tr><td>time/fps</td><td>179</td></tr><tr><td>time/iterations</td><td>200</td></tr><tr><td>time/time_elapsed</td><td>89</td></tr><tr><td>time/total_episodes</td><td>27</td></tr><tr><td>time/total_timesteps</td><td>36000</td></tr><tr><td>train/entropy</td><td>1.79158</td></tr><tr><td>train/entropy_loss</td><td>-0.01792</td></tr><tr><td>train/grad_norm</td><td>0.06179</td></tr><tr><td>train/loss</td><td>-0.10954</td></tr><tr><td>train/n_updates</td><td>423</td></tr><tr><td>train/policy_gradient_loss</td><td>-0.10661</td></tr><tr><td>train/value_loss</td><td>0.01499</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">test</strong> at: <a href='https://wandb.ai/ula_elfray/Homework_ActorCritic/runs/nij2ao0p' target="_blank">https://wandb.ai/ula_elfray/Homework_ActorCritic/runs/nij2ao0p</a><br> View project at: <a href='https://wandb.ai/ula_elfray/Homework_ActorCritic' target="_blank">https://wandb.ai/ula_elfray/Homework_ActorCritic</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20250312_151428-nij2ao0p/logs</code>


Отдача за eval-эпизоды сохраняется в чекпоит в поле `evaluation_history`. При построении графиков для каждого значения `step` необходимо усреднить `returns` по результатам всех эпизодов из 2х запусков обучения. Кривые должны быть представлены в виде: среднее значение +/- [стандартная ошибка](https://ru.wikipedia.org/wiki/%D0%A1%D1%82%D0%B0%D0%BD%D0%B4%D0%B0%D1%80%D1%82%D0%BD%D0%B0%D1%8F_%D0%BE%D1%88%D0%B8%D0%B1%D0%BA%D0%B0) .


```python
torch.load('/content/drive/My Drive/rl_homework/a2c/test/2/checkpoint.pt', weights_only=False)['evaluation_history']
```

    <ipython-input-13-8280669dfc51>:1: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      torch.load('/content/drive/My Drive/rl_homework/a2c/test/2/checkpoint.pt')['evaluation_history']





    [{'step': 9008,
      'returns': [-21.0,
       -21.0,
       -21.0,
       -21.0,
       -21.0,
       -21.0,
       -20.0,
       -21.0,
       -21.0,
       -21.0,
       -21.0,
       -21.0,
       -20.0,
       -21.0,
       -21.0,
       -20.0,
       -20.0,
       -19.0,
       -20.0,
       -20.0,
       -20.0,
       -19.0,
       -19.0,
       -20.0,
       -19.0,
       -18.0,
       -19.0,
       -21.0,
       -18.0,
       -16.0],
      'lengths': [3056,
       3056,
       3168,
       3171,
       3280,
       3304,
       3350,
       3371,
       3408,
       3480,
       3544,
       3612,
       3614,
       3648,
       3656,
       3700,
       3721,
       3746,
       3915,
       3919,
       3970,
       4067,
       4098,
       4154,
       4168,
       4179,
       4230,
       4372,
       4866,
       5293]},
     {'step': 18000,
      'returns': [-21.0,
       -21.0,
       -21.0,
       -21.0,
       -21.0,
       -21.0,
       -21.0,
       -21.0,
       -20.0,
       -21.0,
       -20.0,
       -20.0,
       -20.0,
       -21.0,
       -21.0,
       -20.0,
       -21.0,
       -20.0,
       -20.0,
       -21.0,
       -19.0,
       -20.0,
       -20.0,
       -20.0,
       -20.0,
       -19.0,
       -19.0,
       -19.0,
       -20.0,
       -19.0],
      'lengths': [3056,
       3056,
       3056,
       3056,
       3174,
       3240,
       3304,
       3371,
       3367,
       3408,
       3479,
       3483,
       3548,
       3544,
       3611,
       3661,
       3725,
       3725,
       3726,
       3752,
       3746,
       3791,
       3794,
       3805,
       3836,
       3994,
       4047,
       4126,
       4581,
       4791]},
     {'step': 27008,
      'returns': [-21.0,
       -21.0,
       -21.0,
       -21.0,
       -21.0,
       -20.0,
       -20.0,
       -20.0,
       -21.0,
       -21.0,
       -20.0,
       -21.0,
       -20.0,
       -20.0,
       -20.0,
       -21.0,
       -19.0,
       -20.0,
       -20.0,
       -21.0,
       -21.0,
       -21.0,
       -20.0,
       -19.0,
       -20.0,
       -20.0,
       -19.0,
       -19.0,
       -19.0,
       -19.0],
      'lengths': [3056,
       3056,
       3168,
       3280,
       3296,
       3350,
       3479,
       3514,
       3544,
       3595,
       3590,
       3648,
       3721,
       3706,
       3727,
       3739,
       3749,
       3798,
       3804,
       3843,
       3860,
       3896,
       3962,
       4100,
       4107,
       4163,
       4174,
       4254,
       4280,
       4304]},
     {'step': 36000,
      'returns': [-21.0,
       -21.0,
       -21.0,
       -21.0,
       -21.0,
       -21.0,
       -20.0,
       -21.0,
       -21.0,
       -21.0,
       -21.0,
       -20.0,
       -21.0,
       -21.0,
       -21.0,
       -20.0,
       -21.0,
       -21.0,
       -21.0,
       -21.0,
       -19.0,
       -20.0,
       -20.0,
       -19.0,
       -18.0,
       -21.0,
       -20.0,
       -18.0,
       -18.0,
       -17.0],
      'lengths': [3056,
       3168,
       3174,
       3296,
       3296,
       3364,
       3367,
       3408,
       3416,
       3478,
       3484,
       3485,
       3528,
       3536,
       3546,
       3591,
       3619,
       3648,
       3662,
       3806,
       3989,
       4097,
       4106,
       4185,
       4367,
       4378,
       4507,
       4636,
       5008,
       5164]}]



# Запуск обучения A2C


```python
run_algorithm(
    save_checkpoint_path='/content/drive/rl_homework/a2c/seed-0',
    project='Homework_ActorCritic', run_name='a2c_seed-0', seed=0,
    eval_interval=300000, total_timesteps=20000000,
)
```


```python
run_algorithm(
    save_checkpoint_path='/content/drive/rl_homework/a2c/seed-100',
    project='Homework_ActorCritic', run_name='a2c_seed-100', seed=100,
    eval_interval=300000, total_timesteps=20000000,
)
```
