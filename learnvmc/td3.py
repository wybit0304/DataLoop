"""Twin-critic TD3, replay, checkpoint persistence and physical-unit export."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn

TD3_CONFIG = {'actor_hidden': [128, 64], 'critic_hidden': [128, 64], 'actor_lr': 0.0001, 'critic_lr': 0.0003, 'gamma': 0.99, 'tau': 0.005, 'batch_size': 256, 'buffer_size': 1000000, 'policy_delay': 2, 'target_noise': 0.1, 'noise_clip': 0.2, 'exploration_std': 0.1, 'replay_reward_scale': 0.1}
class Actor(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())
        nn.init.uniform_(self.net[-2].weight, -0.003, 0.003)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, s):
        return self.net(s)

class Critic(nn.Module):

    def __init__(self):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(7, 128), nn.ReLU(), nn.Linear(128, 64))
        self.action = nn.Linear(1, 64)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(64, 1))

    def forward(self, s, a):
        return self.head(self.state(s) + self.action(a))

class Replay:

    def __init__(self, n, seed):
        self.s = np.zeros((n, 7), np.float32)
        self.ns = self.s.copy()
        self.a = np.zeros((n, 1), np.float32)
        self.r = np.zeros((n, 1), np.float32)
        self.d = self.r.copy()
        self.pos = 0
        self.size = 0
        self.n = n
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, ns, terminated):
        i = self.pos
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.ns[i] = ns
        self.d[i] = terminated
        self.pos = (i + 1) % self.n
        self.size = min(self.size + 1, self.n)

    def sample(self, n):
        i = self.rng.integers(self.size, size=n)
        return [torch.from_numpy(z[i]) for z in [self.s, self.a, self.r, self.ns, self.d]]

class TD3:

    def __init__(self):
        self.actor = Actor()
        self.q1 = Critic()
        self.q2 = Critic()
        self.at = copy.deepcopy(self.actor)
        self.qt1 = copy.deepcopy(self.q1)
        self.qt2 = copy.deepcopy(self.q2)
        self.ao = torch.optim.Adam(self.actor.parameters(), lr=TD3_CONFIG['actor_lr'], weight_decay=0.0001)
        self.qo = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=TD3_CONFIG['critic_lr'], weight_decay=0.0001)
        self.updates = 0

    def act(self, s):
        with torch.no_grad():
            return self.actor(torch.as_tensor(s).reshape(1, -1)).numpy()[0]

    def update(self, replay):
        s, a, r, ns, d = replay.sample(TD3_CONFIG['batch_size'])
        with torch.no_grad():
            noise = (torch.randn_like(a) * TD3_CONFIG['target_noise']).clamp(-TD3_CONFIG['noise_clip'], TD3_CONFIG['noise_clip'])
            na = (self.at(ns) + noise).clamp(-1, 1)
            target = r + TD3_CONFIG['gamma'] * (1 - d) * torch.minimum(self.qt1(ns, na), self.qt2(ns, na))
        loss = nn.functional.mse_loss(self.q1(s, a), target) + nn.functional.mse_loss(self.q2(s, a), target)
        self.qo.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 1.0)
        self.qo.step()
        self.updates += 1
        if self.updates % TD3_CONFIG['policy_delay'] == 0:
            aloss = -self.q1(s, self.actor(s)).mean()
            self.ao.zero_grad()
            aloss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.ao.step()
            with torch.no_grad():
                for source, target_net in [(self.actor, self.at), (self.q1, self.qt1), (self.q2, self.qt2)]:
                    for a0, b0 in zip(source.parameters(), target_net.parameters()):
                        b0.mul_(1 - TD3_CONFIG['tau']).add_(a0, alpha=TD3_CONFIG['tau'])

    def save(self, path, **metadata):
        """Save inference weights plus aligned critic/target/optimizer states."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(), 'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(), 'actor_target': self.at.state_dict(),
            'q1_target': self.qt1.state_dict(), 'q2_target': self.qt2.state_dict(),
            'actor_optimizer': self.ao.state_dict(), 'critic_optimizer': self.qo.state_dict(),
            'updates': self.updates, 'td3_config': TD3_CONFIG, **metadata,
        }, path)

    def load(self, path, training=False):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self.actor.load_state_dict(checkpoint['actor'])
        if training:
            if checkpoint['td3_config'] != TD3_CONFIG:
                raise ValueError('Optimizer configuration differs from the checkpoint')
            for name, network in [('q1',self.q1),('q2',self.q2),('actor_target',self.at),
                                  ('q1_target',self.qt1),('q2_target',self.qt2)]:
                network.load_state_dict(checkpoint[name])
            self.ao.load_state_dict(checkpoint['actor_optimizer'])
            self.qo.load_state_dict(checkpoint['critic_optimizer'])
            self.updates = checkpoint['updates']
        self.actor.eval()
        return checkpoint

    def export(self, path, observation_scale, max_moment):
        """Standalone TorchScript: seven raw SI states -> requested moment (Nm)."""
        class PhysicalPolicy(nn.Module):
            def __init__(self, actor, scale, maximum):
                super().__init__()
                self.actor = actor
                self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))
                self.maximum = float(maximum)
            def forward(self, states):
                return self.actor(states / self.scale) * self.maximum
        policy = PhysicalPolicy(copy.deepcopy(self.actor).eval(), observation_scale, max_moment)
        example = torch.zeros(1,7)
        traced = torch.jit.freeze(torch.jit.trace(policy, example).eval())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # File objects also work in Windows paths containing non-ASCII text.
        with Path(path).open('wb') as stream:
            torch.jit.save(traced, stream)
        return traced
