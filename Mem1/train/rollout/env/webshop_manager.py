
from typing import List, Dict
from .webshop import WebShopEnv
from dataclasses import dataclass


class WebShopEnvManager:
    def __init__(self):
        # set up config for webshop environment
        test_env = WebShopEnv()
        server = test_env.server
        self.total_envs = len(server.goals)
        self.envs = {i: WebShopEnv(server=server) for i in range(self.total_envs)}
        self.active_envs = {}

    # given a dict of session_id -> action, step the environment, return the observation
    def step(self, actions, environment_ids) -> Dict[str, str]: 
        assert len(actions) == len(environment_ids)

        observations, rewards, dones, infos = [], [], [], []

        for environment_id, action in zip(environment_ids, actions):
            observation, reward, done, info = self.envs[environment_id].step(action)
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return observations, rewards, dones, infos

    # given a list of session ids, reset the environment and make a dict of session_id -> env
    def reset(self, session_ids: List[str]) -> Dict[str, WebShopEnv]:
        self.active_envs = {session_id: self.envs[session_id] for session_id in session_ids}
        for session_id, env in self.active_envs.items():
            env.reset(session=session_id)
        
