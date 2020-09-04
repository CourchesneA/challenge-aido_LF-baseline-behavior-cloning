from concurrent.futures import ThreadPoolExecutor
from log_schema import Episode, Step, SCHEMA_VERSION
from typing import Generator
import os
import pickle
import numpy as np
import tensorflow as tf

class Reader:
    def __init__(self, log_file):
        self.file_name = log_file
        self.episode_data = None
        self.episode_index = 0
        self.episodes_size = 0
        self.observations_size = 0

        with open(self.file_name, 'rb') as data_file:
            print("Computing dataset size...")
            ep_count = 0
            obs_count = 0
            while True:
                try:
                    ep = pickle.load(data_file)
                    ep_count+=1
                    obs_count += len(ep.steps)
                except EOFError:
                    self.episodes_size = ep_count
                    self.observations_size = obs_count
                    data_file.seek(0)
                    print(f"found {obs_count} data point in {ep_count} episodes")
                    break

    # def read(self):
    #     end = False
    #     Observation=[]
    #     Linear=[]
    #     Angular=[]

    #     while not end:
    #         try:
    #             log = pickle.load(self._log_file)
    #             for entry in log:
    #                 step = entry['step']
    #                 Observation.append(step[0])
    #                 action = step[1]
    #                 Linear.append(action[0])
    #                 Angular.append(action[1])

    #         except EOFError:
    #             end = True

    #     return Observation,Linear,Angular

    def get_dataset(self) -> Generator[Step, None, None]:
        """
        Generator function that loads data. Should be compatible with tf.data.Dataset.from_generator()
        yields (input, target)
        more specifically: (obs: np.ndarray, [linear_v: float, angular_v: float])
        """
        with open(self.file_name, 'rb') as data_file:
            while True:
                if self.episode_data is None:
                    try:
                        self.episode_data = pickle.load(data_file)
                        self.episode_index = 0
                        assert self.episode_data.version == SCHEMA_VERSION, "Schema version do not match"
                    except EOFError:
                        print("End of log file !")
                        break
                try:
                    step: Step = self.episode_data.steps[self.episode_index]
                    self.episode_index += 1
                except IndexError:
                    print("Step is out of bound, starting new episode")
                    self.episode_data = None
                    continue
                yield (step.obs, step.action)
