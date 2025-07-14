import random
import numpy as np
from contextlib import contextmanager
from omegaconf import OmegaConf
import dataclasses

@contextmanager
def all_seed(seed):
    random_state = random.getstate()
    np_random_state = np.random.get_state()

    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)

def register_resolvers():
    try:
        OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
        OmegaConf.register_new_resolver("int_div", lambda x, y: int(float(x) / float(y)))
        OmegaConf.register_new_resolver("not", lambda x: not x)
    except:
        pass # already registered


