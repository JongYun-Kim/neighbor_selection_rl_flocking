import os
from copy import deepcopy
import warnings
import gym  # gym 0.23.1
from gym.utils import seeding
from gym.spaces import Box, Discrete, Dict, MultiDiscrete, MultiBinary
import numpy as np  # numpy 1.23.4
from ray.rllib.utils.typing import (
    AgentID,
    MultiAgentDict,
)
from ray.tune.logger import pretty_print
from utils.utils import (wrap_to_pi, wrap_to_rectangle,
                              get_rel_pos_dist_in_periodic_boundary, map_periodic_to_continuous_space)
from typing import List, Optional
# from pydantic import BaseModel, field_validator, model_validator, ConfigDict, conlist, conint, confloat  # v2
from pydantic import BaseModel, Field, conlist, conint, validator, root_validator  # v1
import yaml


class ControlConfig(BaseModel):
    speed: float = 15.0  # Speed in m/s.
    max_turn_rate: float = 8/15  # Maximum turn rate in rad/s.
    initial_position_bound: float = 250.0  # Initial position bound in meters.
    # ACS specific controls
    beta: float = 1./3.  # communication decay rate
    lam: float =  5.  # inter agent strength
    sig: float = 1.  # bonding strength
    k1: float = 1.
    k2: float = 3.
    r0: float = 60.  # predefined (almost-)desired distance
    rho: float = 1.0  # cost weighting factor b/w alignment and cohesion


class EnvConfig(BaseModel):
    is_training: bool = False
    seed: Optional[int] = None
    obs_dim: int = 4  # periodic: 6, non-periodic: 4
    agent_name_prefix: str = 'agent_'
    env_mode: str = 'single_env'
    action_type: str = 'binary_vector'
    num_agents_pool: List[conint(ge=1)]  # Must clarify it !!
    dt: float = 0.1
    comm_range: Optional[float] = None
    max_time_steps: int = 1000
    use_fixed_episode_length: bool = False
    get_state_hist: bool = False
    get_action_hist: bool = False
    ignore_comm_lost_agents: bool = False
    periodic_boundary: bool = False
    task_type: str = 'acs'
    # Observation type: 'ego_centric' (default) or 'centralized'
    # 'ego_centric': each agent has its own local view (N, N, obs_dim) - relative positions/headings to neighbors
    # 'centralized': all agents share the same global view - positions/headings relative to swarm center/avg heading
    #                The centralized obs is replicated for each agent to maintain (N, N, obs_dim) shape
    observation_type: str = 'ego_centric'
    # Vicsek specific parameters in the env
    alignment_goal: float = 0.97
    alignment_rate_goal: float = 0.03
    alignment_window_length: int = 32
    # ACS specific parameters in the env
    entropy_p_goal: Optional[float] = None  # Standard deviation of the goal position; if None, it will be set to 0.7 * r0
    entropy_v_goal: float = 0.1  # Standard deviation of the goal velocity
    entropy_p_rate_goal: float = 0.1  # Rate of the goal position entropy (50 steps)
    entropy_v_rate_goal: float = 0.2  # Rate of the goal velocity entropy (50 steps)
    entropy_rate_window_length: int = 50  # Window length for the entropy rate
    acs_train_w_pos: float = 1.0
    acs_train_w_vel: float = 0.2
    acs_train_w_ctrl: float = 0.02


class Config(BaseModel):
    control: ControlConfig
    env: EnvConfig
    # nn: Optional[dict] = None  # Implement this later with a pydantic config class for the nn settings

    # model_config = ConfigDict(extra='forbid')

    # @field_validator('control')
    @validator('control')
    def validate_control(cls, v):
        # You can add validation logic for the ControlConfig here if needed
        return v

    # @field_validator('env')
    @validator('env')
    def validate_env(cls, v):
        # You can add validation logic for the LazyVicsekEnvConfig here if needed
        return v

    # # @model_validator(mode='after')
    # @root_validator
    # def set_dependent_defaults(cls, values):
    #     # if values.env.std_p_goal is None:
    #     #     values.env.std_p_goal = 0.7 * values.control.predefined_distance
    #     env_config = values.get('env')
    #     control_config = values.get('control')
    #     if env_config and control_config:
    #         if env_config.std_p_goal is None:
    #             env_config.std_p_goal = 0.7 * control_config.predefined_distance
    #     return values


def load_dict(path: str) -> dict:
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    # return Config(**config_dict)
    return config_dict


def load_config(something):
    if something is None:
        print("Warning: No config is provided; using the default config.")
        return Config(**load_dict('./env/default_env_config.yaml'))
    elif isinstance(something, dict):
        return Config(**something)
    elif isinstance(something, str):
        if os.path.exists(something):
            return Config(**load_dict(something))
        else:
            raise FileNotFoundError(f"File not found: {something}")
    elif isinstance(something, Config):
        return something
    else:
        raise TypeError(f"Invalid type: {type(something)}")


def config_to_env_input(config_instance: Config, seed_id: Optional[int] = None) -> dict:
    return {"seed_id": seed_id, "config": config_instance.dict()}


class NeighborSelectionFlockingEnv(gym.Env):
    def __init__(self, env_context: dict):
        super().__init__()
        seed_id = env_context['seed_id'] if 'seed_id' in env_context else None
        self.seed(seed_id)

        self.config = load_config(env_context['config'])

        self.num_agents: Optional[int] = None  # defined in reset()
        self.num_agents_min: Optional[int] = None  # defined in _validate_config()
        self.num_agents_max: Optional[int] = None  # defined in _validate_config()

        # # States
        # # # state: dict := {"agent_states":   ndarray,  # shape (num_agents_max, data_dim); absolute states!!
        #                                        [x, y, vx, vy, theta]; absolute states!!
        #                     "neighbor_masks": ndarray,  # shape (num_agents_max, num_agents_max)
        #                                        1 if neighbor, 0 if not;  self-loop is 1 (check your self-loops).
        #                     "padding_mask":   ndarray,  # shape (num_agents_max)
        #                                        1 if agent,    0 if padding
        # # # rel_state: dict := {"rel_agent_positions": ndarray,   # shape (num_agents_max, num_agents_max, 2)
        #                         "rel_agent_velocities": ndarray,  # shape (num_agents_max, num_agents_max, 2)
        #                         "rel_agent_headings": ndarray,    # shape (num_agents_max, num_agents_max)  # 2-D !!!
        #                         "rel_agent_dists": ndarray        # shape (num_agents_max, num_agents_max)
        #                         }
        #  }
        self.state, self.rel_state, self.initial_state = None, None, None
        self.agent_states_hist, self.neighbor_masks_hist, self.action_hist = None, None, None
        # self.padding_mask_hist = None
        self.has_lost_comm = None
        self.lost_comm_step = None
        self.time_step = None
        # self.agent_time_step = None
        # Vicsek hist
        self.alignment_hist = None
        # ACS hist
        self.spatial_entropy_hist = None
        self.velocity_entropy_hist = None

        self._validate_config()

        # Define ACTION SPACE
        self.action_dtype = None
        if self.config.env.env_mode == "single_env":
            if self.config.env.action_type == "binary_vector":
                self.action_dtype = np.int8
                self.action_space = Box(low=0, high=1,
                                        shape=(self.num_agents_max, self.num_agents_max), dtype=self.action_dtype)
            else:
                raise NotImplementedError("action_type must be binary_vector. "
                                          "The radius and continuous_vector are still in alpha, sorry.")
        elif self.config.env.env_mode == "multi_env":
            print("WARNING (env.__init__): multi_env is experimental; not fully implemented yet")
            if self.config.env.action_type == "binary_vector":
                self.action_space = Dict({
                    self.config.env.agent_name_prefix + str(i): Box(low=0, high=1,
                                                                    shape=(self.num_agents_max,), dtype=np.bool_)
                    for i in range(self.num_agents_max)
                })
            else:
                raise NotImplementedError("action_type must be binary_vector. "
                                          "The radius and continuous_vector are still in alpha, sorry.")
        else:
            raise NotImplementedError("env_mode must be either single_env or multi_env")

        # Define OBSERVATION SPACE
        if self.config.env.env_mode == "single_env":
            self.observation_space = Dict({
                "local_agent_infos": Box(low=-np.inf, high=np.inf,
                                         shape=(self.num_agents_max, self.num_agents_max, self.config.env.obs_dim),
                                         dtype=np.float64),
                "neighbor_masks": Box(low=0, high=1, shape=(self.num_agents_max, self.num_agents_max), dtype=np.bool_),
                "padding_mask": Box(low=0, high=1, shape=(self.num_agents_max,), dtype=np.bool_),
                "is_from_my_env": Box(low=0, high=2, shape=(), dtype=np.float16),
            })
        elif self.config.env.env_mode == "multi_env":
            self.observation_space = Dict({
                self.config.env.agent_name_prefix + str(i): Dict({
                    "centralized_agent_info": Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float64),
                    "neighbor_mask": Box(low=0, high=1, shape=(self.num_agents_max,), dtype=np.bool_),
                    "padding_mask": Box(low=0, high=1, shape=(self.num_agents_max,), dtype=np.bool_)
                }) for i in range(self.num_agents_max)
            })

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @staticmethod
    def get_default_config_dict():
        try:
            default_config = Config(**load_dict('envs/default_env_config.yaml'))
            print('-------------------DEFAULT CONFIG-------------------')
            print(pretty_print(default_config.dict()))
            # print(default_config.model_dump())
            print('----------------------------------------------------')
            return deepcopy(default_config.dict())
        except FileNotFoundError:
            warnings.warn("Warning: 'default_env_config.yaml' not found. Check the file path.")
            return None

    def _validate_config(self):
        # # env_mode: must be either "single_env" or "multi_env"
        # assert self.env_mode in ["single_env", "multi_env"], "env_mode must be either single_env or multi_env"

        # num_agents_pool: must be a tuple(range)/ndarray of int-s (list is also okay instead of ndarray for list-pool)
        self.num_agents_pool_np = self.config.env.num_agents_pool
        if isinstance(self.num_agents_pool_np, int):
            assert self.num_agents_pool_np > 1, "num_agents_pool must be > 1"
            self.num_agents_pool_np = np.array([self.num_agents_pool_np])
        assert isinstance(self.num_agents_pool_np, (tuple, np.ndarray, list)), "num_agents_pool must be a tuple or ndarray"
        assert all(
            np.issubdtype(type(x), int) for x in self.num_agents_pool_np), "all values in num_agents_pool must be int-s"
        if isinstance(self.num_agents_pool_np, list):
            self.num_agents_pool_np = np.array(self.num_agents_pool_np)  # convert to np-array
        if isinstance(self.num_agents_pool_np, tuple):
            assert len(self.num_agents_pool_np) == 2, "num_agents_pool must be a tuple of length 2, as (min, max); a range"
            assert self.num_agents_pool_np[0] <= self.num_agents_pool_np[1], "min of num_agents_pool must be <= max"
            assert self.num_agents_pool_np[0] > 1, "min of num_agents_pool must be > 1"
            self.num_agents_pool_np = np.arange(self.num_agents_pool_np[0], self.num_agents_pool_np[1] + 1)
        elif isinstance(self.num_agents_pool_np, np.ndarray):
            assert self.num_agents_pool_np.size > 0, "num_agents_pool must not be empty"
            assert len(self.num_agents_pool_np.shape) == 1, "num_agents_pool must be a np-array of shape (n, ), n > 1"
            assert all(self.num_agents_pool_np > 1), "all values in num_agents_pool must be > 1"
        else:
            raise NotImplementedError("Something wrong; check _validate_config() of LazyVicsekEnv; must not reach here")
        # Note: Now self.num_agents_pool is a ndarray of possible num_agents; ㅇㅋ?

        # Set num_agents_min and num_agents_max
        self.num_agents_min = self.num_agents_pool_np.min()
        self.num_agents_max = self.num_agents_pool_np.max()

        # # max_time_step: must be an int and > 0
        # assert isinstance(self.max_time_steps, int), "max_time_step must be an int"
        # assert self.max_time_steps > 0, "max_time_step must be > 0"

        # ACS specific checks
        # entropy_p_goal: set to the default value if None
        if self.config.env.task_type == "acs":
            if self.config.env.entropy_p_goal is None:
                self.config.env.entropy_p_goal = 0.7 * self.config.control.r0
            assert self.config.env.entropy_p_goal > 0, "entropy_p_goal must be > 0"
            assert self.config.env.entropy_v_goal > 0, "entropy_v_goal must be > 0"
            assert self.config.env.entropy_p_rate_goal > 0, "entropy_p_rate_goal must be > 0"
            assert self.config.env.entropy_v_rate_goal > 0, "entropy_v_rate_goal must be > 0"
        elif self.config.env.task_type == "vicsek":
            if self.config.env.entropy_p_goal is not None:
                warnings.warn("entropy_p_goal is not used in vicsek task; it will be ignored.")
        else:
            raise NotImplementedError("task_type must be either vicsek or acs at the moment")

        # Observation type validation
        assert self.config.env.observation_type in ["ego_centric", "centralized"], \
            f"observation_type must be 'ego_centric' or 'centralized', got '{self.config.env.observation_type}'"
        if self.config.env.observation_type == "centralized" and self.config.env.periodic_boundary:
            warnings.warn("Centralized observation with periodic boundary is not fully tested; "
                          "consider using ego_centric observation for periodic boundary environments.")

    def show_current_config(self):
        print('-------------------CURRENT CONFIG-------------------')
        print(pretty_print(self.config.dict()))
        # print(self.config.model_dump())
        print('----------------------------------------------------')

    def custom_reset(self, p_, v_, th_, num_agents_max=None, comm_range=None):
        """
        Custom reset method to reset the environment with the given initial states
        :param p_: ndarray of shape (num_agents, 2)
        :param v_: ndarray of shape (num_agents, 2)
        :param th_: ndarray of shape (num_agents, )
        :param comm_range: float; communication range
        :param num_agents_max: int; maximum number of agents
        :return: obs
        """
        # Dummy reset
        self.reset()
        # Init time steps
        self.time_step = 0
        # self.agent_time_step = np.zeros(self.num_agents_max, dtype=np.int32)

        # Get initial num_agents
        num_agents = len(p_)
        self.num_agents = num_agents
        num_agents_max = num_agents if num_agents_max is None else num_agents_max

        # Check dimension
        assert p_.shape[0] == v_.shape[0] == th_.shape[0], "p_, v_, th_ must have the same shape[0]"
        assert self.num_agents_min <= num_agents <= self.num_agents_max, "num_agents_max must be <= self.num_agents_max"
        assert num_agents_max == self.num_agents_max, "num_agents_max must be == self.num_agents_max"
        assert p_.shape[1] == v_.shape[1] == 2, "p_, v_ must have shape[1] == 2"
        assert th_.shape[1] == 1, "th_ must have shape[1] == 1"

        # Get initial agent states
        # # agent_states: [x, y, vx, vy, theta]
        p = np.zeros((num_agents_max, 2), dtype=np.float64)  # (num_agents_max, 2)
        p[:num_agents, :] = p_
        v = np.zeros((num_agents_max, 2), dtype=np.float64)  # (num_agents_max, 2)
        v[:num_agents, :] = v_
        th = np.zeros(num_agents_max, dtype=np.float64)  # (num_agents_max, )
        th[:num_agents] = th_.squeeze()  # squeeze (num_agents, 1) -> (num_agents,)
        # Concatenate p v th
        agent_states = np.concatenate([p, v, th[:, np.newaxis]], axis=1)  # (num_agents_max, 5)
        # # padding_mask
        padding_mask = np.zeros(num_agents_max, dtype=np.bool_)  # (num_agents_max, )
        padding_mask[:num_agents] = True
        # # neighbor_masks
        self.config.env.comm_range = comm_range
        if self.config.env.comm_range is None:
            neighbor_masks = np.ones((num_agents_max, num_agents_max), dtype=np.bool_)
        else:
            neighbor_masks = self.compute_neighbor_agents(
                agent_states=agent_states, padding_mask=padding_mask, communication_range=self.config.env.comm_range)[0]
        # # state!
        self.state = {"agent_states": agent_states, "neighbor_masks": neighbor_masks, "padding_mask": padding_mask}
        self.initial_state = self.state
        self.has_lost_comm = False

        # Get relative state
        self.rel_state = self.get_relative_state(state=self.state)

        # Get obs
        obs = self.get_obs(state=self.state, rel_state=self.rel_state, control_inputs=np.zeros(num_agents_max))

        return obs

    def reset(self):
        # Init time steps
        self.time_step = 0
        # self.agent_time_step = np.zeros(self.num_agents_max, dtype=np.int32)

        # Get initial num_agents
        self.num_agents = self.np_random.choice(self.num_agents_pool_np)  # randomly choose the num_agents
        padding_mask = np.zeros(self.num_agents_max, dtype=np.bool_)  # (num_agents_max, )
        padding_mask[:self.num_agents] = True

        # Init the state: agent_states [x,y,vx,vy,theta], neighbor_masks[T/F (n,n)], padding_mask[T/F (n)]
        # # Generate initial agent states
        p = np.zeros((self.num_agents_max, 2), dtype=np.float64)  # (num_agents_max, 2)
        l2 = self.config.control.initial_position_bound / 2
        p[:self.num_agents, :] = self.np_random.uniform(-l2, l2, size=(self.num_agents, 2))
        th = np.zeros(self.num_agents_max, dtype=np.float64)  # (num_agents_max, )
        th[:self.num_agents] = self.np_random.uniform(-np.pi, np.pi, size=(self.num_agents,))
        v = self.config.control.speed * np.stack([np.cos(th), np.sin(th)], axis=1)
        v[self.num_agents:] = 0
        # # Concatenate p v th
        agent_states = np.concatenate([p, v, th[:, np.newaxis]], axis=1)  # (num_agents_max, 5)
        if self.config.env.comm_range is None:
            neighbor_masks = np.ones((self.num_agents_max, self.num_agents_max), dtype=np.bool_)
        else:
            neighbor_masks = self.compute_neighbor_agents(
                agent_states=agent_states, padding_mask=padding_mask, communication_range=self.config.env.comm_range)[0]
        self.state = {"agent_states": agent_states, "neighbor_masks": neighbor_masks, "padding_mask": padding_mask}
        self.has_lost_comm = False

        # Get relative state
        self.rel_state = self.get_relative_state(state=self.state)

        # Get obs
        obs = self.get_obs(state=self.state, rel_state=self.rel_state, control_inputs=np.zeros(self.num_agents_max))

        # Historical data
        self.alignment_hist = np.zeros(self.config.env.max_time_steps)
        self.spatial_entropy_hist = np.zeros(self.config.env.max_time_steps)
        self.velocity_entropy_hist = np.zeros(self.config.env.max_time_steps)

        # Other settings
        if self.config.env.get_state_hist:
            self.agent_states_hist = np.zeros((self.config.env.max_time_steps, self.num_agents_max, 5))
            self.neighbor_masks_hist = np.zeros((self.config.env.max_time_steps, self.num_agents_max, self.num_agents_max))
            self.initial_state = self.state
        if self.config.env.get_action_hist:
            self.action_hist = np.zeros((self.config.env.max_time_steps, self.num_agents_max, self.num_agents_max), dtype=np.bool_)

        return obs

    def step(self, action):
        """
        Step the environment
        :param action: your_model_output; ndarray of shape (num_agents_max, num_agents_max) expected under the default
        :return: obs, reward, done, info
        """
        state = self.state  # state of the class (flock);
        rel_state = self.rel_state  # did NOT consider the communication network, DELIBERATELY

        # Interpret the action (i.e. model output)
        action_interpreted = self.interpret_action(model_output=action)
        joint_action = self.multi_to_single(action_interpreted) if self.config.env.env_mode == "multi_env" \
            else action_interpreted
        joint_action = self.to_binary_action(joint_action)  # (num_agents_max, num_agents_max)
        self.validate_action(action=joint_action,
                             neighbor_masks=state["neighbor_masks"], padding_mask=state["padding_mask"])

        # Step the environment in *single agent* setting!, which may be faster due to vectorization-like things
        # # s` = T(s, a)
        next_state, control_inputs, comm_loss_agents = self.env_transition(state, rel_state, joint_action)
        next_rel_state = self.get_relative_state(state=next_state)
        # # r = R(s, a, s`)
        rewards = self._compute_rewards(
            state=state, action=joint_action, next_state=next_state, control_inputs=control_inputs)
        # # o = H(s`)
        obs = self.get_obs(state=next_state, rel_state=next_rel_state, control_inputs=control_inputs)

        # Check episode termination
        done = self.check_episode_termination(state=next_state, rel_state=next_rel_state,
                                              comm_loss_agents=comm_loss_agents)

        # Get custom reward if implemented
        custom_reward = self.compute_custom_reward(state, rel_state, control_inputs, rewards, done)
        _reward = rewards.sum() / self.num_agents if self.config.env.env_mode == "single_env" else self.single_to_multi(rewards)
        reward = custom_reward if custom_reward is not NotImplemented else _reward

        # Collect info
        info = {
            "spatial_entropy": self.spatial_entropy_hist[self.time_step] if self.config.env.task_type == "acs" else None,
            "velocity_entropy": self.velocity_entropy_hist[self.time_step] if self.config.env.task_type == "acs" else None,
            "alignment": self.alignment_hist[self.time_step] if self.config.env.task_type == "vicsek" else None,
            "original_reward": _reward,
            "comm_loss_agents": comm_loss_agents,
        }
        info = self.get_extra_info(info, next_state, next_rel_state, control_inputs, rewards, done)
        if self.config.env.get_state_hist:
            self.agent_states_hist[self.time_step] = next_state["agent_states"]
            self.neighbor_masks_hist[self.time_step] = next_state["neighbor_masks"]
        if self.config.env.get_action_hist:
            self.action_hist[self.time_step] = joint_action

        # Update self.state and the self.rel_state
        self.state = next_state
        self.rel_state = next_rel_state
        # Update time steps
        self.time_step += 1
        # self.agent_time_step[state["padding_mask"]] += 1
        return obs, reward, done, info

    def get_relative_state(self, state):
        """
        Get the relative state (positions, velocities, headings, distances) from the absolute state
        """
        agent_positions = state["agent_states"][:, :2]
        agent_velocities = state["agent_states"][:, 2:4]
        agent_headings = state["agent_states"][:, 4, np.newaxis]  # shape (num_agents_max, 1): 2-D array
        # neighbor_masks = state["neighbor_masks"]  # shape (num_agents_max, num_agents_max)
        padding_mask = state["padding_mask"]  # shape (num_agents_max)

        # Get relative positions and distances
        if self.config.env.periodic_boundary:
            l = self.config.control.initial_position_bound
            # Get relative positions in normal boundary
            rel_agent_positions, _ = self.get_relative_info(
                data=agent_positions, mask=padding_mask, get_dist=False, get_active_only=False)
            # Transform the relative positions to the periodic boundary
            rel_agent_positions, rel_agent_dists = get_rel_pos_dist_in_periodic_boundary(
                rel_pos_normal=rel_agent_positions, width=l, height=l)
            # Remove padding agents (make zero)
            rel_agent_positions[~padding_mask, :, :][:, ~padding_mask, :] = 0  # (num_agents_max, num_agents_max, 2)
            rel_agent_dists[~padding_mask, :][:, ~padding_mask] = 0  # (num_agents_max, num_agents_max)
        else:
            rel_agent_positions, rel_agent_dists = self.get_relative_info(
                data=agent_positions, mask=padding_mask, get_dist=True, get_active_only=False)

        # Get relative velocities
        rel_agent_velocities, _ = self.get_relative_info(
            data=agent_velocities, mask=padding_mask, get_dist=False, get_active_only=False)

        # Get relative headings
        _, rel_agent_headings = self.get_relative_info(
            data=agent_headings, mask=padding_mask, get_dist=True, get_active_only=False)

        # rel_state: dict
        rel_state = {"rel_agent_positions": rel_agent_positions,
                     "rel_agent_velocities": rel_agent_velocities,
                     "rel_agent_headings": rel_agent_headings,
                     "rel_agent_dists": rel_agent_dists
                     }

        return rel_state

    def interpret_action(self, model_output):
        """
        Please implement this method as you need. Currently, it just passes the model_output.
        Interprets the model output
        :param model_output
        :return: interpreted_action
        """
        return model_output

    def validate_action(self, action, neighbor_masks, padding_mask):
        """
        Validates the action by checking the neighbor_mask and padding_mask
        :param action:  (num_agents_max, num_agents_max)
        :param neighbor_masks: (num_agents_max, num_agents_max)
        :param padding_mask: (num_agents_max)
        :return: None
        """
        # Check the dtype and shape of the action
        if self.config.env.action_type == "binary_vector":
            assert isinstance(action, np.ndarray), "action must be a numpy ndarray"
            assert np.issubdtype(action.dtype, np.integer), "action must be a numpy integer type"
            assert action.shape == (self.num_agents_max, self.num_agents_max), \
                "action must be a ndarray of shape (num_agents_max, num_agents_max)"

        # Ensure the diagonal elements are all ones (all with self-loops); if not, set them to ones
        if not np.all(np.diag(action) == 1):
            np.fill_diagonal(action, 1)  # Directly modifies the action array to set diagonal elements to 1
            print("WARNING (env.validate_action): diag(action) not all 1; Self-loops fixed in 'action'.")

        # Check action value based on the neighbor_mask and padding_mask
        # Note: your model might output a masked action. If that's not available, you can ignore this part.
        assert np.all((neighbor_masks | ~action)), "action[i, j] == 1 must not found if neighbor_mask[i, j] == 0"
        assert np.all((padding_mask[:, None] | ~action)), "action[i, j] == 1 must not found if padding_mask[j] == 0"

        # # Efficiently check for rows with all zeros (excluding self-loops)
        # if self.time_step != 0:
        #     assert np.all(action.sum(axis=1) - np.diag(action) > 0), \
        #         "Each row in action, except self-loops, must have at least one True value"

    def get_vicsek_action(self):
        neighbor_masks = self.state["neighbor_masks"]  # shape (num_agents_max, num_agents_max)
        padding_mask = self.state["padding_mask"]  # shape (num_agents_max)
        padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]  # (num_agents_max, num_agents_max)

        # Vicsek action: logical and between the neighbor_masks and the padding_mask_2d
        vicsek_action = neighbor_masks & padding_mask_2d  # (num_agents_max, num_agents_max)
        # Make vicsek_action an integer subtype numpy array
        vicsek_action = vicsek_action.astype(np.int8)

        return vicsek_action

    def to_binary_action(self, action_in_another_type):
        if self.config.env.action_type == "binary_vector":
            return action_in_another_type
        elif self.config.env.action_type == "radius":
            # action_in_another_type: ndarray of shape (num_agents_max, )
            # # action_in_another_type[i] is the radius of the communication range of agent i

            # Check if the radius is positive and less than the communication range (non-padded agents only)
            assert np.all(action_in_another_type[self.state["padding_mask"]] > 0), \
                "action_in_another_type[i] must be > 0 for all non-padded agents"
            assert np.all(action_in_another_type[self.state["padding_mask"]] <= 1), \
                "action_in_another_type[i] must be <= self.comm_range for all non-padded agents"

            # Set the action
            agent_wise_comm_range = self.config.env.comm_range * action_in_another_type[
                self.state["padding_mask"], np.newaxis]  # (num_agents, 1)
            action_in_binary = self.compute_neighbor_agents(
                agent_states=self.state["agent_states"], padding_mask=self.state["padding_mask"],
                communication_range=agent_wise_comm_range)[0]
            return action_in_binary  # (num_agents_max, num_agents_max)
        elif self.config.env.action_type == "continuous_vector":
            raise NotImplementedError("continuous_vector action_type is not implemented yet")
        return None

    def multi_to_single(self, variable_in_multi: MultiAgentDict):
        """
        Converts a multi-agent variable to a single-agent variable
        Assumption: homogeneous agents
        :param variable_in_multi: dict {agent_name_suffix + str(i): variable_in_single[i]}; {str: ndarray}
        :return: variable_in_single: ndarray of shape (num_agents, data...)
        """
        # Add extra dimension of each agent's variable on axis=0
        assert variable_in_multi[self.config.env.agent_name_prefix + str(0)].shape[0] == self.num_agents, \
            "num_agents must == variable_in_multi['agent_0'].shape[0]"
        variable_in_single = np.array(variable_in_multi.values())  # (num_agents, ...)

        return variable_in_single

    def single_to_multi(self, variable_in_single: np.ndarray):
        """
        Converts a single-agent variable to a multi-agent variable
        Assumption: homogeneous agents
        :param variable_in_single: ndarray of shape (num_agents, data...)
        :return: variable_in_multi
        """
        # Remove the extra dimension of each agent's variable on axis=0 and use self.agent_name_suffix with i as keys
        variable_in_multi = {}
        assert variable_in_single.shape[0] == self.num_agents_max, "variable_in_single[0] must be self.num_agents_max"
        for i in range(self.num_agents_max):
            variable_in_multi[self.config.env.agent_name_prefix + str(i)] = variable_in_single[i]

        return variable_in_multi

    def env_transition(self, state, rel_state, action, action_lazy_control=None):
        """
        Transition the environment; all args in single-rl-agent settings
        s` = T(s, a); deterministic
        :param state: dict:
        :param rel_state: dict:
        :param action: ndarray of shape (num_agents_max, num_agents_max)
        :param action_lazy_control: ndarray of shape (num_agents_max, )
        :return: next_state: dict; control_inputs: (num_agents_max, )
        """
        # Validate the laziness_vectors
        # self.validate_action(action=action, neighbor_masks=state["neighbor_masks"], padding_mask=state["padding_mask"])

        # 0. Apply lazy message actions: alters the neighbor_masks!
        lazy_listening_msg_masks = np.logical_and(state["neighbor_masks"], action)  # (num_agents_max, num_agents_max)

        # 1. Get control inputs based on the flocking control algorithm with the lazy listener's network
        if self.config.env.task_type == "vicsek":
            control_inputs = self.get_vicsek_control(state, rel_state, lazy_listening_msg_masks)  # (num_agents_max, )
        elif self.config.env.task_type == "acs":
            control_inputs = self.get_acs_control(state, rel_state, lazy_listening_msg_masks)  # (num_agents_max, )
        else:
            raise NotImplementedError("task_type must be either vicsek or acs")

        # # 2. Apply lazy control actions: alters the control_inputs!
        # control_inputs = action_lazy_control * control_inputs if action_lazy_control is not None else control_inputs

        # 3. Update the agent states based on the control inputs
        next_agent_states = self.update_agent_states(state=state, control_inputs=control_inputs)

        # 4. Update network topology (i.e. neighbor_masks) based on the new agent states
        if self.config.env.comm_range is None:
            next_neighbor_masks = state["neighbor_masks"]
            comm_loss_agents = None
        else:
            next_neighbor_masks, comm_loss_agents = self.compute_neighbor_agents(
                agent_states=next_agent_states, padding_mask=state["padding_mask"],
                communication_range=self.config.env.comm_range)

        # 5. Update the active agents (i.e. padding_mask); you may lose or gain agents
        # next_padding_mask = self.update_active_agents(
        #     agent_states=next_agent_states, padding_mask=state["padding_mask"], communication_range=self.comm_range)
        # self.num_agents = next_padding_mask.sum()  # update the number of agents

        # 6. Update the state
        next_state = {"agent_states": next_agent_states,
                      "neighbor_masks": next_neighbor_masks,
                      "padding_mask": state["padding_mask"]
                      }

        return next_state, control_inputs, comm_loss_agents

    def get_vicsek_control(self, state, rel_state, new_network):
        """
        Get the control inputs based on the agent states using the Vicsek Model
        :return: u (num_agents_max)
        """
        # Please Work with Active Agents Only

        # Get rel_pos, rel_dist, rel_vel, rel_ang, abs_ang, padding_mask, neighbor_masks
        # rel_pos = rel_state["rel_agent_positions"]  # (num_agents_max, num_agents_max, 2)
        # rel_dist = rel_state["rel_agent_dists"]  # (num_agents_max, num_agents_max)
        # rel_vel = rel_state["rel_agent_velocities"]  # (num_agents_max, num_agents_max, 2)
        rel_ang = rel_state["rel_agent_headings"]  # (num_agents_max, num_agents_max)
        # abs_ang = state["agent_states"][:, 4]  # (num_agents_max, )
        padding_mask = state["padding_mask"]  # (num_agents_max)
        neighbor_masks = new_network  # (num_agents_max, num_agents_max)

        # Get data of the active agents
        active_agents_indices = np.nonzero(padding_mask)[0]  # (num_agents, )
        active_agents_indices_2d = np.ix_(active_agents_indices, active_agents_indices)  # (num_agents,num_agents)
        # p = rel_pos[active_agents_indices_2d]  # (num_agents, num_agents, 2)
        # r = rel_dist[active_agents_indices_2d] + (
        #             np.eye(self.num_agents) * np.finfo(float).eps)  # (num_agents, num_agents)
        # v = rel_vel[active_agents_indices_2d]  # (num_agents, num_agents, 2)
        th = rel_ang[active_agents_indices_2d]  # (num_agents, num_agents)
        # th_i = abs_ang[padding_mask]  # (num_agents, )
        net = neighbor_masks[active_agents_indices_2d]  # (num_agents, num_agents) might: no self-loops (i.e. 0 on diag)
        n = (net + (np.eye(self.num_agents) * np.finfo(float).eps)).sum(axis=1)  # (num_agents, )

        # Get control for Vicsek Model
        relative_heading_network_filtered = th * net  # (num_agents, num_agents)
        average_heading = relative_heading_network_filtered.sum(axis=1) / n  # (num_agents, )
        average_heading_rate = average_heading / self.config.env.dt  # (num_agents, )

        # Get control config
        u_max = self.config.control.max_turn_rate

        # 3. Saturation
        u_active = np.clip(average_heading_rate, -u_max, u_max)  # (num_agents, )

        # 4. Padding
        u = np.zeros(self.num_agents_max, dtype=np.float32)  # (num_agents_max, )
        u[padding_mask] = u_active  # (num_agents_max, )

        return u

    def get_acs_control(self, state, rel_state, new_network):
        """
        Get the control inputs based on the agent states using the ACS Model
        :return: u (num_agents_max)
        """
        """
        Get the control inputs based on the agent states
        :return: control_inputs (num_agents_max)
        """
        #  PLEASE WORK WITH ACTIVE AGENTS ONLY

        # Get rel_pos, rel_dist, rel_vel, rel_ang, abs_ang, padding_mask, neighbor_masks
        rel_pos = rel_state["rel_agent_positions"]   # (num_agents_max, num_agents_max, 2)
        rel_dist = rel_state["rel_agent_dists"]      # (num_agents_max, num_agents_max)
        rel_vel = rel_state["rel_agent_velocities"]  # (num_agents_max, num_agents_max, 2)
        rel_ang = rel_state["rel_agent_headings"]    # (num_agents_max, num_agents_max)
        abs_ang = state["agent_states"][:, 4]        # (num_agents_max, )
        padding_mask = state["padding_mask"]         # (num_agents_max)
        neighbor_masks = new_network  # (num_agents_max, num_agents_max)

        # Get data of the active agents
        active_agents_indices = np.nonzero(padding_mask)[0]  # (num_agents, )
        active_agents_indices_2d = np.ix_(active_agents_indices, active_agents_indices)  # (num_agents,num_agents)
        p = rel_pos[active_agents_indices_2d]  # (num_agents, num_agents, 2)
        r = rel_dist[active_agents_indices_2d] + (np.eye(self.num_agents)*np.finfo(float).eps) #(num_agents, num_agents)
        v = rel_vel[active_agents_indices_2d]  # (num_agents, num_agents, 2)
        th = rel_ang[active_agents_indices_2d]  # (num_agents, num_agents)
        th_i = abs_ang[padding_mask]  # (num_agents, )
        net = neighbor_masks[active_agents_indices_2d]  # (num_agents, num_agents) may be no self-loops (i.e. 0 on diag)
        N = (net + (np.eye(self.num_agents) * np.finfo(float).eps)).sum(axis=1)  # (num_agents, )

        # Get control config
        beta = self.config.control.beta
        lam = self.config.control.lam
        k1 = self.config.control.k1
        k2 = self.config.control.k2
        spd = self.config.control.speed
        u_max = self.config.control.max_turn_rate
        r0 = self.config.control.r0
        sig = self.config.control.sig

        # 1. Compute Alignment Control Input
        # # u_cs = (lambda/n(N_i)) * sum_{j in N_i}[ psi(r_ij)sin(θ_j - θ_i) ],
        # # where N_i is the set of neighbors of agent i,
        # # psi(r_ij) = 1/(1+r_ij^2)^(beta),
        # # r_ij = ||X_j - X_i||, X_i = (x_i, y_i),
        psi = (1 + r**2)**(-beta)  # (num_agents, num_agents)
        alignment_error = np.sin(th)  # (num_agents, num_agents)
        u_cs = (lam / N) * (psi * alignment_error * net).sum(axis=1)  # (num_agents, )

        # 2. Compute Cohesion and Separation Control Input
        # # u_coh[i] = (sigma/N*V)
        # #            * sum_(j in N_i)
        # #               [
        # #                   {
        # #                       (K1/(2*r_ij^2))*<-rel_vel, -rel_pos> + (K2/(2*r_ij^2))*(r_ij-R)
        # #                   }
        # #                   * <[-sin(θ_i), cos(θ_i)]^T, rel_pos>
        # #               ]
        # # where N_i is the set of neighbors of agent i,
        # # r_ij = ||X_j - X_i||, X_i = (x_i, y_i),
        # # rel_vel = (vx_j - vx_i, vy_j - vy_i),
        # # rel_pos = (x_j - x_i, y_j - y_i),
        sig_NV = sig / (N * spd)  # (num_agents, )
        k1_2r2 = k1 / (2 * r**2)  # (num_agents, num_agents)
        k2_2r = k2 / (2 * r)  # (num_agents, num_agents)
        v_dot_p = np.einsum('ijk,ijk->ij', v, p)  # (num_agents, num_agents)
        r_minus_r0 = r - r0  # (num_agents, num_agents)
        sin_th_i = -np.sin(th_i)  # (num_agents, )
        cos_th_i = np.cos(th_i)   # (num_agents, )
        dir_dot_p = sin_th_i[:, np.newaxis]*p[:, :, 0] + cos_th_i[:, np.newaxis]*p[:, :, 1]  # (num_agents, num_agents)
        u_coh = sig_NV * np.sum((k1_2r2 * v_dot_p + k2_2r * r_minus_r0) * dir_dot_p * net, axis=1)  # (num_agents, )

        # 3. Saturation
        u_active = np.clip(u_cs + u_coh, -u_max, u_max)  # (num_agents, )

        # 4. Padding
        u = np.zeros(self.num_agents_max, dtype=np.float32)  # (num_agents_max, )
        u[padding_mask] = u_active  # (num_agents_max, )

        return u

    @staticmethod
    def filter_active_agents_data(data, padding_mask):
        """
        Filters out the data of the inactive agents
        :param data: (num_agents_max, num_agents_max, ...)
        :param padding_mask: (num_agents_max)
        :return: active_data: (num_agents, num_agents, ...)
        """
        # Step 1: Find indices of active agents
        active_agents_indices = np.nonzero(padding_mask)[0]  # (num_agents, )

        # Step 2: Use these indices to index into the data array
        active_data = data[np.ix_(active_agents_indices, active_agents_indices)]

        return active_data

    def update_agent_states(self, state, control_inputs):
        padding_mask = state["padding_mask"]

        # 0. <- 3. Positions
        next_agent_positions = (state["agent_states"][:, :2]
                                + state["agent_states"][:, 2:4] * self.config.env.dt)  # (n_a_max, 2)
        if self.config.env.periodic_boundary:
            w = h = self.config.control.initial_position_bound
            next_agent_positions = wrap_to_rectangle(next_agent_positions, w, h)
        # 1. Headings
        next_agent_headings = state["agent_states"][:, 4] + control_inputs * self.config.env.dt  # (num_agents_max, )
        # next_agent_headings = np.mod(next_agent_headings, 2 * np.pi)  # (num_agents_max, )
        # 2. Velocities
        v = self.config.control.speed
        next_agent_velocities = np.zeros((self.num_agents_max, 2), dtype=np.float32)  # (num_agents_max, 2)
        next_agent_velocities[padding_mask] = v * np.stack([np.cos(next_agent_headings[padding_mask]),
                                                            np.sin(next_agent_headings[padding_mask])], axis=1)
        # 3. Positions
        # next_agent_positions = state["agent_states"][:, :2] + next_agent_velocities * self.dt  # (num_agents_max, 2)
        # 4. Concatenate
        next_agent_states = np.concatenate(  # (num_agents_max, 5)
            [next_agent_positions, next_agent_velocities, next_agent_headings[:, np.newaxis]], axis=1)

        return next_agent_states  # This did not update the neighbor_masks; it is done in the env_transition

    def compute_neighbor_agents(self, agent_states, padding_mask, communication_range, includes_self_loops=True):
        """
        1. Computes the neighbor matrix based on communication range
        2. Excludes the padding agents (i.e. mask_value==0)
        3. (By default) Includes the self-loops
        """
        self_loop = includes_self_loops  # True if includes self-loops; False otherwise
        agent_positions = agent_states[:, :2]  # (num_agents_max, 2)
        # Get active relative distances
        if self.config.env.periodic_boundary:
            rel_pos_normal, _ = self.get_relative_info(data=agent_positions, mask=padding_mask,
                                                       get_dist=False, get_active_only=True)
            width = height = self.config.control.initial_position_bound
            _, rel_dist = get_rel_pos_dist_in_periodic_boundary(rel_pos_normal, width, height)
        else:
            _, rel_dist = self.get_relative_info(data=agent_positions, mask=padding_mask,
                                                 get_dist=True, get_active_only=True)

        # Get active neighbor masks
        active_neighbor_masks = rel_dist <= communication_range  # (num_agents, num_agents)
        if not includes_self_loops:
            np.fill_diagonal(active_neighbor_masks, self_loop)  # Set the diagonal to 0 if the mask don't include loops
        # Get the next neighbor masks
        next_neighbor_masks = np.zeros((self.num_agents_max, self.num_agents_max),
                                       dtype=np.bool_)  # (num_agents_max, num_agents_max)
        active_agents_indices = np.nonzero(padding_mask)[0]  # (num_agents, )
        next_neighbor_masks[np.ix_(active_agents_indices, active_agents_indices)] = active_neighbor_masks

        # Check no neighbor agents (be careful neighbor mask may not include self-loops)
        neighbor_nums = next_neighbor_masks.sum(axis=1)  # (num_agents_max, )
        if not includes_self_loops:
            neighbor_nums += 1  # Add 1 for artificial self-loops
        comm_loss_agents = np.logical_and(padding_mask, neighbor_nums == 1)  # is alone in the network?

        return next_neighbor_masks, comm_loss_agents  # (num_agents_max, num_agents_max), (num_agents_max)

    def get_relative_info(self, data, mask, get_dist=False, get_active_only=False):
        """
        Returns the *relative information(s)* of the agents (e.g. relative position, relative angle, etc.)
        :param data: (num_agents_max, data_dim) ## EVEN IF YOU HAVE 1-D data (i.e. data_dim==1), USE 2-D ARRAY
        :param mask: (num_agents_max)
        :param get_dist:
        :param get_active_only:
        :return: rel_data, rel_dist

        Note:
            - Assumes fully connected communication network
            - If local network needed,
            - Be careful with the **SHAPE** of the input **MASK**;
            - Also, the **MASK** only accounts for the *ACTIVE* agents (similar to padding_mask)
        """

        # Get dimension of the data
        assert data.ndim == 2  # we use a 2D array for the data
        assert data[mask].shape[0] == self.num_agents  # validate the mask
        assert data.shape[0] == self.num_agents_max
        data_dim = data.shape[1]

        # Compute relative data
        # rel_data: shape (num_agents_max, num_agents_max, data_dim); rel_data[i, j] = data[j] - data[i]
        # rel_data_active: shape (num_agents, num_agents, data_dim)
        # rel_data_active := data[mask] - data[mask, np.newaxis, :]
        rel_data_active = data[np.newaxis, mask, :] - data[mask, np.newaxis, :]
        if get_active_only:
            rel_data = rel_data_active
        else:
            rel_data = np.zeros((self.num_agents_max, self.num_agents_max, data_dim), dtype=np.float32)
            rel_data[np.ix_(mask, mask, np.arange(data_dim))] = rel_data_active
            # rel_data[mask, :, :][:, mask, :] = rel_data_active  # not sure; maybe 2-D array (not 3-D) if num_true = 1

        # Compute relative distances
        # rel_dist: shape (num_agents_max, num_agents_max)
        # Note: data are all non-negative!!
        if get_dist:
            rel_dist = np.linalg.norm(rel_data, axis=2) if data_dim > 1 else rel_data.squeeze()
        else:
            rel_dist = None

        # get_active_only==False: (num_agents_max, num_agents_max, data_dim), (num_agents_max, num_agents_max)
        # get_active_only==True: (num_agents, num_agents, data_dim), (num_agents, num_agents)
        # get_dist==False: (n, n, d), None
        return rel_data, rel_dist

    def _compute_rewards(self, state, action, next_state, control_inputs: np.ndarray):
        """
        Compute the rewards; Be careful with the **dimension** of *rewards*
        :param control_inputs: (num_agents_max)
        :return: rewards: (num_agents_max)
        """
        if self.config.env.task_type=='vicsek':
            speed = self.config.control.speed
            velocities = state["agent_states"][:, 2:4]  # (num_agents_max, 2)
            average_velocity = np.mean(velocities, axis=0)  # (2, )
            alignment = np.linalg.norm(average_velocity) / speed  # scalar in range [0, 1]
            self.alignment_hist[self.time_step] = alignment

            rewards = np.repeat(alignment, self.num_agents_max)  # (num_agents_max, )
            rewards[~state["padding_mask"]] = 0

            return rewards  # (num_agents_max, )
        elif self.config.env.task_type=='acs':
            rho = self.config.control.rho

            # Heading rate control cost
            heading_rate_costs = (self.config.env.dt * self.config.control.speed) * np.abs(control_inputs)  # (num_agents_max, )
            # Cruise cost (time penalty)
            cruise_costs = self.config.env.dt * np.ones(self.num_agents_max, dtype=np.float32)  # (num_agents_max, )

            rewards = - (heading_rate_costs + (rho * cruise_costs))  # (num_agents_max, )
            rewards[~state["padding_mask"]] = 0

            return rewards  # (num_agents_max, )
        else:
            raise NotImplementedError("task_type not implemented yet")

    def get_obs(self, state, rel_state, control_inputs):
        """
        Get the observation
        i-th agent's observation: [x, y, vx, vy] with its neighbors' info (and padding info) if necessary
        If periodic boundary, the position will be transformed to sin-cos space
          i.e. o_i := [cos(x), sin(x), cos(y), sin(y), vx, vy]
        :return: obs
        """
        # (0) Get masks
        # # We assume that the neighbor_masks are up-to-date and include the paddings (0) and self-loops (1)
        neighbor_masks = state["neighbor_masks"]  # (num_agents_max, num_agents_max); self not included
        padding_mask = state["padding_mask"]
        active_agents_indices = np.nonzero(padding_mask)[0]  # (num_agents, )
        active_agents_indices_2d = np.ix_(active_agents_indices, active_agents_indices)
        # # Add self-loops only for the active agents
        # neighbor_masks_with_self_loops = neighbor_masks.copy()
        # neighbor_masks_with_self_loops[active_agents_indices_2d] = 1

        # Branch based on observation_type
        obs_type = getattr(getattr(self.config, "env", None), "observation_type", None) or "ego_centric"
        if obs_type == "centralized":
            agents_obs = self._get_centralized_obs(state, active_agents_indices, padding_mask)
        elif obs_type == "ego_centric":
            agents_obs = self._get_ego_centric_obs(
                rel_state, active_agents_indices, active_agents_indices_2d
            )
        else: # None is fine, but typo is not accepted!
            raise ValueError(f"Invalid observation_type: {obs_type}")

        # Construct observation
        post_processed_obs = self.post_process_obs(agents_obs, neighbor_masks, padding_mask)
        # # In case of post-processing applied
        if post_processed_obs is not NotImplemented:
            return post_processed_obs
        # # In case of the base implementation (with no post-processing)
        if self.config.env.env_mode == "single_env":
            obs = {"local_agent_infos": agents_obs,     # (num_agents_max, num_agents_max, obs_dim)
                   "neighbor_masks": neighbor_masks,    # (num_agents_max, num_agents_max)
                   "padding_mask": padding_mask,        # (num_agents_max)
                   "is_from_my_env": np.array(True, dtype=np.bool_),
                   }
            return obs
        else:
            raise ValueError(f"self.env_mode: 'single_env' / 'multi_env'; not {self.config.env.env_mode}; in get_obs()")

    def _get_ego_centric_obs(self, rel_state, active_agents_indices, active_agents_indices_2d):
        """
        Get ego-centric observation: each agent has its own local view of neighbors' relative positions/headings.
        Shape: (num_agents_max, num_agents_max, obs_dim)
        """
        # TODO: MUST BE UPDATED -- currently position coordinate uses relative in translation but absolute in rotation!!
        # (1) Get [x, y], [vx==cos(th), vy] in rel_state (active agents only)
        active_agents_rel_positions = rel_state["rel_agent_positions"][active_agents_indices_2d]  # (n, n, 2)
        active_agents_rel_headings = rel_state["rel_agent_headings"][active_agents_indices_2d]
        active_agents_rel_headings = active_agents_rel_headings[:, :, np.newaxis]  # (num_agents, num_agents, 1)

        # (2) Map periodic to continuous space if necessary: [x, y] -> [cos(x), sin(x), cos(y), sin(y)]
        l = self.config.control.initial_position_bound
        if self.config.env.periodic_boundary:
            # (num_agents, num_agents, 4)
            active_agents_rel_positions = map_periodic_to_continuous_space(active_agents_rel_positions, l, l)
        else:  # needs normalization
            # (num_agents, num_agents, 2)
            active_agents_rel_positions = active_agents_rel_positions / (l/2.0)

        # (3) Concat all
        active_agents_obs = np.concatenate(
            [active_agents_rel_positions,    # (num_agents, num_agents, 4 or 2)
             np.cos(active_agents_rel_headings),  # (num_agents, num_agents, 1)
             np.sin(active_agents_rel_headings),  # (num_agents, num_agents, 1)
             ],
            axis=2
        )  # (num_agents, num_agents, obs_dim)
        agents_obs = np.zeros((self.num_agents_max, self.num_agents_max, self.config.env.obs_dim), dtype=np.float64)
        agents_obs[active_agents_indices_2d] = active_agents_obs  # (num_agents_max, num_agents_max, obs_dim)

        return agents_obs

    def _get_centralized_obs(self, state, active_agents_indices, padding_mask):
        """
        Get centralized observation: all agents share the same global view.
        Positions and headings are relative to the swarm center and average heading.
        The single centralized obs (num_agents_max, obs_dim) is replicated to (num_agents_max, num_agents_max, obs_dim)
        so each agent "row" sees the same global info for all agents (columns).
        
        This follows the same logic as lazy-message-listener-flocking's get_obs().
        """
        # Get positions and headings (active agents only)
        active_agent_positions = state["agent_states"][:, :2][active_agents_indices]  # (num_agents, 2)
        active_agent_headings = state["agent_states"][:, 4][active_agents_indices, np.newaxis]  # (num_agents, 1)
        active_agent_headings = wrap_to_pi(active_agent_headings)  # MUST be wrapped to [-pi, pi] before averaging

        # Transform to translation-rotation-invariant space (swarm coordinate system)
        # # Compute swarm center and average heading
        center = np.mean(active_agent_positions, axis=0)  # (2, )
        average_heading = np.mean(active_agent_headings)  # scalar

        # # Rotation matrix to rotate by -average_heading (align swarm heading to +X axis)
        cos_neg_avg = np.cos(-average_heading)
        sin_neg_avg = np.sin(-average_heading)
        rot_mat = np.array([[cos_neg_avg, -sin_neg_avg],
                            [sin_neg_avg, cos_neg_avg]])  # (2, 2)

        # # Transform positions: translate to center, then rotate
        active_p_transformed = active_agent_positions - center  # (num_agents, 2)
        # Apply rotation: R @ p.T -> (2, num_agents), then transpose back
        active_p_transformed = (rot_mat @ active_p_transformed.T).T  # (num_agents, 2)

        # # Transform headings: relative to average heading
        active_th_transformed = active_agent_headings - average_heading  # (num_agents, 1)

        # # Concat [x, y, cos(th), sin(th)]
        active_pvu = np.concatenate(
            [active_p_transformed,  # (num_agents, 2)
             np.cos(active_th_transformed),  # (num_agents, 1)
             np.sin(active_th_transformed),  # (num_agents, 1)
             ],
            axis=1
        )  # (num_agents, obs_dim=4)

        # # Fill into (num_agents_max, obs_dim) with padding zeros
        centralized_obs = np.zeros((self.num_agents_max, self.config.env.obs_dim), dtype=np.float64)
        centralized_obs[padding_mask] = active_pvu  # (num_agents_max, obs_dim)

        # # Normalize by [init_bound/2, init_bound/2, 1, 1] (same as legacy)
        init_bound = self.config.control.initial_position_bound
        centralized_obs[:, :2] /= (init_bound / 2)

        # Replicate the centralized obs for each agent (row) to form (num_agents_max, num_agents_max, obs_dim)
        # Each row i contains the same global observation of all agents
        agents_obs = np.tile(centralized_obs[np.newaxis, :, :], (self.num_agents_max, 1, 1))
        # agents_obs shape: (num_agents_max, num_agents_max, obs_dim)

        return agents_obs

    def post_process_obs(self, agent_observations, neighbor_masks, padding_mask):
        """
        Implement your logic; e.g. flatten the obs for MLP if the MLP doesn't use action masks or so...
        """
        return NotImplemented

    def check_episode_termination(self, state, rel_state, comm_loss_agents):
        """
        Check if the episode is terminated:
        1. If the alignment is achieved
        2. If the max_time_step is reached
        3. If communication is lost
        :return: done(s)
        """
        padding_mask = state["padding_mask"]
        done = False

        # 1. Check if the control task is done
        if self.config.env.task_type=='vicsek':
            # Check alignment
            if self.alignment_hist[self.time_step] > self.config.env.alignment_goal:
                if not self.config.env.use_fixed_episode_length:
                    win_len = self.config.env.alignment_window_length - 1
                    if self.time_step >= win_len:
                        last_n_alignments = self.alignment_hist[self.time_step - win_len:self.time_step + 1]
                        max_alignment = np.max(last_n_alignments)
                        min_alignment = np.min(last_n_alignments)
                        if max_alignment - min_alignment < self.config.env.alignment_rate_goal:
                            done = True
        elif self.config.env.task_type=='acs':
            # Get the spatial and velocity entropy and assign to the histograms
            spatial_entropy, velocity_entropy = self._get_entropy(state)
            self.spatial_entropy_hist[self.time_step] = spatial_entropy
            self.velocity_entropy_hist[self.time_step] = velocity_entropy

            # Check if the spatial and velocity entropies are within the goals
            if (spatial_entropy < self.config.env.entropy_p_goal) and (velocity_entropy < self.config.env.entropy_v_goal):
                if not self.config.env.use_fixed_episode_length:
                    # Check if the entropies are stable over the last N steps (entropy rate checks)
                    effective_win_len = self.config.env.entropy_rate_window_length - 1
                    if self.time_step >= effective_win_len:
                        last_n_spatial_entropies = self.spatial_entropy_hist[self.time_step - effective_win_len:self.time_step + 1]
                        last_n_velocity_entropies = self.velocity_entropy_hist[self.time_step - effective_win_len:self.time_step + 1]
                        spatial_entropy_rate = np.max(last_n_spatial_entropies) - np.min(last_n_spatial_entropies)
                        velocity_entropy_rate = np.max(last_n_velocity_entropies)
                        if (spatial_entropy_rate < self.config.env.entropy_p_rate_goal) and (velocity_entropy_rate < self.config.env.entropy_v_rate_goal):
                            done = True
        else:
            raise NotImplementedError(f"task_type({self.config.env.task_type}) not implemented/supported yet")

        # 2. Check max_time_step
        if self.time_step >= self.config.env.max_time_steps - 1:
            done = True

        # 3. Check communication loss
        if self.config.env.comm_range is not None:
            if comm_loss_agents.any() and not done:
                done = False if self.config.env.ignore_comm_lost_agents else True
                self.lost_comm_step = self.time_step if self.has_lost_comm is not None else self.lost_comm_step
                self.has_lost_comm = True

        # 4. (Optional) Handle dones in the multi-env
        if self.config.env.env_mode == "single_env":
            return done
        elif self.config.env.env_mode == "multi_env":
            # padding agents: False
            dones_in_array = np.ones(self.num_agents_max, dtype=np.bool_)
            # done for swarm agents
            dones_in_array[padding_mask] = done
            dones = self.single_to_multi(dones_in_array)
            # Add "__all__" key to the dones dict
            dones["__all__"] = done
            return dones
        else:
            raise ValueError(f"self.env_mode: 'single_env' / 'multi_env'; not {self.config.env.env_mode}; in check_episode_termination()")

    def _get_entropy(self, state):
        padding_mask = state["padding_mask"]
        masked_states = state["agent_states"][padding_mask]  # (num_agents, 4)  <- for optimized indexing
        agent_positions = masked_states[:, :2]    # (num_agents, 2)
        agent_velocities = masked_states[:, 2:4]  # (num_agents, 2)

        # Get spatial and velocity entropy
        spatial_entropy = np.sqrt(np.sum(np.var(agent_positions, axis=0)))    # scalar
        velocity_entropy = np.sqrt(np.sum(np.var(agent_velocities, axis=0)))  # scalar

        return spatial_entropy, velocity_entropy

    def get_extra_info(self, info, state, rel_state, control_inputs, rewards, done):
        return info

    def compute_custom_reward(self, state, rel_state, control_inputs, rewards, done):
        """
        Impelment your custom reward logic
        :return: custom_reward
        """
        if self.config.env.is_training and self.config.env.task_type=='acs':
            # Get spatial entropy errors
            std_pos = self.spatial_entropy_hist[self.time_step]  # scalar
            std_pos_target = self.config.env.entropy_p_goal - 2.5
            std_pos_error = (std_pos - std_pos_target) ** 2  # (100-40)**2 = 3600
            pos_error_reward = - (1 / 3600) * np.maximum(std_pos_error, 0)

            # Get velocity entropy errors
            std_vel = self.velocity_entropy_hist[self.time_step]
            std_vel_target = self.config.env.entropy_v_goal - 0.05
            std_vel_error = (std_vel - std_vel_target) ** 2  # (15-0.05)**2 = 223.5052
            vel_error_reward = - (1 / 220) * np.maximum(std_vel_error, 0.0)

            # Get control cost
            rho = self.config.control.rho
            control_cost = rewards.sum() / self.num_agents
            control_cost = control_cost + (rho * self.config.env.dt)

            # Get the custom reward
            custom_reward = (self.config.env.acs_train_w_pos * pos_error_reward
                             + self.config.env.acs_train_w_vel * vel_error_reward
                             - self.config.env.acs_train_w_ctrl * control_cost)
            return custom_reward

        return NotImplemented

    def render(self, mode='human'):
        """
        Render the environment
        :param mode:
        :return:
        """
        pass


if __name__ == "__main__":
    my_seed_id = 0
    env = NeighborSelectionFlockingEnv(yaml_path='envs/default_env_config.yaml', seed_id=my_seed_id)
    print(pretty_print(env.config.dict()))
    # env.get_default_config_dict()
    print("Paused here for demonstration")

    obs = env.reset()
    num_agents_ = env.num_agents

    # fully_connected_action = np.ones(shape=(n, n), dtype=np.int8)
    individually_isolated_action = np.eye(num_agents_, dtype=np.int8)
    for _ in range(3):
        # obs, reward, done, info = env.step(fully_connected_action)
        obs_, reward_, done_, info_ = env.step(individually_isolated_action)

    print("Paused here for demonstration")
