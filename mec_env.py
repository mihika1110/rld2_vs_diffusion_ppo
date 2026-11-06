from typing import Dict, Tuple, Optional, Any
import numpy as np
from mumec import World, Simulator, Globe, CFields, FTYPE


class MECEnv:

    def __init__(
        self,
        n_stations: int = 2,
        n_users: int = 4,
        weights: Tuple[float, float] = (0.5, 0.5),
        partial: bool = False,
        seed: Optional[int] = None,
        max_steps: int = Globe.max_steps,
    ):
        self.n_stations = int(n_stations)
        self.n_users = int(n_users)
        self.weights = np.array(weights, dtype=FTYPE)
        self.partial = bool(partial)
        self.max_steps = int(max_steps)
        self._seed = seed

        #build world config
        self.world = World.New(N=self.n_stations, M=self.n_users)
        self.sim = Simulator(
            world=self.world,
            weights=self.weights,
            partial=self.partial,
            frozen=False, #will move to next state 
            seed=self._seed,
        )

        self.observation_dim = self.sim.observation_dim
        self.action_dim = self.n_users
        self.n_actions = self.n_stations + 1  # 0 = local, 1..N = BS

    #update seed initialisation
    def seed(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self.sim.seed = seed
        self.sim.rng = np.random.default_rng(seed)


    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed(seed)

        self.sim.Reset()
        obs = self.sim.GetObs().astype(np.float32)

        info = {
            "step": self.sim.ts,
            "horizon": self.sim.horizon,
            "user_cpu_capacity": float(np.mean(self.sim.U[:, 5])),  # UFields.F
            "bs_cpu_capacity": float(np.mean(self.sim.B[:, 4])),    # BFields.F
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if action is not None:
            action = np.asarray(action, dtype=np.int64).reshape(-1)
            assert len(action) == self.n_users, (
                f"Action length {len(action)} does not match n_users {self.n_users}"
            )
            assert np.all(action>=0) and np.all(action < self.n_actions), (
                f"Action values must be in [0, {self.n_actions - 1}], got {action}"
            )

        #Simulator.Step returns per-user weighted cost array of shape (M,)
        user_costs = self.sim.Step(action)
        executed_action = np.array(self.sim.last_action, dtype=np.int64)

        #mean cost and reward (scaled for critic convergence and RL stability)
        mean_cost = float(np.mean(user_costs))
        reward = -0.1 * mean_cost

        #extract underlying time and energy costs
        #sim.costs has shape: (lenCFields, M, N+1)
        time_costs = self.sim.costs[CFields.Time, np.arange(self.n_users), executed_action]
        energy_costs = self.sim.costs[CFields.Energy, np.arange(self.n_users), executed_action]
        mean_time = float(np.mean(time_costs))
        mean_energy = float(np.mean(energy_costs))

        #offloading metrics
        offload_count = int(np.sum(executed_action > 0))
        offload_ratio = float(offload_count/self.n_users)

        obs = self.sim.GetObs().astype(np.float32)
        terminated = bool(self.sim.terminated)
        truncated = bool(self.sim.truncated)

        info = {
            "step": self.sim.ts,
            "cost_mean": mean_cost,
            "cost_per_user": user_costs.tolist(),
            "time_mean": mean_time,
            "energy_mean": mean_energy,
            "offload_ratio": offload_ratio,
            "user_cpu_capacity": float(np.mean(self.sim.U[:, 5])),
            "bs_cpu_capacity": float(np.mean(self.sim.B[:, 4])),
            "action": executed_action.tolist(),
        }

        return obs, reward, terminated, truncated, info

    def get_greedy_action(self) -> np.ndarray:
        return np.argmin(self.sim.weighted_costs, axis=1).astype(np.int64)

    def sample_random_action(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            return np.random.randint(0, self.n_actions, size=self.n_users, dtype=np.int64)
        return rng.integers(0, self.n_actions, size=self.n_users, dtype=np.int64)
