from enum import IntEnum
import numpy as np
import math
import matplotlib.pyplot as plt


FTYPE = np.float32


class Globe:
    prop_speed = 3.0e8     # (propagation speed) speed of light
    carrier_freq = 3.5e9   # carrier freq (Hz)
    wavelength = prop_speed/carrier_freq
    npsd = 2e-20           # noise power spectral density (Watts/Hz)
    exmin = 0.1            # minimum execution capacity
    dmax = 6000.0          # normalization for distance
    dmin = 1.0             # minimum distance for numerical stability
    range_speed = (15.0, 150.0)      # absolute maximum speed to travel
    range_eps_traj = (0.1, 0.5)     # direction change prob
    range_task = (1.0e9, 6.0e9)     # user task size
    range_data = (1.0e7, 10.0e7)    # user data size
    range_eps_task = (0.0, 1.0)     # zero/low task prob

    range_ser = (8.0e9, 10.0e9)  # station execution rate cc/s
    range_sqgr = (0.01, 0.05)    # station queue growth coeff
    range_sqre = (0.01, 0.05)    # station queue recovery ratio
    range_txbw = (0.5e7, 3.5e7)  # station-user transmission bandwidth

    range_uer = (2.0e9, 4.0e9)  # user execution rate cc/s
    range_uqgr = (0.01, 0.07)   # user queue growth coeff
    range_uqre = (0.01, 0.07)   # user queue recovery ratio
    range_txpow = (0.5, 1.5)    # watts - user transmission power
    range_expow = (1.0, 3.0)    # watts - user execution power

    range_path_loss = (2.0, 3.0)  # path loss exponent alpha = 2 -> original free-space model alpha > 2 -> stronger path loss
    range_fade_gain = (1.0, 2.0)  # small-scale fading power gain - fading_power_gain = 1 -> no fading
    range_shadow_db = (0.0, 2.0)  # shadowing attenuation (in dB) - shadowing_db = 0 -> no additional shadowing (positive value -> additional attenuation)

    max_steps = 200
    xrange = (-dmax*0.5, dmax*0.5)
    yrange = (-dmax*0.5, dmax*0.5)


def Remap(X, input_low, input_high, output_low, output_high):
    return ((X - input_low)*(output_high-output_low)/(input_high-input_low)) + output_low


def DeNorm(X, output_low, output_high):
    return Remap(X, 0.0, 1.0, output_low, output_high)


def ReNorm(X, input_low, input_high):
    return Remap(X, input_low, input_high, 0.0, 1.0)


class World:

    @staticmethod
    def New(N, M):
        return dict(
            N = N, # no of base stations
            M = M, # no of users

            sx = [0.5+0.75*math.cos(2*math.pi*n/N) for n in range(N)],  # station position
            sy = [0.5+0.75*math.sin(2*math.pi*n/N) for n in range(N)],  # station position

            ser =  [0.5 for _ in range(N)], # station execution rate cc/s
            sqgr = [0.5 for _ in range(N)], # station queue growth coeff
            sqre = [0.12 for _ in range(N)], # station queue recovery ratio
            txbw = [0.5 for _ in range(N)], # station-user transmission bandwidth

            uer = [0.5 for _ in range(M)],   # user execution rate cc/s
            uqgr = [0.2 for _ in range(M)],  # user queue growth coeff
            uqre = [0.1 for _ in range(M)],  # user queue recovery ratio
            txpow = [0.5 for _ in range(M)], # user transmission power
            expow = [0.5 for _ in range(M)], # users execution power

            eps_task = [0.1 for _ in range(M)],   # task low prob
            task_low = [0.1 for _ in range(M)],   # user task size
            task_high = [0.7 for _ in range(M)],  # user task size
            data_low = [0.1 for _ in range(M)],   # user data size
            data_high = [0.8 for _ in range(M)],  # user data size

            speed_low = [0.1 for _ in range(M)],  # speed for user
            speed_high = [1.0 for _ in range(M)], # speed for user
            eps_trajectory = [0.0 for _ in range(M)], # prob of selecting new target

            path_loss = 0.1,    # path loss exponent alpha = 2  -> original free-space model  alpha > 2  -> stronger path loss
            fade_gain = 0.1,    # small-scale fading power gain - fading_power_gain = 1 -> no fading
            shadow_db = 0.1,    # shadowing attenuation (in dB) - shadowing_db = 0 -> no additional shadowing (positive value  -> additional attenuation)
        )

    @staticmethod
    def SetPosition(world, n, sx, sy):
        # all arguments are ratios
        world['sx'][n] = sx
        world['sy'][n] = sy

    @staticmethod
    def SetStation(world, n, ser, sqgr, sqre, txbw):
        # all arguments are ratios
        world['ser'][n] = ser
        world['sqgr'][n] = sqgr
        world['sqre'][n] = sqre
        world['txbw'][n] = txbw

    @staticmethod
    def SetUser(world, m, uer, uqgr, uqre, txpow, expow):
        # all arguments are ratios
        world['uer'][m] = uer
        world['uqgr'][m] = uqgr
        world['uqre'][m] = uqre
        world['txpow'][m] = txpow
        world['expow'][m] = expow

    @staticmethod
    def SetTask(world, m, eps_task, task_low, task_high, data_low, data_high):
        # all arguments are ratios
        world['eps_task'][m] = eps_task
        world['task_low'][m] = task_low
        world['task_high'][m] = task_high
        world['data_low'][m] = data_low
        world['data_high'][m] = data_high

    @staticmethod
    def SetMob(world, m, speed_low, speed_high, eps_trajectory):
        # all arguments are ratios
        world['speed_low'][m] = speed_low
        world['speed_high'][m] = speed_high
        world['eps_trajectory'][m] = eps_trajectory

    @staticmethod
    def SetWireless(world, path_loss, fade_gain, shadow_db):
        # all arguments are ratios
        world['path_loss'] = path_loss
        world['fade_gain'] = fade_gain
        world['shadow_db'] = shadow_db

    @staticmethod
    def Descriptor(world):
        pairwise_distance = []
        for i in range(world['N']):
            for j in range(i+1, world['N']):
                pairwise_distance.append(((world['sx'][i] - world['sx'][j])**2 + (world['sy'][i] - world['sy'][j])**2)**0.5)

        return [
            *pairwise_distance,
            *world['ser'],
            *world['sqgr'],
            *world['sqre'],
            *world['txbw'],
            *world['uer'],
            *world['uqgr'],
            *world['uqre'],
            *world['txpow'],
            *world['expow'],
            *world['eps_task'],
            *world['task_low'],
            *world['task_high'],
            *world['data_low'],
            *world['data_high'],
            *world['speed_low'],
            *world['speed_high'],
            *world['eps_trajectory'],
            world['path_loss'],
            world['fade_gain'],
            world['shadow_db'],
        ]

    @staticmethod
    def Remap(world, remapper):
        res = __class__.New(world['N'], world['M'])
        for n in range(world['N']):
            res['sx'][n] = remapper(world['sx'][n], *Globe.xrange)
            res['sy'][n] = remapper(world['sy'][n], *Globe.yrange)

            res['ser'][n] = remapper(world['ser'][n], *Globe.range_ser)
            res['sqgr'][n] = remapper(world['sqgr'][n], *Globe.range_sqgr)
            res['sqre'][n] = remapper(world['sqre'][n], *Globe.range_sqre)
            res['txbw'][n] = remapper(world['txbw'][n], *Globe.range_txbw)

        for m in range(world['M']):
            res['uer'][m] = remapper(world['uer'][m], *Globe.range_uer)
            res['uqgr'][m] = remapper(world['uqgr'][m], *Globe.range_uqgr)
            res['uqre'][m] = remapper(world['uqre'][m], *Globe.range_uqre)
            res['txpow'][m] = remapper(world['txpow'][m], *Globe.range_txpow)
            res['expow'][m] = remapper(world['expow'][m], *Globe.range_expow)

            res['eps_task'][m] = remapper(world['eps_task'][m], *Globe.range_eps_task)
            res['task_low'][m] = remapper(world['task_low'][m], *Globe.range_task)
            res['task_high'][m] = remapper(world['task_high'][m], *Globe.range_task)
            res['data_low'][m] = remapper(world['data_low'][m], *Globe.range_data)
            res['data_high'][m] = remapper(world['data_high'][m], *Globe.range_data)

            res['speed_low'][m] = remapper(world['speed_low'][m], *Globe.range_speed)
            res['speed_high'][m] = remapper(world['speed_high'][m], *Globe.range_speed)
            res['eps_trajectory'][m] = remapper(world['eps_trajectory'][m], *Globe.range_eps_traj)

        res['path_loss'] = remapper(world['path_loss'], *Globe.range_path_loss)
        res['fade_gain'] = remapper(world['fade_gain'], *Globe.range_fade_gain)
        res['shadow_db'] = remapper(world['shadow_db'], *Globe.range_shadow_db)

        return res

    @staticmethod
    def DeNorm(world): return __class__.Remap(world, DeNorm)

    @staticmethod
    def ReNorm(world): return __class__.Remap(world, ReNorm)


class UFields(IntEnum):
    X = 0
    Y = 1
    C = 2
    D = 3
    E = 4
    F = 5


class BFields(IntEnum):
    X = 0
    Y = 1
    E = 2
    V = 3
    F = 4


class CFields(IntEnum):
    Time = 0
    Energy = 1


lenUFields = len(UFields)
lenBFields = len(BFields)
lenCFields = len(CFields)


def Create(N, M):
    return np.hstack((
        np.zeros((M*lenUFields,), dtype=FTYPE),
        np.zeros((N*lenBFields,), dtype=FTYPE),
    ))


def Split(state, N, M):
    users_end = M*lenUFields
    U = state[0:users_end].reshape(M, lenUFields)
    B = state[users_end:users_end+N*lenBFields].reshape(N, lenBFields)
    return U, B


def RandomTrajectory(rng, xrange, yrange, srange, crange, drange, eps_task, eps_trajectory, max_steps):
    epxy, epx, epy = rng.random(), rng.random(), rng.random()
    xe, ye = ((xrange[0] if rng.random() < epx else xrange[1]) , (rng.uniform(*yrange))) if rng.random() < epxy else ((rng.uniform(*xrange)), (yrange[0] if rng.random() < epy else yrange[1]))

    x, y = rng.uniform(*xrange), rng.uniform(*yrange)

    trajectory = []
    for _ in range(max_steps):
        if rng.random() < eps_trajectory:
            xe, ye = ((xrange[0] if rng.random() < epx else xrange[1]) , (rng.uniform(*yrange))) if rng.random() < epxy else ((rng.uniform(*xrange)), (yrange[0] if rng.random() < epy else yrange[1]))
        dx, dy = xe-x, ye-y
        dis = np.hypot(dx, dy)
        s = rng.uniform(*srange)
        x, y = x + s*dx/dis, y + s*dy/dis
        c, d = (crange[0], drange[0]) if rng.random() < eps_task else (rng.uniform(*crange), rng.uniform(*drange),)
        trajectory.append((x, y, c, d))

    return np.array(trajectory, dtype=FTYPE)


class Simulator:
    COST_PER_UNIT = {
        CFields.Time: 1.0,
        CFields.Energy: 1.0,
    }

    def __init__(self, world, weights, partial=False, frozen=False, seed=None):
        self.world = world # supposed to be normalized
        descriptor = World.Descriptor(self.world) # descriptor is normalized
        self.descriptor = np.array(descriptor, dtype=FTYPE)
        self.cfg = World.DeNorm(self.world)
        self.partial = bool(partial)
        self.frozen = bool(frozen)
        self.seed = seed # can be none
        self.rng = np.random.default_rng(self.seed)

        self.N = self.cfg['N']
        self.M = self.cfg['M']
        self.state = Create(self.N, self.M)
        self.U, self.B = Split(self.state, self.N, self.M)
        self.distances = np.zeros((self.M, self.N), dtype=FTYPE)
        self.costs = np.zeros((lenCFields, self.M, self.N+1), dtype=FTYPE)
        self.weights = np.zeros((lenCFields,), dtype=FTYPE)
        for c in CFields:
            self.weights[c] = weights[c]
        self.weighted_costs = np.zeros((self.M, self.N+1), dtype=FTYPE)

        self.observation_dim = self.M*(2 + self.N) + (self.N if not self.partial else 0)
        self.action_dim = self.M # len of action vector
        self.n_actions = self.N+1 # values for each dim in action vector

        if self.frozen:
            self.trajectories = self.NextTrajectories()

    def NextTrajectory(self, m):
        return RandomTrajectory(
            self.rng,
            xrange = Globe.xrange,
            yrange = Globe.yrange,
            srange = (self.cfg['speed_low'][m], self.cfg['speed_high'][m]),
            crange = (self.cfg['task_low'][m], self.cfg['task_high'][m]),
            drange = (self.cfg['data_low'][m], self.cfg['data_high'][m]),
            eps_task = self.cfg['eps_task'][m],
            eps_trajectory = self.cfg['eps_trajectory'][m],
            max_steps=Globe.max_steps,
        )

    def NextTrajectories(self):
        return np.stack([self.NextTrajectory(m) for m in range(self.M)], axis=0)

    def UpdateCosts(self):
        for m in range(self.M):
            tx_time = 0.0
            tx_energy = tx_time*self.cfg['txpow'][m]
            ex_time = (self.U[m, UFields.C]) / (self.U[m, UFields.E]*self.U[m, UFields.F])
            ex_energy = ex_time*self.cfg['expow'][m]
            self.costs[CFields.Time, m, 0] = (tx_time + ex_time) * self.COST_PER_UNIT[CFields.Time]
            self.costs[CFields.Energy, m, 0] = (tx_energy + ex_energy) * self.COST_PER_UNIT[CFields.Energy]

            for n in range(self.N):
                de = max(Globe.dmin, self.distances[m, n])
                Pn = self.cfg['txpow'][m] * (
                    (
                        self.cfg['fade_gain']**2
                    )/(
                        (((4*np.pi*de)/(Globe.wavelength))**self.cfg['path_loss'])
                    )*(
                        10**(-self.cfg['shadow_db']/10)
                    )
                )
                Mn = self.B[n, BFields.V] * np.log2(1+(Pn/(Globe.npsd*self.B[n, BFields.V])))
                tx_time = self.U[m, UFields.D]/Mn
                tx_energy = tx_time*self.cfg['txpow'][m]
                ex_time = (self.U[m, UFields.C]) / (self.B[n, BFields.E]*self.B[n, BFields.F])
                ex_energy = 0.0 # ignore energy consumption of base stations
                self.costs[CFields.Time, m, n+1] = (tx_time + ex_time ) * self.COST_PER_UNIT[CFields.Time]
                self.costs[CFields.Energy, m, n+1] = (tx_energy + ex_energy ) * self.COST_PER_UNIT[CFields.Energy]

        self.weighted_costs = np.sum(self.weights[:, None, None] * self.costs, axis=0)

    def GetObs(self):
        return np.array([
            *[
                value
                for m in range(self.M)
                for value in (
                    Remap(self.U[m, UFields.C], self.cfg['task_low'][m], self.cfg['task_high'][m], 0.0, 1.0),
                    Remap(self.U[m, UFields.D], self.cfg['data_low'][m], self.cfg['data_high'][m], 0.0, 1.0),
                    *[self.distances[m, n]/Globe.dmax for n in range(self.N)],
                )
            ],
            *([self.B[n][BFields.F] for n in range(self.N)] if not self.partial else []),
        ], dtype=FTYPE)

    def Reset(self):
        if not self.frozen:
            self.trajectories = self.NextTrajectories()
        self.horizon = Globe.max_steps-1
        self.ts = 0
        self.last_action = [0 for _ in range(self.M)]

        # initial state
        for m in range(self.M):
            self.U[m, UFields.X], self.U[m, UFields.Y], self.U[m, UFields.C], self.U[m, UFields.D] = self.trajectories[m, self.ts]
            self.U[m, UFields.E] = self.cfg['uer'][m]
            self.U[m, UFields.F] = 1.0

        for n in range(self.N):
            self.B[n, BFields.X], self.B[n, BFields.Y] = self.cfg['sx'][n], self.cfg['sy'][n]
            self.B[n, BFields.E], self.B[n, BFields.V] = self.cfg['ser'][n], self.cfg['txbw'][n]
            self.B[n, BFields.F] = 1.0

        self.UpdateDistances()
        self.UpdateCosts()
        self.terminated = False
        self.truncated = False

    def UpdateDistances(self):
        for m in range(self.M):
            self.distances[m] = np.hypot(
                self.U[m, UFields.X] - self.B[:, BFields.X],
                self.U[m, UFields.Y] - self.B[:, BFields.Y],
            )

    def Step(self, action):
        if action is None:
            action = np.argmin(self.weighted_costs, axis=1)
        self.last_action = [int(a) for a in action]
        cost = self.weighted_costs[np.arange(self.M), self.last_action].copy()

        # action causes some queuing delay at the CPU
        for m in range(self.M):
            if self.last_action[m] == 0:
                self.U[m, UFields.F] -= self.cfg['uqgr'][m]
            else:
                self.B[self.last_action[m]-1, BFields.F] -= self.cfg['sqgr'][self.last_action[m]-1]

        self.ts += 1

        # clipping
        self.U[:, UFields.F] = np.clip(self.U[:, UFields.F] + np.asarray(self.cfg['uqre']), Globe.exmin, 1.0)
        self.B[:, BFields.F] = np.clip(self.B[:, BFields.F] + np.asarray(self.cfg['sqre']), Globe.exmin, 1.0)

        # stochastic next state
        for m in range(self.M):
            self.U[m, UFields.X], self.U[m, UFields.Y], self.U[m, UFields.C], self.U[m, UFields.D] = self.trajectories[m, self.ts]

        self.UpdateDistances()
        self.UpdateCosts()
        self.truncated = self.ts >= self.horizon
        self.terminated = False
        return cost

    def Render(self):
        plt.figure()
        plt.xlim(-Globe.dmax, Globe.dmax)
        plt.ylim(-Globe.dmax, Globe.dmax)
        plt.gca().set_aspect('equal')
        plt.xticks([])
        plt.yticks([])

        for m in range(self.M):
            plt.scatter(self.trajectories[m,:self.ts,0], self.trajectories[m,:self.ts,1], color='tab:gray', marker='.')
            plt.scatter(self.trajectories[m,self.ts:,0], self.trajectories[m,self.ts:,1], color='tab:olive', marker='.')
            ux, uy = self.U[m, UFields.X], self.U[m, UFields.Y]
            plt.scatter([ux], [uy], marker='o')

        for n in range(self.N):
            bx, by = self.B[n, BFields.X], self.B[n, BFields.Y]
            plt.scatter(bx, by, marker='o')
            for m in range(self.M):
                ux, uy = self.U[m, UFields.X], self.U[m, UFields.Y]
                if self.last_action[m] == n+1:
                    plt.plot([ux, bx], [uy, by], linestyle='solid', color='k')
                else:
                    plt.plot([ux, bx], [uy, by], linestyle='dotted')

        plt.show()
