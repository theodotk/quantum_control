import numpy as np

import gym
from gym import spaces
from stable_baselines3 import SAC

import optuna

from qutip import tensor, basis, sigmax, sigmay, sigmaz, sigmam, Bloch, mesolve, sesolve, qeye, fidelity, sigmap
from argparse import ArgumentParser



sm = sigmam()
class QD_T_Env(gym.Env):
    '''
    Environement for putting the qubit into a target state.
    The agent is not aware what the target is, so it must be trained and evaluated on the same target.
    
    It can be used with limited pulse areas, but the agent must be optimized for it,
    and I'm the current observables are not particularly good for this task.
    '''
    def __init__(self,
                 rho0,
                 detuning = 0,
                 gamma = 0,
                 gamma_s = 0,
                 eta = 0,
                 target = None,
                 steps_max = 1000,
                 max_power = 1,
                 dt = 0.001,
                 steps = 100,
                 pulse_area = float('inf'),
                 seed = 1,
                 reward_on_arrival = False):
        super(QD_T_Env, self).__init__()
        
        self.reward_on_arrival = reward_on_arrival
        self.steps = steps     
        self.n_steps = 0
        self.steps_max = steps_max
        
        # target
        np.random.seed(seed = seed)
        
        self.generated_target = False
        if target is None:
            target = self._rand_target()
            self.generated_target = True
        self.target = target
        
        ot = [((self.target)*op).tr().real for op in [sigmax(), sigmay(), sigmaz()]]
        ph = np.arctan2(ot[1],ot[0])
        th = np.arctan2(np.sqrt(ot[0]**2 + ot[1]**2), ot[2])
        
        self.t_angles = [ph,th]
        
        
        self.state0 = rho0
        self.state = rho0
        self.fidelities = []
        
        # Hamiltonian, dissipation and time
        
        self.detuning = detuning
        self.gamma = gamma
        self.gamma_s = gamma_s
        self.eta = eta #coupling rate
        
        self.max_power = max_power
        self.pulse_area = np.sqrt(pulse_area)
        self.remaining_pulse_area = 1
        
        largest_time_scale = np.max([detuning, gamma, gamma_s, max_power])
        
        if dt > 0.1 * 1./largest_time_scale:
            c = dt/0.01*largest_time_scale
            dt = 0.01/largest_time_scale
            self.steps *= int(c)
            print(f'Initializing dt to {dt}, increasing n. substeps to {self.steps}')
        self.dt = dt
        
        self.c_ops = [np.sqrt(gamma)*sigmap(), np.sqrt(gamma_s/2)*sigmaz()]
        
        pulse_coef = np.sqrt(gamma*self.eta) if np.sqrt(gamma*self.eta) else 1
        def H(detuning, phi, p):
            return detuning*sigmaz()/2 + 1j * p * pulse_coef * (sm*np.exp(-1j*np.pi*phi) - sm.dag()*np.exp(1j*np.pi*phi))
        
        self.H = H
        
        # observation and action spaces
        n_coords = 3
        n_target_diff = 2
        self.observation_space = spaces.Box(low=np.array([-1]*n_coords+[-2]*n_target_diff),
                                            high=np.array([1]*n_coords+[ 2]*n_target_diff),
                                            shape=(n_target_diff+n_coords,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-self.max_power, -1]), high=np.array([self.max_power,1]), shape=(2,), dtype=np.float32
        )
        
    def _rand_target(self):
        '''
        Generate a target state density matrix.
        Because the problem in symmetrical wrt rotations around Z,
        it seems unnecessary to provide a phase, it can be added post hoc via rotation of the whore result.
        '''
        phase, amp = 0, np.random.rand()
        #phase = (phase*2 - 1)*np.pi
        target = ((1-amp)**0.5 *basis(2,0) + amp**0.5 *np.exp(-1j*phase)*basis(2,1))
        ot = [((self.target)*op).tr().real for op in [sigmax(), sigmay(), sigmaz()]]
        ph = np.arctan2(ot[1],ot[0])
        th = np.arctan2(np.sqrt(ot[0]**2 + ot[1]**2), ot[2])
        
        self.t_anglself.statees = [ph,th]
        return target*target.dag()
    
    def _get_obs(self):
        rho = self.state
        
        o = [(rho*op).tr().real for op in [sigmax(), sigmay(), sigmaz()]]
        
        ph = np.arctan2(o[1],o[0])
        th = np.arctan2(np.sqrt(o[0]**2 + o[1]**2), o[2])
        #p = sum([v**2 for v in o])
        
        t_ph, t_th = self.t_angles
        
        ot = [(th-t_th)/np.pi, (ph-t_ph)/np.pi]
        return np.array(o+ot, dtype=np.float32)
        
    def reset(self):
        if self.generated_target:
            self.target = self._rand_target()
        self.remaining_pulse_area = 1
        self.state = self.state0
        self.n_steps = 0
        self.last_p = 0
        self.fidelities = []
        return self._get_obs()
       
    def step(self, p):
        
        rho = self.state
        H = self.H
        c_ops = self.c_ops
        target = self.target
        
        dt = self.dt

        p, phi = p

        #self.last_p = p
        
        # if the pulse area is finite, we keep track of the remaining proportion of it
        p = np.sign(p) * min(np.abs(p), self.remaining_pulse_area * self.pulse_area)
        self.remaining_pulse_area = max(0, self.remaining_pulse_area - p/self.pulse_area)
        
        try:
            rho = mesolve(H(self.detuning, phi, p),
                          rho,
                          np.linspace(0, self.steps*dt, self.steps+1),
                          c_ops).states[-1]
        # qutip error where it wants to increase num steps. unfortunately it's 'Exception' and not something less general
        except Exception as e: 
            print(e)
            print('bruteforce calculation')
            for _ in range(self.steps):
                L = -1j * (H(self.detuning, phi, p)*rho - rho*H(self.detuning, phi, p))
                for c in c_ops:
                    L += c*rho*c.dag() - 0.5*(c.dag()*c*rho + rho*c.dag()*c)
                rho += L * dt
                rho = rho.unit()
        
        self.fidelities += [fidelity(target, rho)]
        reward = 0 if self.reward_on_arrival else self.fidelities[-1]
        done = False
        if self.n_steps == self.steps_max:
            reward = np.max(self.fidelities) if self.reward_on_arrival else reward
            done = True
        self.state = rho
        self.n_steps += 1
        
        return self._get_obs(), reward, done, {}

    
    def render(self):
        raise NotImplementedError()
    def close (self):
        pass    
    

def eval_model(env, model, n_steps = 100, verbose = 1):
    obs = env.reset()

    actions = [] 
    points = [] # trajectory
    rewards = [] 
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
    
        if verbose==2:
            print("Step {}".format(step + 1))
            print("Action: ", action)
        points.append(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        
        if verbose==2:
            print('obs=', obs, 'reward=', reward, 'done=', done)

        if done:
            obs = env.reset()
    points.append(obs)
    if verbose:
        print(f"Mean reward={np.mean(rewards)}")
    return np.array(points), np.array(actions), np.array(rewards)

    

def sample_sac_params(trial: optuna.Trial):
    """Sampler for SAC hyperparameters."""
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-2, log=True)
    learning_starts = trial.suggest_int('learning_starts', 100, 10000, log=True)
    tau = trial.suggest_float("tau", 1e-4, 1, log=True)

    return {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "learning_starts":learning_starts,
        'tau' : tau
    }

def make_env(seed = None):
    '''
    Create an instance of environement with random parameters
    '''
    
    if seed is not None:
        np.random.seed(seed)
        
    return QD_T_Env(rho0 = basis(2,0)*basis(2,0).dag(),
                     detuning = 0.1*(np.random.rand()-0.5),
                     gamma = 0.1*np.random.rand(),
                     gamma_s = 0,
                     target = basis(2,1)*basis(2,1).dag(),
                     steps_max = 500,
                     max_power = 1,
                     dt = 0.001,
                     steps = 100,
                     seed = 1)

def q_objective(trial):
    '''
    The function to optimize while looking for the model hyperparameters (provided via optuna trial).
    We want the model be able to adapt to different conditions, so the script semi-randomly generates them.
    We want different trials to be in the same conditions, hence all the seed machinations.
    '''
    
    n_trains = 10
    env = make_env(seed = 0)
    
    params = sample_sac_params(trial)
    
    model = SAC(policy = "MlpPolicy", env = env, verbose = 0, **params)
    total_timesteps = trial.suggest_int('total_timesteps', 100, 50_000, log=True)
    
    nan_encountered = False
    try:
        for i in range(n_trains):
            model.learn(total_timesteps//n_trains)
            env = make_env(seed = 1+i) # generate env with different params
            model.set_env(env)
        
    except AssertionError as e:
        print(e)
        nan_encountered = True
    finally:
        # Free memory
        model.env.close()
        env.close()
    if nan_encountered:
        return float("nan")
    
    # evaluate the model on a group of different random environements
    return np.mean(
                    np.array(
                        [eval_model(make_env(seed = i + n_trains + 1),
                                    model,
                                    n_steps = 500,
                                    verbose = 0
                                   )[2] for i in range(20)]
                    )
                )    
    
    
    
def optimize(n_trials=50):
    study = optuna.create_study(direction='maximize')
    study.optimize(q_objective, n_trials=n_trials, catch = (ValueError,))

    trial = study.best_trial

    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
    return trial
    
def train_best_model(params, n_trains = 10, offset = 1000):
    '''
    train a model with the hyperparameters corresponding to the best result
    '''
    total_timesteps =  params.pop('total_timesteps', None)

    qenv = make_env(offset)
    qenv2 = make_env(2*offset)

    best_model = SAC(policy = "MlpPolicy", env = qenv, verbose = 0, **params)

    for i in range(n_trains):
        qenv = make_env(offset+i)
        best_model.set_env(qenv)
        best_model.learn(total_timesteps//n_trains,
                         eval_env = qenv2,
                         eval_freq = total_timesteps//n_trains//2)    
    
    return best_model
    
    
if __name__ == "__main__":
    
    parser = ArgumentParser(
        description="Optimize and train a model to excite a TLS"
    )
    parser.add_argument(
        "--n-trials",
        "-t",
        default=50,
        type=int,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--name",
        '-n',
        type=str,
        default = 'my_model',
        help="Name of the trained and saved model.",
    )
    
    args = parser.parse_args()
    
    
    trial = optimize(n_trials=args.n_trials)
    best_model = train_best_model(trial.params, n_trains = 10, offset = 1000)
    best_model.save(args.name)
    print('Finished')
    
