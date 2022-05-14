import numpy as np

import gym
from stable_baselines3 import SAC

import matplotlib.pyplot as plt

from qutip import basis, sigmax, sigmay, sigmaz, sigmam, Bloch, mesolve, qeye, fidelity, sigmap
from mpl_toolkits.mplot3d import Axes3D


from ipywidgets import Label, interact, fixed, Select, Dropdown, Checkbox, interactive_output, VBox, HBox, Output, FloatText, IntText, interactive, FloatSlider, Button,Layout
from IPython.display import display

from training import QD_T_Env, eval_model

class Interface:
    '''
    Interface to make the simulation more interactive
    '''
    def __init__(self, model, figsize=(9,5)):
        self.figsize = figsize
        self.model = model
            
        self.theta = FloatSlider( # initial state angle with Z axis
                            value=0,
                            max = 1,
                            step = 0.01,
                            description=r'Initial $\theta/\pi$',
                            disabled=False)
        
        self.phi = FloatSlider( # in case we need it
                value=0,
                min = -1,
                max =  1,
                step = 1,
                description=r'Initial $\phi/\pi$',
                disabled=False)

        self.w = FloatText( # detuning
            value=0,
            #min = -0.05,
            #max = 0.05,
            step = 0.1,
            description='',
            disabled=False
        )
        
        self.gamma = FloatText( # SE rate
            value=0,
            min=0,
            step = 0.1,
            description='',
            disabled=False
        )
        self.gamma_s = FloatText( # dephasing
            value=0,
            min=0,
            step = 0.01,
            description='',
            disabled=False
        )

        self.p = FloatText( # max laser power (useless because it is defined by a trained model)
            value=1,
            description='',
            disabled=False
        )

        self.t = IntText( # max time steps
            value=100,
            description='',
            disabled=False
        )
        
        
        self.button = Button(description="Run simulation",
                         layout=Layout(width='90%', height='80px'),
                         button_style='danger')
        self.button.on_click(self.on_button_clicked)
        
        self.res = None
        self.clicked = IntText(
            value=0,
            disabled=True
        )

        
        def draw_init_state(**kwargs):
            bl = Bloch()
            if 'phi' in kwargs:
                phi = kwargs['phi'] * np.pi
            else:
                phi = 0
            theta = kwargs['theta'] * np.pi
            init_state = self._construct_state(phi, theta)
            bl.add_states([init_state])
            # target:
            bl.add_vectors([0,0,-1])
            bl.render()

        self.interactive_plot = interactive(draw_init_state, **{'theta':self.theta})#, 'phi':self.phi})
    
    def _construct_state(self, phi, theta):
        x, y, z = np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)
        return (qeye(2)+z*sigmaz()+y*sigmay()+x*sigmax())/2
    
    def _get_init_state(self):
        ph, th = self.phi.value*np.pi, self.theta.value*np.pi
        return self._construct_state(ph, th)
    
    def _run_simulation(self):
        rho0 = self._get_init_state()
        max_power = self.model.action_space.high[0]
        
        if self.gamma.value<0:
            raise ValueError(f'Spontaneous emission rate must be nonnegative, but received {self.gamma.value}')
        if self.gamma_s.value<0:
            raise ValueError(f'Dephasing rate must be nonnegative, but received {self.gamma_s.value}')
        #if self.p.value<0:
        #    raise ValueError(f'Max laser power must be positive, but received {self.p.value}')
        
        print(fr'Simulation with parameters: $\delta$ = {self.w.value}, $\gamma$ = {self.gamma.value}, $\gamma^*$ = {self.gamma_s.value}, max laser power = {max_power}, max time steps = {self.t.value}.')
        
        
        qenv = QD_T_Env(rho0,
                 detuning = self.w.value,
                 gamma = self.gamma.value,
                 gamma_s = self.gamma_s.value,
                 eta = 0,
                 pulse_area = float('inf'),
                 target = basis(2,1)*basis(2,1).dag(),
                 steps_max = self.t.value,
                 max_power = max_power,
                 dt = 0.001,
                 steps = 100,
                 seed = 1,
                 reward_on_arrival = 0)
        
        self.res = eval_model(qenv, self.model, n_steps = self.t.value, verbose = 0)
    
    
    def on_button_clicked(self, b):
        '''
        Run the simulation and produce plots
        '''
        self._run_simulation()
        self.clicked.value += 1

    def show(self):
        units = r'arbitrary $T^{-1}$ units'#r'meV/$\hbar$'
        form_items = [Label(value=f'Max laser power = 1 in {units}.'),
                      Label(value=f'Detuning $\delta$ ({units}):'),
                      self.w,
                      #self.p,
                      Label(value=r'Max time steps:'),
                      self.t,
                      Label(value=fr'Spontaneous emission rate $\gamma$ ({units}):'),
                      self.gamma,
                      Label(value=fr'Dephasing rate $\gamma^*$ ({units}):'),
                      self.gamma_s]

        form = VBox([
                    HBox([
                        VBox(form_items),
                        VBox([Label(value="Initial state (green):"),
                              self.interactive_plot])]
                        ),
                    self.button],
                    layout=Layout(
                    display='flex',
                    flex_flow='column',
                    border='solid 2px',
                    align_items='stretch',
                    width='80%'
                    ))
        
        out = interactive_output(
            self.draw, {'b':self.clicked}
        )
        
        
        display(form, out)
        
        
    def draw(self, b):
        if self.res is not None:
            plots(*self.res, simple = True)




def plots(points, acs, r, simple = False, n_steps = 1000, dt = 0.1):
    
    lp = len(points)
    time = np.linspace(0, (lp-1)*dt, lp)
    x = points[:,0]
    if not simple and len(points[0]) == 3:
        y = np.zeros_like(x)
        z = points[:,1]
    else:
        y = points[:,1]
        z = points[:,2]
    
    fig = plt.figure(figsize = (12, 12))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(time, x, label = 'x')
    ax1.plot(time, y, label = 'y')
    ax1.plot(time, z, label = 'z')
    ax1.plot(time, [np.sqrt(x_**2+y_**2+z_**2) for x_,y_,z_ in zip(x,y,z)], label = 'purity')
    ax1.set_xlabel(r'Time, $t/T$')
    ax1.legend()
    
    time = time[1:]
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    b = Bloch(fig=fig, axes=ax2)
    b.clear()
    b.add_vectors([x[0], y[0], z[0]])
    if len(points[0])==6:
        target = points[0, :3] - points[0, 3:]
        b.add_vectors(target)
    elif len(points[0])==5:
        t = points[0, 3:]        
        o = points[0]
        phi = np.arctan2(o[1],o[0]) - t[1]*np.pi
        theta = np.arctan2(np.sqrt(o[0]**2 + o[1]**2), o[2]) - t[0]*np.pi
        target = np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)
        b.add_vectors(target)
    else:
        b.add_vectors([0,0,-1])
    
    b.add_points([x, y, z])
    b.render()
    ax2.set_box_aspect([1, 1, 1]) # required for mpl > 3.1
    
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.bar(time, np.array(acs)[:,0], width = time[0], label = r'Pulse power $\Omega$', alpha = 0.6)
    if len(acs[0])>1:
        ax3.bar(time, np.array(acs)[:,1], width = time[0], label = r'Pulse phase $\phi$', alpha = 0.4)
    else:
        ax3.bar(np.append([0],time), points[:,2], width = time[0], label = r'Remaining pulse', alpha = 0.4)
    ax3.legend()
    ax3.set_xlabel(r'Time, $t/T$')
    ax3.set_title('Actions')
    
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(time, r)
    ax4.set_xlabel(r'Time, $t/T$')
    ax4.set_ylabel(r'Reward: $F(\rho(t), \rho_{target})$')
    ax4.set_title('Reward')
    
    plt.show()    
   
    
class QDLEnv(gym.Env):
    '''
    LEGACY
    
    Environement destined for putting the qubit into a target state.
    The agent is not aware what the target is, so it must be trained and evaluated on the same target.
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, rho0, H, c_ops = [], target = None,
                 steps_max = 1000,
                 max_power = 1,
                 dt = 0.001,
                 steps = 100):
        super(QDLEnv, self).__init__()
        
        self.state = rho0
        self.state0 = rho0
        self.H = H
        self.c_ops = c_ops
        self.target = target if target is not None else basis(2,1)*basis(2,1).dag()
        
        self.steps = steps
        self.dt = dt
        
        self.n_steps = 0
        self.last_p = 0
        self.steps_max = steps_max
        
        self.max_power = max_power
        
        
        self.action_space = spaces.Box(
            low=np.array([-self.max_power, -1]), high=np.array([self.max_power,1]), shape=(2,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        
    def step(self, p):
        
        rho = self.state
        H = self.H
        c_ops = self.c_ops
        target = self.target
        
        dt = self.dt
        
        #p = np.clip(p, -self.max_power, self.max_power)
        p, phi = p
        #p += self.last_p
        self.last_p = p
        
        try:
            if len(c_ops):
                rho = mesolve(H(phi, p), rho, np.linspace(0, self.steps*dt, self.steps+1), c_ops).states[-1]
            else:
                rho = sesolve(H(phi, p), rho, np.linspace(0, self.steps*dt, self.steps+1)).states[-1]
        except Exception as e: #strange qutip thingy
            print(e)
            print('bruteforce calculation')
            for _ in range(self.steps):
                L = -1j * (H(phi, p)*rho - rho*H(phi, p))
                for c in c_ops:
                    L += c*rho*c.dag() - 0.5*(c.dag()*c*rho - rho*c.dag()*c)
                rho += L * dt
                rho = rho.unit()
        
        
        reward = (target*rho).tr().real
        done = False
        if reward == 1 or self.n_steps == self.steps_max:
            done = True
        self.state = rho
        self.n_steps += 1
        
        return self._get_obs(), reward, done, {}
    
    def _get_obs(self):
        rho = self.state
        return np.array([(rho*op).tr().real for op in [sigmax(), sigmay(), sigmaz()]], dtype=np.float32)
        
    def reset(self):
        self.state = self.state0
        self.n_steps = 0
        self.last_p = 0
        return self._get_obs()
    
    def render(self, mode='human'):
        raise NotImplementedError()
    def close (self):
        pass
class Interface:
    '''
    Interface to make the simulation more interactive
    '''
    def __init__(self, model, figsize=(9,5)):
        self.figsize = figsize
        self.model = model
            
        self.theta = FloatSlider( # initial state angle with Z axis
                            value=0,
                            max = 1,
                            step = 0.01,
                            description=r'Initial $\theta/\pi$',
                            disabled=False)
        
        self.phi = FloatSlider( # in case we need it
                value=0,
                min = -1,
                max =  1,
                step = 1,
                description=r'Initial $\phi/\pi$',
                disabled=False)

        self.w = FloatText( # detuning
            value=0,
            #min = -0.05,
            #max = 0.05,
            step = 0.1,
            description='',
            disabled=False
        )
        
        self.gamma = FloatText( # SE rate
            value=0,
            min=0,
            step = 0.1,
            description='',
            disabled=False
        )
        self.gamma_s = FloatText( # dephasing
            value=0,
            min=0,
            step = 0.01,
            description='',
            disabled=False
        )

        self.p = FloatText( # max laser power (useless because it is defined by a trained model)
            value=1,
            description='',
            disabled=False
        )

        self.t = IntText( # max time steps
            value=100,
            description='',
            disabled=False
        )
        
        
        self.button = Button(description="Run simulation",
                         layout=Layout(width='90%', height='80px'),
                         button_style='danger')
        self.button.on_click(self.on_button_clicked)
        
        self.res = None
        self.clicked = IntText(
            value=0,
            disabled=True
        )

        
        def draw_init_state(**kwargs):
            bl = Bloch()
            if 'phi' in kwargs:
                phi = kwargs['phi'] * np.pi
            else:
                phi = 0
            theta = kwargs['theta'] * np.pi
            init_state = self._construct_state(phi, theta)
            bl.add_states([init_state])
            # target:
            bl.add_vectors([0,0,-1])
            bl.render()

        self.interactive_plot = interactive(draw_init_state, **{'theta':self.theta})#, 'phi':self.phi})
    
    def _construct_state(self, phi, theta):
        x, y, z = np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)
        return (qeye(2)+z*sigmaz()+y*sigmay()+x*sigmax())/2
    
    def _get_init_state(self):
        ph, th = self.phi.value*np.pi, self.theta.value*np.pi
        return self._construct_state(ph, th)
    
    def _run_simulation(self):
        rho0 = self._get_init_state()
        max_power = self.model.action_space.high[0]
        
        if self.gamma.value<0:
            raise ValueError(f'Spontaneous emission rate must be nonnegative, but received {self.gamma.value}')
        if self.gamma_s.value<0:
            raise ValueError(f'Dephasing rate must be nonnegative, but received {self.gamma_s.value}')
        #if self.p.value<0:
        #    raise ValueError(f'Max laser power must be positive, but received {self.p.value}')
        
        print(fr'Simulation with parameters: $\delta$ = {self.w.value}, $\gamma$ = {self.gamma.value}, $\gamma^*$ = {self.gamma_s.value}, max laser power = {max_power}, max time steps = {self.t.value}.')
        
        
        qenv = QD_T_Env(rho0,
                 detuning = self.w.value,
                 gamma = self.gamma.value,
                 gamma_s = self.gamma_s.value,
                 eta = 0,
                 pulse_area = float('inf'),
                 target = basis(2,1)*basis(2,1).dag(),
                 steps_max = self.t.value,
                 max_power = max_power,
                 dt = 0.001,
                 steps = 100,
                 seed = 1,
                 reward_on_arrival = 0)
        
        self.res = eval_model(qenv, self.model, n_steps = self.t.value, verbose = 0)
    
    
    def on_button_clicked(self, b):
        '''
        Run the simulation and produce plots
        '''
        self._run_simulation()
        self.clicked.value += 1

    def show(self):
        units = r'arbitrary $T^{-1}$ units'#r'meV/$\hbar$'
        form_items = [Label(value=f'Max laser power = 1 in {units}.'),
                      Label(value=f'Detuning $\delta$ ({units}):'),
                      self.w,
                      #self.p,
                      Label(value=r'Max time steps:'),
                      self.t,
                      Label(value=fr'Spontaneous emission rate $\gamma$ ({units}):'),
                      self.gamma,
                      Label(value=fr'Dephasing rate $\gamma^*$ ({units}):'),
                      self.gamma_s]

        form = VBox([
                    HBox([
                        VBox(form_items),
                        VBox([Label(value="Initial state (green):"),
                              self.interactive_plot])]
                        ),
                    self.button],
                    layout=Layout(
                    display='flex',
                    flex_flow='column',
                    border='solid 2px',
                    align_items='stretch',
                    width='80%'
                    ))
        
        out = interactive_output(
            self.draw, {'b':self.clicked}
        )
        
        
        display(form, out)
        
        
    def draw(self, b):
        if self.res is not None:
            plots(*self.res, simple = True)



def eval_model(env, model, n_steps = 100, verbose = 1):
    obs = env.reset()

    acs = []
    points = []
    r = []
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
    
        if verbose==2:
            print("Step {}".format(step + 1))
            print("Action: ", action)
        points.append(obs)
        obs, reward, done, info = env.step(action)
        r.append(reward)
        acs.append(action)
        
        if verbose==2:
            print('obs=', obs, 'reward=', reward, 'done=', done)

        if done:
            obs = env.reset()
    points.append(obs)
    if verbose:
        print(f"Mean reward={np.mean(r)}")
    return np.array(points), np.array(acs), np.array(r)


def plots(points, acs, r, simple = False, n_steps = 1000, dt = 0.1):
    
    lp = len(points)
    time = np.linspace(0, (lp-1)*dt, lp)
    x = points[:,0]
    if len(points[0]) == 4 or (not simple and len(points[0]) == 3):
        y = np.zeros_like(x)
        z = points[:,1]
    else:
        y = points[:,1]
        z = points[:,2]
    
    fig = plt.figure(figsize = (12, 12))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(time, x, label = 'x')
    ax1.plot(time, y, label = 'y')
    ax1.plot(time, z, label = 'z')
    ax1.plot(time, [np.sqrt(x_**2+y_**2+z_**2) for x_,y_,z_ in zip(x,y,z)], label = 'purity')
    ax1.set_xlabel(r'Time, $t/T$')
    ax1.legend()
    
    time = time[1:]
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    b = Bloch(fig=fig, axes=ax2)
    b.clear()
    b.add_vectors([x[0], y[0], z[0]])
    if len(points[0])==6:
        target = points[0, :3] - points[0, 3:]
        b.add_vectors(target)
    elif len(points[0])==5:
        t = points[0, 3:]        
        o = points[0]
        phi = np.arctan2(o[1],o[0]) - t[1]*np.pi
        theta = np.arctan2(np.sqrt(o[0]**2 + o[1]**2), o[2]) - t[0]*np.pi
        target = np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)
        b.add_vectors(target)
    else:
        b.add_vectors([0,0,-1])
    
    b.add_points([x, y, z])
    b.render()
    ax2.set_box_aspect([1, 1, 1]) # required for mpl > 3.1
    
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.bar(time, np.array(acs)[:,0], width = time[0], label = r'Pulse power $\Omega$', alpha = 0.6)
    if len(acs[0])>1:
        ax3.bar(time, np.array(acs)[:,1], width = time[0], label = r'Pulse phase $\phi$', alpha = 0.4)
    else:
        ax3.bar(np.append([0],time), points[:,2], width = time[0], label = r'Remaining pulse', alpha = 0.4)
    ax3.legend()
    ax3.set_xlabel(r'Time, $t/T$')
    ax3.set_title('Actions')
    
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(time, r)
    ax4.set_xlabel(r'Time, $t/T$')
    ax4.set_ylabel(r'Reward: $F(\rho(t), \rho_{target})$')
    ax4.set_title('Reward')
    
    plt.show()    

    

    
    
class QDLEnv(gym.Env):
    '''
    LEGACY
    
    Environement destined for putting the qubit into a target state.
    The agent is not aware what the target is, so it must be trained and evaluated on the same target.
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, rho0, H, c_ops = [], target = None,
                 steps_max = 1000,
                 max_power = 1,
                 dt = 0.001,
                 steps = 100):
        super(QDLEnv, self).__init__()
        
        self.state = rho0
        self.state0 = rho0
        self.H = H
        self.c_ops = c_ops
        self.target = target if target is not None else basis(2,1)*basis(2,1).dag()
        
        self.steps = steps
        self.dt = dt
        
        self.n_steps = 0
        self.last_p = 0
        self.steps_max = steps_max
        
        self.max_power = max_power
        
        
        self.action_space = spaces.Box(
            low=np.array([-self.max_power, -1]), high=np.array([self.max_power,1]), shape=(2,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        
    def step(self, p):
        
        rho = self.state
        H = self.H
        c_ops = self.c_ops
        target = self.target
        
        dt = self.dt
        
        #p = np.clip(p, -self.max_power, self.max_power)
        p, phi = p
        #p += self.last_p
        self.last_p = p
        
        try:
            if len(c_ops):
                rho = mesolve(H(phi, p), rho, np.linspace(0, self.steps*dt, self.steps+1), c_ops).states[-1]
            else:
                rho = sesolve(H(phi, p), rho, np.linspace(0, self.steps*dt, self.steps+1)).states[-1]
        except Exception as e: #strange qutip thingy
            print(e)
            print('bruteforce calculation')
            for _ in range(self.steps):
                L = -1j * (H(phi, p)*rho - rho*H(phi, p))
                for c in c_ops:
                    L += c*rho*c.dag() - 0.5*(c.dag()*c*rho + rho*c.dag()*c)
                rho += L * dt
                rho = rho.unit()
        
        
        reward = (target*rho).tr().real
        done = False
        if reward == 1 or self.n_steps == self.steps_max:
            done = True
        self.state = rho
        self.n_steps += 1
        
        return self._get_obs(), reward, done, {}
    
    def _get_obs(self):
        rho = self.state
        return np.array([(rho*op).tr().real for op in [sigmax(), sigmay(), sigmaz()]], dtype=np.float32)
        
    def reset(self):
        self.state = self.state0
        self.n_steps = 0
        self.last_p = 0
        return self._get_obs()
    
    def render(self, mode='human'):
        raise NotImplementedError()
    def close (self):
        pass
    

class DetModel:
    '''
    A very lazy deterministic model for a benchmark
    '''
    def __init__(self, p = 1):
        self.p_max = p
        self.turn = False
    def predict(self, obs, deterministic=True):
        if obs[2] <= 0 and obs[0] < 0:
            if not self.turn:
                self.turn = True
                self.p_max /= 2
            return (self.p_max, -1), 0
        else:
            self.turn = False
            return (self.p_max,0.), 0