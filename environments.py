#%%
import numpy as np
from time import sleep
from random import random, randint, choice
import pickle
from math import cos, sin, sqrt, exp, pi
from matplotlib import animation, rc
import matplotlib.pyplot as plt
from IPython.display import HTML

import winsound
frequency = 1000
duration = 1000



#%%
from tqdm import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)



#%% region[cyan]
class CatchTheFruitSimplified:
    def __init__(self, width=3, height=3, rewards=(0,1), prob=0.2):
        self.width = width
        self.height = height
        self.rewards = rewards
        self.prob = prob
        self.Rs = {}
        self.Tss = {}
    
    def randomPolicy(self):
        π = {state: choice(self.actions(state)) for state in self.states()}
        return π

    def states(self):
        return (k for k in range(self.width * 2**(self.width*self.height)))

    def startingStates(self):
        return (int(self.width/2),)

    def actions(self, state=None):
        return (-1,0,+1)

    def R(self, state):
        if state in self.Rs: return self.Rs[state]
        pos,arr = self.unhash(state)
        catch = arr[0][pos]
        r = self.rewards[catch]
        self.Rs[state] = r
        return r
    
    def Ts(self,state,action):
        if (state,action) in self.Tss: return self.Tss[(state,action)]
        pos,arr = self.unhash(state)
        pos = min(self.width-1,max(0,pos+action))
        arr.pop(0)
        ts = {}
        for tophash in range(2**self.width):
            top = self.inttotern(tophash,1,self.width)[0]
            p = (1-self.prob)**top.count(0)*self.prob**top.count(1)
            state = self.hash(arr+[top],pos)
            ts[state] = p
        self.Tss[(state,action)] = ts
        return ts
    
    def hash(self, arr, pos):
        h = pos
        for row in range(self.height):
            for col in range(self.width):
                h += arr[row][col]*self.width*2**(col+row*self.width)
        return h

    def unhash(self, h):
        pos = h%self.width
        h//=self.width
        arr = self.inttotern(h, self.height, self.width)
        return pos,arr
    
    def inttotern(self, h, width, height):
        arr = []
        for _ in range(width*height):
            arr.append(h%2)
            h//=2
        return np.array(arr).reshape(width,height).tolist()

    def display(self, state):
        symbols = ['⚫️', '🍎', '🤲']
        pos,arr = self.unhash(state)
        print((self.width+2)*'⬜️')
        for row in range(self.height):
            print('⬜️',end='')
            for col in range(self.width):
                print(symbols[arr[self.height-row-1][col]],end='')
            print('⬜️')
        print('⬜️',end='')
        for k in range(self.width):
            if k==pos:
                print(symbols[2],end='')
            else:
                print(symbols[0],end='')
        print('⬜️')
        print((self.width+2)*'⬜️')

    def play(self, π, nbr_runs=10, wait=1, start=0, display=True):
        state = start
        score = 0
        if display:
            print(5*'\n')
            self.display(state)
            print("Score = ",score)
        sleep(wait)
        for _ in range(nbr_runs):
            action = 0
            if state not in π:
                if display: print("State not in policy")
            else:
                action = π[state]
            Ts = self.Ts(state,action)
            rng = random()
            p_accumul = 0
            for next_state, p in Ts.items():
                p_accumul += p
                if rng <= p_accumul:
                    state = next_state
                    break
            if display:
                print(5*'\n')
                self.display(state)
                print("Score = ",score)
            score += self.R(state)
            sleep(wait)
        return score
    
    def performance(self, π, nbr_runs=200, nbr_turns=200):
        scores = []
        for run in tqdm(range(nbr_runs)):
            scores.append(self.play(π, nbr_turns, wait=0, display=False))
        return min(scores) , sum(scores)/len(scores) , max(scores)
#endregion




#%% region[cyan]
class CatchTheFruit:
    def __init__(self, width=3, height=3, rewards=(0,1,-1), probs=(0.2,0.2)):
        self.width = width
        self.height = height
        self.rewards = rewards
        self.probs = probs
        self.Rs = {}
        self.Tss = {}
    
    def randomPolicy(self):
        π = {state: choice(self.actions(state)) for state in self.states()}
        return π

    def states(self):
        return (k for k in range(self.width * 3**(self.width*self.height)))

    def startingStates(self):
        return (int(self.width/2),)

    def actions(self, state=None):
        return (-1,0,+1)

    def R(self, state):
        if state in self.Rs: return self.Rs[state]
        pos,arr = self.unhash(state)
        catch = arr[0][pos]
        r = self.rewards[catch]
        self.Rs[state] = r
        return r
    
    def Ts(self,state,action):
        if (state,action) in self.Tss: return self.Tss[(state,action)]
        pos,arr = self.unhash(state)
        pos = min(self.width-1,max(0,pos+action))
        arr.pop(0)
        ts = {}
        for tophash in range(3**self.width):
            top = self.inttotern(tophash,1,self.width)[0]
            p = (1-self.probs[0]-self.probs[1])**top.count(0)*self.probs[0]**top.count(1)*self.probs[1]**top.count(2)
            state = self.hash(arr+[top],pos)
            ts[state] = p
        self.Tss[(state,action)] = ts
        return ts
    
    def hash(self, arr, pos):
        h = pos
        for row in range(self.height):
            for col in range(self.width):
                h += arr[row][col]*self.width*3**(col+row*self.width)
        return h

    def unhash(self, h):
        pos = h%self.width
        h//=self.width
        arr = self.inttotern(h, self.height, self.width)
        return pos,arr
    
    def inttotern(self, h, width, height):
        arr = []
        for _ in range(width*height):
            arr.append(h%3)
            h//=3
        return np.array(arr).reshape(width,height).tolist()

    def display(self, state):
        symbols = ['⚫️', '🍎', '💩', '🤲']
        pos,arr = self.unhash(state)
        print((self.width+2)*'⬜️')
        for row in range(self.height):
            print('⬜️',end='')
            for col in range(self.width):
                print(symbols[arr[self.height-row-1][col]],end='')
            print('⬜️')
        print('⬜️',end='')
        for k in range(self.width):
            if k==pos:
                print(symbols[3],end='')
            else:
                print(symbols[0],end='')
        print('⬜️')
        print((self.width+2)*'⬜️')

    def play(self, π, nbr_runs=10, wait=1, start=0, display=True):
        state = start
        score = 0
        if display:
            print(5*'\n')
            self.display(state)
            print("Score = ",score)
        sleep(wait)
        for _ in range(nbr_runs):
            action = 0
            if state not in π:
                if display: print("State not in policy")
            else:
                action = π[state]
            Ts = self.Ts(state,action)
            rng = random()
            p_accumul = 0
            for next_state, p in Ts.items():
                p_accumul += p
                if rng <= p_accumul:
                    state = next_state
                    break
            if display:
                print(5*'\n')
                self.display(state)
                print("Score = ",score)
            score += self.R(state)
            sleep(wait)
        return score
    
    def performance(self, π, nbr_runs=200, nbr_turns=200):
        scores = []
        for run in tqdm(range(nbr_runs)):
            scores.append(self.play(π, nbr_turns, wait=0, display=False))
        return min(scores) , sum(scores)/len(scores) , max(scores)
#endregion




#%% region[yellow]
class Reacher:
    def __init__(self, arm_nbr=2, α_res=100, x_res=100, reward_reach=100, reach_distance=0.5, reward_gradient_multiplier=1, respawn_target=False):
        self.arm_nbr = arm_nbr
        self.arm_length = sqrt(2)*x_res/(2*arm_nbr)
        self.α_res = α_res
        self.x_res = x_res
        self.reward_reach = reward_reach
        self.reach_distance = reach_distance
        self.reward_gradient_multiplier = reward_gradient_multiplier
        self.respawn_target = respawn_target
        self.Rs = {}
        self.Tss = {}
    
    def randomPolicy(self):
        π = {state: choice(self.actions(state)) for state in self.states()}
        return π

    def states(self):
        if self.arm_nbr==1:
            return tuple((α1,x1,x2) for α1 in range(self.α_res) for x1 in range(int(-self.x_res/2),int(self.x_res/2)) for x2 in range(int(-self.x_res/2),int(self.x_res/2)))
        elif self.arm_nbr==2:
            return tuple((α1,α2,x1,x2) for α1 in range(self.α_res) for α2 in range(self.α_res) for x1 in range(int(-self.x_res/2),int(self.x_res/2)) for x2 in range(int(-self.x_res/2),int(self.x_res/2)))
        elif self.arm_nbr==3:
            return tuple((α1,α2,α3,x1,x2) for α1 in range(self.α_res) for α2 in range(self.α_res) for α3 in range(self.α_res) for x1 in range(int(-self.x_res/2),int(self.x_res/2)) for x2 in range(int(-self.x_res/2),int(self.x_res/2)))

    def startingStates(self):
        if self.arm_nbr==1:
            return tuple((0,x1,x2) for x1 in range(int(-self.x_res/2),int(self.x_res/2)) for x2 in range(int(-self.x_res/2),int(self.x_res/2)))
        elif self.arm_nbr==2:
            return tuple((0,self.α_res/2,x1,x2) for x1 in range(int(-self.x_res/2),int(self.x_res/2)) for x2 in range(int(-self.x_res/2),int(self.x_res/2)))
        elif self.arm_nbr==3:
            return tuple((0,self.α_res/2,self.α_res/2,x1,x2) for x1 in range(int(-self.x_res/2),int(self.x_res/2)) for x2 in range(int(-self.x_res/2),int(self.x_res/2)))

    def actions(self, state=None):
        if self.arm_nbr==1:
            return tuple(tuple(a1) for a1 in [-1,0,+1])
        elif self.arm_nbr==2:
            return tuple((a1,a2) for a1 in [-1,0,+1] for a2 in [-1,0,+1])
        elif self.arm_nbr==3:
            return tuple((a1,a2,a3) for a1 in [-1,0,+1] for a2 in [-1,0,+1] for a3 in [-1,0,+1]) 

    def R(self, state):
        if state in self.Rs: return self.Rs[state]
        if self.arm_nbr==1:
            α1,x1,x2 = state
            α1 = 2*pi*α1/float(self.α_res)
            y1,y2 = (self.arm_length*cos(α1) , self.arm_length*sin(α1))
        elif self.arm_nbr==2:
            α1,α2,x1,x2 = state
            α1 = 2*pi*α1/float(self.α_res)
            α2 = 2*pi*(α2/float(self.α_res)-0.5)
            y1,y2 = (self.arm_length*(cos(α1)+cos(α1+α2)) , self.arm_length*(sin(α1)+sin(α1+α2)))
        elif self.arm_nbr==3:
            α1,α2,α3,x1,x2 = state
            α1 = 2*pi*α1/float(self.α_res)
            α2 = 2*pi*(α2/float(self.α_res)-0.5)
            α3 = 2*pi*(α3/float(self.α_res)-0.5)
            y1,y2 = (self.arm_length*(cos(α1)+cos(α1+α2)+cos(α1+α2+α3)) , self.arm_length*(sin(α1)+sin(α1+α2)+sin(α1+α2+α3)))
        d = sqrt((x1-y1)**2+(x2-y2)**2)
        r = self.reward_reach if d<=self.reach_distance else self.reward_gradient_multiplier*exp(-0.05*d)/self.reward_reach
        self.Rs[state] = r
        return r
    
    def T(self,state,action):
        # if (state,action) in self.Tss: return self.Tss[(state,action)]
        state_t = state
        if self.arm_nbr==1:
            α1,x1,x2 = state
            if self.respawn_target:
                y1,y2 = (self.arm_length*cos(2*pi*α1/float(self.α_res)) , self.arm_length*sin(2*pi*α1/float(self.α_res)))
                d = sqrt((x1-y1)**2+(x2-y2)**2)
                if d<=self.reach_distance:
                    x1 = randint(-self.x_res/2,self.x_res/2-1)
                    x2 = randint(-self.x_res/2,self.x_res/2-1)
            α1 += action[0]
            α1 %= self.α_res
            state = (α1,x1,x2)
        elif self.arm_nbr==2:
            α1,α2,x1,x2 = state
            if self.respawn_target:
                y1,y2 = (self.arm_length*(cos(2*pi*α1/float(self.α_res))+cos(2*pi*α1/float(self.α_res)+2*pi*(α2/float(self.α_res)-0.5))) , self.arm_length*(sin(2*pi*α1/float(self.α_res))+sin(2*pi*α1/float(self.α_res)+2*pi*(α2/float(self.α_res)-0.5))))
                d = sqrt((x1-y1)**2+(x2-y2)**2)
                if d<=self.reach_distance:
                    x1 = randint(-int(self.x_res/2),int(self.x_res/2)-1)
                    x2 = randint(-int(self.x_res/2),int(self.x_res/2)-1)
            α1 += action[0]
            α2 += action[1]
            α1 %= self.α_res
            α2 %= self.α_res
            state = (α1,α2,x1,x2)
        elif self.arm_nbr==3:
            α1,α2,α3,x1,x2 = state
            if self.respawn_target:
                y1,y2 = (self.arm_length*(cos(2*pi*α1/float(self.α_res))+cos(2*pi*α1/float(self.α_res)+2*pi*(α2/float(self.α_res)-0.5))+cos(2*pi*α1/float(self.α_res)+2*pi*(α2/float(self.α_res)-0.5)+2*pi*(α3/float(self.α_res)-0.5))) , self.arm_length*(sin(2*pi*α1/float(self.α_res))+sin(2*pi*α1/float(self.α_res)+2*pi*(α2/float(self.α_res)-0.5))+sin(2*pi*α1/float(self.α_res)+2*pi*(α2/float(self.α_res)-0.5)+2*pi*(α3/float(self.α_res)-0.5))))
                d = sqrt((x1-y1)**2+(x2-y2)**2)
                if d<=self.reach_distance:
                    x1 = randint(-self.x_res/2,self.x_res/2-1)
                    x2 = randint(-self.x_res/2,self.x_res/2-1)
            α1 += action[0]
            α2 += action[1]
            α3 += action[1]
            α1 %= self.α_res
            α2 %= self.α_res
            α3 %= self.α_res
            state = (α1,α2,α3,x1,x2)
        # self.Tss[(state_t,action)] = state
        return state
    
    def Ts(self,state,action):
        return {self.T(state,action):1}
    
    def play(self, π, nbr_turns=0, position=(0,0), display=True, respawn_target=True):
        respawn_target_backup = self.respawn_target
        self.respawn_target = respawn_target
        if nbr_turns==0:
            nbr_turns = 3*self.α_res
        state = choice(self.startingStates())
        if position!=(0,0):
            state = (*state[:self.arm_nbr], *position)
        score = 0
        if display:
            fig, ax = plt.subplots()
            ax.set_xlim(-sqrt(2)*self.x_res/2, +sqrt(2)*self.x_res/2)
            ax.set_ylim(-sqrt(2)*self.x_res/2, +sqrt(2)*self.x_res/2)
            arm, = ax.plot([], [], lw=3, marker='o', color="Goldenrod")
            target, = ax.plot([], [], markersize=10, marker='o', color="IndianRed")
            def init():
                arm.set_data([], [])
                return (arm,)
            def animate(i):
                nonlocal state
                nonlocal score
                nonlocal ax
                if state not in π:
                    if display: print("State not in policy : {}".format(state))
                else:
                    action = π[state]
                state = self.T(state,action)
                score += self.R(state)
                if self.arm_nbr==1:
                    α1,x1,x2 = state
                    α1 = 2*pi*α1/float(self.α_res)
                    X = [0, self.arm_length*cos(α1)]
                    Y = [0, self.arm_length*sin(α1)]
                elif self.arm_nbr==2:
                    α1,α2,x1,x2 = state
                    α1 = 2*pi*α1/float(self.α_res)
                    α2 = 2*pi*(α2/float(self.α_res)-0.5)
                    X = [0, self.arm_length*cos(α1), self.arm_length*(cos(α1)+cos(α1+α2))]
                    Y = [0, self.arm_length*sin(α1), self.arm_length*(sin(α1)+sin(α1+α2))]
                elif self.arm_nbr==3:
                    α1,α2,α3,x1,x2 = state
                    α1 = 2*pi*α1/float(self.α_res)
                    α2 = 2*pi*(α2/float(self.α_res)-0.5)
                    α3 = 2*pi*(α3/float(self.α_res)-0.5)
                    X = [0, self.arm_length*cos(α1), self.arm_length*(cos(α1)+cos(α1+α2)), self.arm_length*(cos(α1)+cos(α1+α2)+cos(α1+α2+α3))]
                    Y = [0, self.arm_length*sin(α1), self.arm_length*(sin(α1)+sin(α1+α2)), self.arm_length*(sin(α1)+sin(α1+α2)+sin(α1+α2+α3))]
                arm.set_data(X, Y)
                target.set_data([x1],[x2])
                ax.set_title("Score : {:0.2f}".format(score))
                return (arm,)
            anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nbr_turns, interval=100, blit=True)
            HTML(anim.to_jshtml())
            self.respawn_target = respawn_target_backup
            return score, anim
        else:
            for _ in range(nbr_turns):
                if state not in π:
                    if display: print("State not in policy")
                else:
                    action = π[state]
                state = self.T(state,action)
                score += self.R(state)
            self.respawn_target = respawn_target_backup
        return score

    def performance(self, π, nbr_runs=200, nbr_turns=0, respawn_target=True):
        scores = []
        for run in tqdm(range(nbr_runs)):
            scores.append(self.play(π, nbr_turns, display=False, respawn_target=respawn_target))
        return min(scores) , sum(scores)/len(scores) , max(scores)
#endregion

# %%
