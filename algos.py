#%%
from random import choice, random
import numpy as np

from tqdm import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)

def policyDistance(π1,π2):
    return np.linalg.norm(np.array(list(π1.values()))-np.array(list(π2.values())))




#%% region[red] VALUE ITERATION
def valueIteration(mdp, ε, γ, max_iter=1000, record=True, verbose=True):
    rec = []
    itr = 0
    Us = {state: 0 for state in mdp.states()}
    π = mdp.randomPolicy()
    Δ = 2*ε
    if record: rec.append((mdp.performance(π), 0))
    while Δ > ε and itr<max_iter:
        print("Iteration #{}".format(itr))
        Δ = 0
        π_temp = π.copy()
        for state in tqdm(mdp.states()):
            U_temp = Us[state]
            U_max = 0
            for action in mdp.actions(state):
                Ts = mdp.Ts(state, action)
                U_max_temps = sum([Ts[state_t]*(mdp.R(state_t)+γ*Us[state_t]) for state_t in Ts.keys()])
                if U_max_temps > U_max:
                    U_max = U_max_temps
                    π[state] = action
            Δ = max(Δ, abs(U_temp-U_max))
            Us[state] = U_max
        if record: rec.append((mdp.performance(π), policyDistance(π,π_temp)))
        if verbose: print("Δ = {}\nperf = {}".format(Δ,mdp.performance(π)))
        itr +=1
    return π, rec
#endregion




#%% region[blue] POLICY ITERATION
def policyIteration(mdp, γ, eval_iter, max_iter, record=True, verbose=True):
    rec = []
    Us = {state: 0 for state in mdp.states()}
    π = {state: choice(mdp.actions(state)) for state in mdp.states()}
    π = mdp.randomPolicy()
    if record: rec.append((mdp.performance(π), 0))
    for itr in range(max_iter):
        if verbose: print("Iteration #{}".format(itr))
        Us = policyEvaluation(mdp, γ, Us, π, eval_iter, verbose)
        π_temp = π.copy()
        π = policyImprovement(mdp, γ, Us, π)
        if record: rec.append((mdp.performance(π), policyDistance(π,π_temp)))
        if verbose: print("perf = {}".format(mdp.performance(π)))
        if (π_temp==π):
            break
    return π, rec

def policyEvaluation(mdp, γ, Us, π, eval_iter, verbose=True):
    for _ in tqdm(range(eval_iter)):
        for state in mdp.states():
            action = π[state]
            Ts = mdp.Ts(state, action)
            Us[state] = sum([Ts[state_t]*(mdp.R(state_t)+γ*Us[state_t]) for state_t in Ts.keys()])
    return Us

def policyImprovement(mdp, γ, Us, π):
    for state in tqdm(mdp.states()):
        π_temp = π.copy()
        U_max = 0
        for action in mdp.actions(state):
            Ts = mdp.Ts(state, action)
            U_max_temps = sum([Ts[state_t]*(mdp.R(state_t)+γ*Us[state_t]) for state_t in Ts.keys()])
            if U_max_temps > U_max:
                U_max = U_max_temps
                π[state] = action
    return π
#endregion




# %% region[green] Q LEARNING
def Qlearning(mdp, γ, α, qpolicy, qpolicy_args={}, qinit=0, nbr_turns=100, nbr_episode=1000, rec_period=100, record=True, verbose=True):
    rec = []
    Qs = {(state,action):qinit for state in mdp.states() for action in mdp.actions(state)}
    π = mdp.randomPolicy()
    if record: rec.append((mdp.performance(π), 0))
    for episode in tqdm(range(1,nbr_episode+1)):
        π_temp = π
        for state in mdp.startingStates():
            for turn in range(nbr_turns):
                action = qpolicy(mdp, Qs, state, **qpolicy_args)
                # Q_temp = Qs[(state,action)]
                Ts = mdp.Ts(state,action)
                rng = random()
                p_accumul = 0
                for next_state, p in Ts.items():
                    p_accumul += p
                    if rng <= p_accumul:
                        break
                R = mdp.R(next_state)
                Qs[(state,action)] += α * (R + γ*max([Qs[(next_state,action_t)] for action_t in mdp.actions(next_state)]) - Qs[(state,action)])
                state = next_state
        π = {state:qpolicy(mdp, Qs, state, **qpolicy_args) for state in mdp.states()}
        if record and episode%rec_period==0: rec.append((mdp.performance(π,nbr_runs=120), policyDistance(π,π_temp)))
        if verbose: print("perf = {}".format(mdp.performance(π)))
    return π, rec

def εGreedy(mdp, Qs, state, epsilon):
    rng = random()
    if rng>epsilon:
        Q_max = -99999999
        for act in mdp.actions(state):
            Q_temp = Qs[(state,act)]
            if Q_temp>Q_max:
                Q_max = Q_temp
                action = act
    else:
        action = choice(mdp.actions(state))
    return action

def εpRandom(mdp, Qs, state, epsilon=0.2):
    rng = random()
    εp = epsilon/float(len(mdp.actions(state)))
    Q_sum = sum([Qs[(state,act)] for act in mdp.actions(state)]) + 0.00001
    ps = {act:εp+(1-epsilon)*(Qs[(state,act)]+0.00001/float(len(mdp.actions(state))))/Q_sum for act in mdp.actions(state)}
    rng = random()
    p_accumul = 0
    for action, p in ps.items():
        p_accumul += p
        if rng <= p_accumul:
            break
    return action

def εβpRandom(mdp, Qs, state, epsilon=0.1, beta=2):
    rng = random()
    εp = epsilon/float(len(mdp.actions(state)))
    Q_sum = sum([Qs[(state,act)]**beta for act in mdp.actions(state)]) + 0.00001
    ps = {act:εp+(1-epsilon)*(Qs[(state,act)]**beta+0.00001/float(len(mdp.actions(state))))/Q_sum for act in mdp.actions(state)}
    rng = random()
    p_accumul = 0
    for action, p in ps.items():
        p_accumul += p
        if rng <= p_accumul:
            break
    return action

#endregion