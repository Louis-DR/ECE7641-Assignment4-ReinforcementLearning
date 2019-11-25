#%%
from environments import CatchTheFruit, CatchTheFruitSimplified, Reacher
from algos import valueIteration, policyIteration, Qlearning, εGreedy, εpRandom, εβpRandom

import pickle
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn-whitegrid')
from IPython.display import HTML

import winsound
frequency = 1000
duration = 1000





#region[white]
def analysis(records):
    # plotArgs = {'γ=0.50':{'dashes':[2,1]}, 'γ=0.90':{'dashes':[4,1]}, 'γ=0.99':{'dashes':[2,0]}, 'default':{'dashes':[2,0]}}
    # plotArgs = {'ε=0.00':{'alpha':1}, 'ε=0.25':{'alpha':0.8}, 'ε=0.50':{'alpha':0.6}, 'ε=0.75':{'alpha':0.4}, 'ε=1.00':{'alpha':0.2}, 'default':{'dashes':[2,0]}}
    # plotArgs = {'α=0.01':{'alpha':1}, 'α=0.04':{'alpha':0.8}, 'α=0.10':{'alpha':0.6}, 'α=0.25':{'alpha':0.4}, 'α=0.50':{'alpha':0.2}, 'α=0.90':{'alpha':0.1}, 'default':{'dashes':[2,0]}}
    plotArgs = {'Optimistic init':{'dashes':[2,0]}, 'ε-Greedy':{'dashes':[5,1]}, 'εp-Random':{'dashes':[3,1]}, 'εβp-Random':{'dashes':[1,1]}, 'default':{'dashes':[2,0]}}
    for label,rec in records.items():
        perfs, dists = zip(*rec)
        perfs = perfs[:20]
        dists = dists[:21]
        plt.figure(1)
        plt.plot(range(len(perfs)), [entry[1] for entry in perfs], color="IndianRed", linewidth=3, label=label, **plotArgs.get(label, plotArgs['default']))
        # plt.fill_between(range(len(perfs)), [entry[0] for entry in perfs], [entry[2] for entry in perfs], color="IndianRed", alpha=0.15)
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.legend()
        plt.figure(2)
        plt.plot(range(1,len(dists)), dists[1:], color="SteelBlue", linewidth=3, label=label, **plotArgs.get(label, plotArgs['default']))
        plt.xlabel("Iteration")
        plt.ylabel("Policy change")
        plt.legend()
    plt.show()
#endregion




# region[red]
#
#                         VALUE ITERATION 
#
# endregion 
#%% region[cyan] CATCH THE FRUITS SIMPLIFIED
# γs = [0.1,0.5,0.9]
γs = [0.5,0.9,0.99] # don't forget to change the dict key resolution
width = 3
height = 3
ε = 0.1
max_iter = 100
policies = {}
records = {}
for γ in γs:
    game = CatchTheFruitSimplified(width,height)
    π,rec = valueIteration(game, ε, γ, max_iter=max_iter)
    policies["γ={:0.2f}".format(γ)] = π
    records["γ={:0.2f}".format(γ)] = rec
filename = 'CTFS_{}x{}_VI_γs={}'.format(width,height,γs)
outfile = open(filename,'wb')
pickle.dump((policies,records),outfile)
outfile.close()
analysis(records)
#endregion
#%% region[cyan] CATCH THE FRUITS
# γs = [0.1,0.5,0.9]
γs = [0.5,0.9,0.99] # don't forget to change the dict key resolution
width = 3
height = 3
ε = 0.1
max_iter = 100
policies = {}
records = {}
for γ in γs:
    game = CatchTheFruit(width,height)
    π,rec = valueIteration(game, ε, γ, max_iter=max_iter, verbose=False)
    policies["γ={:0.2f}".format(γ)] = π
    records["γ={:0.2f}".format(γ)] = rec
filename = 'CTF_{}x{}_VI_γs={}'.format(width,height,γs)
outfile = open(filename,'wb')
pickle.dump((policies,records),outfile)
outfile.close()
analysis(records)
#endregion
#%% region[yellow] REACHER
# γs = [0.1,0.5,0.9]
γs = [0.5,0.9,0.99] # don't forget to change the dict key resolution
arm_nbr = 2
α_res = 100
x_res = 10
ε = 0.001
max_iter = 50
policies = {}
records = {}
for γ in γs:
    game = Reacher(arm_nbr=arm_nbr, α_res=α_res, x_res=x_res, reward_reach=20, reward_gradient_multiplier=0)
    π,rec = valueIteration(game, ε, γ, max_iter=max_iter)
    policies["γ={:0.2f}".format(γ)] = π
    records["γ={:0.2f}".format(γ)] = rec
filename = 'R{}_{}_{}_VI_noDistanceReward_γs={}'.format(arm_nbr,α_res,x_res,γs)
outfile = open(filename,'wb')
pickle.dump((policies,records),outfile)
outfile.close()
analysis(records)
#endregion
# region[red]
#
#                         VALUE ITERATION
#
# endregion 





# region[blue]
#
#                         POLICY ITERATION 
#
# endregion 
#%% region[cyan] CATCH THE FRUITS SIMPLIFIED
γs = [0.5,0.9,0.99]
width = 3
height = 3
max_iter = 50
eval_iter = 10
policies = {}
records = {}
for γ in γs:
    game = CatchTheFruitSimplified(width,height)
    π,rec = policyIteration(game, γ, eval_iter=eval_iter, max_iter=max_iter, verbose=False)
    policies["γ={:0.2f}".format(γ)] = π
    records["γ={:0.2f}".format(γ)] = rec
filename = 'CTFS_{}x{}_PI_γs={}'.format(width,height,γs)
outfile = open(filename,'wb')
pickle.dump((policies,records),outfile)
outfile.close()
analysis(records)


#endregion
#%% region[cyan] CATCH THE FRUITS
γs = [0.5]
width = 3
height = 4
max_iter = 5
eval_iter = 10
policies = {}
records = {}
for γ in γs:
    game = CatchTheFruit(width,height)
    π,rec = policyIteration(game, γ, eval_iter=eval_iter, max_iter=max_iter)
    policies["γ={:0.1f}".format(γ)] = π
    records["γ={:0.1f}".format(γ)] = rec
filename = 'CTF_{}x{}_PI_γs={}'.format(width,height,γs)
outfile = open(filename,'wb')
pickle.dump((policies,records),outfile)
outfile.close()
analysis(records)


#endregion
#%% region[yellow] REACHER
γs = [0.5]
arm_nbr = 3
α_res = 50
x_res = 6
max_iter = 30
eval_iter = 10
policies = {}
records = {}
for γ in γs:
    game = Reacher(arm_nbr=arm_nbr, α_res=α_res, x_res=x_res, reward_reach=20)
    π,rec = policyIteration(game, γ, eval_iter=eval_iter, max_iter=max_iter)
    policies["γ={:0.1f}".format(γ)] = π
    records["γ={:0.1f}".format(γ)] = rec
filename = 'R{}_{}_{}_PI_γs={}'.format(arm_nbr,α_res,x_res,γs)
outfile = open(filename,'wb')
pickle.dump((policies,records),outfile)
outfile.close()
analysis(records)


#endregion
# region[blue]
#
#                         POLICY ITERATION
#
# endregion 




# region[blue]
#
#                         Q-LEARNING 
#
# endregion 
#%% region[cyan] CATCH THE FRUITS SIMPLIFIED STRATEGIES
width = 3
height = 3
γ = 0.5
α = 0.5
qinit = 0
qpolicy = εGreedy
nbr_episode = 1250
nbr_turns = 200
rec_period = 50
policies = {}
records = {}

qpolicy_args = {'epsilon':0.2}
game = CatchTheFruitSimplified(width,height)
π,rec = Qlearning(game, γ, α, qpolicy=qpolicy, qpolicy_args=qpolicy_args, qinit=0, nbr_turns=nbr_turns, nbr_episode=nbr_episode, rec_period=rec_period, verbose=False)
policies["ε-Greedy"] = π
records["ε-Greedy"] = rec

qpolicy_args = {'epsilon':0.1}
game = CatchTheFruitSimplified(width,height)
π,rec = Qlearning(game, γ, α, qpolicy=εpRandom, qpolicy_args=qpolicy_args, qinit=0, nbr_turns=nbr_turns, nbr_episode=nbr_episode, rec_period=rec_period, verbose=False)
policies["εp-Random"] = π
records["εp-Random"] = rec

qpolicy_args = {'epsilon':0.1, 'beta':3}
game = CatchTheFruitSimplified(width,height)
π,rec = Qlearning(game, γ, α, qpolicy=εβpRandom, qpolicy_args=qpolicy_args, qinit=0, nbr_turns=nbr_turns, nbr_episode=nbr_episode, rec_period=rec_period, verbose=False)
policies["εβp-Random"] = π
records["εβp-Random"] = rec

qpolicy_args = {'epsilon':0.05}
game = CatchTheFruitSimplified(width,height)
π,rec = Qlearning(game, γ, α, qpolicy=qpolicy, qpolicy_args=qpolicy_args, qinit=5, nbr_turns=nbr_turns, nbr_episode=nbr_episode, rec_period=rec_period, verbose=False)
policies["Optimistic init"] = π
records["Optimistic init"] = rec

filename = 'CTFS_{}x{}_QL_exploStrats'.format(width,height)
outfile = open(filename,'wb')
pickle.dump((policies,records),outfile)
outfile.close()
analysis(records)



#endregion
#%% region[cyan] CATCH THE FRUITS SIMPLIFIED GAMMA
γs = [0.5,0.9,0.99]
width = 3
height = 3
α = 0.5
qinit = 0
qpolicy = εGreedy
qpolicy_args = {'epsilon':0.2}
nbr_episode = 2000
nbr_turns = 200
rec_period = 40
policies = {}
records = {}
for γ in γs:
    game = CatchTheFruitSimplified(width,height)
    π,rec = Qlearning(game, γ, α, qpolicy=qpolicy, qpolicy_args=qpolicy_args, qinit=0, nbr_turns=nbr_turns, nbr_episode=nbr_episode, rec_period=rec_period, verbose=False)
    policies["γ={:0.2f}".format(γ)] = π
    records["γ={:0.2f}".format(γ)] = rec
filename = 'CTFS_{}x{}_QL_a={}_e={}_γs={}'.format(width,height,α,qpolicy_args['epsilon'],γs)
outfile = open(filename,'wb')
pickle.dump((policies,records),outfile)
outfile.close()
analysis(records)


#endregion
#%% region[cyan] CATCH THE FRUITS SIMPLIFIED GAMMA 2
γs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.999]
width = 3
height = 3
α = 0.5
qinit = 0
qpolicy = εGreedy
qpolicy_args = {'epsilon':0.2}
nbr_episode = 1000
nbr_turns = 200
rec = []
for γ in γs:
    game = CatchTheFruitSimplified(width,height)
    π,_ = Qlearning(game, γ, α, qpolicy=qpolicy, qpolicy_args=qpolicy_args, qinit=0, nbr_turns=nbr_turns, nbr_episode=nbr_episode, record=False, verbose=False)
    rec.append(game.performance(π, nbr_runs=1000))
plt.plot(γs, [r[1] for r in rec], linewidth=3, color='IndianRed')
plt.fill_between(γs, [r[0] for r in rec], [r[2] for r in rec], linewidth=3, color='IndianRed', alpha=0.15)
plt.xlabel('Learning rate')
plt.xscale('log')
plt.ylabel('Score')
plt.show()

#endregion
#%% region[cyan] CATCH THE FRUITS SIMPLIFIED ALPHA
αs = [0.01,0.035,0.1,0.25,0.5,0.9]
width = 3
height = 3
γ = 0.5
qinit = 0
qpolicy = εGreedy
qpolicy_args = {'epsilon':0.2}
nbr_episode = 4000
nbr_turns = 200
rec_period = 100
policies = {}
records = {}
for α in αs:
    game = CatchTheFruitSimplified(width,height)
    π,rec = Qlearning(game, γ, α, qpolicy=qpolicy, qpolicy_args=qpolicy_args, qinit=0, nbr_turns=nbr_turns, nbr_episode=nbr_episode, rec_period=rec_period, verbose=False)
    policies["α={:0.2f}".format(α)] = π
    records["α={:0.2f}".format(α)] = rec
filename = 'CTFS_{}x{}_QL_as={}_e={}_γ={}'.format(width,height,αs,qpolicy_args['epsilon'],γ)
outfile = open(filename,'wb')
pickle.dump((policies,records),outfile)
outfile.close()
analysis(records)


#endregion
#%% region[cyan] CATCH THE FRUITS SIMPLIFIED ALPHA 2
αs = [0.0001,0.00035,00.001,0.0035,0.01,0.035,0.1,0.35,0.9]
width = 3
height = 3
γ = 0.5
qinit = 0
qpolicy = εGreedy
qpolicy_args = {'epsilon':0.2}
nbr_episode = 1000
nbr_turns = 200
rec = []
for α in αs:
    game = CatchTheFruitSimplified(width,height)
    π,_ = Qlearning(game, γ, α, qpolicy=qpolicy, qpolicy_args=qpolicy_args, qinit=0, nbr_turns=nbr_turns, nbr_episode=nbr_episode, record=False, verbose=False)
    rec.append(game.performance(π, nbr_runs=1000))
plt.plot(αs, [r[1] for r in rec], linewidth=3, color='IndianRed')
plt.fill_between(αs, [r[0] for r in rec], [r[2] for r in rec], linewidth=3, color='IndianRed', alpha=0.15)
plt.xlabel('Learning rate')
plt.xscale('log')
plt.ylabel('Score')
plt.show()

#endregion
#%% region[cyan] CATCH THE FRUITS SIMPLIFIED EPSILON
# εs = [0,0.1,0.5]
εs = [0,0.25,0.5,0.75,1.0]
width = 3
height = 3
γ = 0.5
α = 0.5
qinit = 0
qpolicy = εGreedy
nbr_episode = 4000
nbr_turns = 200
rec_period = 100
policies = {}
records = {}
for ε in εs:
    qpolicy_args = {'epsilon':ε}
    game = CatchTheFruitSimplified(width,height)
    π,rec = Qlearning(game, γ, α, qpolicy=qpolicy, qpolicy_args=qpolicy_args, qinit=0, nbr_turns=nbr_turns, nbr_episode=nbr_episode, rec_period=rec_period, verbose=False)
    policies["ε={:0.2f}".format(ε)] = π
    records["ε={:0.2f}".format(ε)] = rec
filename = 'CTFS_{}x{}_QL_a={}_es={}_γ={}'.format(width,height,α,εs,γ)
outfile = open(filename,'wb')
pickle.dump((policies,records),outfile)
outfile.close()
analysis(records)


#endregion
#%% region[cyan] CATCH THE FRUITS SIMPLIFIED EPSILON 2
# εs = [0,0.1,0.5]
εs = [0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
width = 3
height = 3
γ = 0.5
α = 0.5
qinit = 0
qpolicy = εGreedy
nbr_episode = 1000
nbr_turns = 200
rec = []
for ε in εs:
    qpolicy_args = {'epsilon':ε}
    game = CatchTheFruitSimplified(width,height)
    π,_ = Qlearning(game, γ, α, qpolicy=qpolicy, qpolicy_args=qpolicy_args, qinit=0, nbr_turns=nbr_turns, nbr_episode=nbr_episode, record=False, verbose=False)
    rec.append(game.performance(π, nbr_runs=1000))
plt.plot(εs, [r[1] for r in rec], linewidth=3, color='IndianRed')
plt.fill_between(εs, [r[0] for r in rec], [r[2] for r in rec], linewidth=3, color='IndianRed', alpha=0.15)
plt.xlabel('Epsilon')
plt.ylabel('Score')
plt.show()


#endregion
#%% region[cyan] CATCH THE FRUITS STRATEGIES
width = 3
height = 3
γ = 0.5
α = 0.5
qinit = 0
qpolicy = εGreedy
nbr_episode = 1250
nbr_turns = 200
rec_period = 50
policies = {}
records = {}

qpolicy_args = {'epsilon':0.2}
game = CatchTheFruit(width,height)
π,rec = Qlearning(game, γ, α, qpolicy=qpolicy, qpolicy_args=qpolicy_args, qinit=0, nbr_turns=nbr_turns, nbr_episode=nbr_episode, rec_period=rec_period, verbose=False)
policies["ε-Greedy"] = π
records["ε-Greedy"] = rec

qpolicy_args = {'epsilon':0.3}
game = CatchTheFruit(width,height)
π,rec = Qlearning(game, γ, α, qpolicy=εpRandom, qpolicy_args=qpolicy_args, qinit=0, nbr_turns=nbr_turns, nbr_episode=nbr_episode, rec_period=rec_period, verbose=False)
policies["εp-Random"] = π
records["εp-Random"] = rec

qpolicy_args = {'epsilon':0.1, 'beta':3}
game = CatchTheFruit(width,height)
π,rec = Qlearning(game, γ, α, qpolicy=εβpRandom, qpolicy_args=qpolicy_args, qinit=0, nbr_turns=nbr_turns, nbr_episode=nbr_episode, rec_period=rec_period, verbose=False)
policies["εβp-Random"] = π
records["εβp-Random"] = rec

qpolicy_args = {'epsilon':0.05}
game = CatchTheFruit(width,height)
π,rec = Qlearning(game, γ, α, qpolicy=qpolicy, qpolicy_args=qpolicy_args, qinit=5, nbr_turns=nbr_turns, nbr_episode=nbr_episode, rec_period=rec_period, verbose=False)
policies["Optimistic init"] = π
records["Optimistic init"] = rec

filename = 'CTF_{}x{}_QL_exploStrats'.format(width,height)
outfile = open(filename,'wb')
pickle.dump((policies,records),outfile)
outfile.close()
analysis(records)



#endregion
#%% region[yellow] REACHER STRATEGIES
arm_nbr = 2
α_res = 60
x_res = 6
γ = 0.5
α = 0.5
qinit = 0
qpolicy = εGreedy
nbr_episode = 1000
nbr_turns = 100
rec_period = 50
policies = {}
records = {}

qpolicy_args = {'epsilon':0.2}
game = Reacher(arm_nbr=arm_nbr, α_res=α_res, x_res=x_res, reward_reach=20)
π,rec = Qlearning(game, γ, α, qpolicy=qpolicy, qpolicy_args=qpolicy_args, qinit=0, nbr_turns=nbr_turns, nbr_episode=nbr_episode, rec_period=rec_period, verbose=False)
policies["ε-Greedy"] = π
records["ε-Greedy"] = rec

qpolicy_args = {'epsilon':0.05}
game = Reacher(arm_nbr=arm_nbr, α_res=α_res, x_res=x_res, reward_reach=20)
π,rec = Qlearning(game, γ, α, qpolicy=qpolicy, qpolicy_args=qpolicy_args, qinit=5, nbr_turns=nbr_turns, nbr_episode=nbr_episode, rec_period=rec_period, verbose=False)
policies["Optimistic init"] = π
records["Optimistic init"] = rec

filename = 'R{}_{}_{}_QL_exploStrats'.format(arm_nbr,α_res,x_res)
outfile = open(filename,'wb')
pickle.dump((policies,records),outfile)
outfile.close()
analysis(records)



#endregion
#%% region[yellow] REACHER
γs = [0.5,0.9,0.99]
arm_nbr = 2
α_res = 60
x_res = 6
α = 0.5
qinit = 0
qpolicy = εGreedy
qpolicy_args = {'epsilon':0.2}
nbr_episode = 1000
nbr_turns = 200
rec_period = 100
policies = {}
records = {}
for γ in γs:
    game = Reacher(arm_nbr=arm_nbr, α_res=α_res, x_res=x_res, reward_reach=20)
    π,rec = Qlearning(game, γ, α, qpolicy=qpolicy, qpolicy_args=qpolicy_args, qinit=0, nbr_turns=nbr_turns, nbr_episode=nbr_episode, rec_period=rec_period, verbose=False)
    policies["γ={:0.1f}".format(γ)] = π
    records["γ={:0.1f}".format(γ)] = rec
filename = 'R{}_{}_{}_QL_γs={}'.format(arm_nbr,α_res,x_res,γs)
outfile = open(filename,'wb')
pickle.dump((policies,records),outfile)
outfile.close()
analysis(records)


#endregion
# region[blue]
#
#                         Q-LEARNING
#
# endregion 






#%% #region[white] COMPUTATION TIME VALUE ITERATION
comput_time_VI_CTF = [0.1 , 45 , 35*60 ,45*60*60]
nbr_s_VI_CTF = [3*2**(3*3) , 3*3**(3*3) , 3*3**(4*3) , 4*3**(4*3)]
nbr_sa_VI_CTF = [3*3*2**(3*3) , 3*3*3**(3*3) , 3*3*3**(4*3) , 4*4*3**(4*3)]

comput_time_VI_R = [3, 30 , 6*60]
nbr_s_VI_R = [60*60*6*6 , 100*100*6*6 , 60*60*60*6*6]
nbr_sa_VI_R = [9*60*60*6*6 , 9*100*100*6*6 , 27*60*60*60*6*6]

plt.plot(nbr_s_VI_CTF, comput_time_VI_CTF, marker='o', color="DarkSeaGreen", linewidth=3, label="Catch The Fruit")
plt.plot(nbr_s_VI_R, comput_time_VI_R, marker='o', color="GoldenRod", linewidth=3, label="Reacher")
plt.xscale('log')
# plt.xlim(10**5,10**7)
plt.yscale('log')
plt.xlabel("Number of states")
plt.ylabel("Computation time (sec)")
plt.legend()
plt.show()
plt.plot(nbr_sa_VI_CTF, comput_time_VI_CTF, marker='o', color="DarkSeaGreen", linewidth=3, label="Catch The Fruit")
plt.plot(nbr_sa_VI_R, comput_time_VI_R, marker='o', color="GoldenRod", linewidth=3, label="Reacher")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Number of state-action combination")
plt.ylabel("Computation time (sec)")
plt.legend()
plt.show()
#endregion




#%% #region[white] COMPUTATION TIME VALUE ITERATION
comput_time_PI_CTF = [1 , 5.5*60]
nbr_s_PI_CTF = [3*2**(3*3) , 3*3**(3*3)]
nbr_sa_PI_CTF = [3*3*2**(3*3) , 3*3*3**(3*3)]

comput_time_PI_R = [20*60 , 20*60*60 , 500*60*60]
nbr_s_PI_R = [60*60*6*6 , 100*100*6*6 , 60*60*60*6*6]
nbr_sa_PI_R = [9*60*60*6*6 , 9*100*100*6*6 , 27*60*60*60*6*6]

plt.plot(nbr_s_PI_CTF, comput_time_PI_CTF, marker='o', color="DarkSeaGreen", linewidth=3, label="Catch The Fruit")
plt.plot(nbr_s_PI_R, comput_time_PI_R, marker='o', color="GoldenRod", linewidth=3, label="Reacher")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Number of states")
plt.ylabel("Computation time (sec)")
plt.legend()
plt.show()
plt.plot(nbr_sa_PI_CTF, comput_time_PI_CTF, marker='o', color="DarkSeaGreen", linewidth=3, label="Catch The Fruit")
plt.plot(nbr_sa_PI_R, comput_time_PI_R, marker='o', color="GoldenRod", linewidth=3, label="Reacher")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Number of state-action combination")
plt.ylabel("Computation time (sec)")
plt.legend()
plt.show()
#endregion




#%% #region[white] COMPUTATION TIME Q-LEARNING
comput_time_QL_CTF = [25 , 120 , 240 , 2700]
nbr_s_QL_CTF = [3*2**(3*3) , 3*3**(3*3) , 3*3**(4*3) , 4*3**(4*3)]
nbr_sa_QL_CTF = [3*3*2**(3*3) , 3*3*3**(3*3) , 3*3*3**(4*3) , 4*4*3**(4*3)]

comput_time_QL_R = [900, 6600]
nbr_s_QL_R = [60*60*6*6 , 100*100*6*6]
nbr_sa_QL_R = [9*60*60*6*6 , 9*100*100*6*6]

plt.plot(nbr_s_QL_CTF, comput_time_QL_CTF, marker='o', color="DarkSeaGreen", linewidth=3, label="Catch The Fruit")
plt.plot(nbr_s_QL_R, comput_time_QL_R, marker='o', color="GoldenRod", linewidth=3, label="Reacher")
plt.xscale('log')
# plt.xlim(10**5,10**7)
plt.yscale('log')
plt.xlabel("Number of states")
plt.ylabel("Computation time (sec)")
plt.legend()
plt.show()
plt.plot(nbr_sa_QL_CTF, comput_time_QL_CTF, marker='o', color="DarkSeaGreen", linewidth=3, label="Catch The Fruit")
plt.plot(nbr_sa_QL_R, comput_time_QL_R, marker='o', color="GoldenRod", linewidth=3, label="Reacher")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Number of state-action combination")
plt.ylabel("Computation time (sec)")
plt.legend()
plt.show()
#endregion



#%% region[white]
plt.plot(nbr_s_PI_CTF, comput_time_PI_CTF, marker='o', color="DarkSeaGreen", linewidth=3, label="Catch The Fruit PI", dashes=[1,1])
plt.plot(nbr_s_PI_R, comput_time_PI_R, marker='o', color="GoldenRod", linewidth=3, label="Reacher PI", dashes=[1,1])
plt.plot(nbr_s_VI_CTF, comput_time_VI_CTF, marker='o', color="DarkSeaGreen", linewidth=3, label="Catch The Fruit VI", dashes=[2,1])
plt.plot(nbr_s_VI_R, comput_time_VI_R, marker='o', color="GoldenRod", linewidth=3, label="Reacher VI", dashes=[2,1])
plt.plot(nbr_s_QL_CTF, comput_time_QL_CTF, marker='o', color="DarkSeaGreen", linewidth=3, label="Catch The Fruit QL", dashes=[2,0])
plt.plot(nbr_s_QL_R, comput_time_QL_R, marker='o', color="GoldenRod", linewidth=3, label="Reacher QL", dashes=[2,0])
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Number of states")
plt.ylabel("Computation time (sec)")
plt.legend()
plt.show()
plt.plot(nbr_sa_PI_CTF, comput_time_PI_CTF, marker='o', color="DarkSeaGreen", linewidth=3, label="Catch The Fruit PI", dashes=[1,1])
plt.plot(nbr_sa_PI_R, comput_time_PI_R, marker='o', color="GoldenRod", linewidth=3, label="Reacher PI", dashes=[1,1])
plt.plot(nbr_sa_VI_CTF, comput_time_VI_CTF, marker='o', color="DarkSeaGreen", linewidth=3, label="Catch The Fruit VI", dashes=[2,1])
plt.plot(nbr_sa_VI_R, comput_time_VI_R, marker='o', color="GoldenRod", linewidth=3, label="Reacher VI", dashes=[2,1])
plt.plot(nbr_sa_QL_CTF, comput_time_QL_CTF, marker='o', color="DarkSeaGreen", linewidth=3, label="Catch The Fruit QL", dashes=[2,0])
plt.plot(nbr_sa_QL_R, comput_time_QL_R, marker='o', color="GoldenRod", linewidth=3, label="Reacher QL", dashes=[2,0])
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Number of state-action combination")
plt.ylabel("Computation time (sec)")
plt.legend()
plt.show()
#endregion




# %%
ε = 0.01
γ = 0.80
width = 3
height = 3
# game = CatchTheFruit(width,height)

arm_nbr = 3
α_res = 50
x_res = 6
# game = Reacher(arm_nbr=arm_nbr, α_res=α_res, x_res=x_res, reward_reach=20)

filename = 'CTFS_3x3_QL_a=0.5_e=0.2_γs=[0.5, 0.9, 0.99]'
infile = open(filename,'rb')
(policies,records) = pickle.load(infile)
infile.close()

# π = policies['γ=0.9']

# records['γ=0.50'][0] = (records['γ=0.50'][0],0)
# records['γ=0.90'][0] = (records['γ=0.90'][0],0)
# records['γ=0.99'][0] = (records['γ=0.99'][0],0)

# records['γ=0.1'][0] = (records['γ=0.1'][0],0)
# records['γ=0.5'][0] = (records['γ=0.5'][0],0)
# records['γ=0.9'][0] = (records['γ=0.9'][0],0)
analysis(records)

# game.play(π,nbr_runs=100,wait=1)














# %%
plt.rcParams['figure.figsize'] = [5,5]
score,anim = game.play(π, nbr_turns=1000)
HTML(anim.to_jshtml())
plt.rcParams['figure.figsize'] = [6.4,4.8]








# %%
