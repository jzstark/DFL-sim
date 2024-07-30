import numpy as np
import matplotlib.pyplot as plt
import random 

import pickle
# Set global font size
plt.rcParams.update({'font.size': 14})


def plot1():

    #fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, layout="constrained")
    fig, axes = plt.subplots(1, 2,figsize=(9, 4))

    with open('p2p_iter100_agents4_10K_kfix0_neighborfix0.pkl', 'rb') as file:
        acc1 = pickle.load(file)
    with open('p2p_iter100_agents4_10K_kfix0_neighborfix1.pkl', 'rb') as file:
        acc2 = pickle.load(file)
    with open('p2p_iter100_agents4_10K_kfix0_neighborfix2.pkl', 'rb') as file:
        acc3 = pickle.load(file)
    xlen = len(acc1)
    axes[0].plot(list(range(xlen)), acc1, linewidth=2, linestyle="-", label='All neighbors')
    axes[0].plot(list(range(xlen)), acc2, linewidth=2, linestyle="-.", label='1 less neighbor')
    axes[0].plot(list(range(xlen)), acc3, linewidth=2, linestyle="--",label='2 less neighbor')
    axes[0].set_ylim([-1, 40000])
    axes[0].set_xlim([-1, 25])
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Norm Error')
    axes[0].set_title('4 Participants')
    axes[0].legend()


    with open('p2p_iter100_agents8_10K_kfix0_neighborfix0.pkl', 'rb') as file:
        acc1 = pickle.load(file)
    with open('p2p_iter100_agents8_10K_kfix0_neighborfix1.pkl', 'rb') as file:
        acc2 = pickle.load(file)
    with open('p2p_iter100_agents8_10K_kfix0_neighborfix2.pkl', 'rb') as file:
        acc3 = pickle.load(file)
    xlen = len(acc1)
    axes[1].plot(list(range(xlen)), acc1, linewidth=2, linestyle="-", label='All neighbors')
    axes[1].plot(list(range(xlen)), acc2, linewidth=2, linestyle="-.", label='1 less neighbor')
    axes[1].plot(list(range(xlen)), acc3, linewidth=2, linestyle="--", label='2 less neighbor')
    axes[1].set_title('8 Participants')
    axes[1].set_xlabel('Iteration')
    #axes[1].set_ylabel('Norm Error')
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def plot2():
    fig, axes = plt.subplots(1, 2,figsize=(9, 3.5))
    
    with open('p2p_iter100_agents4_10K_kfix0_neighborfix0.pkl', 'rb') as file:
        acc1 = pickle.load(file)
    with open('p2p_iter100_agents6_10K_kfix0_neighborfix0.pkl', 'rb') as file:
        acc2 = pickle.load(file)
    with open('p2p_iter100_agents8_10K_kfix0_neighborfix0.pkl', 'rb') as file:
        acc3 = pickle.load(file)
    xlen = len(acc1)
    axes[0].plot(list(range(xlen)), acc1, linewidth=2, linestyle="-", label='n=4')
    axes[0].plot(list(range(xlen)), acc2, linewidth=2, linestyle="-.",label='n=6')
    axes[0].plot(list(range(xlen)), acc3, linewidth=2, linestyle="--",label='n=8')
    axes[0].set_title('All neighbors')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Norm Error')
    axes[0].legend()


    with open('p2p_iter100_agents4_10K_kfix0_neighborfix2.pkl', 'rb') as file:
        acc1 = pickle.load(file)
    with open('p2p_iter100_agents6_10K_kfix0_neighborfix2.pkl', 'rb') as file:
        acc2 = pickle.load(file)
    with open('p2p_iter100_agents8_10K_kfix0_neighborfix2.pkl', 'rb') as file:
        acc3 = pickle.load(file)
    xlen = len(acc1)
    axes[1].plot(list(range(xlen)), acc1, linewidth=2, linestyle="-", label='n=4')
    axes[1].plot(list(range(xlen)), acc2, linewidth=2, linestyle="-.",label='n=6')
    axes[1].plot(list(range(xlen)), acc3, linewidth=2, linestyle="--",label='n=8')
    axes[1].set_ylim([-1, 36000])
    axes[1].set_xlim([-1, 25])
    axes[1].set_title('2 less neighbor')
    axes[1].set_xlabel('Iteration')
    axes[1].legend()

    plt.tight_layout()
    plt.show()



def plot3():
    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    
    with open('p2p_iter100_agents4_10K_kfix0_neighborfix0.pkl', 'rb') as file:
        acc11 = pickle.load(file)
    with open('p2p_iter100_agents6_10K_kfix0_neighborfix0.pkl', 'rb') as file:
        acc12 = pickle.load(file)
    with open('p2p_iter100_agents8_10K_kfix0_neighborfix0.pkl', 'rb') as file:
        acc13 = pickle.load(file)
    with open('p2p_iter100_agents4_10K_kfix0_neighborfix1.pkl', 'rb') as file:
        acc21 = pickle.load(file)
    with open('p2p_iter100_agents6_10K_kfix0_neighborfix1.pkl', 'rb') as file:
        acc22 = pickle.load(file)
    with open('p2p_iter100_agents8_10K_kfix0_neighborfix1.pkl', 'rb') as file:
        acc23 = pickle.load(file)
    with open('p2p_iter100_agents4_10K_kfix0_neighborfix2.pkl', 'rb') as file:
        acc31 = pickle.load(file)
    with open('p2p_iter100_agents6_10K_kfix0_neighborfix2.pkl', 'rb') as file:
        acc32 = pickle.load(file)
    with open('p2p_iter100_agents8_10K_kfix0_neighborfix2.pkl', 'rb') as file:
        acc33 = pickle.load(file)

    foo1 = [(np.mean(x[-5:]), np.std(x[-5:])) for x in [acc11, acc12, acc13]]
    y1, std1 = zip(*foo1)
    foo2 = [(np.mean(x[-5:]), np.std(x[-5:])) for x in [acc21, acc22, acc23]]
    y2, std2 = zip(*foo2)
    foo3 = [(np.mean(x[-5:]), np.std(x[-5:])) for x in [acc31, acc32, acc33]]
    y3, std3 = zip(*foo3)
    x = [4, 6, 8]
    axes.errorbar(x,y1, yerr=std1, linewidth=2, linestyle='solid', marker='o', label="All neighbors")
    axes.errorbar(x,y2, yerr=std2, linewidth=2, linestyle='dashed', marker='^',label="1 less neighbor")
    axes.errorbar(x,y3, yerr=std3, linewidth=2, linestyle='dashdot', marker='s', label="2 less neighbor")
    #axes[0].set_title('K0_N0')
    axes.set_xlabel('Participants #')
    axes.set_ylabel('Norm Error')
    axes.legend()
    plt.tight_layout()

    plt.show()


def plot4():
    fig, axes = plt.subplots(figsize=(6, 4))
    
    with open('p2p_iter100_agents6_10K_kfix0_neighborfix0.pkl', 'rb') as file:
        acc1 = pickle.load(file)
    with open('p2p_iter100_agents6_10K_kfix20_neighborfix0.pkl', 'rb') as file:
        acc2 = pickle.load(file)
    with open('p2p_iter100_agents6_10K_kfix50_neighborfix0.pkl', 'rb') as file:
        acc3 = pickle.load(file)
    with open('p2p_iter100_agents6_10K_kfix100_neighborfix0.pkl', 'rb') as file:
        acc4 = pickle.load(file)
    with open('p2p_iter100_agents6_10K_kfix150_neighborfix0.pkl', 'rb') as file:
        acc5 = pickle.load(file)
    xlen = len(acc1)
    axes.plot(list(range(xlen)), acc1, linewidth=2, linestyle="dotted", label='k=0')
    axes.plot(list(range(xlen)), acc2, linewidth=2, linestyle=":",label='k=20')
    axes.plot(list(range(xlen)), acc3, linewidth=2, linestyle="-.",label='k=50')
    axes.plot(list(range(xlen)), acc4, linewidth=2, linestyle="--",label='k=100')
    axes.plot(list(range(xlen)), acc5, linewidth=2, linestyle="-",label='k=150')
    axes.set_title('All neighbors with 6 participants')
    axes.set_xlabel('Iteration')
    axes.set_ylabel('Norm Error')
    axes.legend()

    plt.tight_layout()
    plt.show()


plot4()