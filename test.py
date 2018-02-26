import numpy as np
import gym
import random
import time


"""Types of Noise
   Sensor noise type = 0   => No Sensor Noise
   Sensor noise type = 1   => Uniform Angle Sensor Noise
   Sensor noise type = 2   => Gaussian Angle sensor Noise
   Sensor noise type = 3   => Uniform Sensor Noise for all states
   Sensor noise type = 4   => Gaussian Sensor Noise for all states
   Actuator noise type = 0 => No Actuator Noise
   Actuator noise type = 1 => Uniform Actuator Noise
   Actuator noise type = 2 => Gaussian Actuator Noise

"""

sensor_noise_type=2
actuator_noise_type=1

rs=10/100             #Percent Uniform Sensor Noise
ra=10/100             #Percent Uniform Actuator Noise
rsig=0.2              #Sigma Gaussian Sensor Noise
r_seed=10             #Random Seed

max_steps=200      #Max Number of steps
max_trials=1       #Max of Trial in a Run
max_runs=100           #Max Number of Runs


Rad2Ang=180/np.pi
Ang2Rad=np.pi/180

alpha=0.95	    #discount factor
Initlc=0.3	    #initial learning rate for critic
Initla=0.3	    #initial learning rate for action
Ta = 0.005          #Threshold Action Error
Tc = 0.05           #Threshold Critic Error

NF_Theta = (12.0*Ang2Rad)
NF_ThetaDot = (120.0*Ang2Rad)
NF_x = 2.4
NF_xDot = 1.5

Ncrit = 5
Nact = 10

n_hidden = 6 #Number of hidden nodes

#Objective value of the cost function
Uc=0

#Number of Input Units
wc_inputs=5
wa_inputs=4

Exphist=[]

class Environ:

    def __init__(self):
        self.NF=[NF_x,NF_xDot,NF_Theta,NF_ThetaDot]

    def sensor_noise(self, state, type, rs,rsig):
        if type==0:
            state=np.divide(state, self.NF)
            return state
        elif type==1:
            state=state*[1,1,1+np.random.uniform(-rs,rs),1]
            return np.divide(state,self.NF)
        elif type==2:
            state=state*[1,1,1+np.random.normal(0,rsig),1]
            return np.divide(state, self.NF)
        elif type==3:
            state=state*[1+np.random.uniform(-rs,rs),1+np.random.uniform(-rs,rs),1+np.random.uniform(-rs,rs),1+np.random.uniform(-rs,rs)]
            return np.divide(state,self.NF)
        elif type==4:
            state=state*[1+np.random.normal(0,rsig),1+np.random.normal(0,rsig),1+np.random.normal(0,rsig),1+np.random.normal(0,rsig)]
            return np.divide(state, self.NF)

    def actuator_noise(self,new_action,type,ra):
        if type==0:
            return new_action
        elif type==1:
            return new_action+np.random.uniform(-ra,ra)
        elif type==2:
            return new_action+np.random.normal(-ra,ra)

class Agent:

    def crit_input(self,inputs,new_action):
        x=list(inputs)
        x.append(new_action)
        return np.array(x)

    def act(self, inputs, wa1, wa2):
        ha=np.matmul(inputs,wa1)
        g= (1-np.exp(-ha))/(1+np.exp(-ha))
        va = np.matmul(g,wa2)
        new_action = (1 - np.exp(-va))/(1 + np.exp(-va))
        return new_action,g

    def crit(self, inp, wc1, wc2):
        qc=np.matmul(inp,wc1)
        p = (1 - np.exp(-qc))/(1 + np.exp(-qc))
        J=np.matmul(p,wc2)
        return J,p

    def normalize(self, w1,w2):
        w1 = w1 / np.max(np.absolute(w1))
        w2 = w2 / np.max(np.absolute(w2))
        return w1,w2

    def error(self,e):
        E=0.5 * np.square(e)
        return E

def start():
    env = gym.make('CartPole-v0')
    env._max_episode_steps=None
    env.seed(r_seed)
    agent=Agent()
    environ=Environ()
    count=0
    weights = 1
    w_t = np.load('weights.npy')
    values = w_t
    total_reward = 0
    for runs in range(1,max_runs+1):

      for trial in range(1,max_trials+1):

        steps = 0
        lc = Initlc
        la = Initla

        #Actor and Critic Network Random Weight Initialization
        if not weights:
            wc1 = np.square((np.random.uniform(0, 1, (wc_inputs, n_hidden)) - 0.5))
            wc2 = np.square((np.random.uniform(0, 1, (n_hidden, 1)) - 0.5))
            wa1 = np.square((np.random.uniform(0, 1, (wa_inputs, n_hidden)) - 0.5))
            wa2 = np.square((np.random.uniform(0, 1, (n_hidden, 1)) - 0.5))
            delta_wa1 = np.random.uniform(1, n_hidden)
            delta_wa2 = np.random.uniform(n_hidden, 1)
            values= [wc1,wc2,wa1,wa2,delta_wa1,delta_wa2]
        else:
            wc1=w_t[0]
            wc2=w_t[1]
            wa1=w_t[2]
            wa2=w_t[3]
            delta_wa1=w_t[4]
            delta_wa2=w_t[5]

        # Start Trial
        initstate=env.reset()
        inputs = environ.sensor_noise(initstate, sensor_noise_type, rs, rsig)
        new_action,g=agent.act(inputs,wa1,wa2)
        new_action = environ.actuator_noise(new_action, actuator_noise_type, ra)
        x=agent.crit_input(inputs,new_action)
        J,p=agent.crit(x,wc1,wc2)
        Jprev = J

        #Exceute Trial until Failure
        while(steps<max_steps):
            if (new_action >= 0):
                action = 1
            else:
                action = 0
            #env.render()
            next_state, reward, done, info = env.step(action)
            total_reward = total_reward + reward
            inputs = environ.sensor_noise(next_state, sensor_noise_type, rs,rsig)
            new_action,g = agent.act(inputs,wa1,wa2)
            x = agent.crit_input(inputs, new_action)
            J,p = agent.crit(x,wc1,wc2)
            if not done:
                reinf = 0
            else:
                reinf = -1
            # Learning Weights update
            if steps%5==0:
                lc = lc - 0.05
                la = la - 0.05

            if (lc<0.01):
                lc=0.005

            if (la<0.01):
                la=0.005

            cyc = 0
            ecrit = alpha*J-(Jprev-reinf)
            Ec = agent.error(ecrit)

            # Critic Network Update
            while (Ec> Tc and cyc< Ncrit):
                gradEcJ=alpha*ecrit
                x=agent.crit_input(inputs,new_action)
                gradqwc1 = x.T
                for i in range(n_hidden):
                    gradJp=wc2[i]
                    gradpq=0.5*(1-np.square(p[i]))
                    wc1[:,i]-=lc*np.matmul(gradEcJ,gradJp)*gradpq*gradqwc1

                gradJwc2=p.T
                wc2[:,0] -= lc*gradEcJ*gradJwc2

                x = agent.crit_input(inputs, new_action)
                J,p=agent.crit(x,wc1,wc2)

                cyc+=1
                ecrit = alpha*J-(Jprev-reinf)
                Ec = 0.5 * np.square(ecrit)

            #normalize weights
            wc1,wc2 = agent.normalize(wc1,wc2)

            cyc = 0
            eact=J-Uc
            Ea = agent.error(eact)
            #Action network update
            while (Ea>Ta and cyc<=Nact):

                graduv = 0.5*(1-np.square(new_action))
                gradEaJ = eact
                gradJu = 0

                for i in range(n_hidden):
                    gradJu = gradJu + wc2[i]*0.5*(1-np.square(p[i]))*wc1[wc_inputs-1,i]

                for i in range(n_hidden):
                    gradvg=wa2[i]
                    gradgh=0.5*(1-np.square(g[i]))
                    gradhwa1=inputs.T
                    delta_wa1=-la*gradEaJ*gradJu*graduv*gradvg*gradgh*gradhwa1
                    wa1[:,i] += delta_wa1

                gradvwa2 = g.T

                delta_wa2=-la*gradEaJ*gradJu*graduv*gradvwa2

                wa2[:,0] += delta_wa2

                new_action,g=agent.act(inputs,wa1,wa2)
                x = agent.crit_input(inputs, new_action)
                J,p=agent.crit(x,wc1,wc2)
                cyc = cyc+1
                Ea = agent.error(J-Uc)

            wa1, wa2 = agent.normalize(wa1, wa2)

            if done:
                break
            else:
                Jprev = J
            steps+=1

        if not done:

            Exphist.append((runs,trial))
            w_t = values
            weights = 1
            count += 1
            print ("Trial #",trial, "has balanced for", str(steps)," steps in run#",runs," consecutive avg reward", total_reward/count)
            break
        else:
            print ("Trial has", str(steps), "steps now")

        if(trial%100 == 0):
            weights=0
            print("Still continuing",trial)


    print("[Run,Trial]",Exphist)
    trial_sum=0
    for run,trial in Exphist:
        trial_sum+=trial
    avg_trial=trial_sum/max_runs
    success_ratio=count/max_runs*100
    avg_reward = total_reward / max_runs
    print(max_runs,"runs : Average # of trial is ",avg_trial)
    print("Successful Runs",success_ratio,"%")
    print("Total Reward",total_reward)

if __name__ == "__main__":
    start_time = time.time()
    start()
    print("Total Time Taken", (time.time() - start_time))
