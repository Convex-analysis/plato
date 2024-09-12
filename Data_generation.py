import numpy as np
    
class FLclient:
    def __init__(self, in_time=0, out_time=0, communicattion_time=0, computation_time=0, control_param=1, schedule_time=0, actual_cos_time=0):
        self.in_time = in_time
        self.out_time = out_time
        self.communication_time = communicattion_time
        self.computation_time = computation_time
        self.control_param = control_param
        self.actual_cos_time = actual_cos_time
        self.schedule_time = schedule_time
    
    #Set attributions of clients 
    def set_in_time(self, in_time):
        self.in_time = in_time
    
    def set_out_time(self, out_time):
        self.out_time = out_time
    
    def set_communication_time(self, communication_time):
        self.communication_time = communication_time
    
    def set_computation_time(self, computation_time):
        self.computation_time = computation_time
    
    def set_control_param(self):
        #randomly generate control parameters from 1 to 10
        self.control_param = np.random.randint(1, 10)
        
    #set schedule time of client
    def set_schedule_time(self, schedule_time):
        self.schedule_time = schedule_time
        
    def set_actual_cos_time(self):
        self.actual_cos_time = self.communication_time + (self.computation_time/self.control_param)
        
    def get_in_time(self):
        return self.actual_cos_time   
    def get_schedule_time(self):
        return self.schedule_time
    def get_out_time(self):
        return self.out_time
    def get_communication_time(self):
        return self.communication_time
    def get_computation_time(self):
        return self.computation_time
    def get_control_param(self):
        return self.control_param 
    
class Environment:
    def __init__(self, clientstobeAssigned, bandwidth, time):
        #the initial client list when T=0
        self.client_list = clientstobeAssigned
        #time here denotes the one FL round time, the unit is ms, define the time length from t=0 to t=time
        self.time = time
        #bandwidth of 5G NR V2X sidelink, denotes the subchannel amounts of bandwidth, can be configured as 10, 12, 15, 20, 25, 50, 100
        self.bandwidth = bandwidth
        #subchannel list, the length is equal to bandwidth
        self.bandwidth_list = [0]*bandwidth
       
    #generate the resource pool according to the bandwidth and time as a 2D matrix
    def generate_resource_pool(self):
        resource_pool = np.zeros((self.bandwidth, self.time/10))
        return resource_pool
    
    #Generate the number of newly joined clients at each moment based on the Poisson process
    def generate_new_clients_number(self):
        #the average number of clients that join the FL system at each moment
        lamda = 3
        #generate the number of newly joined clients at each moment based on the Poisson queue
        new_clients_number = np.random.poisson(lamda)
        #outputs are rounded to the nearest integer
        new_clients_number = np.round(new_clients_number)
        if new_clients_number <= 0:
            new_clients_number = 0
        return new_clients_number
    
    #set part
    def set_time(self, time):
        self.time = time
        
    def set_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth
        
    def set_client_list(self, client_list):
        self.client_list = client_list
        
    def set_bandwidth_list(self, bandwidth_list):
        self.bandwidth_list = bandwidth_list
    #get part
    def get_time(self):
        return self.time
    
    def get_bandwidth(self):
        return self.bandwidth
    
    def get_client_list(self):  
        return self.client_list
    
    def get_bandwidth_list(self):
        return self.bandwidth_list


#State class is used to store the state of the environment, including the client list and the utilization of resource pool at moment time
class State:
    def __init__(self, client_list, resource_pool, time):
        self.client_list = client_list
        self.resource_pool = resource_pool
        self.time = time
        
    def set_client_list(self, client_list):
        self.client_list = client_list
        
    def set_resource_pool(self, resource_pool):
        self.resource_pool = resource_pool
        
    def get_client_list(self):
        return self.client_list
    
    def get_resource_pool(self):
        return self.resource_pool 

    def get_time(self):
        return self.time    
    
#Action class is used to store the bandwidth and the time length of client in the resource pool
class Action:
    def __init__(self, bandwidth, time, client, control_param):
        self.bandwidth = bandwidth
        self.time = time
        self.client = client
        self.control_param = control_param
        
    def set_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth
        
    def set_time(self, time):
        self.time = time
        
    def get_bandwidth(self):
        return self.bandwidth
    
    def get_time(self):
        return self.time
    
    def set_control_param(self, control_param):
        self.control_param = control_param
        
    def get_control_param(self):
        return self.control_param
    
    def get_client(self):
        return self.client
    
    
#Reward class is used to store the reward of action
class Reward:
    def __init__(self, reward):
        self.reward = reward
        
    def set_reward(self, reward):
        self.reward = reward
        
    def get_reward(self):
        return self.reward    
    
    
def main():
    #the probability of newly joined clients at each moment
    new_joint_clients_probability = 0.3
    '''
        Initialize the environment of this FL system
        initial bandwidth: 100Mhz
        subchannel amount (bandwith in the parameter): 50 means there are 50 subchannels and each subchannel is 2Mhz
        time: 100000ms    
    '''
    env = Environment(clientstobeAssigned = [], bandwidth = 50, time = 100000)
    '''
        Initialize the client list of this FL system from t=0 to t=100000 or 1000
    '''
    for t in range(env.get_time()/10):
        #generate the number of newly joined clients at each moment based on the Poisson queue
        if np.random.rand() < new_joint_clients_probability:
            new_clients_number = env.generate_new_clients_number()
            #generate the new clients
            for i in range(new_clients_number):
                client = FLclient()
                client.set_in_time(t*10)
                #the sojourn time of clients in the FL system is randomly generated from 20s to 60s
                run_time = np.random.randint(20000, 60000)
                client.set_out_time(t*10 + run_time)
                client.set_control_param()
                #let the sum of communication time and computation time be randomly generated from 0.5 to 1.5 times run_time
                total_time = np.random.randint(run_time*0.5, run_time*1.5)
                #the ratio of communication time to computation time is randomly generated from 0.5 to 0.9
                ratio = np.random.uniform(0.6, 0.9)
                client.set_communication_time(total_time*(1-ratio))
                client.set_computation_time(total_time*ratio)
                client.set_actual_cos_time()
                env.get_client_list().append(client)
        #allocate the resource pool to the clients
        else:
            pass       
        #generate the resource pool according to the bandwidth and time as a 2D matrix