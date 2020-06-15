import numpy as np
from copy import deepcopy as dpc
import matplotlib.pyplot as plt
import random as rnd

import tensorflow as tf
from tensorflow import keras
import tensorflow.contrib.eager as tfe
from tensorflow.python.client import device_lib

'''
def update_policy( policy, optimizer ):
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    # if len(rewards) > 1 : rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    f=open("loss.txt","a")
    f.write(loss.item())
    f.close()
    
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode = []
'''

def getGPUs():
    devices = device_lib.list_local_devices()
    return [ device.name for device in devices if device.device_type == 'GPU']

class Policy :

    def __init__( self, dim_state, dim_action, gamma=0.9, load_name=None ) :

        tf.logging.set_verbosity(tf.logging.ERROR)

        self.state_space = dim_state
        self.action_space = dim_action

        self.gamma = gamma

        self.global_step = tfe.Variable(0)
        self.loss_avg = tfe.metrics.Mean()
        # self.accuracy = tfe.metrics.precision()

        Input1 = keras.Input(shape=(self.state_space-14,),name="fichas")
        Input2 = keras.Input(shape=(14,),name="numeros")
        x1 = keras.layers.Dense(512,activation=tf.nn.relu,use_bias=False)(Input1)
        x2 = keras.layers.Dense(512,activation=tf.nn.relu,use_bias=False)(Input2)
        x1 = keras.layers.Dense(256,activation=tf.nn.relu,use_bias=False)(x1)
        x2 = keras.layers.Dense(256,activation=tf.nn.relu,use_bias=False)(x2)
        x = keras.layers.concatenate([x1,x2],axis=-1)
        x = keras.layers.Dense(64, activation=tf.nn.relu, use_bias=False)(x)
        x = keras.layers.Dropout(rate=0.6)(x)
        output = keras.layers.Dense(self.action_space, activation="softmax")(x)
        self.model = keras.Model(inputs=[Input1,Input2], outputs=output)


        # self.model = keras.Sequential( [
        #     keras.layers.Dense(512,activation=tf.nn.relu,use_bias=False,input_shape=(self.state_space,)),
        #     keras.layers.Dense(256,activation=tf.nn.relu,use_bias=False),
        #     keras.layers.Dense( 128, activation=tf.nn.relu, use_bias=False,input_shape=(self.state_space,)),
        #     keras.layers.Dense( 64, activation=tf.nn.relu, use_bias=False),
        #     keras.layers.Dropout( rate=0.6 ),
        #     keras.layers.Dense( self.action_space, activation=tf.nn.softmax )])
        self.model.summary()

        if load_name is not None : self.model = keras.models.load_model( load_name )

        self.optimizer = tf.train.AdamOptimizer()

        self.device = "CPU:0"
        gpus = getGPUs( )
        if gpus : self.device = gpus[0]

        # Episode policy and reward history
        # self.policy_history = Variable(torch.Tensor())
        # self.reward_episode = []

        # Overall reward and loss history
        # self.reward_history = []
        self.loss_history = []
    def load_Model(self,load_name=None):
        self.model = keras.models.load_model( load_name )


    def update_policy_supervised( self, states, actions ) :
        states = np.array(states)
        epochs = 100
        f=open("loss.txt","a")
        for e in range(epochs) :
            with tf.device( self.device ) :
                with tf.GradientTape() as tape:
                    actions_ = self.model( [states[:,:-14],states[:,-14:] ])
                    # actions = tf.multiply(actions,1/np.sum(actions,axis=1,dtype=np.float).reshape(-1,1))
                    loss = tf.losses.softmax_cross_entropy( onehot_labels=tf.multiply(actions,1/np.sum(actions,axis=1,dtype=np.float).reshape(-1,1)), logits=actions_ )
                grads = tape.gradient( loss, self.model.trainable_variables )
                # del tape
                del tape
                self.optimizer.apply_gradients( zip( grads, self.model.trainable_variables ), self.global_step )
            # logits=[]
            # for elem in actions_:
            #     coso = np.zeros((1,57))
            #     ordenados = np.argsort(elem)
            #     coso[0,ordenados[:3]] = 1
            #     logits.append(coso)
            #
            f.write(str(loss.numpy())+"\n")
            # self.accuracy(coso,actions)
            print( f'\tEpoch {e+1:d}/{epochs}... | Loss: {loss:.3f}' )
        f.close()

    def saveModel( self, name ) :
        self.model.save('models/' + name + '.h5')
