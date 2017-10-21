import numpy as np

import tensorflow as tf
import keras.backend as K

K_0 = 2
K_1 = 2 
K_2 = K_1

C_Y_CM_TARGET = 20.
Y_CM_TARGET = .95
C_TORSO_ANGLE_TARGET = 7.
TORSO_ANGLE_TARGET = 0.05
ANKLE_ANGLE_TARGET = 0.00

DIM_ACTION = 18
DIM_STATE = 41

DIM_TRIVIAL_CONTROL = 3
DIM_RED_ACTION = 7
DIM_RED_STATE = 9


REDUCE_STATE_W = np.zeros(shape=(DIM_STATE, DIM_RED_STATE))
REDUCE_STATE_W[22][0] = 1.
REDUCE_STATE_W[24][0] = -1.
REDUCE_STATE_W[6][1] = 1.
REDUCE_STATE_W[7][2] = 1.
REDUCE_STATE_W[8][3] = 1.
REDUCE_STATE_W[9][4] = 1.
REDUCE_STATE_W[10][5] = 1.
REDUCE_STATE_W[11][6] = 1.
REDUCE_STATE_W[19][7] = 1.
REDUCE_STATE_W[20][8] = 1.
REDUCE_STATE_B = np.zeros(shape=(DIM_RED_STATE))

INVERSE_REDUCE_ACTION_W = np.zeros(shape=(DIM_RED_ACTION, DIM_ACTION))
INVERSE_REDUCE_ACTION_W[0][2] = -1.
INVERSE_REDUCE_ACTION_W[0][4] = 1.
INVERSE_REDUCE_ACTION_W[1][6] = 1.
INVERSE_REDUCE_ACTION_W[1][7] = 1.
INVERSE_REDUCE_ACTION_W[1][8] = -1.
INVERSE_REDUCE_ACTION_W[2][6+9] = 1.
INVERSE_REDUCE_ACTION_W[2][7+9] = 1.
INVERSE_REDUCE_ACTION_W[2][8+9] = -1.
INVERSE_REDUCE_ACTION_W[3][1] = -1.
INVERSE_REDUCE_ACTION_W[3][5] = 1.
INVERSE_REDUCE_ACTION_W[3][1+9] = -1.
INVERSE_REDUCE_ACTION_W[3][5+9] = 1.
INVERSE_REDUCE_ACTION_W[5][0] = 1.
INVERSE_REDUCE_ACTION_W[5][3] = -1.
INVERSE_REDUCE_ACTION_W[5][0+9] = 1.
INVERSE_REDUCE_ACTION_W[5][3+9] = -1.
INVERSE_REDUCE_ACTION_B = np.zeros(shape=(DIM_ACTION))

REDUCE_ACTION_W = INVERSE_REDUCE_ACTION_W.T
REDUCE_ACTION_B = np.zeros(shape=(DIM_RED_ACTION))


PROP_CONTROL_W = np.zeros(shape=(DIM_RED_STATE, DIM_RED_ACTION))
PROP_CONTROL_W[0][0] = -K_0
PROP_CONTROL_W[1][1] = -K_1
PROP_CONTROL_W[2][2] = -K_2
PROP_CONTROL_B = np.zeros(shape=(DIM_RED_ACTION))




def residual_controller(residual_red_state):
    '''
    '''
    print 'residual_red_state in :', residual_red_state
    
    residual_red_action = np.zeros(DIM_RED_ACTION-DIM_TRIVIAL_CONTROL)
    
    print 'residual_red_action out :', residual_red_action

    return residual_red_action





#############################################################################################################

def reduce_state_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] = DIM_RED_ACTION
    return tuple(shape)



def reduce_state( state ):
    '''
    '''
#position of the pelvis (rotation, x, y) 0,1,2    -0.05, 0.0, 0.91,
#velocity of the pelvis (rotation, x, y) 3,4,5     0.0, 0.0, 0.0,
#rotation of each ankle, knee and hip (6 values) 6,7,8,9,10,11               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#angular velocity of each ankle, knee and hip (6 values) 12,13,14,15,16,17,              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  
#position of the center of mass (2 values)  18,19   -0.06973405523475405, 0.9707652519778237, 
#velocity of the center of mass (2 values)  20, 21        0.0, 0.0,  
#positions (x, y) of head, pelvis, torso, left and right toes, left and right talus (14 values) head 22,23,pel 24,25,tor 26,27,ltoe 28,29,rtoe 30,31,ltal 32,33,rtal 34,35 0.007169537779780744, 1.5365721883823606, 0.0, 0.91, -0.09650084892621281, 0.9964310485677471, 0.007987580127344573, -0.027441466796053905, 0.007987580127344573, -0.027441466796053905, -0.11968333174236659, 0.022952398528571172, -0.11968333174236659, 0.022952398528571172, 
#strength of left and right psoas: 1 for  difficulty < 2  , otherwise a random normal variable with mean 1 and standard, 36,37 1,1
#deviation 0.1 fixed for the entire simulation
#next obstacle: x distance from the pelvis, y position of the center relative to the the ground, radius. 38,39,40  [ 100, 0, 0]

    pelvis_angle = state[0]
    x_pelvis = state[1]
    y_pelvis = state[2]
    dx_pelvis = state[4]
    dy_pelvis = state[5]
    
    torso_angle = state[22] - state[24] #diff in x approx
    x_cm = state[18]
    y_cm = state[19]
    dx_cm = state[20]
    dy_cm = state[21]
    
    l_ankle_angle = state[6]
    r_ankle_angle = state[7]
    
    #x_l_tal = state[32]
    #x_r_tal = state[34]
    #y_l_tal = state[33]
    #y_r_tal = state[35]

    l_knee_angle = state[8]
    r_knee_angle = state[9]
    
    l_upper_angle = state[10]
    r_upper_angle = state[11]
  
    red_state = [           
                            ## trivial control
                        #pelvis_angle, 
                        torso_angle, 
                        l_ankle_angle, 
                        r_ankle_angle, 
                            ##
                        l_knee_angle, 
                        r_knee_angle, 
                        l_upper_angle, 
                        r_upper_angle,
                            ##
                        #y_pelvis,
                        y_cm,
                        dx_cm,
                        ]
    
    return red_state

def inverse_reduce_action( red_action ):
    ''' transfo from reduced action space to original
    '''
    #hamstrings,
    #biceps femoris,
    #gluteus maximus,
    #iliopsoas,
    #rectus femoris,
    #vastus,
    #gastrocnemius,
    #soleus,
    #tibialis anterior.
    ## 18 conps: 9 muscles of the right leg first, then 9 muscles of the left leg
    action = np.zeros(DIM_ACTION)

    action[2] = -red_action[0] # mapping upper body angle action to original action space 
    action[4] = red_action[0] # mapping body angle angle action to original action space, positive corresp to bending back
      
    action[6] = red_action[1] # mapping r ankle, pos corresp to opening ankle
    action[7] = red_action[1] # mapping r ankle
    action[8] = -red_action[1] # mapping r ankle
    action[6+9] = red_action[2] # mapping l ankle, pos corresp to opening ankle
    action[7+9] = red_action[2] # mapping l ankle
    action[8+9] = -red_action[2] # mapping l ankle
    
    action[1] = -red_action[3] # mapping r knee, positive corresp to bending
    action[5] = red_action[3] # mapping r knee
    action[1+9] = -red_action[4] # mapping l knee, positive corresp to bending
    action[5+9] = red_action[4] # mapping l knee
   
    action[0] = red_action[5] # mapping left upper angle
    action[3] = -red_action[5] # mapping left upper angle
    action[0+9] = red_action[6] # mapping left upper angle
    action[3+9] = -red_action[6] # mapping left upper angle
    
    return action


def proportional_controller( state, target_state=0., constant=-1 ):
    '''
    '''
    action = constant*(state-target_state)
    
    return action



def red_controller( red_state ):
    '''
    '''
    red_action = np.zeros(DIM_RED_ACTION)

    red_action[0] =  proportional_controller(red_state[0], target_state=TORSO_ANGLE_TARGET, constant=-K_0)
    red_action[1] =  proportional_controller(red_state[1], target_state=0.0, constant=-K_1) #positive means corr to unbends
    red_action[2] =  proportional_controller(red_state[2], target_state=0.0, constant=-K_2) #positive means corr to unbends
    
    residual_red_action = red_action[DIM_TRIVIAL_CONTROL:]
    residual_red_state = red_state[DIM_TRIVIAL_CONTROL:]
    
    resisual_red_action = residual_controller(residual_red_state)
    
    return red_action

def controller( state ):
    ''' control function
    calls reduce_con
    '''
        ## reduction of state   
    red_state = reduce_state( state )
        
        ## controller in reduced coord
    red_action = red_controller( red_state )
    #return env.action_space.sample()

        ## inverse reduction of action   
    action = inverse_reduce_action( red_action )
    #action = np.zeros(18)
        #a[0]=1. #pull leg back
        #a[1]=1. #bends knee #
        #a[2]=1. #pulls trunk back #
        #a[3]=1. #lifts leg
        #a[4]=1. #pulls trunk forward #
        #a[5]=1. #unbends knee #
        #a[6]=1. #unbends ankle
        #a[7]=1. #unbends ankle
        #a[8]=1. #bends ankle
    return action, red_state, red_action

##################################################################################################


def custom_reward(state):
    '''
    '''
    red_state = reduce_state(state)
    
    term1 = red_state[-1]
    term2 = C_Y_CM_TARGET*min(red_state[-2]-Y_CM_TARGET, 0.)
    term3 = -C_TORSO_ANGLE_TARGET*abs(red_state[0]-TORSO_ANGLE_TARGET)
    
    #print 'y_cm', red_state[-2]
    #print 'term1', term1
    #print 'term2', term2
    #print 'term3', term3

    return term1 + term2 + term3 # dx_cm**2 minus penalty on low y_cm


