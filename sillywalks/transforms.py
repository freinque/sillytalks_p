import numpy as np

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


