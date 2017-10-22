import numpy as np

import sillywalks.transforms
import sillywalks.params


##################################################################################################
## reward functions

def custom_reward(state):
    '''
    '''
    red_state = sillywalks.transforms.reduce_state(state)                   # reward defined at red_state level
    
    term1 = red_state[-1]                                                   # dx_cm term
    term2 = sillywalks.params.C_Y_CM_TARGET*min(red_state[-2]-sillywalks.params.Y_CM_TARGET, 0.)             # low y_cm penalty term
    term3 = -sillywalks.params.C_TORSO_ANGLE_TARGET*abs(red_state[0]-sillywalks.params.TORSO_ANGLE_TARGET)   # torso_angle penalty term
    
    #print 'y_cm', red_state[-2]
    #print 'term1', term1
    #print 'term2', term2
    #print 'term3', term3

    return term1 + term2 + term3


