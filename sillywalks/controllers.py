# old stuff
import numpy as np

import sillywalks.transforms
import sillywalks.params




def residual_controller(residual_red_state):
    '''
    '''
    #print 'residual_red_state in :', residual_red_state
    
    ## TODO this will become neural network
    residual_red_action = np.zeros(sillywalks.params.DIM_RED_ACTION-sillywalks.params.DIM_TRIVIAL_CONTROL)
    
    #print 'residual_red_action out :', residual_red_action

    return residual_red_action





#############################################################################################################

def proportional_controller( state, target_state=0., constant=-1 ):
    '''
    '''
    action = constant*(state-target_state)
    
    return action

def red_controller( red_state ):
    '''
    '''
    print 'red state ', red_state

    red_action = np.zeros(sillywalks.params.DIM_RED_ACTION)
    
    red_action[0] =  proportional_controller(red_state[0], target_state=0.05, constant=-10)
    red_action[1] =  proportional_controller(red_state[1], target_state=1.2, constant=-10) #positive means corr to unbends
    red_action[2] =  proportional_controller(red_state[2], target_state=0.0, constant=-10) #positive means corr to unbends
    
    residual_red_action = red_action[sillywalks.params.DIM_TRIVIAL_CONTROL:]
    residual_red_state = red_state[sillywalks.params.DIM_TRIVIAL_CONTROL:]
    
    resisual_red_action = residual_controller(residual_red_state)
    
    print 'red_action ', red_action

    return red_action

def controller( state ):
    ''' control function
    calls reduce_con
    '''
        ## reduction of state   
    red_state = sillywalks.transforms.reduce_state( state )
        
        ## controller in reduced coord
    red_action = red_controller( red_state )
    #return env.action_space.sample()

        ## inverse reduction of action   
    action = sillywalks.transforms.inverse_reduce_action( red_action )
    #action = np.zeros(18)
        #a[0]=1. #pull leg back
    #action[1]=1. #bends knee #
    #a[2]=1. #pulls trunk back #
    #action[3]=1. #lifts leg
    #a[4]=1. #pulls trunk forward #
    #a[5]=1. #unbends knee #
    #action[6]=10. #unbends r ankle
    #action[7]=10. #unbends ankle
    #action[8]=-10. #bends ankle
    #action[6+9]=10. #unbends l ankle
    #action[7+9]=10. #unbends ankle
    #action[8+9]=-10. #bends ankle
    #action = -action
    return action, red_state, red_action

