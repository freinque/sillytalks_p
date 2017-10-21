import osim.env.run

import sillywalks.controllers

class RunEnv(osim.env.run.RunEnv):
    '''
    '''
    
    def compute_reward(self):
        
        #reduced_state = sillywalks.controllers.reduce_state(self.current_state)
    
        return sillywalks.controllers.custom_reward(self.current_state)


