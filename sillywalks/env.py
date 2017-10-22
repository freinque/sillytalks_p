import osim.env.run

import sillywalks.controllers

class RunEnv(osim.env.run.RunEnv):
    ''' overwriting reward function from environment class
    '''
    
    def compute_reward(self):
        
        return sillywalks.controllers.custom_reward(self.current_state)


