import osim.env.run

import sillywalks.rewards

class RunEnv(osim.env.run.RunEnv):
    ''' overwriting reward function from environment class
    '''
    
    def compute_reward(self):
        
        return sillywalks.rewards.custom_reward(self.current_state)


