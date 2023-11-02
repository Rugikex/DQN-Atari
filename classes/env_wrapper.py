import random

import gymnasium as gym


class AtariWrapper(gym.Wrapper):
    def __init__(self, env, skip_frames=4, play=False):
        super(AtariWrapper, self).__init__(env)
        self.env = env
        self.skip_frames = skip_frames
        self.play = play
        self.lives = 0
        self.has_to_reset = True
        # Get the no-op and fire action
        self.no_op_action = [action for action, meaning in enumerate(env.unwrapped.get_action_meanings()) if meaning == 'NOOP'][0]
        self.did_fire_action = True
        # self.fire_action = None
        # check_fire_action = [action for action, meaning in enumerate(env.unwrapped.get_action_meanings()) if meaning == 'FIRE']
        # if len(check_fire_action) > 0:
        #     self.fire_action = check_fire_action[0]
        #     self.did_fire_action = False

    def reset(self):
        if self.has_to_reset:
            self.has_to_reset = False
            state, info = self.env.reset()
            self.lives = info['lives']
            # Play no-op action between 1 and 30 frames at the beginning of the game
            for _ in range(random.randint(1, 30)):
                state, _, done, _, info = self.env.step(self.no_op_action)
                if info['lives'] != self.lives:
                    done = True
                if done:
                    state, info = self.env.reset()
            
            # Fire action if exists to not get stuck in the game
            if not self.did_fire_action:
                _, _, done, _, info = self.env.step(self.fire_action)
                if info['lives'] != self.lives:
                    done = True
                if done:
                    return self.reset()
        else:
            state, _, _, _, info = self.env.step(self.no_op_action)
        
        return state, info
    
    def step(self, action):
        next_state, reward, done, not_use, info = self.env.step(action)
        self.has_to_reset = done
        previous_state = next_state
        if info['lives'] != self.lives:
            self.lives = info['lives']
            if self.play:
                done = done
            else:
                done = True
            info['previous_state'] = previous_state
            return next_state, reward, done, not_use, info
        # elif info['lives'] != self.lives and not done and self.fire_action is not None:
        #     # Play fire action if exists to not get stuck in the game
        #     next_state, reward, done, not_use, info = self.env.step(self.fire_action)
        #     self.lives = info['lives']
            
        #     info['previous_state'] = previous_state
        #     return next_state, reward, done, not_use, info


        sum_reward = reward
        skip_frames = self.skip_frames - 1 # -1 because we already did one step
        for skip in range(skip_frames):
            if skip <= skip_frames - 2:
                previous_state = next_state

            next_state, reward, done, not_use, info = self.env.step(action)
            self.has_to_reset = done
            if info['lives'] != self.lives:
                self.lives = info['lives']
                done = True

            sum_reward += reward

            if done:
                if self.play:
                    done = self.has_to_reset
                break

        info['previous_state'] = previous_state
        return next_state, sum_reward, done, not_use, info
    