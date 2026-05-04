import numpy as np
from stable_baselines3 import PPO
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
import torch as th
import matplotlib.pyplot as plt
from collections import deque
import time
import math
import cv2
import gym
import os

color1 = "#95d0fc"
color2 = "#e50000"
color3 = "#3620f8"
color4 = "#e50000"

def hex_to_bgr(hex_color):
    # Remove '#' symbol
    hex_color = hex_color.lstrip('#')
    
    # Split into R, G, B channels and convert to integers
    r = int(hex_color[0:2], 16)  # Red channel
    g = int(hex_color[2:4], 16)  # Green channel
    b = int(hex_color[4:6], 16)  # Blue channel
    
    # Return BGR format for OpenCV
    return (b, g, r)

Color1 = hex_to_bgr(color1)
Color2 = hex_to_bgr(color2)
Color3 = hex_to_bgr(color3)
Color4 = hex_to_bgr(color4)


def find_particles(image_data, target_ratio, target_angle):
    try:
        if image_data.dtype == np.float32 or image_data.dtype == np.float64:
            image_data = (image_data * 255).astype(np.uint8)
        
        # Convert to BGR format
        if len(image_data.shape) == 3 and image_data.shape[2] == 3:
            img = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
        else:
            print("Unexpected image format")
            return None, None, None, None
        
        # Process black particle swarm
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)

        # Find contours of the particle swarm
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("No contours found")
            return img, None, None, None
        
        # Find the largest contour (particle swarm)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create mask
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, (255), -1)
        
        # Calculate centroid of the particle swarm (center of target rectangle)
        M = cv2.moments(mask)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            center = (center_x, center_y)
        else:
            # Use bounding box center if moments cannot be calculated
            x, y, w, h = cv2.boundingRect(largest_contour)
            center = (x + w//2, y + h//2)
        
        # -------------------------- New: Calculate target rectangle parameters --------------------------
        # 1. Calculate area of the extracted rectangle (target rectangle has equal area)
        extract_area = cv2.contourArea(largest_contour)*1.2
        
        # 2. Calculate width and height of target rectangle based on target aspect ratio
        target_height = np.sqrt(extract_area / target_ratio)
        target_width = target_height * target_ratio
        
        # 3. Construct target rectangle in minAreaRect format
        target_angle_norm = target_angle % 180
        if target_angle_norm > 90:
            target_width, target_height = target_height, target_width
            target_angle_norm -= 90
        if target_angle_norm > 0:
            target_angle_norm -= 180
        target_rect = (center, (target_width, target_height), target_angle_norm)
        
        # 4. Draw target rectangle
        target_box = cv2.boxPoints(target_rect)
        target_box = np.int0(target_box)
        # --------------------------------------------------------------------------
        
        # Calculate parameters of extracted rectangle
        extract_rect = cv2.minAreaRect(largest_contour)
        (_, (extract_w, extract_h), extract_angle) = extract_rect
        if extract_w < extract_h:
            extract_w, extract_h = extract_h, extract_w
            extract_angle += 90
        extract_aspect_ratio = extract_w / extract_h
        extract_angle = extract_angle % 180
        
        # Draw extracted rectangle and center point
        result = img.copy()
        extract_box = cv2.boxPoints(extract_rect)
        extract_box = np.int0(extract_box)
        cv2.drawContours(result, [target_box], 0, Color1, 2)
        cv2.circle(result, center, 3, Color4, -1)

        cv2.imwrite('Episode_End.png', result)
        return result, center, extract_aspect_ratio, extract_angle
    
    except Exception as e:
        print(f"Error in find_particles: {e}")
        return None, None, None, None

class MeanRatio:
    def __init__(self, window_size=2000):
        self.window_size = window_size
        self.ratio_history = deque(maxlen=window_size)
        
    def update(self, error):                          
        # Add to history
        self.ratio_history.append(error**2)      
        avg_r = max(self.ratio_history)       
        return avg_r
    
    def reset(self):
        self.ratio_history.clear()

class AvrRatio:
    def __init__(self, window_size=2000):
        self.window_size = window_size
        self.ratio_history = deque(maxlen=window_size)
        
    def update(self, ratio):                          
        # Add to history
        self.ratio_history.append(ratio)      
        avg_r = sum(self.ratio_history) / len(self.ratio_history)      
        return avg_r
    
    def reset(self):
        self.ratio_history.clear()

class CustomGymWrapper(gym.Wrapper):
    def __init__(self, env, decision_interval=4, max_steps_per_episode=2048):
        super().__init__(env)
        # Target position parameters (normal distribution)
        self.noice = np.random.randint(100, 1000)
        self.aspect_ratio = 1.0
        self.angle = 0.0
        self.angle_error = 0.0
        self.ratio_error = 0.0
        self.mag_ratio = 3.0
        self.mag_angle = 0.0
        self.Target_ratio = np.random.randint(2, 6)
        self.Target_angle = np.random.randint(0, 180)
        self.ratio_tracker = MeanRatio(window_size=100)
        self.angle_tracker = MeanRatio(window_size=100)
        self.avrratio_tracker = AvrRatio(window_size=100)
        self.avrangle_tracker = AvrRatio(window_size=100)
        self.preaction = 0
        self.action_space = gym.spaces.MultiDiscrete([3, 3])
        self.meanR = 4
        self.meanA = 900
        self.avrR = 2
        self.avrA = 1
        self.prev_angleerror = 0
        self.angleintegral = 0
        self.prev_ratioerror = 0
        self.ratiointegral = 0
        self.success_hold_steps = 10
        self.success_counter = 0
        
        self.observation_space = gym.spaces.Box(
            low=np.array([-1,-1, -1, -1,-1,-1]),
            high=np.array([1,1, 1, 1,1,1]),
            dtype=np.float32
        )
        
        # Decision interval control
        self.decision_interval = decision_interval
        self.step_counter = 0
        self.cached_action = 0
        
        # Episode step control
        self.max_steps_per_episode = max_steps_per_episode
        self.episode_step_counter = 0
        self.prev_action = None

    def reset(self):
        # Reset all counters and caches
        self.step_counter = 0
        self.cached_action = 0
        self.episode_step_counter = 0
        self.aspect_ratio = 1.0
        self.noice = np.random.randint(100, 1000)
        self.angle = 0.0
        self.angle_error = 0.0
        self.ratio_error = 0.0
        self.mag_ratio = 3.0
        self.mag_angle = 0.0
        self.Target_angle = 0
        self.Target_ratio = 4
        self.avrR = 2
        self.avrA = 1
        self.meanR = 4
        self.meanA = 900
        self.prev_angleerror = 0
        self.angleintegral = 0
        self.prev_ratioerror = 0
        self.ratiointegral = 0
        self.success_counter = 0
        
        # Generate new random target
        random_integer = np.random.randint(0, 299)
        self.prev_action = None
        self.ratio_tracker.reset()
        self.angle_tracker.reset()
        self.avrratio_tracker.reset()
        self.avrangle_tracker.reset()
        
        obs = super().reset()
        modified_obs, result = self.observation(obs)
        return modified_obs

    def step(self, action):
        self.step_counter += 1
        self.episode_step_counter += 1
        
        # Decision interval logic
        if self.step_counter % self.decision_interval == 1:
            self.cached_action = action
            action_diff_penalty = 0
            self.prev_action = action
            discrete_to_angle = {0: 0, 1: 1, 2: -1}
            self.mag_angle += discrete_to_angle[action[0]] * self.decision_interval / 2
            discrete_to_ratio = {0: 0, 1: 0.05, 2: -0.05}
            self.mag_ratio += discrete_to_ratio[action[1]] * self.decision_interval
            self.mag_ratio = max(self.mag_ratio, 1.8)
            self.mag_ratio = min(self.mag_ratio, 6)
            print(f'Mag Angle: {self.mag_angle}, Mag Ratio: {self.mag_ratio}')
        
        obs, reward, done, info = super().step([self.mag_angle, self.mag_ratio])
        modified_obs, result = self.observation(obs)
        
        if self.step_counter % self.decision_interval == 1:
            self.prev_angleerror = self.angle_error
            self.angleintegral += self.angle_error / 100
            self.angleintegral = max(min(self.angleintegral, 90), -90)
            self.prev_ratioerror = self.ratio_error
            self.ratiointegral += self.ratio_error / self.noice
            self.ratiointegral = max(min(self.ratiointegral, 8), -8)
        
        modified_reward = 0
        
        if self.step_counter < 200:
            modified_reward -= -0.1
        else:
            norm_angle_err = self.angle_error / 45.0
            norm_ratio_err = self.ratio_error / 2.0

            # (a) Base squared error penalty
            w_angle = 0.5
            w_ratio = 1.0
            base_penalty = - (w_angle * norm_angle_err ** 2 + w_ratio * norm_ratio_err ** 2)

            # (b) Progress bonus
            progress_bonus = 0.0
            if self.step_counter % self.decision_interval == 1:
                prev_norm_angle_err = self.prev_angleerror / 45.0
                prev_norm_ratio_err = self.prev_ratioerror / 2.0

                d_angle = prev_norm_angle_err**2 - norm_angle_err**2
                d_ratio = prev_norm_ratio_err**2 - norm_ratio_err**2

                progress_bonus = 0.3 * d_angle + 0.7 * d_ratio
                progress_bonus = float(np.clip(progress_bonus, -1.0, 1.0))

            # (c) Stability bonus
            stability_bonus = 0.0
            if abs(self.angle_error) < 8 and abs(self.ratio_error) < 0.3:
                stability_bonus += 0.5
            if abs(self.angle_error) < 5 and abs(self.ratio_error) < 0.2:
                stability_bonus += 0.5

            # (d) Failure condition: large error
            if abs(self.angle_error) > 70 or abs(self.ratio_error) > 5:
                modified_reward += -10.0
                done = True
                info['termination_reason'] = 'large_error'

            # (e) Success condition: maintain small error
            if abs(self.angle_error) < 10 and abs(self.ratio_error) < 0.5:
                self.success_counter += 1
            else:
                self.success_counter = 0

            if self.success_counter >= self.success_hold_steps:
                if self.Target_ratio < 5:
                    modified_reward += 50.0
                    self.Target_angle += 30
                    self.Target_ratio += 0.5
                    self.success_counter = 0
                info['termination_reason'] = 'success'

            modified_reward += base_penalty + progress_bonus + stability_bonus
        
        # Max steps termination
        if self.episode_step_counter >= self.max_steps_per_episode:
            done = True
        
        # Record termination reason
        if done:         
            info['termination_reason'] = 'max_steps_reached'
        
        return modified_obs, modified_reward, done, info, result

    def observation(self, obs):
        result, center, aspect_ratio, angle = find_particles(obs, self.Target_ratio, self.Target_angle)
        if result is not None:
            cv2.imshow('Unity Processed', result)
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        
        if center is not None:
            self.aspect_ratio = aspect_ratio
            self.angle = self.mag_angle
        
        self.angle_error = -self.angle + self.Target_angle
        self.ratio_error = -self.aspect_ratio + self.Target_ratio
        
        if self.angle_error > 90:
            self.angle_error -= 180
        if self.angle_error < -90:
            self.angle_error += 180
        
        combined_obs = np.array([
            self.ratio_error / 8,
            self.ratiointegral / 8,
            (self.ratio_error - self.prev_ratioerror) / 8,
            (self.angle_error + 7) / 90 * 1.5,
            self.angleintegral / 90,
            (self.angle_error - self.prev_angleerror) / 90
        ], dtype=np.float32)
        
        return combined_obs, result


class RewardLoggerCallback(BaseCallback):
    """
    Custom callback: Merge rewards from all parallel environments and output one global average curve
    """
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.all_episode_rewards = []

    def _on_training_start(self) -> None:
        """Initialize reward counter for each environment"""
        n_envs = self.training_env.num_envs
        self.episode_rewards = [0.0 for _ in range(n_envs)]

    def _on_step(self) -> bool:
        """Update rewards after each step, record and calculate global average when episode ends"""
        current_rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        n_envs = self.training_env.num_envs

        for i in range(n_envs):
            # Accumulate reward for current environment
            self.episode_rewards[i] += current_rewards[i]

            # If episode ends for current environment
            if dones[i]:
                # Record episode reward to global history
                episode_reward = self.episode_rewards[i]
                self.all_episode_rewards.append(episode_reward)
                
                # Print episode info
                print(f"Env {i} | Completed global episode {len(self.all_episode_rewards)} | Reward: {episode_reward:.2f}")
                
                # Calculate global average reward
                global_avg_reward = np.mean(self.all_episode_rewards)
                
                # Write global average reward to TensorBoard
                self.logger.record("episode_reward/global_avg", global_avg_reward)
                
                # Dump log immediately
                self.logger.dump(step=self.num_timesteps)
                
                # Reset accumulated reward for current environment
                self.episode_rewards[i] = 0.0

        return True

# Inference Environment
env_name = "RL_shape_env/Swarm.exe"
channel = EngineConfigurationChannel()
unity_env = UnityEnvironment(env_name, side_channels=[channel])
channel.set_configuration_parameters(time_scale=1)

base_env = UnityToGymWrapper(unity_env)
env = CustomGymWrapper(
    base_env,
    decision_interval=2,
    max_steps_per_episode=40480
)

# Load trained model
model = PPO.load("ppo_shape_final.zip", env)
model.set_env(env)

try:
    obs = env.reset()
    video_out = cv2.VideoWriter('RL_shape.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info, result = env.step(action)
        video_out.write(result)
        
        if dones:
            print('Rewards:', rewards)
            env.reset()
            video_out.release()
            break
        
        env.render()

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    # Safely close environments
    try:
        env.close()
    except:
        pass
    
    try:
        unity_env.close()
    except:
        pass
    
    print("Program terminated")