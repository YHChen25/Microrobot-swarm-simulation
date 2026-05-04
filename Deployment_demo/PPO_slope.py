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


class TrajectoryTracker:
    def __init__(self, max_points=50000):
        self.max_points = max_points
        self.points = deque(maxlen=max_points)  # Only store Python tuples
    
    def add_point(self, point):
        # Force convert to Python tuple (ensure hashable)
        if isinstance(point, np.ndarray):
            point = (int(point[0].item()), int(point[1].item()))  # Convert to Python scalar with .item()
        elif isinstance(point, list):
            point = (int(point[0]), int(point[1]))
        elif not isinstance(point, tuple):
            raise TypeError("Point must be a tuple, list, or numpy array")
        self.points.append(point)
    
    def draw_trajectory(self, img):
        if len(self.points) < 2:
            return img
        for i in range(1, len(self.points)):
            start = self.points[i-1]
            end = self.points[i]
            # cv2.line(img, start, end, Color2, 3)
        
        # Save trajectory (avoid repeated writing, keep only the last record)
        if len(self.points) > 1:
            points_np = np.array(self.points, dtype=np.int32)
            np.savetxt(
                'Trajectory.csv',
                points_np,
                delimiter=',',
                header='x,y',
                comments='',
                fmt='%d'
            )
        return img
    
    def reset(self):
        self.points.clear()


def find_particles(image_data, Target_pos, trajectory_tracker):
    try:
        if image_data.dtype == np.float32 or image_data.dtype == np.float64:
            image_data = (image_data * 255).astype(np.uint8)
        
        # Convert to BGR format
        if len(image_data.shape) == 3 and image_data.shape[2] == 3:
            img = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
        else:
            print("Unexpected image format")
            return None, None, None

        # Process black particle swarm
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)

        # Find contours of the particle swarm
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("No contours found")
            return img, None, None
            
        # Find the largest contour (particle swarm)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create mask
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, (255), -1)
        
        # Calculate centroid of the particle swarm
        M = cv2.moments(mask)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            center = (center_x, center_y)
        else:
            # Use bounding box center if moments cannot be calculated
            x, y, w, h = cv2.boundingRect(largest_contour)
            center = (x + w//2, y + h//2)
        
        # Draw on result image
        result = img.copy()
        # Draw particle swarm contour
        # cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 1)
        
        num_points = 6
        r = 100
        x = 350
        y = 225
        
        points = np.array([[x, y-r],
                           [x+r, y-r],
                           [x, y],
                           [x-r, y+r],
                           [x, y+r],
                           [x+r, y+r]])
        points_x = points[:, 0]
        points_y = points[:, 1]
        
        # Draw particle swarm center
        cv2.circle(result, center, 3, Color4, -1)
        trajectory_tracker.add_point(center)
        
        # Draw historical trajectory
        result = trajectory_tracker.draw_trajectory(result)
        return result, mask, center
        
    except Exception as e:
        print(f"Error in image processing: {e}")
        return None, None, None


x = 400
y = 225
r = 180
num_points = 300
theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
points_x = x + r * np.cos(theta)
points_y = y + r * np.sin(theta)


class VelocityTracker:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.vx_history = deque(maxlen=window_size)
        self.vz_history = deque(maxlen=window_size)
        self.prev_center = None
        
    def update(self, current_center):
        if self.prev_center is None:
            self.prev_center = current_center
            return 0, 0
            
        # Calculate current velocity
        current_vx = current_center[0] - self.prev_center[0]
        current_vz = current_center[1] - self.prev_center[1]
        
        # Add to history
        self.vx_history.append(current_vx)
        self.vz_history.append(current_vz)
        
        # Update previous center point
        self.prev_center = current_center
        
        # Calculate average velocity
        avg_vx = sum(self.vx_history) / len(self.vx_history)
        avg_vz = sum(self.vz_history) / len(self.vz_history)
        
        return avg_vx, avg_vz
    
    def reset(self):
        self.vx_history.clear()
        self.vz_history.clear()
        self.prev_center = None


class CustomGymWrapper(gym.Wrapper):
    def __init__(self, env, decision_interval=5, max_steps_per_episode=2048, 
                 action_penalty_threshold=30, action_penalty_weight=0.5):
        super().__init__(env)
        self.center = np.zeros(2, dtype=np.int32)
        
        # Target position parameters (normal distribution)
        self.noice = (50 + np.random.randint(0, 100)) / 75
        self.obd = np.random.randint(25, 45) / 35
        random_integer = np.random.randint(0, 299)
        
        num_points = 6
        r = 100
        x = 350
        y = 225
        
        points = np.array([[x, y-r],
                           [x+r, y-r],
                           [x, y],
                           [x-r, y+r],
                           [x, y+r],
                           [x+r, y+r]])
        self.points_x = points[:, 0]
        self.points_y = points[:, 1]
        self.sign = 0
        self.Target_pos = (self.points_x[self.sign], self.points_y[self.sign])
        self.Obstacle_pos = (0, 0)
        
        self.velocity_tracker = VelocityTracker(window_size=20)
        self.trjactory_tracker = TrajectoryTracker(max_points=50000)
        self.current_vx = 0
        self.current_vy = 0
        self.preaction = 0
        self.action_space = gym.spaces.Discrete(3)
        self.actual_angle = 0
        
        self.observation_space = gym.spaces.Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )
        
        self.angle_error = 0
        self.angleintegral = 0
        self.prev_angleerror = 0
        self.distance = 0
        self.obdistance = 0
        
        # Decision interval control
        self.decision_interval = decision_interval
        self.mag_angle = 0
        self.step_counter = 0
        self.cached_action = 0
        self.currenttheta = 0
        self.currentobtheta = 0
        
        # Max steps per episode control
        self.max_steps_per_episode = max_steps_per_episode
        self.episode_step_counter = 0

        self.action_penalty_threshold = action_penalty_threshold
        self.action_penalty_weight = action_penalty_weight
        self.prev_action = None

    def reset(self):
        # Reset all counters and caches
        self.step_counter = 0
        self.cached_action = 0
        self.episode_step_counter = 0
        self.current_vx = 0
        self.current_vy = 0
        self.actual_angle = 0
        self.mag_angle = 0
        self.distance = 0
        self.obdistance = 0
        self.currenttheta = 0
        self.currentobtheta = 0
        self.angle_error = 0
        self.angleintegral = 0
        self.prev_angleerror = 0
        self.center = np.zeros(2, dtype=np.int32)
        
        self.noice = (50 + np.random.randint(0, 100)) / 75
        self.obd = np.random.randint(25, 45) / 35
        
        # Generate new random target position
        self.sign = 0
        self.Target_pos = (self.points_x[self.sign], self.points_y[self.sign])
        
        self.velocity_tracker.reset()
        self.trjactory_tracker.reset()
        self.prev_action = None
        
        obs = super().reset()
        modified_obs, result = self.observation(obs)
        return modified_obs

    def step(self, action):
        self.step_counter += 1
        self.episode_step_counter += 1
        
        # Decision interval logic
        if self.step_counter % self.decision_interval == 1:
            self.cached_action = int(action.item())
            action_diff_penalty = 0
            self.prev_action = action
            discrete_to_angle = {0: 0, 1: 1, 2: -1}
            self.mag_angle += discrete_to_angle[self.cached_action] * self.decision_interval
        
        obs, reward, done, info = super().step(self.mag_angle)
        modified_obs, result = self.observation(obs)
        
        current_x = self.Target_pos[0] - self.center[0]
        current_y = self.Target_pos[1] - self.center[1]
        
        if self.step_counter % self.decision_interval == 1:
            self.prev_angleerror = self.angle_error
            self.angleintegral += self.angle_error / 100
            self.angleintegral = max(min(self.angleintegral, 180), -180)
        
        modified_reward = 0
        error = self.angle_error
        
        # Out of bounds penalty
        if self.center[0] < 10 or self.center[0] > 630 or self.center[1] < 0 or self.center[1] > 470:
            modified_reward -= 1000
            done = True
        
        # Angle error reward shaping
        if abs(error) < 20:
            modified_reward += 1 / (abs(error) + 1)
        elif abs(error) < 45:
            modified_reward += 0
        elif abs(error) < 90:
            modified_reward -= 0.1
        elif abs(error) < 135:
            modified_reward -= 0.2
        else:
            modified_reward -= 0.4
        
        distance = math.sqrt(current_x **2 + current_y** 2)

        # Reach target reward
        if distance < 5:
            modified_reward += 100
            self.sign += 1
            if self.sign < 8:
                self.Target_pos = (self.points_x[self.sign], self.points_y[self.sign])
                current_x = self.Target_pos[0] - self.center[0]
                current_y = self.Target_pos[1] - self.center[1]
                self.Target_angle = math.atan2(current_y, current_x) * 180 / math.pi
                self.angle = self.Target_angle
            else:
                done = True
        
        # Max steps termination
        if self.episode_step_counter >= self.max_steps_per_episode:
            modified_reward -= 200
            done = True
         
        # Record termination reason
        if done:
            if distance < 3:
                info['termination_reason'] = 'target_reached'
            elif self.center[0] < 10 or self.center[0] > 630 or self.center[1] < 0 or self.center[1] > 470:
                info['termination_reason'] = 'out_of_bounds'
            else:
                info['termination_reason'] = 'max_steps_reached'
        
        return modified_obs, modified_reward, done, info, result

    def observation(self, obs):
        result, mask, center = find_particles(obs, self.Target_pos, self.trjactory_tracker)
        if result is not None:
            cv2.imshow('Unity Processed', result)
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        
        if center is not None:
            self.center = center
            self.current_vx, self.current_vy = self.velocity_tracker.update(center)
            self.angle = math.atan2(self.current_vy, self.current_vx) * 180 / math.pi
            current_x = self.Target_pos[0] - self.center[0]
            current_y = self.Target_pos[1] - self.center[1]
            self.Target_angle = math.atan2(current_y, current_x) * 180 / math.pi
        else:
            self.center = (0, 0)
        
        self.angle_error = -self.angle + self.Target_angle
        if self.angle_error > 180:
            self.angle_error -= 360
        if self.angle_error < -180:
            self.angle_error += 360
        
        combined_obs = np.array([
            self.angle_error / 180,
            self.angleintegral / 180,
            (self.angle_error - self.prev_angleerror) / 180
        ], dtype=np.float32)
        
        return combined_obs, result

    def _modify_action(self, action):
        modified_action = action * 180
        return modified_action


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


# Initialize environment for inference
env_name = "RL_slope_env/Swarm.exe"
channel = EngineConfigurationChannel()
unity_env = UnityEnvironment(env_name, side_channels=[channel])
channel.set_configuration_parameters(time_scale=1)

base_env = UnityToGymWrapper(unity_env)
env = CustomGymWrapper(
    base_env,
    decision_interval=5,
    max_steps_per_episode=40960,
    action_penalty_threshold=0.2,
    action_penalty_weight=0.5
)

# Load trained model
model = PPO.load("ppo_slope_final.zip", env)
model.set_env(env)

try:
    obs = env.reset()
    video_out = cv2.VideoWriter('RL_slope.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info, result = env.step(action)
        video_out.write(result)
        cv2.imwrite('Episode_End.png', result)
        
        if dones:
            print('Rewards:', rewards)
            env.reset()
            video_out.release()
            break
        
        env.render()

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    # Close environments safely
    try:
        env.close()
    except:
        pass
    
    try:
        unity_env.close()
    except:
        pass
    
    print("Program terminated")