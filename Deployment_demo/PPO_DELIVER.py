import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
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

Color1 = (255, 0, 0)    # Blue - target point
Color4 = (0, 255, 0)    # Green - center of black particles
Color_red = (0, 0, 255) # Red - center of red particles (new)


class VelocityTracker:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.vx_history = deque(maxlen=window_size)
        self.vz_history = deque(maxlen=window_size)
        self.prev_center = None
        
    def update(self, current_center):
        if self.prev_center is None:
            self.prev_center = current_center
            return 0.0, 0.0
        
        current_vx = float(current_center[0] - self.prev_center[0])
        current_vz = float(current_center[1] - self.prev_center[1])
        
        self.vx_history.append(current_vx)
        self.vz_history.append(current_vz)
        self.prev_center = current_center
        
        avg_vx = sum(self.vx_history) / len(self.vx_history) if self.vx_history else 0.0
        avg_vz = sum(self.vz_history) / len(self.vz_history) if self.vz_history else 0.0
        return avg_vx, avg_vz
    
    def reset(self):
        self.vx_history.clear()
        self.vz_history.clear()
        self.prev_center = None


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


def find_particles(image_data, Target_pos):
    try:
        if image_data.dtype == np.float32 or image_data.dtype == np.float64:
            image_data = (image_data * 255).astype(np.uint8)
        
        # Convert to BGR format (OpenCV native format)
        if len(image_data.shape) == 3 and image_data.shape[2] == 3:
            img = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
        else:
            print("Unexpected image format")
            return None, None, None, None, None, None
        
        # ========== Original: Black particle detection logic ==========
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        center = None
        largest_contour = None
        mask = None
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, (255), -1)
            
            M = cv2.moments(mask)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                center = (center_x, center_y)
            else:
                x, y, w, h = cv2.boundingRect(largest_contour)
                center = (x + w//2, y + h//2)
        else:
            print("No black particle contours found")
        # ==============================================================

        # ========== Red particle detection core logic ==========
        redcenter = None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = mask_red1 | mask_red2
        
        red_contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if red_contours:
            largest_red_contour = max(red_contours, key=cv2.contourArea)
            M_red = cv2.moments(largest_red_contour)
            if M_red["m00"] != 0:
                red_x = int(M_red["m10"] / M_red["m00"])
                red_y = int(M_red["m01"] / M_red["m00"])
                redcenter = (red_x, red_y)
            else:
                x_r, y_r, w_r, h_r = cv2.boundingRect(largest_red_contour)
                redcenter = (x_r + w_r//2, y_r + h_r//2)
        # =======================================================

        # ========== New core: Red particle position validation ==========
        is_red_valid = False
        if redcenter is None:
            is_red_valid = True
        elif largest_contour is not None:
            point_test = cv2.pointPolygonTest(largest_contour, redcenter, measureDist=False)
            is_red_valid = (point_test >= 0)
        else:
            is_red_valid = False
        # =================================================================

        # Draw visualization
        result = img.copy()
        cv2.circle(result, (int(Target_pos[0]), int(Target_pos[1])), 2, Color1, 2)
        if center is not None:
            cv2.circle(result, center, 3, Color4, -1)
        if redcenter is not None:
            cv2.circle(result, redcenter, 3, Color_red, -1)

        return result, mask, center, redcenter, is_red_valid, img
        
    except Exception as e:
        print(f"Error in image processing: {e}")
        return None, None, None, None, None, None


class CustomGymWrapper(gym.Wrapper):
    def __init__(self, env, decision_interval=5, max_steps_per_episode=2048, 
                 action_penalty_threshold=30, action_penalty_weight=0.5):
        super().__init__(env)
        self.center = np.zeros(2, dtype=np.int32)
        
        # Target position parameters (normal distribution)
        self.noice = (50 + np.random.randint(0, 100)) / 75
        self.obd = np.random.randint(25, 45) / 35
        random_integer = np.random.randint(0, 299)
        self.sign = 0
        
        x = 320
        y = 240
        r = 100 + np.random.randint(-25, 25)
        theta = 0
        points_x = x + r * np.cos(theta / 180 * np.pi)
        points_y = y + r * np.sin(theta / 180 * np.pi)
        
        random_x = np.random.randint(-10, 10)
        random_y = np.random.randint(-10, 10)
        self.Target_pos = (points_x, points_y)
        self.Obstacle_pos = (
            x + r * np.cos(theta / 180 * np.pi) / 2 + random_x,
            y + r * np.sin(theta / 180 * np.pi) / 2 + random_y
        )
        
        self.velocity_tracker = VelocityTracker(window_size=20)
        self.current_vx = 0
        self.current_vy = 0
        self.preaction = 0
        self.action_space = gym.spaces.Discrete(3)
        self.actual_angle = 0
        self.magsign = 0
        
        self.observation_space = gym.spaces.Box(
            low=np.array([0, -1, -1, -1, -1, 0, -1, -1]),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1]),
            dtype=np.float32
        )
        
        self.distance = 0
        self.obdistance = 0
        self.preobd = 0
        
        # Decision interval control
        self.decision_interval = decision_interval
        self.step_counter = 0
        self.cached_action = 0
        self.currenttheta = 0
        self.currentobtheta = 0
        
        # Episode step control
        self.max_steps_per_episode = max_steps_per_episode
        self.episode_step_counter = 0

        self.action_penalty_threshold = action_penalty_threshold
        self.action_penalty_weight = action_penalty_weight
        self.prev_action = None
        self.is_red_valid = False

    def reset(self):
        # Reset all counters and states
        self.is_red_valid = False
        self.step_counter = 0
        self.magsign = 0
        self.cached_action = 0
        self.episode_step_counter = 0
        self.current_vx = 0
        self.current_vy = 0
        self.actual_angle = 0
        self.sign = 0
        self.distance = 0
        self.obdistance = 0
        self.currenttheta = 0
        self.currentobtheta = 0
        
        self.noice = (50 + np.random.randint(0, 100)) / 75
        self.obd = np.random.randint(25, 45) / 35
        
        # Generate new random target position
        random_integer = np.random.randint(0, 299)
        x = 320
        y = 240
        r = 180
        theta = 0
        
        points_x = x + r * np.cos(theta / 180 * np.pi)
        points_y = y + r * np.sin(theta / 180 * np.pi)
        
        random_x = np.random.randint(-10, 10)
        random_y = np.random.randint(-10, 10)
        self.Target_pos = (points_x, points_y)
        self.Obstacle_pos = (
            x + r * np.cos(theta / 180 * np.pi) / 2 + random_x,
            y + r * np.sin(theta / 180 * np.pi) / 2 + random_y
        )
        
        self.velocity_tracker.reset()
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
        else:
            action_diff_penalty = 0
        
        discrete_to_angle = {0: 0, 1: 10, 2: -10}
        self.actual_angle += discrete_to_angle[self.cached_action]
        action = np.array([self.actual_angle, self.magsign])
        
        obs, reward, done, info = super().step(action)
        modified_obs, result = self.observation(obs)
        
        obx = 0
        if self.is_red_valid is not True or self.sign < 1:
            Target_pos = self.Obstacle_pos
        else:
            Target_pos = self.Target_pos
        
        current_x = Target_pos[0] - self.center[0]
        current_y = Target_pos[1] - self.center[1]
        current_xx = self.Obstacle_pos[0] - self.center[0]
        current_yy = self.Obstacle_pos[1] - self.center[1]
        current_xt = self.Target_pos[0] - self.center[0]
        current_yt = self.Target_pos[1] - self.center[1]
        
        if self.current_vx == 0 and self.current_vy == 0:
            currenttheta = math.atan2(current_y, current_x) * 180 / math.pi
        
        targettheta = math.atan2(current_y, current_x) * 180 / math.pi
        self.currenttheta = math.atan2(current_y, current_x)
        self.currentobtheta = math.atan2(current_yy, current_xx)
        currenttheta = math.atan2(self.current_vy, self.current_vx) * 180 / math.pi
        
        error = abs(targettheta - currenttheta)
        error = min(error, 360 - error)
        distance = math.sqrt(current_x **2 + current_y** 2)
        distancet = math.sqrt(current_xt **2 + current_yt** 2)
        obdistance = math.sqrt(current_xx **2 + current_yy** 2)
        
        self.distance = math.sqrt((current_x / 640) **2 + (current_y / 640)** 2) * self.noice
        
        if self.is_red_valid is not True:
            self.obdistance = math.sqrt((current_xx / 640) **2 + (current_yy / 640)** 2) / self.obd
        
        modified_reward = 0
        
        if self.episode_step_counter >= 50:
            if self.center[0] < 10 or self.center[0] > 630 or self.center[1] < 0 or self.center[1] > 470:
                modified_reward -= 600
                done = True
            
            if abs(error) < 45:
                modified_reward += 0.2
            elif abs(error) < 90:
                modified_reward += 0.1
            elif abs(error) < 135:
                modified_reward -= 0.1
            else:
                modified_reward -= 0.2
            
            if self.is_red_valid:
                modified_reward += 1
            else:
                modified_reward -= 0.5
            
            if (self.is_red_valid is True) and self.sign == 0:
                modified_reward += 150
                self.sign += 1
            
            if distancet < 30:
                modified_reward += 600
                self.magsign = 1
                random_integer = np.random.randint(0, 360)
            
            if self.episode_step_counter >= self.max_steps_per_episode / 3:
                modified_reward -= 0.3
            
            if self.episode_step_counter >= self.max_steps_per_episode:
                modified_reward -= 200
                done = True

        # Record termination reason
        if done:
            if distance < 3:
                info['termination_reason'] = 'target_reached'
            elif obdistance < obx:
                info['termination_reason'] = 'hit_obstacle'
            elif self.center[0] < 10 or self.center[0] > 630 or self.center[1] < 0 or self.center[1] > 470:
                info['termination_reason'] = 'out_of_bounds'
            else:
                info['termination_reason'] = 'max_steps_reached'
        
        info['action_diff_penalty'] = action_diff_penalty
        return modified_obs, modified_reward, done, info, result

    def observation(self, obs):
        result, mask, center, redcenter, is_red_valid, img = find_particles(obs, self.Target_pos)
        print(is_red_valid)
        
        if result is not None:
            cv2.imshow('Unity Processed', result)
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        
        if redcenter is not None:
            self.Obstacle_pos = redcenter
        if is_red_valid is not None:
            self.is_red_valid = is_red_valid
        
        if center is not None:
            self.center = center
            self.current_vx, self.current_vy = self.velocity_tracker.update(center)
        
        combined_obs = np.array([
            self.distance,
            math.sin(self.currenttheta),
            math.cos(self.currenttheta),
            math.sin(self.actual_angle * math.pi / 180),
            math.cos(self.actual_angle * math.pi / 180),
            1,
            math.sin(self.currentobtheta),
            math.cos(self.currentobtheta)
        ], dtype=np.float32)
        
        return combined_obs, img

    def _modify_action(self, action):
        modified_action = action * 180
        return modified_action


# Main program entry
env_name = "deliver_env1/Swarm.exe"
channel = EngineConfigurationChannel()
unity_env = UnityEnvironment(env_name, side_channels=[channel])
channel.set_configuration_parameters(time_scale=1.0)

env = UnityToGymWrapper(unity_env)
baseenv = CustomGymWrapper(
    env,
    decision_interval=5,
    max_steps_per_episode=40480,
    action_penalty_threshold=0.2,
    action_penalty_weight=0.5
)

env = baseenv
model = PPO.load("E:\RL/ppo_nav_final.zip", env)
model.set_env(env)

try:
    obs = env.reset()
    video_out = cv2.VideoWriter('RL_deliver.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info, result = env.step(action)
        cv2.imwrite('Episode_End.png', result)
        video_out.write(result)
        
        if dones:
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