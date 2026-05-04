from unittest import result

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

    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) 
    g = int(hex_color[2:4], 16)  
    b = int(hex_color[4:6], 16) 
    

    return (b, g, r)
Color1=hex_to_bgr(color1)
Color2=hex_to_bgr(color2)
Color3=hex_to_bgr(color3)
Color4=hex_to_bgr(color4)

def find_particles(image_data,Target_pos,Obstacle_pos):
    try:
        
        if image_data.dtype == np.float32 or image_data.dtype == np.float64:
            image_data = (image_data * 255).astype(np.uint8)
        
        # 转换为BGR格式
        if len(image_data.shape) == 3 and image_data.shape[2] == 3:
            img = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
        else:
            print("Unexpected image format")
            return None, None, None
        #metaimg=img
        #img = cv2.resize(img, (640, 480))  
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("No contours found")
            return img, None, None

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
        
        result = img.copy()
        #cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 1)  
                   
        cv2.circle(result, (int(Target_pos[0]), int(Target_pos[1])), 2, Color1, 2)
        cv2.circle(result, (int(Obstacle_pos[0]), int(Obstacle_pos[1])), 15, Color2, -1)
        cv2.circle(result, center, 3, Color4, -1)  

        return result, mask, center
        
    except Exception as e:
        print(f"Error in image processing: {e}")
        return None, None, None


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
            
        current_vx = current_center[0] - self.prev_center[0]
        current_vz = current_center[1] - self.prev_center[1]
        
        self.vx_history.append(current_vx)
        self.vz_history.append(current_vz)
        
        self.prev_center = current_center
        
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
        self.noice=(50+np.random.randint(0, 100))/75
        self.obd=np.random.randint(25, 45)/35
        random_integer = np.random.randint(0, 299)
        self.sign=0
        x = 320
        y = 240
        r = 180+np.random.randint(-25, 25)
        theta=np.random.randint(0, 360)
        points_x = x + r * np.cos(theta/180*np.pi)  
        points_y = y + r * np.sin(theta/180*np.pi)  
        random_x=np.random.randint(-10, 10)
        random_y=np.random.randint(-10, 10)
        self.Target_pos = (points_x, points_y)
        self.Obstacle_pos = (x+r * np.cos(theta/180*np.pi)/2+random_x,y+ r * np.sin(theta/180*np.pi)/2+random_y)
        #self.Target_pos = (points_x[random_integer], points_y[random_integer]) 
        #self.Obstacle_pos = (points_x[random_integer]/2+400/2+random_x, points_y[random_integer]/2+225/2+random_y)
        self.velocity_tracker = VelocityTracker(window_size=20)
        self.current_vx = 0
        self.current_vy = 0 
        self.preaction = 0
        self.action_space = gym.spaces.Discrete(3)
        self.actual_angle=0
        
        self.observation_space = gym.spaces.Box(
            low=np.array([0, -1, -1, -1,-1,0,-1,-1]),
            high=np.array([1, 1, 1, 1,1,1,1,1]), 
            dtype=np.float32  
        )
        self.distance=0
        self.obdistance=0
        self.decision_interval = decision_interval
        self.step_counter = 0
        self.cached_action = 0
        self.currenttheta=0
        self.currentobtheta=0
        self.max_steps_per_episode = max_steps_per_episode  
        self.episode_step_counter = 0 

        self.action_penalty_threshold = action_penalty_threshold 
        self.action_penalty_weight = action_penalty_weight      
        self.prev_action = None                           

    def reset(self):
        self.step_counter = 0
        self.cached_action = 0
        self.episode_step_counter = 0
        self.current_vx = 0
        self.current_vy = 0 
        self.actual_angle=0
        self.sign=0
        self.distance=0
        self.obdistance=0
        self.currenttheta=0
        self.currentobtheta=0
        self.noice=(50+np.random.randint(0, 100))/75
        self.obd=np.random.randint(25, 45)/35

        random_integer = np.random.randint(0, 299)
        x = 320
        y = 240
        r = 200+np.random.randint(-25, 25)
        theta=np.random.randint(0, 360)
        points_x = x + r * np.cos(theta/180*np.pi)  
        points_y = y + r * np.sin(theta/180*np.pi)  
        random_x=np.random.randint(-10, 10)
        random_y=np.random.randint(-10, 10)
        self.Target_pos = (points_x, points_y)
        self.Obstacle_pos = (x+r * np.cos(theta/180*np.pi)/2+random_x,y+ r * np.sin(theta/180*np.pi)/2+random_y)
        #self.Target_pos = (points_x[random_integer], points_y[random_integer])
        #self.Obstacle_pos = (points_x[random_integer]/2+400/2+random_x, points_y[random_integer]/2+225/2+random_y)
        self.velocity_tracker.reset()
        self.prev_action = None
        obs = super().reset()
        modified_obs,result = self.observation(obs)
        return modified_obs

    def step(self, action):
        self.step_counter += 1
        self.episode_step_counter += 1  
        
        if self.step_counter % self.decision_interval == 1:
            self.cached_action = int(action.item()) 
            action_diff_penalty = 0
            self.prev_action = action
        else:

            action_diff_penalty = 0
        discrete_to_angle = {0: 0, 1: 10, 2: -10}
        self.actual_angle += discrete_to_angle[self.cached_action]
        obs, reward, done, info = super().step(self.actual_angle)
        modified_obs,result = self.observation(obs)
        obx=35*self.obd

        current_x = self.Target_pos[0] - self.center[0]
        current_y = self.Target_pos[1] - self.center[1]
        current_xx = self.Obstacle_pos[0] - self.center[0]
        current_yy = self.Obstacle_pos[1] - self.center[1]
        if self.current_vx == 0 and self.current_vy == 0:
            currenttheta = math.atan2(current_y, current_x)*180/math.pi
        targettheta = math.atan2(current_y, current_x)*180/math.pi
        self.currenttheta=math.atan2(current_y, current_x)
        self.currentobtheta=math.atan2(current_yy, current_xx)
        currenttheta = math.atan2(self.current_vy, self.current_vx)*180/math.pi
        error = abs(targettheta - currenttheta)
        error = min(error, 360 - error)
        distance = math.sqrt(current_x **2 + current_y** 2)
        obdistance = math.sqrt(current_xx **2 + current_yy** 2)
        self.distance=math.sqrt((current_x/640) **2 + (current_y/640)** 2)*self.noice
        self.obdistance=math.sqrt((current_xx/640) **2 + (current_yy/640)** 2)/self.obd
        #modified_reward = -distance/400  
        modified_reward=0
        if self.episode_step_counter >=20:
            if self.center[0]<10 or self.center[0]>630 or self.center[1]<0 or self.center[1]>470:
                modified_reward -=600
                done = True
            '''if self.cached_action == 0:
                modified_reward += 0.05
            else:
                modified_reward -= 0.05'''
            if abs(error)<45:
                modified_reward += 0.2
            elif abs(error)<90:
                modified_reward += 0.1
            elif abs(error)<135:
                modified_reward -= 0.1
            else:
                modified_reward -= 0.2
            
            if obdistance<obx+20:
                modified_reward -= 2/(obdistance-obx+0.05)
            if obdistance<obx:
                modified_reward -=300
                done = True
            if distance < 80 and self.sign==0:
                modified_reward +=150
                self.sign+=1
            if distance < 10:
                modified_reward +=600 
                random_integer = np.random.randint(0, 360)
                self.Target_pos = (self.Target_pos[0]+30*np.cos(random_integer/180*np.pi), self.Target_pos[1]+30*np.sin(random_integer/180*np.pi))
                done = True
            if self.episode_step_counter >= self.max_steps_per_episode/3:
                modified_reward -= 0.3
            if self.episode_step_counter >= self.max_steps_per_episode:
                modified_reward -= 200
                done = True

        if done:
            if distance < 3:
                info['termination_reason'] = 'target_reached'
            elif obdistance < obx:
                info['termination_reason'] = 'hit_obstacle'
            elif self.center[0]<10 or self.center[0]>630 or self.center[1]<0 or self.center[1]>470:
                info['termination_reason'] = 'out_of_bounds'
            else:
                info['termination_reason'] = 'max_steps_reached'
        info['action_diff_penalty'] = action_diff_penalty
        return modified_obs, modified_reward, done, info,result

    def observation(self, obs):
        result, mask, center = find_particles(obs, self.Target_pos,self.Obstacle_pos)
        if result is not None:
            cv2.imshow('Unity Processed', result)
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):  
                cv2.destroyAllWindows()
        if center is not None:
            self.center = center
            self.current_vx, self.current_vy = self.velocity_tracker.update(center)
            
        combined_obs = np.array([
            self.distance,   
            math.sin(self.currenttheta),
            math.cos(self.currenttheta),
            math.sin(self.actual_angle*math.pi/180),
            math.cos(self.actual_angle*math.pi/180),
            self.obdistance,
            math.sin(self.currentobtheta),
            math.cos(self.currentobtheta)
        ], dtype=np.float32)
        
        return combined_obs,result

    def _modify_action(self, action):
        modified_action = action*180
        return modified_action



env_name = "RL_nav_env/Swarm.exe"
channel = EngineConfigurationChannel()
unity_env = UnityEnvironment(env_name, side_channels=[channel])
channel.set_configuration_parameters(time_scale=1.0)

base_env = UnityToGymWrapper(unity_env) 
env =CustomGymWrapper(
    base_env, 
    decision_interval=5, 
    max_steps_per_episode=40480,
    action_penalty_threshold=0.2,  
    action_penalty_weight=0.5  
)

model = PPO.load("ppo_nav_final.zip", env)
model.set_env(env)
model.learn(total_timesteps=0)
try:
    i=0
    obs = env.reset()
    #video_out = cv2.VideoWriter('RL_navigation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info,result = env.step(action)
        #video_out.write(result)
        if dones:
            #cv2.imwrite(f'Episode_End.png', result)
            env.reset()
            #video_out.release()
            break
        env.render()
except Exception as e:
    print(f"Error occurred: {e}")

finally:
    try:
        env.close()
    except:
        pass
    
    try:
        unity_env.close()
    except:
        pass
    
    print("Program terminated")