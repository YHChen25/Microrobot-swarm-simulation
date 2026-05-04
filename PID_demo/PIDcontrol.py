import numpy as np
import math
from stable_baselines3 import PPO
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import cv2
from collections import deque
import matplotlib.pyplot as plt
import time

color1 = "#95d0fc"
color2 = "#e50000"
color3 = "#3620f8"
color4 = "#e50000"

def hex_to_bgr(hex_color):
    # Remove '#' character
    hex_color = hex_color.lstrip('#')
    
    # Split into R, G, B channels and convert to integers
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    # Return BGR format for OpenCV
    return (b, g, r)

Color1 = hex_to_bgr(color1)
Color2 = hex_to_bgr(color2)
Color3 = hex_to_bgr(color3)
Color4 = hex_to_bgr(color4)

class TrajectoryTracker:
    def __init__(self, max_points=100000):
        self.positions = []
        self.max_points = max_points
    
    def add_point(self, point):
        self.positions.append(point)
        if len(self.positions) > self.max_points:
            self.positions.pop(0)
    
    def draw_trajectory(self, image):
        if len(self.positions) < 2:
            return image
        
        # Convert trajectory points to numpy array for drawing
        points = np.array(self.positions, dtype=np.int32)
        data = np.column_stack((
            points[:, 0],  # X coordinate
            points[:, 1]   # Y coordinate
        ))
        np.savetxt(
            f'Trajectory.csv',
            data,
            delimiter=',',
            header='x,y',
            comments='',
            fmt='%.4f'
        )
        cv2.polylines(image, [points], False, Color2, 2)  # Red trajectory line
        return image
    
    def reset(self):
        self.positions.clear()

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
        
        # Add to history buffer
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


# Data logging and visualization class
class DataLogger:
    def __init__(self):
        self.time_stamps = []
        self.actions = []
        self.errors = []
        self.errors_d = []
        self.start_time = time.time()
    
    def log(self, action, error, error_d):
        current_time = time.time() - self.start_time
        self.time_stamps.append(current_time)
        self.actions.append(action)
        self.errors.append(error)
        self.errors_d.append(error_d)
    
    def plot_results(self):
        # Create a figure with 4 subplots
        fig, (ax2, ax3, ax1, ax4) = plt.subplots(4, 1, figsize=(10, 12))
        
        # Plot control action
        ax2.plot(self.time_stamps, self.actions, 'g-', label='Control Action')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Action')
        ax2.legend()
        ax2.grid(True)
        ax2.set_title('Control Action Over Time')
        
        # Plot angle error
        ax3.plot(self.time_stamps, self.errors, 'r-', label='Error')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Error')
        ax3.legend()
        ax3.grid(True)
        ax3.set_title('Error Over Time')
        
        # Plot derivative error
        ax1.plot(self.time_stamps, self.errors_d, 'b-', label='Error_d')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Error_d')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title('Error Over Time')
        
        # Plot error distribution histogram
        ax4.hist(self.errors, bins=50, color='red', alpha=0.7)
        ax4.set_xlabel('Error')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Error Distribution')
        
        plt.tight_layout()
        plt.savefig('pid_control_results.png')
        plt.show()
        
        # Save data to CSV file
        np.savetxt('pid_data.csv', 
                  np.column_stack((self.time_stamps, self.actions, self.errors)),
                  delimiter=',',
                  header='time,action,error',
                  comments='')


class DampedPIDController:
    def __init__(self, kp, ki, kd, damping_coefficient):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.damping_coefficient = damping_coefficient
        self.prev_error = 0
        self.integral = 0
        self.prev_output = 0
        
    def update(self, error, dt=0.1):
        # Standard PID calculation
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        pid_output = (self.kp * error + 
                     self.ki * self.integral + 
                     self.kd * derivative)
        
        # Add damping to reduce oscillation
        damped_output = pid_output - self.damping_coefficient * (pid_output - self.prev_output)
        
        # Update state variables
        self.prev_error = error
        self.prev_output = damped_output
        
        return damped_output
    
    def reset(self):
        self.prev_error = 0
        self.integral = 0
        self.prev_output = 0

def find_particles(image_data, j, trajectory_tracker):
    try:
        
        if image_data.dtype == np.float32 or image_data.dtype == np.float64:
            image_data = (image_data * 255).astype(np.uint8)
        
        # Convert to BGR format (OpenCV standard)
        if len(image_data.shape) == 3 and image_data.shape[2] == 3:
            img = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
        else:
            print("Unexpected image format")
            return None, None, None

        # Process black particle swarm
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)

        # Find contours of particles
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("No contours found")
            return img, None, None
            
        # Get the largest contour (main particle swarm)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create binary mask
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, (255), -1)
        
        # Calculate centroid using image moments
        M = cv2.moments(mask)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            center = (center_x, center_y)
        else:
            # Fallback: use bounding box center
            x, y, w, h = cv2.boundingRect(largest_contour)
            center = (x + w//2, y + h//2)
        
        # Draw visualization
        result = img.copy()
        
        # Draw circular target path (300 points)
        num_points = 8
        r=100
        x = 350
        y = 225
        points=np.array([[x, y-r],
                        [x+r, y-r],
                        [x+r, y],
                        [x, y],
                        [x-r, y],
                        [x-r, y+r],
                        [x, y+r],
                        [x+r, y+r]])

        points_x = points[:, 0]
        points_y = points[:, 1]

        for i in range(num_points):
            cv2.circle(result, (int(points_x[i]), int(points_y[i])), 2, Color3, 2)
        cv2.circle(result, (int(points_x[j]), int(points_y[j])), 2, Color1, 2)
        
        # Draw particle center
        cv2.circle(result, center, 3, Color4, -1)
        trajectory_tracker.add_point(center)
        
        # Draw movement trajectory
        result = trajectory_tracker.draw_trajectory(result)
        return result, mask, center
        
    except Exception as e:
        print(f"Error in image processing: {e}")
        return None, None, None
        
try:
    # Initialize Unity environment
    env_name = "RL_slope_env/Swarm.exe"
    channel = EngineConfigurationChannel()
    unity_env = UnityEnvironment(env_name, side_channels=[channel])
    channel.set_configuration_parameters(time_scale=1)
    env = UnityToGymWrapper(unity_env)

    # Initialize damped PID controllers
    pid_theta = DampedPIDController(
        kp=0.04, 
        ki=0.001, 
        kd=0.001,
        damping_coefficient=0
    )
    pid_x = DampedPIDController(
        kp=0.05, 
        ki=0.001, 
        kd=0.000,
        damping_coefficient=0
    )
    pid_y = DampedPIDController(
        kp=0.05, 
        ki=0.001, 
        kd=0.000,
        damping_coefficient=0
    )

    data_logger = DataLogger()
    velocity_tracker = VelocityTracker(window_size=50)

    # Generate circular target points
    num_points = 8
    r=100
    x = 350
    y = 225
    points=np.array([[x, y-r],
                    [x+r, y-r],
                    [x+r, y],
                    [x, y],
                    [x-r, y],
                    [x-r, y+r],
                    [x, y+r],
                    [x+r, y+r]])

    points_x = points[:, 0]
    points_y = points[:, 1]

    trajectory_tracker = TrajectoryTracker(max_points=50000)

    # Main loop initialization
    i = 0
    obs = env.reset()
    result, mask, center = find_particles(obs, i, trajectory_tracker)
    video_out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (result.shape[1], result.shape[0]))
    
    Target_pos = (points_x[i], points_y[i])
    current_x = Target_pos[0] - center[0]
    current_y = Target_pos[1] - center[1]
    action_theta = math.atan2(current_y, current_x) * 180 / math.pi
    error = 0
    step_counter = 0
    start_time = time.time()

    # Load trained PPO model


    while True:
        try:
            current_time = time.time() - start_time
            
            # Execute control command
            action = np.array(action_theta)
            obs, rewards, dones, info = env.step(action)

            # Update target position and detect particles
            Target_pos = (points_x[i], points_y[i])
            result, mask, center = find_particles(obs, i, trajectory_tracker)
            cv2.imwrite('final_result.png', result)

            if result is not None:
                cv2.imshow('Unity Processed', result)
                video_out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 
            
            if center is not None:
                # Update velocity estimation
                current_vx, current_vy = velocity_tracker.update(center)
                
                current_x = Target_pos[0] - center[0]
                current_y = Target_pos[1] - center[1]

                # Switch to next target when reached
                if math.sqrt(current_x**2 + current_y**2) <= 1:
                    i += 1
                    Target_pos = (int(points_x[i]), int(points_y[i]))
                    current_x = Target_pos[0] - center[0]
                    current_y = Target_pos[1] - center[1]
                    action_theta = math.atan2(current_y, current_x) * 180 / math.pi

                    if i >= num_points:
                        cv2.imwrite('final_result.png', result)
                        break

                if current_vx == 0 and current_vy == 0:
                    currenttheta = math.atan2(current_y, current_x) * 180 / math.pi

                # Update PID at 10Hz
                if int(current_time / 0.1) >= step_counter:     
                    targettheta = math.atan2(current_y, current_x) * 180 / math.pi
                    currenttheta = math.atan2(current_vy, current_vx) * 180 / math.pi
                    error = targettheta - currenttheta

                    # Normalize angle error to [-180, 180]
                    if error > 180:
                        error -= 360
                    elif error < -180:
                        error += 360

                    # Update control angle
                    action_theta += pid_theta.update(error)
          
                    step_counter += 1
                    
                    # Log control data
                    data_logger.log(action_theta, currenttheta, error)    
                
            if dones:
                cv2.imwrite('final_result.png', result)
                obs = env.reset()
                pid_theta.reset()
                video_out.release()
                step_counter = 0
                trajectory_tracker.reset()
            
            env.render()
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            break

except Exception as e:
    print(f"Error in environment setup: {e}")

finally:
    # Plot results after execution
    #if 'data_logger' in locals():
        #data_logger.plot_results()

    # Clean up resources
    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    try:
        env.close()
    except:
        pass
    
    try:
        unity_env.close()
    except:
        pass
        
    print("Program terminated")