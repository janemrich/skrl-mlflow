import gymnasium as gym
import os

try:
    print(f"Gymnasium version: {gym.__version__}")
    env = gym.make_vec("Pendulum-v1", num_envs=2, render_mode="rgb_array")
    print("Vector env created")
    
    try:
        # Try gym.wrappers.RecordVideo
        print("Attempting gym.wrappers.RecordVideo...")
        env = gym.wrappers.RecordVideo(env, video_folder="test_videos", video_length=100)
        print("gym.wrappers.RecordVideo accepted vector env")
    except Exception as e:
        print(f"gym.wrappers.RecordVideo failed: {e}")
        
    env.close()

except Exception as e:
    print(f"An error occurred: {e}")
