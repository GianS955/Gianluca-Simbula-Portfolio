import rclpy
from rclpy.node import Node
from fetchpush_msgs.msg import Observation, Action, DesiredGoal
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import imageio
import os


class EnvNode(Node):
    def __init__(self):
        super().__init__('env_node')

        self.obs_publisher_ = self.create_publisher(Observation, 'observation', 10)
        self.action_subscriber = self.create_subscription(Action, 'action', self.action_callback, 10)
        self.goal_subscriber = self.create_subscription(DesiredGoal, 'desired_goal', self.goal_callback, 10)

        self.declare_parameter('output_path', '/tmp')
        self.output_path_ = self.get_parameter('output_path').get_parameter_value().string_value

        gym.register_envs(gymnasium_robotics)
        self.env_ = gym.make('FetchPush-v4', render_mode='rgb_array')        
        self.frames_=[]

    def goal_callback(self, goal):
        
        self.obs, _ = self.env_.reset()
        self.env_.unwrapped.goal = np.array(goal.desired_goal)
        self.obs['desired_goal'] = np.array(goal.desired_goal)
        self.uuid = goal.goal_id

        self.publish_observation()

    def publish_observation(self, observation = None):
        
        obs = Observation() 
        if observation is None: 
            observation = self.obs
            observation['terminal'] = False
            observation['truncated'] = False
        
        obs.observation = observation['observation'].tolist()
        obs.desired_goal = observation['desired_goal'].tolist()
        obs.achieved_goal = observation['achieved_goal'].tolist()
        obs.terminal = observation['terminal']
        obs.truncated = observation['truncated']
        self.obs_publisher_.publish(obs)

    def action_callback(self, action):
        obs, _, _, truncated, info = self.env_.step(np.array(action.action, dtype=np.float64))
        obs['terminal'] = bool(info['is_success'])
        obs['truncated'] = truncated

        self.frames_.append(self.env_.render())
        if bool(info['is_success']) or truncated:
            imageio.mimsave(os.path.join(self.output_path_,f'{self.uuid}.gif'), self.frames_, fps=15, loop= 0)
            self.frames_ = []
        self.publish_observation(obs)

def main(args = None):
    rclpy.init(args = args)
    node = EnvNode()
    rclpy.spin(node)
    rclpy.shutdown()