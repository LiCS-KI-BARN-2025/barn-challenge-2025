#!/usr/bin/env python3

import rospy
import rospkg
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.srv import GetPlan, GetPlanRequest
from collections import deque

import numpy as np
import torch as th
import torch.nn as nn

from cnn import CNN1D, StackedCNN1D
from actor import Actor

MAX_LIN_SPEED = 1.5
MAX_ANG_SPEED = 1.5
SPEED_SCALER = 1.5
USE_PLANNER = False
NUM_OBS_STACK = 1
MODEL_PATH = 'sac_cnn_5'

rospack = rospkg.RosPack()

def rotate_vector(vector: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 2D vector by a given angle."""
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
    return rotation_matrix @ vector

def quaternion_to_euler_angle(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0,+1.0,t2)
    #t2 = +1.0 if t2 > +1.0 else t2

    t2 = np.where(t2<-1.0, -1.0, t2)
    #t2 = -1.0 if t2 < -1.0 else t2
    Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.arctan2(t3, t4)

    return X, Y, Z 


class Planner:
    def __init__(
            self,
            slam=True,
            global_planner=USE_PLANNER,
            num_obs_stack=NUM_OBS_STACK,
            goal_x=0.0,
            goal_y=0.0,
            laser_topic='/front/scan',
            pose_topic='/pose_tracked',
            odom_topic='/odometry/filtered',
        ):

        self.ready = not global_planner

        self.laser_data = None
        self.odom_data = None
        self.pose_data = None
        
        self.goal_x = goal_x
        self.goal_y = goal_y

        # self.local_goal_x = goal_x
        # self.local_goal_y = goal_y
        self.last_pos = None
        self.last_update = None

        self.num_obs_stack = num_obs_stack
        self.path = None

        if self.num_obs_stack > 1:
            self.stacked_observations = deque(maxlen=self.num_obs_stack)
            for _ in range(self.num_obs_stack):
                self.stacked_observations.append(np.zeros(5 + 64))

        if self.num_obs_stack > 1:
            feature_extractor = StackedCNN1D(
                stack_size=self.num_obs_stack,
                single_obs_dim=5 + 64,
                inner_cnn_features_dim=16,
                skip_front=5
            )
        else:
            feature_extractor = CNN1D(features_dim=16, skip_front=5)

        load_success = False
        for dim in [8, 16]:
            self.actor = Actor(
                action_dim=2,
                net_arch=[64, 64],
                features_extractor=feature_extractor,
                features_dim=self.num_obs_stack * (dim + 5),
                activation_fn=nn.ReLU,
            )
        
            model_path = rospack.get_path('cnn_rl') + f'/models/{MODEL_PATH}/policy.pth'
            try:
                self.actor.load(model_path)
            except Exception as e:
                rospy.logerr(f"Error loading model: {e}")
                continue
            load_success = True
            break

        if not load_success:
            rospy.logerr("Failed to load model. Exiting.")
            exit(1)

        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_callback)
        self.slam = slam
        if self.slam:
            # Removed old subscriber for pose_topic and added tf2 listener instead.
            self.pose_sub = rospy.Subscriber(pose_topic, PoseStamped, self.pose_callback)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback) 

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        if global_planner:
            self.plan_timer = rospy.Timer(rospy.Duration(0.5), self.call_get_plan)

    def laser_callback(self, data):
        # Make sure the length of ranges is 720
        assert len(data.ranges) == 720, "Laser scan data should have 720 ranges."
        self.laser_data = data

    def get_observation(self, num_ranges=64):
        if self.laser_data is None:
            rospy.logwarn("Laser data not available yet.")
            return None
        
        if self.odom_data is None:
            rospy.logwarn("Odometry data not available yet.")
            return None
        
        observations = np.zeros(5 + num_ranges)

        # Get the robot_pos and yaw
        robot_pos = np.array([self.pose_data.pose.position.x, self.pose_data.pose.position.y])
        _, _, yaw = quaternion_to_euler_angle(
        self.pose_data.pose.orientation.w,
            self.pose_data.pose.orientation.x,
            self.pose_data.pose.orientation.y,
            self.pose_data.pose.orientation.z
        )
    
        self.local_goal_x = self.goal_x
        self.local_goal_y = self.goal_y

        if self.path is not None and len(self.path) > 0:
            last_pose_x = robot_pos[0]
            last_pose_y = robot_pos[1]
            cumm_dist = 0.0
            for i, pose_stamped in enumerate(self.path):
                dx = pose_stamped.pose.position.x - last_pose_x
                dy = pose_stamped.pose.position.y - last_pose_y
                cumm_dist += np.sqrt(dx * dx + dy * dy)
                if cumm_dist > 1.5:
                    self.local_goal_x = pose_stamped.pose.position.x
                    self.local_goal_y = pose_stamped.pose.position.y
                    break

                last_pose_x = pose_stamped.pose.position.x
                last_pose_y = pose_stamped.pose.position.y
            
            # prune the path
            self.path = self.path[i:]

        target = np.array([self.local_goal_x, self.local_goal_y])

        diff = target - robot_pos
        dist = np.linalg.norm(diff)
        
        observations[:2] = rotate_vector(diff, -yaw) / dist
        observations[2] = dist

        # Observe linear and angular velocity
        observations[3] = self.odom_data.twist.twist.linear.x
        observations[4] = self.odom_data.twist.twist.angular.z

        # print("Observations:", observations[:5])
        range_indices = np.round(np.linspace(0, 719, num_ranges)).astype(int)

        for i, range_idx in enumerate(range_indices):
            # observations[5 + i] = min(self.laser_data.ranges[np.floor(range_idx).astype(int)], self.laser_data.ranges[np.ceil(range_idx).astype(int)])
            observations[5 + i] = self.laser_data.ranges[range_idx]

        observations[5:] = np.clip(observations[5:], 0.0, 10.0)
        # Reverse the order of the laser ranges

        return observations
    
    def pose_callback(self, data):
        self.pose_data = data
                    
    def odometry_callback(self, data):
        self.odom_data = data
        
    def unscale_action(self, scaled_action: np.ndarray, low, high) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def plan(self):
        if not self.ready:
            rospy.logwarn("Planner is not ready yet.")
            return
        
        if self.laser_data is None:
            rospy.logwarn("Laser data not available yet.")
            return
        
        if self.odom_data is None:
            rospy.logwarn("Odometry data not available yet.")
            return
        if self.pose_data is None:
            rospy.logwarn("Pose data not available yet.")
            return
        
        observations = self.get_observation()
        if self.num_obs_stack > 1:
            self.stacked_observations.append(observations)
            observations = np.concatenate(list(self.stacked_observations), axis=0)

        if observations is None:
            rospy.logwarn("No observations available.")
            return

        with th.no_grad():
            observations = th.tensor(observations, dtype=th.float32).unsqueeze(0)

            action = self.actor(observations)
            action = action.squeeze().cpu().numpy() * SPEED_SCALER
            action[0] = np.clip(action[0], -MAX_LIN_SPEED, MAX_LIN_SPEED)
            action[1] = np.clip(action[1], -MAX_ANG_SPEED, MAX_ANG_SPEED)
                        
        cmd = Twist()
        cmd.linear.x = action[0]
        cmd.angular.z = action[1]
        self.cmd_pub.publish(cmd)

    def call_get_plan(self, event):
        # Obtain current robot pose from tf2
        if self.pose_data is None:
            rospy.logwarn("Pose data not available yet.")
            return

        start = PoseStamped()
        start.header.stamp = rospy.Time.now()
        start.header.frame_id = "map"
        start.pose = self.pose_data.pose

        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"
        goal.pose.position.x = self.goal_x
        goal.pose.position.y = self.goal_y
        goal.pose.position.z = 0.0
        goal.pose.orientation = Quaternion(0, 0, 0, 1)

        req = GetPlanRequest()
        req.start = start
        req.goal = goal
        req.tolerance = 0.5  # Tolerance value

        try:
            rospy.wait_for_service("/move_base/make_plan", timeout=1.0)
            get_plan = rospy.ServiceProxy("/move_base/make_plan", GetPlan)
            resp = get_plan(req)

            self.local_goal_x = self.goal_x
            self.local_goal_y = self.goal_y

            # If cannot find a plan, the service will return an empty plan
            if len(resp.plan.poses) > 0:
                self.path = resp.plan.poses
            
            self.ready = True
        except Exception as e:
            rospy.logwarn("GetPlan service call failed: " + str(e))
        
if __name__ == "__main__":
    rospy.init_node('planner_node')
    
    freq = rospy.get_param('~freq', 20.0)
    slam = rospy.get_param('~slam', True)
    goal_x = rospy.get_param('~goal_x', 0.0)
    goal_y = rospy.get_param('~goal_y', 0.0)

    # Print all parameters from current node
    node_params = rospy.get_param_names()
    node_params = [param for param in node_params if param.startswith('/planner_node')]
    rospy.loginfo("Current node parameters:")
    for param in node_params:
        param_value = rospy.get_param(param)
        rospy.loginfo(f"{param}: {param_value}")

    planner = Planner(slam=slam, goal_x=goal_x, goal_y=goal_y)

    try:
        while not rospy.is_shutdown():
            planner.plan()
            rospy.sleep(1.0 / freq)
    except rospy.ROSInterruptException:
        pass
