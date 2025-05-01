#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/PoseStamped.h>

int main(int argc, char** argv) {
    ros::init(argc, argv, "transform_listener");
    ros::NodeHandle nh;

    // Create publisher for pose
    ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/pose_tracked", 10);

    // Create TF listener
    tf::TransformListener listener;

    ros::Rate rate(10.0); // 10 Hz

    while (ros::ok()) {
        tf::StampedTransform transform;
        try {
            // Look up transform from map to base_link
            listener.waitForTransform("/map", "/base_link", ros::Time(0), ros::Duration(1.0));
            listener.lookupTransform("/map", "/base_link", ros::Time(0), transform);

            // Create and fill PoseStamped message
            geometry_msgs::PoseStamped pose;
            pose.header.stamp = ros::Time::now();
            pose.header.frame_id = "map";

            // Set position
            pose.pose.position.x = transform.getOrigin().x();
            pose.pose.position.y = transform.getOrigin().y();
            pose.pose.position.z = transform.getOrigin().z();

            // Set orientation
            pose.pose.orientation.x = transform.getRotation().x();
            pose.pose.orientation.y = transform.getRotation().y();
            pose.pose.orientation.z = transform.getRotation().z();
            pose.pose.orientation.w = transform.getRotation().w();

            // Publish the pose
            pose_pub.publish(pose);
        }
        catch (tf::TransformException &ex) {
            ROS_WARN("%s", ex.what());
            ros::Duration(1.0).sleep();
            continue;
        }

        rate.sleep();
    }

    return 0;
}
