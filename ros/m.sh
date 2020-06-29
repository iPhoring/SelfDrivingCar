rm -rf build
export ROS_IP=$( hostname -I | awk '{print $1}')
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch