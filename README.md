# f110_gymnasium_ros2_jazzy
Hacky update of f110_gym and f110_gym_ros.

I have combined the following 4 repositories.

https://github.com/f1tenth/f1tenth_gym

https://github.com/f1tenth/f1tenth_gym_ros

https://github.com/CL2-UWaterloo/f1tenth_ws

https://github.com/CL2-UWaterloo/Raceline-Optimization




## wsl2 ubuntu stuff
Install vscode on windows. it has a server that you can call from the ubuntu command line. makes editing files easier. 
Install ubuntu 24.04
in a windows powershell terminal
```
wsl --install -d Ubuntu-24.04 --name <WhateverNameYouWant>
```

## python, pip and breaking system packages
```
sudo apt-get install python3-pip
```
```
python3 -m pip config set global.break-system-packages true
```

```
cd f110_gymnasium_ros2_jazzy/f110_gymnasium
```
```
pip install -e .
```

## ROS2 jazzy stuff
Follow this link and install the optional development tools as well.
https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html

i like to add this to .bashrc so idont have to everytime i open a new terminal
```
source /opt/ros/jazzy/setup.bash
```



## test run with keyboard input
make sure sim runs a controls work with teleop

## extra tools for waypoints and racelines
go over the race line optimization stuff.

