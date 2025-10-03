# Goal

The goal of this project is to train an autonomous vehicle to navigate through a series of finish gates within a given time limit.

![](./materials/back_view.png)

# Observation

Two kinds of observation are provided to the agent:
1. The image recorded by the car's camera. Shape is $(\text{num\_envs}, H, W, 3)$
2. The car's state which includes
    * the root link linear and angular velocities in simulation world frame, and 
    * the actions applied on the throttle and steering.
    
    Shape is $(\text{num\_envs}, 8)$.

# Action

The agent needs to perform two kinds of actions:
1. The throttle joint velocity. This is the wheel velocity.
2. The steering angle. It determins the direction of the car.

# Model

A custom model is used because the flexibility of SKRL’s model configuration is relatively limited.

The custom model consists of three components:
* `backbone`: A CNN module that extracts the feature map from the image of the car camera.
* `policy_layer`: A module consisting of two linear layers. It accepts the feature map and the state of the car, and then computes the policy to perform action. The output dimension is $(\text{num\_envs}, 2)$.
* `value_layer`: A module consisting of two linear layers. It accepts the feature map and the state of the car, and then predict the value of the current state.  The output dimension is $(\text{num\_envs}, 1)$.

# How to Train?

Go to the folder `finish_gate_drive` and run the following command:
```
python skrl_train.py \
    --task Finish-Gate-Drive-Direct \
    --num_envs 64 \
    --headless \
    --video --video_length 100 --video_interval 1000
```