# Goal

The goal of this project is to train an autonomous vehicle to navigate through a series of finish gates within a given time limit.

![](./materials/back_view.png)

# Observation

The agent receives two types of observations:
1. The image captured by the car’s camera, with shape `(num_envs, H, W, 3)`.
2. The car’s state with shape `(num_envs, 8)`. It includes:
    * the root link’s linear and angular velocities in the simulation world frame, and
    * the actions applied to the throttle and steering.

# Action

The agent needs to perform two types of actions:
1. Throttle joint velocity – controls the wheel velocity.
2. Steering angle – determines the direction of the car.

# Model

A custom model is used because SKRL’s model configuration offers limited flexibility.
The custom model consists of three components:
* `backbone`: A CNN module that extracts feature maps from the car camera images.
* `policy_layer`: A module with two linear layers. It takes as input both the feature map and the car’s state, then computes the policy for action selection. The output shape is `(num_envs, 2)`.
* `value_layer`: A module with two linear layers. It takes as input both the feature map and the car’s state, then predicts the value of the current state. The output shape is `(num_envs, 1)`.

# How to Train?

Navigate to the finish_gate_drive folder and run the following command:

```
python skrl_train.py \
    --task Finish-Gate-Drive-Direct \
    --num_envs 64 \
    --headless \
    --video --video_length 100 --video_interval 1000
```