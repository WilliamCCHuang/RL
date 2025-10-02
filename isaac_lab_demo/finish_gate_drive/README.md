# Goal

The goal of this project is to train an autonomous vehicle to navigate through a series of finish gates within a given time limit.

![](./materials/back_view.png)

# Observation



# Model

A custom model is used because the flexibility of SKRL’s model configuration is relatively limited.

The custom model consists of three components:
* `backbone`: A CNN module that extracts the feature map from the image of the car camera.
* `policy_layer`: A module consisting of two linear layers. It accepts the feature map and the state of the car, and then computes the policy to perform action.
* `value_layer`: A module consisting of two linear layers. It accepts the feature map and the state of the car, and then predict the value of the current state.

# How to Train?

Go to the folder `finish_gate_drive` and run the following command:
```
python skrl_train.py \
    --task Finish-Gate-Drive-Direct \
    --num_envs 64 \
    --headless \
    --video --video_length 100 --video_interval 1000
```