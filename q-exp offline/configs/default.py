DEFAULT_ENV = {
    "HalfCheetah": {
        " --discrete_control ": 0,
        " --state_dim ": 17,
        " --action_dim ": 6,
        " --action_min ": -1,
        " --action_max ": 1,
    },
    "Hopper": {
        " --discrete_control ": 0,
        " --state_dim ": 11,
        " --action_dim ": 3,
        " --action_min ": -1,
        " --action_max ": 1,
    },
    "Walker2d": {
        " --discrete_control ": 0,
        " --state_dim ": 17,
        " --action_dim ": 6,
        " --action_min ": -1,
        " --action_max ": 1,
    },
}

DEFAULT_AGENT = {
    "HalfCheetah-expert": {
        "IQL": {
            " --expectile ": [0.7],
            " --tau ": [1./3.],
        },
        "InAC": {
            " --tau ": [0.01],
        },
        "TAWAC":{
            " --tau ": [1.0],
        },
        "AWAC":{
            " --tau ": [1.0],
        },
        "TD3BC":{
            " --tau ": [2.5],
        },
    },
    "HalfCheetah-medexp": {
        "IQL": {
            " --expectile ": [0.7],
            " --tau ": [1./3.],
        },
        "InAC": {
            " --tau ": [0.1],
        },
        "TAWAC":{
            " --tau ": [1.0],
        },
        "AWAC":{
            " --tau ": [1.0],
        },
        "TD3BC":{
            " --tau ": [2.5],
        },
    },
    "HalfCheetah-medium": {
        "IQL": {
            " --expectile ": [0.7],
            " --tau ": [1./3.],
        },
        "InAC": {
            " --tau ": [0.33],
        },
        "TAWAC":{
            " --tau ": [1.0, 0.5, 0.01],
        },
        "AWAC":{
            " --tau ": [0.5],
        },
        "TD3BC":{
            " --tau ": [2.5],
        },
    },
    "HalfCheetah-medrep": {
        "IQL": {
            " --expectile ": [0.7],
            " --tau ": [1./3.],
        },
        "InAC": {
            " --tau ": [0.5],
        },
        "TAWAC":{
            " --tau ": [0.01],
        },
        "AWAC":{
            " --tau ": [1.0],
        },
        "TD3BC":{
            " --tau ": [2.5],
        },
    },

    "Hopper-expert": {
        "IQL": {
            " --expectile ": [0.7],
            " --tau ": [1./3.],
        },
        "InAC": {
            " --tau ": [0.01],
        },
        "TAWAC":{
            " --tau ": [1.0],
        },
        "AWAC":{
            " --tau ": [1.0],
        },
        "TD3BC":{
            " --tau ": [2.5],
        },
    },
    "Hopper-medexp": {
        "IQL": {
            " --expectile ": [0.7],
            " --tau ": [1./3.],
        },
        "InAC": {
            " --tau ": [0.01],
        },
        "TAWAC":{
            " --tau ": [0.5],
        },
        "AWAC":{
            " --tau ": [1.0],
        },
        "TD3BC":{
            " --tau ": [2.5],
        },
    },
    "Hopper-medium": {
        "IQL": {
            " --expectile ": [0.7],
            " --tau ": [1./3.],
        },
        "InAC": {
            " --tau ": [0.1],
        },
        "TAWAC":{
            " --tau ": [1.0, 0.5, 0.01],
        },
        "AWAC":{
            " --tau ": [0.5],
        },
        "TD3BC":{
            " --tau ": [2.5],
        },
    },
    "Hopper-medrep": {
        "IQL": {
            " --expectile ": [0.7],
            " --tau ": [1./3.],
        },
        "InAC": {
            " --tau ": [0.5],
        },
        "TAWAC":{
            " --tau ": [0.5],
        },
        "AWAC":{
            " --tau ": [0.5],
        },
        "TD3BC":{
            " --tau ": [2.5],
        },
    },

    "Walker2d-expert": {
        "IQL": {
            " --expectile ": [0.7],
            " --tau ": [1./3.],
        },
        "InAC": {
            " --tau ": [0.01],
        },
        "TAWAC":{
            " --tau ": [1.0],
        },
        "AWAC":{
            " --tau ": [1.0],
        },
        "TD3BC":{
            " --tau ": [2.5],
        },
    },
    "Walker2d-medexp": {
        "IQL": {
            " --expectile ": [0.7],
            " --tau ": [1./3.],
        },
        "InAC": {
            " --tau ": [0.1],
        },
        "TAWAC":{
            " --tau ": [0.01],
        },
        "AWAC":{
            " --tau ": [0.1],
        },
        "TD3BC":{
            " --tau ": [2.5],
        },
    },
    "Walker2d-medium": {
        "IQL": {
            " --expectile ": [0.7],
            " --tau ": [1./3.],
        },
        "InAC": {
            " --tau ": [0.33],
        },
        "TAWAC":{
            " --tau ": [1.0, 0.5, 0.01],
        },
        "AWAC":{
            " --tau ": [0.1],
        },
        "TD3BC":{
            " --tau ": [2.5],
        },
    },
    "Walker2d-medrep": {
        "IQL": {
            " --expectile ": [0.7],
            " --tau ": [1./3.],
        },
        "InAC": {
            " --tau ": [0.5],
        },
        "TAWAC":{
            " --tau ": [0.5],
        },
        "AWAC":{
            " --tau ": [0.1],
        },
        "TD3BC":{
            " --tau ": [2.5],
        },
    },
}
