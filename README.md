# inverse_kinematic_3d_model

Implement body inverse kinematic with machine learning and display on 3d model with three.js

<img src="https://github.com/qianyuez/inverse_kinematic_3d_model/blob/master/data/ik_model.gif" width="500px">

## Requirements
python3.6
- `numpy`
- `keras (tensorflow backend)`
- `tornado`


## Train
`cd inverse_kinematic_3d_model`

`python train.py`


## Display
```
optional arguments:
  -h, --help   show this help message and exit
  --port PORT  local server port, default 8000
```

`cd inverse_kinematic_3d_model`

`python server.py --port PORT`


## Data 
longbow locomotion pack

This 3d model is created by FireFoxFighters and published on Sketchfab.

To download:

https://sketchfab.com/3d-models/longbow-locomotion-pack-1e6c169185574572891900f239756c42


## Defect
The hand joint angles predicted by the model are imprecise, especially for those target points that beyond the reach of the hand. The movement of the hands also lack coherence.
