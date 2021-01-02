<!-- # lyft-motion-prediction-for-autonomous-vehicle -->

### Description


### Table of Contents

- [Environment Setup](#environment-setup)
- [Training](#training)

### Environment Setup
- Python 3.* is installed
- Install dependencies
```
pip install -r requirements.txt
```

### Training
- After installing all the requirements, run the following command for trainig
```
python train.py -env cartpole
```
- `-env`: (REQUIRED) RL Environment, availables are `[cartpole, aimsun]`. `aimsun` requires simulation software and data collecting scripts


