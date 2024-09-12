# Smuggle-and-seek

This repository contains the code of the smuggle-and-seek game, which is part of my Master Thesis: "Tackling Drug Criminality in Seaports: a Theory of Mind Approach". This repository includes code for running the model in a GUI (_run.py_), and code for running the experiments (_batch_run.py_). 

## How to run

1. Download the code from this repository, or clone the repository.
2. Install all dependencies : `pip install -r requirements.txt`.

#### Run the model in the GUI:
3. Adjust the parameter _l_ at the top of the file _server.py_ to change the game environment.
4. Run the file _run.py_ : `python run.py`.
5. Set the parameters to your preferences in the GUI, 'Reset' the model and 'Start' the model.

#### Run the experiments:
3. Adjust the parameters at the top of the file _batch_run.py_ to the parameters that you want to test.
4. Run the file _batch_run.py_: `python batch_run.py`.
