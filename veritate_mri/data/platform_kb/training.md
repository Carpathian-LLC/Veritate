# How training works

Training in Veritate is driven from the dashboard's Training tab. You pick a trainer (which fixes the model size), pick a corpus, set any training options you care about, and start the run. The trainer then runs a PyTorch training loop in the background while the dashboard shows you live progress.

You start a run by choosing an action. The two main training actions are "scratch" (build a brand new model from random initialization or from a named base) and "continue" (resume training an existing model from its latest checkpoint). Other actions in the same flow include RAG (continue-train a model to answer from supplied context) and synth (generate synthetic training data with a teacher model).

As a run trains, the dashboard charts loss, learning rate, throughput (tokens per second), and gradient norm. These all come from one training log file that the trainer appends a row to at every logging step. A separate live feed streams per-step internal state of the model so you can watch the model's neurons and predictions evolve.

A plateau detector summarizes the health of the loss curve into one of six states: improving, plateau, regressing, slowing, bouncing, or warming. Very short runs (under roughly fifty steps) usually sit in warming the whole time because the detector needs some history. This verdict is a quick read on whether the run is still making progress.

Checkpoints are saved at a configurable interval. Every checkpoint is written into a per-model folder under models/<name>/. Inside that folder, checkpoints/ holds the saved PyTorch files (step_<N>.pt), train.csv holds the training log, config.json holds the model shape and training settings, and hooks/ holds a suite of analysis files captured at each checkpoint. The whole models/ area is local to your machine and is not committed to the repository.

There is also an Auto tune feature: a measured benchmark that runs the selected trainer on throwaway weights, finds the largest batch size that fits in memory and a throughput sweet spot, and writes those values back into the form so you do not have to guess them.
