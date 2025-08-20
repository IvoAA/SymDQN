# SymDQN Setup Guide

## Prerequisites

This project requires Python 3.8+ and several dependencies.

## Installation Steps

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **ClearML Setup** (Optional - for experiment tracking):
   If you want to use experiment tracking, you'll need to configure ClearML:
   ```bash
   clearml-init
   ```
   Follow the prompts to connect to your ClearML server or create a free account.

## Running the Project

The main entry point is `sym_dqn/run_experiment_symdqn.py`.

**Usage:**
```bash
cd sym_dqn
py run_experiment_symdqn.py <experiment_name> <model_index>
```

**Parameters:**
- `experiment_name`: A string name for your experiment run
- `model_index`: Integer 1-5 selecting the model variant:
  1. Baseline (Standard DQN)
  2. Modules (DQN + Shape Recognition + Reward Prediction)  
  3. Modules-Axiom (Above + LTN axioms)
  4. Modules-Action (Modules + action guidance)
  5. Modules-Axiom-Action (Full system)

**Example:**
```bash
py run_experiment_symdqn.py "test_run" 1
```

## Project Structure

- `sym_dqn/models/checkpoints/` - Model save directory (created automatically)
- `sym_dqn/grid_shapes_env/` - Custom grid environment
- `sym_dqn/models/SymDQN.py` - Main model architecture
- `sym_dqn/agent.py` - Training agent with symbolic reasoning

## Notes

- The project will create a 5x5 grid environment with shapes (cross, circle, square)
- Training runs for 250 epochs with 50 episodes each
- Models are automatically saved to the checkpoints directory
- ClearML logging can be disabled by setting `logger=False` in the experiment function
