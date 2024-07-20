from clearml import Task 
from agent import AgentSym
import argparse
from gymnasium.envs.registration import register
import torch

def register_envs(envs, name):
    env_ids = []
    
    for i, env_setting in enumerate(envs):
        env_id = f"ShapesGridCustom_{i}-{name}"
        register(
            id=env_id,
            entry_point='grid_shapes_env.shapes_grid_base:ShapesGridBase',
            kwargs=env_setting,
            max_episode_steps=training_settings["max_steps"],
        )
        env_ids.append(env_id)
    
    return env_ids

def experiment(name, env_settings, model_settings, training_settings, logger=True):    
    if logger:
        task = Task.init(project_name='Thesis', task_name=name)
        task.connect(model_settings)
        task.connect(training_settings)
        task.connect({"training_settings": env_settings})
        logger = task.get_logger()

    agent = AgentSym(name, model_settings, logger)
    print(f"----> Agent {agent.name} created")

    env_ids = register_envs(env_settings, name)
    agent.train_on_envs(env_ids, training_settings)

def get_model_settings(arch=False, module=False, ltn=False, action=False, action_guiding_training=False):
    return {
        "explore_steps": 25000, "update_steps": 1000,
        "init_epsilon": 0.95, "final_epsilon": 0.05, 
        "memory_capacity": 1000, "batch_size": 16, 
        "gamma": 0.99, "learning_rate": 0.0001,
        "max_gradient_norm": 1,
        "reset_epsilon": False,
        "max_shapes": 5, 
        "use_shape_recognizer": arch, "use_reward": arch, 
        "module_training": module,
        "use_ltn": ltn,
        "action_guiding": action,
        "action_guiding_training": action_guiding_training
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receive the index of the model_settings to be used")
    parser.add_argument("name", type=str, help="Name of Run")
    parser.add_argument("idx", type=int, help="Index of model settings to use")
    args = parser.parse_args()
    idx = args.idx - 1
    
    env_settings = [{'background_color':'white','entities_color':'black'}]
    
    model_settings = [
        get_model_settings(),
        get_model_settings(arch=True, module=True),
        get_model_settings(arch=True, module=True, ltn=True),
        get_model_settings(arch=True, module=True, action=True),
        get_model_settings(arch=True, module=True, ltn=True, action=True),
    ]

    names = [
        "Baseline",
        "Modules",
        "Modules-Axiom",
        "Modules-Action",
        "Modules-Axiom-Action",
    ]
    name = f"{args.name}_{names[idx]}"

    assert idx < len(model_settings)

    training_settings = {
        "epochs": 250,
        "episodes_per_epoch": 50,
        "max_steps": 50,
        "no_reward_cost": -.1 # motivate agent to act faster
    }

    experiment(name, env_settings, model_settings[idx], training_settings)
