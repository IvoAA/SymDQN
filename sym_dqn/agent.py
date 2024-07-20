from utils import get_device, same_shape, extract_patches
from models.SymDQN import SymDQN, ShapeRecognizer, MLP, Memory
import torch.nn.functional as F
from datetime import datetime
from collections import deque
import gymnasium as gym
import numpy as np
import traceback
import random
import torch
import ltn

class AgentSym:
    def __init__(self, name, model_settings, writer=False, load_model=False) -> None:
        self.name = name
        self.device = get_device()
        torch.set_default_device(self.device)

        self.max_shapes = model_settings["max_shapes"]

        self.use_ltn = model_settings["use_ltn"]
        self.use_reward = model_settings["use_reward"]
        self.use_shape_recognizer = model_settings["use_shape_recognizer"]
        self.module_training = model_settings["module_training"]
        self.action_guiding = model_settings["action_guiding"]
        self.action_guiding_training = model_settings["action_guiding_training"]
        
        
        self.onlineQNetwork = SymDQN(max_shapes=self.max_shapes, use_shape_recognizer=self.use_shape_recognizer, use_reward=self.use_reward)
        self.targetQNetwork = SymDQN(max_shapes=self.max_shapes, use_shape_recognizer=self.use_shape_recognizer, use_reward=self.use_reward)
        
        if load_model:
            self.onlineQNetwork.load_state_dict(torch.load(load_model))
        else:
            torch.save(self.onlineQNetwork.state_dict(), f"sym_dqn/models/checkpoints/{self.name}_{0}.pth")
        
        self.targetQNetwork.load_state_dict(self.onlineQNetwork.state_dict())

        lr = model_settings["learning_rate"]

        if self.module_training == False:
            self.params = list(self.onlineQNetwork.parameters())
            self.shape_recognizer = ShapeRecognizer(self.max_shapes)
            self.reward_predictor = MLP(num_classes=self.max_shapes)
        else:
            self.params = list(self.onlineQNetwork.global_conv.parameters()) + list(self.onlineQNetwork.duel.parameters())
            self.shape_recognizer = self.onlineQNetwork.shape_recognizer
            self.reward_predictor = self.onlineQNetwork.reward_predictor

        self.optimizer = torch.optim.Adam(self.params, lr=lr)

        self.shape_recognizer_optimizer = torch.optim.Adam(self.shape_recognizer.parameters(), lr=lr)

        self.sat_agg = ltn.fuzzy_ops.SatAgg()
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.And = ltn.Connective(ltn.fuzzy_ops.AndProd(stable=True))
        self.Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum(stable=True))
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach(stable=True))
        self.Equiv = ltn.Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(stable=True), ltn.fuzzy_ops.ImpliesReichenbach(stable=True)))

        self.p_op = 8
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=self.p_op, stable=True), quantifier="f")
        self.Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=self.p_op, stable=True), quantifier="e")
        self.Eq = ltn.Predicate(func=lambda x, y: torch.exp(-torch.norm(x - y, dim=1))) # predicate measuring similarity

        self.IsShape = ltn.Predicate(self.shape_recognizer)
        self.s1 = ltn.Variable("shape_types1", torch.eye(self.max_shapes))
        self.s2 = ltn.Variable("shape_types2", torch.eye(self.max_shapes))

            # Reward Pred
        self.Agent = lambda s1, count: ltn.Constant(s1.value[torch.argmin(count)])
        self.CountShapes = lambda state_patches, shapes: abs((self.IsShape(state_patches, shapes).value > 0.8).sum(dim=0)-1)
        self.GetShapeIndex = lambda s_p1, agent: torch.argmax(self.IsShape(s_p1, agent).value)
        self.ShapeAtIndex = lambda s_p1, i, s1: ltn.Constant(self.IsShape(ltn.Constant(s_p1.value[i]), s1).value)
        self.IndexFromAction = lambda index, action: (
            index - 5 if action == 0 and index >= 5 else
            index + 1 if action == 1 and (index + 1) % 5 != 0 else
            index + 5 if action == 2 and index < 5 * (5 - 1) else
            index - 1 if action == 3 and index % 5 != 0 else
            index
        )

        diff_multiplier = 2
        self.LessThan = ltn.Predicate(func=lambda x, y: torch.sigmoid(diff_multiplier*(y - x)))
        self.MoreThan = ltn.Predicate(func=lambda x, y: torch.sigmoid(diff_multiplier*(x - y)))

        self.multiple_appearance_count = torch.zeros(self.max_shapes) # Agent Count
        self.RewardPredictor = ltn.Function(model=self.reward_predictor)
        self.reward_optimizer = torch.optim.Adam(self.reward_predictor.parameters(), lr=0.01)

        self.memory = Memory(model_settings["memory_capacity"])
        self.batch_size = model_settings["batch_size"]

        self.gamma = model_settings["gamma"]
        self.epsilon = model_settings["init_epsilon"]
        self.init_epsilon = model_settings["init_epsilon"]
        self.final_epsilon = model_settings["final_epsilon"]

        self.learn_steps = 0
        self.explore_steps = model_settings["explore_steps"]
        self.update_steps = model_settings["update_steps"]
        self.reset_epsilon = model_settings["reset_epsilon"]
        self.max_gradient_norm = model_settings["max_gradient_norm"]

        self.writer = writer
        self.begin_learn = False

    def update_unique_images(self, unique_shapes, batches):
        for batch in batches:
            for image in batch:
                # Check if the image already exists in unique_images
                if not any(same_shape(image, unique_image) for unique_image in unique_shapes):
                    
                    unique_shapes.append(image)

    def update_dqn(self, batch):
        batch_state, batch_next_state, batch_action, batch_reward, batch_done = batch

        with torch.no_grad():
            onlineQ_next = self.onlineQNetwork(batch_next_state)
            targetQ_next = self.targetQNetwork(batch_next_state)
            online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
            y = batch_reward + (1 - batch_done) * self.gamma * targetQ_next.gather(1, online_max_action)

                
        loss = F.mse_loss(self.onlineQNetwork(batch_state).gather(1, batch_action), y)
        
        self.optimizer.zero_grad()

        loss.backward()

        total_norm = torch.nn.utils.clip_grad_norm_(self.params, self.max_gradient_norm)   

        if not torch.isnan(total_norm).any():
            self.optimizer.step()
        else:
            print(f"update_dqn: {total_norm}")
        torch.cuda.empty_cache()

        return loss.item(), total_norm
        
    def AxiomPatchHasShape(self, p1, s1, p):
        # Each patch has at least 1 shape
        return self.Forall([p1], self.Exists([s1], self.IsShape(p1, s1), p=p), p=p)

    def AxiomMaxOneShapePerPatch(self, p1, s1, s2, p):
        # There is no patch with several shapes
        return self.Not(self.Exists(
            [p1, s1, s2], 
            self.And(self.And(self.IsShape(p1, s1), self.IsShape(p1, s2)), self.Not(self.Eq(s1, s2)))),
            p = p
            )

    def AxiomDiffPatchDiffShape(self, p1, p2, s1, p):
        # Different Patches have different shapes
        return self.Forall([p1, p2, s1], 
            self.Implies(
                self.And(self.IsShape(p1, s1), self.Not(self.Eq(p1, p2))), 
                self.Not(self.IsShape(p2, s1))),
                p=p
        )
    
    def AxiomHigherReward(self, q1, q2, r1, r2, p=8):
        ltn.diag(q1, r1)
        ltn.diag(q2, r2)
        # Q is higher for actions that lead to shapes with higher reward
        return self.Forall(
            [q1, q2, r1, r2],
                self.MoreThan(q1, q2),
            cond_vars=[r1, r2],
            cond_fn=lambda r1, r2: r1.value > r2.value + 0.5,
            p=p
        )
    
    def update_shape_recognizer(self, unique_patches):
        p = torch.stack([p.flatten() for p in torch.stack(unique_patches)])
        p1 = ltn.Variable("patches1", p)
        p2 = ltn.Variable("patches2", p)

        # Query loss
        query_loss = 1. - self.sat_agg(
                    # # Each patch has at least 1 shape
                    self.AxiomPatchHasShape(p1, self.s1, 50),

                    # There is no patch with several shapes
                    self.AxiomMaxOneShapePerPatch(p1, self.s1, self.s2, 50),

                    # Different Patches have different shapes
                    self.AxiomDiffPatchDiffShape(p1, p2, self.s1, 50)
                    )
        

        # learning
        self.shape_recognizer_optimizer.zero_grad()
        
        loss = 1. - self.sat_agg(
                # # Each patch has at least 1 shape
                self.AxiomPatchHasShape(p1, self.s1, self.p_op),

                # There is no patch with several shapes
                self.AxiomMaxOneShapePerPatch(p1, self.s1, self.s2, self.p_op),

                # Different Patches have different shapes
                self.AxiomDiffPatchDiffShape(p1, p2, self.s1, self.p_op)
                )
            
        try:
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.shape_recognizer.parameters(), self.max_gradient_norm) 
            if not norm.isnan(): # Prevent exploding gradients
                self.shape_recognizer_optimizer.step()
        except:
            pass

        return query_loss
    
    def update_reward_predictor(self, state_patches, next_s_p, reward):
        s_p1 = ltn.Variable("state_patches1", state_patches[0])
        next_s_p1 = ltn.Variable("next_state_patches1", next_s_p[0])

        self.multiple_appearance_count = self.multiple_appearance_count/2 + self.CountShapes(s_p1, self.s1)

        agent_next_index = self.GetShapeIndex(next_s_p1, self.Agent(self.s1, self.multiple_appearance_count))

        pred = self.RewardPredictor(self.ShapeAtIndex(s_p1, agent_next_index, self.s1)).value[0]

        loss = F.mse_loss(pred, torch.tensor(reward, dtype=torch.float32))
        
        self.reward_optimizer.zero_grad()
        loss.backward()
        self.reward_optimizer.step()

        return loss

    def update_sym_dqn(self, state):
        state_patches = extract_patches(state)[0]
        s_p = torch.stack([p.flatten() for p in state_patches])
        s_p1 = ltn.Variable("state_patches1", s_p)

        agent_index = self.GetShapeIndex(s_p1, self.Agent(self.s1, self.multiple_appearance_count))
        
        rewardsFromActions = []
        for i in range(4):
            shapeFromAction = self.ShapeAtIndex(s_p1, self.IndexFromAction(agent_index, i), self.s1)
            rewardsFromActions.append(self.RewardPredictor(shapeFromAction).value[0])
        rewardsPred1 = ltn.Variable("rewardsPred1", torch.stack(rewardsFromActions))
        rewardsPred2 = ltn.Variable("rewardsPred2", torch.stack(rewardsFromActions))

        q_values = self.onlineQNetwork(state)[0]
        q_values1 = ltn.Variable("qValues1", q_values)
        q_values2 = ltn.Variable("qValues2", q_values)

        loss = 1. - self.sat_agg(self.AxiomHigherReward(q_values1, q_values2, rewardsPred1, rewardsPred2))
        if self.use_ltn:
            self.optimizer.zero_grad()
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(self.params, self.max_gradient_norm) 

            if not torch.isnan(total_norm).any():
                self.optimizer.step()
            else:
                print(f"update_sym_dqn: {total_norm}")

        return loss


    def update_model(self, batch):
        batch_state, batch_next_state, batch_action, batch_reward, batch_done = batch
        batch_state = torch.tensor(batch_state, dtype=torch.float32)
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32)
        batch_action = torch.tensor(batch_action, dtype=torch.int64).unsqueeze(1)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1)
        batch_done = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1)
        tensor_batch = batch_state, batch_next_state, batch_action, batch_reward, batch_done

        loss, total_norm = self.update_dqn(tensor_batch)

        return loss, total_norm

    def get_best_action(self, state):
        with torch.no_grad():
            if self.action_guiding:
                state_patches = extract_patches(state)[0]
                s_p = torch.stack([p.flatten() for p in state_patches])
                s_p1 = ltn.Variable("state_patches1", s_p)
                agent_index = self.GetShapeIndex(s_p1, self.Agent(self.s1, self.multiple_appearance_count))

                rewardsFromActions = []
                for i in range(4):
                    shapeFromAction = self.ShapeAtIndex(s_p1, self.IndexFromAction(agent_index, i), self.s1)
                    rewardsFromActions.append(self.RewardPredictor(shapeFromAction).value[0])
            else:
                rewardsFromActions = [0, 0, 0, 0]
            best_action = self.onlineQNetwork.select_guided_action(state, rewardsFromActions)
        torch.cuda.empty_cache()
        return best_action

    def epoch_eval(self, env, training_settings, num_episodes=50):
        """Evaluates the model for a given number of episodes and returns the average score."""
        total_score = 0
        total_pos = 0
        total_neg = 0
        total_len = 0
        self.onlineQNetwork.eval() # Switch model to eval mode
        
        with torch.no_grad(): 
            for _ in range(num_episodes):
                _, _ = env.reset()
                state = env.unwrapped.get_state_img()
                max_reward = env.unwrapped.episode_max_reward()
                min_reward = env.unwrapped.episode_min_reward()
                epi_reward = 0
                epi_pos = 0
                epi_neg = 0

                for step in range(training_settings["max_steps"]):
                    tensor_state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
                    action = self.get_best_action(tensor_state) 
                    _, reward, done, _, _ = env.step(action)
                    epi_reward += reward

                    if reward == 1:
                        epi_pos += 1
                    if reward == -1:
                        epi_neg += 1

                    if done:
                        break
                    state = env.unwrapped.get_state_img()

                total_len += step+1
                total_score += epi_reward/max_reward
                total_pos += epi_pos/max_reward
                total_neg += epi_neg/max(min_reward, 1)

        self.onlineQNetwork.train()  # Switch back to training mode

        return total_score / num_episodes, total_len / num_episodes, total_pos / num_episodes , total_neg / num_episodes

    def train(self, env, training_settings, env_i=0):
        epochs = training_settings["epochs"]
        episodes_per_epoch = training_settings["episodes_per_epoch"]
        max_steps = training_settings["max_steps"]

        env.unwrapped.seed(env_i)
        n_action = env.action_space.n

        episode_avg_reward, episode_avg_len, episode_avg_pos, episode_avg_neg = self.epoch_eval(env, training_settings, num_episodes=episodes_per_epoch)

        curr_epoch = env_i*epochs

        if self.writer:
            self.writer.report_scalar(title="Epoch Reward", series="ratio", value=episode_avg_reward, iteration=curr_epoch)
            self.writer.report_scalar(title="Epoch Reward", series="pos", value=episode_avg_pos, iteration=curr_epoch)
            self.writer.report_scalar(title="Epoch Reward", series="neg", value=episode_avg_neg, iteration=curr_epoch)
            self.writer.report_scalar(title="Episode Length", series="linear", value=episode_avg_len, iteration=curr_epoch)

        unique_patches = []
        
        for epoch in range(epochs):
            epoch_reward = 0
            epoch_loss = 0
            epoch_norm = 0
            epoch_ltn_loss = 0
            epoch_reward_loss = 0
            epoch_symdqn_loss = 0

            for episode in range(episodes_per_epoch):
                _, _ = env.reset()
                state = env.unwrapped.get_state_img()
                max_reward = env.unwrapped.episode_max_reward()
                
                # Extract patches and identify unique shapes
                patches = extract_patches(torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0))
                self.update_unique_images(unique_patches, patches) 
                
                episode_reward = 0
                episode_loss = 0
                episode_norm = 0
                episode_ltn_loss = 0
                episode_reward_loss = 0
                episode_symdqn_loss = 0

                for step in range(max_steps):
                    if random.random() < self.epsilon:
                        action = random.randint(0, n_action - 1)
                    else:
                        tensor_state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
                        if self.action_guiding_training:
                            action = self.get_best_action(tensor_state)
                        else:
                            action = self.onlineQNetwork.select_action(tensor_state)

                    _, reward, done, _, _ = env.step(action)
                    next_state = env.unwrapped.get_state_img()
                    episode_reward += reward

                    if reward == 0:
                        reward = training_settings["no_reward_cost"]

                    # Train ltn
                    episode_ltn_loss = self.update_shape_recognizer(unique_patches)

                    patches = extract_patches(torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0))
                    next_s_p = extract_patches(torch.tensor(np.array(next_state), dtype=torch.float32).unsqueeze(0))

                    episode_reward_loss += self.update_reward_predictor(patches, next_s_p, reward)
                    episode_symdqn_loss += self.update_sym_dqn(torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0))


                    # Train DuelingDQN
                    self.memory.add((state, next_state, action, reward, done))
                    
                    if len(self.memory) > self.batch_size:
                        if not self.begin_learn:
                            self.begin_learn = True
                        self.learn_steps += 1
                        if self.learn_steps % self.update_steps == 0:
                            self.targetQNetwork.load_state_dict(self.onlineQNetwork.state_dict())
                        
                        batch = self.memory.sample(self.batch_size)
                        loss, norm = self.update_model(batch)
                        
                        episode_loss += loss
                        episode_norm += norm
                        if self.epsilon > self.final_epsilon:
                            self.epsilon -= (self.init_epsilon - self.final_epsilon) / self.explore_steps

                    if done:
                        break
                    state = next_state
                    
                epoch_reward += episode_reward/max_reward

                curr_episode = episode + epoch*episodes_per_epoch + env_i*epochs*episodes_per_epoch
  
                epoch_reward += episode_reward/max_reward
                epoch_loss += episode_loss/max(1, step)
                epoch_norm += episode_norm/max(1, step)
                epoch_ltn_loss += episode_ltn_loss
                epoch_reward_loss += episode_reward_loss/max(1, step)
                epoch_symdqn_loss += episode_symdqn_loss/max(1, step)

            self.onlineQNetwork.last_reward_loss = epoch_reward_loss/episodes_per_epoch
            episode_avg_reward, episode_avg_len, episode_avg_pos, episode_avg_neg = self.epoch_eval(env, training_settings, num_episodes=episodes_per_epoch)

            curr_epoch = env_i*epochs + epoch + 1

            if self.writer:
                self.writer.report_scalar(title="Epoch Reward", series="ratio", value=episode_avg_reward, iteration=curr_epoch)
                self.writer.report_scalar(title="Epoch Reward", series="pos", value=episode_avg_pos, iteration=curr_epoch)
                self.writer.report_scalar(title="Epoch Reward", series="neg", value=episode_avg_neg, iteration=curr_epoch)
                self.writer.report_scalar(title="Episode Length", series="linear", value=episode_avg_len, iteration=curr_epoch)
                self.writer.report_scalar(title="Episode Loss", series="linear", value=epoch_loss/episodes_per_epoch, iteration=curr_epoch)
                self.writer.report_scalar(title="Episode Norm", series="linear", value=epoch_norm/episodes_per_epoch, iteration=curr_epoch)
            
                if self.use_shape_recognizer:    
                    self.writer.report_scalar(title="ShapeRecognizer Loss", series="linear", value=epoch_ltn_loss/episodes_per_epoch, iteration=curr_epoch)

                if self.use_reward:    
                    self.writer.report_scalar(title="Reward Loss", series="linear", value=epoch_reward_loss/episodes_per_epoch, iteration=curr_epoch)
                self.writer.report_scalar(title="Axiom 4", series="linear", value=epoch_symdqn_loss/episodes_per_epoch, iteration=curr_epoch)
              
            print(f'Epoch {epoch}\t\tEpoch Reward: {episode_avg_reward:.3f}\t\tTime: {datetime.now().strftime("%H:%M:%S")}')
                
        torch.save(self.onlineQNetwork.state_dict(), f"sym_dqn/models/checkpoints/{self.name}_{env_i}.pth")


    def train_on_envs(self, envs, training_settings) -> None:
        for i, env_id in enumerate(envs):
            env = gym.make(env_id)

            try:
                self.train(env, training_settings, env_i=i)
            except Exception as e:
                print("An error occurred:", e)
                traceback.print_exc()
            
            env.close()
