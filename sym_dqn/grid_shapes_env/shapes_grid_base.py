import gymnasium as gym
from gymnasium.utils import seeding
import sys
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
import imageio
from skimage.transform import resize


class ShapesGridBase(gym.Env):
    metadata = {
        'render_modes': ['console','human'],
        'entity_decoding' : {0:'empty', 1:'agent', 2:'cross', 3:'circle', 4:'square'},
        'entity_encoding' : {'empty':0,'agent':1, 'cross':2, 'circle':3,'square':4},
        'entity_cmdrender' : {0:'.', 1:'+', 2:'x', 3:'o',4:'□'},
        'action_decoding' : {0:'up', 1:'right', 2:'down', 3:'left'},
        'action_mov_vec' : {0:[-1,0], 1:[0,1], 2:[1,0], 3:[0,-1]}, #[mov_row,mov_col]
                #rows: 0=top, nrow= bottom; cols: 0=left, ncol=right
        'background_color' : 'white',
        'entity_colors' : {'agent':'black','cross':'black','circle':'black','square':'black'}
    }

    '''Base grid version of the CrossCircle environment '''
    def __init__(self, ncol=5, nrow=5, min_entities=18, max_entities=18, entity_size=10, 
                entt_to_reward={'empty':0,'cross':1,'circle':-1,'square':0}, 
                entt_to_include=['cross','circle','square'],
                background_color='white', entities_color='black',
                version='random'):
        super(ShapesGridBase, self).__init__()
        
        self.version = version

        # creating the grid vector
        assert(nrow >= 3)
        assert(ncol >= 3)
        self.ncol = ncol
        self.nrow = nrow
        self.grid = [self.metadata['entity_encoding']['empty']]*ncol*nrow #1-D storage

        # creating the entities sets
        assert(min_entities <= max_entities)
        assert(min_entities > 0)
        self.min_entities = min_entities
        self.max_entities = max_entities
        self.entities = {'circle':set([]), 'cross':set([]), 'square':set([])}

        # mapping reward and entities
        self.entt_to_include = entt_to_include.copy()
        self.entt_to_include.sort()
        self.entt_to_reward = entt_to_reward.copy()
        self.reward_to_entt = {0:[],1:[],-1:[]}
        for entt, reward in self.entt_to_reward.items():
            self.reward_to_entt[reward].append(entt)
        assert(len(self.reward_to_entt[1]) > 0)

        # gym action space
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.MultiDiscrete([5] * (nrow * ncol))


        self.last_action = None

        # preparing the visuals
        self.entity_size = entity_size
        self.frame_width = entity_size * ncol
        self.frame_height = entity_size * nrow
        self.viewer = None

        self.masks = {}
        for entity_type in self.entt_to_include+['agent']:
            f = os.path.join(os.path.dirname(__file__), "images", "{}.png".format(entity_type))
            mask = imageio.imread(f)
            mask = resize(mask, (self.entity_size, self.entity_size), mode='edge', preserve_range=True)
            self.masks[entity_type] = np.tile(mask[..., 3:], (1, 1, 3)) / 255.

        self.color_desc = entities_color+'on'+background_color
        background_color = to_rgb(background_color)
        self.background_color = np.array(background_color)[None, None, :]

        self.entity_colors = {}
        for entity_type in self.entt_to_include+['agent']:
            entity_color = to_rgb(entities_color)
            self.entity_colors[entity_type] = np.array(entity_color)[None, None, :]

        self.seed(42)
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:  # Check if a new seed is provided
            self.seed(seed)
        ''' Clear entities and state, call setup_field()'''
        self.grid = np.zeros((self.ncol*self.nrow), dtype=int)
        self.entities = {'circle':set([]), 'cross':set([]), 'square':set([])}
        self.last_action = None
        self.setup_field()
        self._episode_steps = 0

        return self.grid, {}

    def episode_max_reward(self):
        types_to_collect = self.reward_to_entt[1]
        collectable_rewards = 0
        for entt_type in types_to_collect:
            collectable_rewards += len(self.entities[entt_type])
        return collectable_rewards

    def episode_min_reward(self):
        types_to_collect = self.reward_to_entt[-1]
        collectable_rewards = 0
        for entt_type in types_to_collect:
            collectable_rewards += len(self.entities[entt_type])
        return collectable_rewards

    def setup_field(self):
        if self.version == 'fixed':
            return self.layout_fixed()
        elif self.version == 'random':
            return self.layout_random()

        print(f"Invalid Shapes Env Version requestes: {self.version}")

        '''Calls layout. Meant as a chance for subclasses to alter layout() call'''
        raise NotImplementedError('Needs to be implemented in subclasses')

    def layout_fixed(self):
        ''' setup agent and entities; does not clear field '''
        # agent: starts top left
        self.agent_pos = 0
        self.grid[self.agent_pos] = self.metadata['entity_encoding']['agent']
        # entities : one per row
        for row in range(self.nrow):
            col = (row+3)%self.ncol #when row=0, col!=0 because of the agent's position
            entt_pos = self.rowcol_to_gridindex(row, col)
            if row == 0:
                # at least one entity must be a positive reward
                entt_type = self.reward_to_entt[1][0]
            else:
                entt_type = self.entt_to_include[(row%len(self.entt_to_include))]
            self.grid[entt_pos] = self.metadata['entity_encoding'][entt_type]
            self.entities[entt_type].add(entt_pos)
        return

    def layout_random(self, min_entities=None, max_entities=None):
        ''' setup agent and entities; does not clear field '''
        if min_entities == None:
            min_entities = self.min_entities
        if max_entities == None:
            max_entities = self.max_entities
        
        # agent
        self.agent_pos = self.np_random.integers(len(self.grid))
        self.grid[self.agent_pos] = self.metadata['entity_encoding']['agent']

        # entities
        n_entities = self.np_random.integers(min_entities, max_entities+1)
        free_pos = np.delete(np.arange(len(self.grid)), self.agent_pos)
        self.np_random.shuffle(free_pos)
        entities_pos = free_pos[:n_entities]

        for i in range(len(entities_pos)):
            entt_pos = entities_pos[i]
            if i == 0:
                # at least one entity must be a positive reward
                entt_type = self.reward_to_entt[1][0]
            else:
                entt_type = self.np_random.choice(self.entt_to_include)
            self.grid[entt_pos] = self.metadata['entity_encoding'][entt_type]
            self.entities[entt_type].add(entt_pos)
        return

    def step(self, action):
        action_type = self.metadata['action_decoding'][action]
        self.last_action = action_type
        
        reward = self.move_agent(action)
        self._episode_steps += 1
        
        types_to_collect = self.reward_to_entt[1]
        done_per_type = [len(self.entities[entt_type])==0 for entt_type in types_to_collect]
        done = all(done_per_type)

        return self.grid, reward, done, False, {}

    def get_agent_new_pos(self, action):
        move_vec = self.metadata['action_mov_vec'][action]

        ag_row, ag_col = self.gridindex_to_rowcol(self.agent_pos)
        ag_row = np.clip(ag_row+move_vec[0], 0, self.nrow-1)
        ag_col = np.clip(ag_col+move_vec[1], 0, self.ncol-1)

        return self.rowcol_to_gridindex(ag_row, ag_col)

    def move_agent(self, action):
        '''empty previous agent cell, move agent position (taking into account grid's borders),
        returns reward depending on what the arrival cell contains'''
        self.grid[self.agent_pos] = self.metadata['entity_encoding']['empty']

        self.agent_pos = self.get_agent_new_pos(action)

        entt_in_arrivalcell = self.metadata['entity_decoding'][self.grid[self.agent_pos]]
        if entt_in_arrivalcell != 'empty':
            self.entities[entt_in_arrivalcell].remove(self.agent_pos) # clears entity eaten
        reward = self.entt_to_reward[entt_in_arrivalcell]

        self.grid[self.agent_pos] = self.metadata['entity_encoding']['agent']
        return reward

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(42+seed)
        return [seed]

    def get_image(self):
        image = np.tile(self.background_color, (self.frame_width, self.frame_height, 1))
        for i in range(len(self.grid)):
            entity_type = self.metadata['entity_decoding'][self.grid[i]]
            if entity_type != 'empty':
                row_entity, col_entity = self.gridindex_to_rowcol(i)
                image = self._render_shape(image, entity_type, row_entity, col_entity)
        return image
    
    def get_state_img(self):
        return np.transpose(self.get_image(), (2, 0, 1))

    def render(self, mode='human'):
        if mode=='console':
            outfile = sys.stdout
            desc = [
                    [self.metadata['entity_cmdrender'][self.grid[r*self.ncol+c]] for c in range(self.ncol)]
                     for r in range(self.nrow)]
            ag_row, ag_col = self.gridindex_to_rowcol(self.agent_pos)
            desc[ag_row][ag_col] = gym.utils.colorize(desc[ag_row][ag_col], "red", highlight = True)
            if self.last_action != None:
                outfile.write(self.last_action + "\n")
            outfile.write("\n".join(''.join(line) for line in desc)+"\n")
        
        elif mode=='human':
            image = self.get_image()
            plt.clf()
            plt.tick_params(axis='both', which='both', bottom=False, top=False, 
                    labelbottom=False, right=False, left=False, labelleft=False)
            plt.imshow(image)
            plt.pause(1)

    def _render_shape(self, image, entity_type, row_entity, col_entity):
        alpha = self.masks[entity_type]
        tile = np.tile(self.entity_colors[entity_type], (self.entity_size, self.entity_size, 1))

        top = row_entity*self.entity_size
        bottom = (row_entity+1)*self.entity_size

        left = col_entity*self.entity_size
        right = (col_entity+1)*self.entity_size

        image[top:bottom, left:right, ...] = alpha * tile + (1 - alpha) * image[top:bottom, left:right, ...]
        return image

    def rowcol_to_gridindex(self, row, col):
        return col + row*self.ncol

    def gridindex_to_rowcol(self, i):
        col = i%self.ncol
        row = i//self.ncol
        return row, col
    
    def get_reward_per_action(self):
        rewards = []
        for action in range(len(self.metadata["action_decoding"])):
            agent_pos = self.get_agent_new_pos(action)
            entt_in_arrivalcell = self.metadata['entity_decoding'][self.grid[agent_pos]]
            reward = self.entt_to_reward[entt_in_arrivalcell]
            rewards.append(reward)

        return rewards



