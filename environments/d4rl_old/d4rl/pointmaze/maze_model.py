""" A pointmass maze env."""
from gym.envs.mujoco import mujoco_env
from gym import utils
from d4rl import offline_env
from d4rl.pointmaze.dynamic_mjc import MJCModel
import numpy as np
import random

from gym.spaces import Box

WALL = 10
EMPTY = 11
START = 12

HUGE_MAZE = \
    "######################\\" + \
    "#S0#000#0#00000#000#0#\\" + \
    "#0000#0#0#00#00#0#0#0#\\" + \
    "######0#0#0###0###0#0#\\" + \
    "#000000#0#00000000000#\\" + \
    "#0####0#0#####0#0###0#\\" + \
    "#00#00000000000#0#0#0#\\" + \
    "#0##0#0###0#####0#000#\\" + \
    "#00#0###0#000#0#0#####\\" + \
    "#00000000#0#000#00000#\\" + \
    "####00###########0####\\" + \
    "#000000000#0000#00000#\\" + \
    "#00#00#0#0####0#####0#\\" + \
    "#00#00#0#0#0000#00000#\\" + \
    "#0#00#0#00#0##0#0#####\\" + \
    "#00#0000#0#00#0000000#\\" + \
    "#0######00##0#0###0###\\" + \
    "#00000#000#00#000#000#\\" + \
    "#0#00#00#0##########0#\\" + \
    "#00#000#00#000#000#00#\\" + \
    "#0#00#00#0#0#000#0000#\\" + \
    "######################"

LARGE_MAZE = \
    "############\\" + \
    "#OOOO#OOOOO#\\" + \
    "#O##O#O#O#O#\\" + \
    "#OOOOOO#OOO#\\" + \
    "#O####O###O#\\" + \
    "#OO#O#OOOOO#\\" + \
    "##O#O#O#O###\\" + \
    "#SO#OOO#OOO#\\" + \
    "############"

LARGE_MAZE_EVAL = \
    "############\\" + \
    "#OO#OOO#OOO#\\" + \
    "##O###O#O#O#\\" + \
    "#OO#O#OOOOO#\\" + \
    "#O##O#OO##O#\\" + \
    "#OOOOOO#OOO#\\" + \
    "#O##O#O#O###\\" + \
    "#SOOO#OOOOO#\\" + \
    "############"

MEDIUM_MAZE = \
    '########\\' + \
    '#OO##OO#\\' + \
    '#OO#OOO#\\' + \
    '##OOO###\\' + \
    '#OO#OOO#\\' + \
    '#O#OO#O#\\' + \
    '#SOO#OO#\\' + \
    "########"

MEDIUM_MAZE_EVAL = \
    '########\\' + \
    '#OOOOOO#\\' + \
    '#O#O##O#\\' + \
    '#OOOO#O#\\' + \
    '###OO###\\' + \
    '#OOOOOO#\\' + \
    '#SO##OO#\\' + \
    "########"

SMALL_MAZE = \
    "######\\" + \
    "#OOOO#\\" + \
    "#O##O#\\" + \
    "#SOOO#\\" + \
    "######"

U_MAZE = \
    "#####\\" + \
    "#OOO#\\" + \
    "###O#\\" + \
    "#SOO#\\" + \
    "#####"

U_MAZE_EVAL = \
    "#####\\" + \
    "#OOO#\\" + \
    "#O###\\" + \
    "#OOO#\\" + \
    "#####"

OPEN = \
    "#######\\" + \
    "#OOOOO#\\" + \
    "#OOOOO#\\" + \
    "#OOOOO#\\" + \
    "#######"


def parse_maze(maze_str):
    lines = maze_str.strip().split('\\')
    width, height = len(lines), len(lines[0])
    maze_arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == '#':
                maze_arr[w][h] = WALL
            elif tile == 'S':
                maze_arr[w][h] = START
            elif tile == ' ' or tile == 'O' or tile == '0':
                maze_arr[w][h] = EMPTY
            else:
                raise ValueError('Unknown tile type: %s' % tile)
    return maze_arr


def point_maze(maze_str):
    maze_arr = parse_maze(maze_str)

    mjcmodel = MJCModel('point_maze')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(damping=1, limited='false')
    default.geom(friction=".5 .1 .1", density="1000", margin="0.002", condim="1", contype="2", conaffinity="1")

    asset = mjcmodel.root.asset()
    asset.texture(type="2d", name="groundplane", builtin="checker", rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3", width=100,
                  height=100)
    asset.texture(name="skybox", type="skybox", builtin="gradient", rgb1=".4 .6 .8", rgb2="0 0 0",
               width="800", height="800", mark="random", markrgb="1 1 1")
    asset.material(name="groundplane", texture="groundplane", texrepeat="20 20")
    asset.material(name="wall", rgba=".7 .5 .3 1")
    asset.material(name="target", rgba=".6 .3 .3 1")

    visual = mjcmodel.root.visual()
    visual.headlight(ambient=".4 .4 .4", diffuse=".8 .8 .8", specular="0.1 0.1 0.1")
    visual.map(znear=.01)
    visual.quality(shadowsize=2048)

    worldbody = mjcmodel.root.worldbody()
    worldbody.geom(name='ground', size="40 40 0.25", pos="0 0 -0.1", type="plane", contype=1, conaffinity=0,
                   material="groundplane")

    particle = worldbody.body(name='particle', pos=[1.2, 1.2, 0])
    particle.geom(name='particle_geom', type='sphere', size=0.1, rgba='0.0 0.0 1.0 0.0', contype=1)
    particle.site(name='particle_site', pos=[0.0, 0.0, 0], size=0.2, rgba='0.3 0.6 0.3 1')
    particle.joint(name='ball_x', type='slide', pos=[0, 0, 0], axis=[1, 0, 0])
    particle.joint(name='ball_y', type='slide', pos=[0, 0, 0], axis=[0, 1, 0])

    worldbody.site(name='target_site', pos=[0.0, 0.0, 0], size=0.2, material='target')

    width, height = maze_arr.shape
    for w in range(width):
        for h in range(height):
            if maze_arr[w,h] == WALL:
                worldbody.geom(conaffinity=1,
                               type='box',
                               name='wall_%d_%d'%(w,h),
                               material='wall',
                               pos=[w+1.0, h+1.0, 0],
                               size=[0.5, 0.5, 0.2])

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)

    return mjcmodel


class MazeEnv(mujoco_env.MujocoEnv, utils.EzPickle, offline_env.OfflineEnv):

    # IMAGE GENERATION SETTINGS

    def __init__(self,
                 maze_spec=U_MAZE,
                 reset_target=True,
                 reward_type='sparse',
                 **kwargs):
        offline_env.OfflineEnv.__init__(self, **kwargs)
        self.str_maze_spec = maze_spec
        self.maze_arr = parse_maze(maze_spec)
        self.maze_width = len(self.maze_arr[0])
        self.maze_height = len(self.maze_arr)

        reset_locations = list(zip(*np.where(self.maze_arr == START)))
        assert len(reset_locations) == 1, "Map parsing error: more than one reset position detected."
        self.reset_location = reset_locations[0]
        self.pretrain_goals_box = self.get_tile_box(*self.reset_location)

        self.reward_type = reward_type
        self.empty_tiles = list(zip(*np.where(self.maze_arr == EMPTY)))
        self.empty_tiles.sort()

        self._target = np.array([0.0, 0.0])

        model = point_maze(maze_spec)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, model_path=f.name, frame_skip=1)
        utils.EzPickle.__init__(self)

    def get_tile_box(self, x, y):
        """
        return a gym.spaces.Box object that allow to sample states from a maze tile.
        Inputs: x, y coordinates of the desired tile.
        """
        return Box(np.array([x - 0.63, y - 0.63]), np.array([x + 0.22, y + 0.22]))

    def sample_goal(self):
        # TODO
        pass

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.clip_velocity()
        self.do_simulation(action, self.frame_skip)
        self.set_marker()
        ob = self._get_obs()
        reward = 1.0 if np.linalg.norm(ob[0:2] - self._target) <= 0.5 else 0.0
        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def get_target(self):
        return self._target

    def set_target(self, target_location=None):
        if target_location is None:
            target_location = self.get_tile_box(*random.choice(self.empty_tiles)).sample()
        self._target = target_location

    def set_marker(self):
        self.data.site_xpos[self.model.site_name2id('target_site')] = np.array([self._target[0]+1, self._target[1]+1, 0.0])

    def clip_velocity(self):
        qvel = np.clip(self.sim.data.qvel, -5.0, 5.0)
        self.set_state(self.sim.data.qpos, qvel)

    def reset_model(self):
        reset_location = np.array(self.reset_location).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.set_target()
        return self._get_obs(), self.get_target()

    def reset_to_location(self, location):
        self.sim.reset()
        reset_location = np.array(location).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        pass

    def get_background_image(self):
        """
        return a np_array of pixels, that represent an image of the environment.
        """
        tile_size = 10  # in pixels, tiles are square.
        walls_color = [0, 0, 0]
        empty_color = [250, 250, 250]
        start_area_color = [0, 250, 0]

        width_px = tile_size * self.maze_width
        height_px = tile_size * self.maze_height
        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for x, row in enumerate(self.maze_arr):
            for y, cell in enumerate(row):
                color = []
                if cell == 10:
                    color = walls_color
                elif cell == 11 or cell == 12:
                    color = empty_color

                tile_img = np.zeros(shape=(tile_size, tile_size, 3), dtype=np.uint8)
                tile_img[:, :, :] = color
                if cell == 12:
                    # Draw spawn range
                    low = tile_size // 2 - int(tile_size * 0.1)
                    high = tile_size // 2 + int(tile_size * 0.1)

                    tile_img[low:high, low:high, :] = start_area_color

                x_min = x * tile_size
                x_max = (x + 1) * tile_size
                y_min = y * tile_size
                y_max = (y + 1) * tile_size
                img[x_min:x_max, y_min:y_max, :] = tile_img

                img[x_min:x_max, y_min:y_max, :] = tile_img
        return img
