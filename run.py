from math import floor
from blessed import Terminal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import signal
import time
import random


class LifeSim:

    SPECIES_TO_START_WITH = 15 # number of species to spawn when the simulation starts
    SPECIES_LIMIT = 20 # maximum number of species allowed, kills lowest population species if more than this value exist
    FRAMES_PER_EVOLUTION = 30 # number of frames before a species is selected for evolution
    EVOLUTION_PROTECTION_FRAMES = 120 # number of frames a species has reduced chances of evolving for after evolving once
    KERNEL_SIZE = 4 # size of the kernel to use for species neighbor convolution

    def __init__(self, width=None, height=None):
        self.term = Terminal()
        self.p = 0.5
        self.width = width or self.term.width
        self.height = height or self.term.height
        self.fig = None
        self.img = self.img if hasattr(self, 'img') else None
        self.img_stack = []
        self.frame = 0
        self.frame_times = []
        self.general_map = np.zeros((self.width, self.height)) > 0.5
        self.color_map = np.zeros((self.width, self.height), dtype=int) 
        self.change_map = np.zeros((self.width, self.height)) > 0.5
        self.species_info = {}
        self.build_random()
    
    def build_random(self):
        for i in range(LifeSim.SPECIES_TO_START_WITH):
            species_name = self.create_species()
            self.handle_spawns(
                species_name,
                (np.random.rand(self.width, self.height) < 0.15) & ~self.general_map,
                np.zeros((self.width, self.height)) > 0.5
            )

    def random_kernel(parent_kernel=None):
        if parent_kernel is None:
            kernel = np.random.rand(LifeSim.KERNEL_SIZE, LifeSim.KERNEL_SIZE)
            if LifeSim.KERNEL_SIZE % 2:
                kernel[2][2] = 0
            kernel = 8 * kernel/np.sum(kernel, keepdims=False)
            return kernel
        else:
            kernel = parent_kernel + (1 * np.random.rand(LifeSim.KERNEL_SIZE, LifeSim.KERNEL_SIZE) * (np.random.rand(LifeSim.KERNEL_SIZE, LifeSim.KERNEL_SIZE) < 0.1))
            if LifeSim.KERNEL_SIZE % 2:
                kernel[2][2] = 0
            kernel = 8 * kernel/np.sum(kernel, keepdims=False)
            return kernel

    def create_species(self, parent=None):
        species_kernel = LifeSim.random_kernel(self.species_info[parent]['species_kernel'] if parent else None)
        replication_kernel = LifeSim.random_kernel(self.species_info[parent]['enemy_kernel'] if parent else species_kernel)
        enemy_kernel = LifeSim.random_kernel(self.species_info[parent]['enemy_kernel'] if parent else None)
        species_name = hash(species_kernel.tobytes() + enemy_kernel.tobytes())
        if species_name in self.species_info:
            return self.create_species(parent)
        self.species_info[species_name] = {
            'species_kernel': species_kernel,
            'replication_kernel': replication_kernel,
            'enemy_kernel': enemy_kernel,
            'map': np.zeros((self.width, self.height)) > 0.5,
            'count': 0,
            'last_evolution_frame': 0,
            'color': sum([ 256**i*num for i, num in enumerate(np.random.randint(0, 256, (3,))) ]),
        }
        return species_name

    
    def delete_nonexistent_species(self):
        for species_name in list(self.species_info.keys()):
            if self.species_info[species_name]['count'] == 0:
                del self.species_info[species_name]
    
    def handle_spawns(self, species_name, births, deaths):
        self.general_map |= births
        self.general_map &= ~deaths
        self.color_map -= (births | deaths) * 16777216
        np.putmask(self.color_map, self.color_map < 0, 0)
        self.color_map += births * self.species_info[species_name]['color']
        self.species_info[species_name]['map'] |= births
        self.species_info[species_name]['map'] &= ~deaths
        self.species_info[species_name]['count'] = np.sum(self.species_info[species_name]['map'], keepdims=False)
        self.change_map |= births | deaths

    def evolve(self):
        species_evolution_odds = np.array(list(map(
            lambda species : species['count'] * min(1, (self.frame - species['last_evolution_frame']) / LifeSim.EVOLUTION_PROTECTION_FRAMES) ** 2,
            self.species_info.values()
        )))
        species_name = np.random.choice(list(self.species_info.keys()), p=species_evolution_odds/sum(species_evolution_odds))
        new_species_name = self.create_species(species_name)
        self.species_info[species_name]['last_evolution_frame'] = self.frame
        self.species_info[new_species_name]['last_evolution_frame'] = self.frame
        new_spawn_spots = (np.random.rand(self.width, self.height) < 0.5) & self.species_info[species_name]['map']
        self.handle_spawns(species_name, np.zeros((self.width, self.height)) > 0.5, new_spawn_spots)
        self.handle_spawns(new_species_name, new_spawn_spots, np.zeros((self.width, self.height)) > 0.5)
        if len(self.species_info) > LifeSim.SPECIES_LIMIT:
            smallest_name = min(self.species_info.keys(), key=lambda species_name : self.species_info[species_name]['count'])
            self.handle_spawns(smallest_name, np.zeros((self.width, self.height)) > 0.5, self.species_info[smallest_name]['map'])

    def render_state(self):
        text = ''
        for x, column in enumerate(self.change_map):
            for y, cell in enumerate(column):
                if cell:
                    color_num = self.color_map[x, y]
                    color = [ color_num % 256, (color_num >> 8) % 256, color_num >> 16 ]
                    text += self.term.move_xy(floor(x), floor(y)) + (f"{self.term.color_rgb(*color)}█{self.term.normal}" if self.general_map[x][y] else ' ')
        self.change_map = np.zeros((self.width, self.height)) > 0.5
        print(text, end='', flush=True)

    def calc_next_state_game_of_life(self, old_map):
        #process species in random order so there is no collision advantage
        species_info_items = list(self.species_info.items())
        np.random.shuffle(species_info_items)
        for species_name, species in species_info_items:
            self_convolution = signal.convolve2d(species['map'].astype(int), species['species_kernel'], mode='same', boundary='wrap')
            replication_convolution = self_convolution
            enemy_convolution = signal.convolve2d(old_map & ~species['map'], species['enemy_kernel'], mode='same', boundary='wrap')
            birth_conditions_met = (replication_convolution >= 2.5) & (replication_convolution <= 3.5) & (enemy_convolution <= 1)
            life_conditions_met = (self_convolution >= 1.5) & (self_convolution <= 3.5) & (enemy_convolution <= 2.25)
            births = birth_conditions_met & ~species['map'] & ~old_map & ~self.general_map
            deaths = ~life_conditions_met & species['map']
            self.handle_spawns(species_name, births, deaths)
        if self.frame % LifeSim.FRAMES_PER_EVOLUTION == 0:
            self.evolve()
        self.delete_nonexistent_species()

    def check_for_errors(self, pt=''):
        combined_maps = np.zeros((self.width, self.height)) > 0.5
        for species_name_a, species_a in self.species_info.items():
            combined_maps |= species_a['map']
            for species_name_b, species_b in self.species_info.items():
                if species_name_a != species_name_b and np.sum(species_a['map'] & species_b['map'], keepdims=False) != 0:
                    print(np.sum(species_a['map'] & species_b['map'], keepdims=False))
                    raise Exception('2 species sharing square: ' + pt)
            if np.sum(species_a['map'] & ~self.general_map, keepdims=False) != 0:
                print(np.sum(species_a['map'] & ~self.general_map, keepdims=False))
                raise Exception('Species has untracked map location: ' + pt)
        if np.sum(combined_maps ^ self.general_map, keepdims=False) != 0:
            print(np.sum(combined_maps ^ self.general_map, keepdims=False))
            raise Exception('Map has location unocupied by any species: ' + pt)

    def start(self, single_frame=False):
        with self.term.cbreak(), self.term.hidden_cursor():
            if not single_frame:
                print(self.term.home + self.term.clear)
            while self.term.inkey(timeout=0.0) != 'q':
                self.frame += 1
                start = time.time()
                old_map = self.general_map
                self.calc_next_state_game_of_life(old_map)
                if len(self.species_info) == 1:
                    self.__init__(self.width, self.height)
                calc_time = time.time() - start
                if not single_frame:
                    self.render_state()
                render_time = time.time() - start
                self.frame_times.append(render_time)
                self.frame_times = self.frame_times[-10:]
                fps = self.term.move_xy(0, self.height) + str(round(len(self.frame_times)/(sum(self.frame_times) + 0.00001), 1)) + f" fps ({self.frame})   calc_time: {round(calc_time, 4)}   render_time: {round(render_time, 4)}   species_count: {len(self.species_info)}     "
                if self.frame % 10 == 0:
                    print(fps, end='', flush=True)
                if single_frame:
                    img = np.array([self.color_map % 256, (self.color_map >> 8) % 256, self.color_map >> 16])
                    img = np.moveaxis(img, 0, -1)
                    self.img_stack.append(img)
                    self.img_stack = self.img_stack[-10:]
                    self.img.set_data(np.mean(self.img_stack, axis=0, dtype=int))
                    return
        
    def start_img(self):
        fig = plt.figure(figsize=(10, 10),)
        fig.subplots_adjust(0,0,1,1)
        self.img = plt.imshow(self.general_map, vmin=0, vmax=1)
        anim = animation.FuncAnimation(fig, lambda i : self.start(single_frame=True), interval=1, save_count=1000000)
        writer = animation.writers['ffmpeg'](fps=30, bitrate=-1)
        anim.save('demo.mp4', writer=writer, dpi=100)
        # plt.show()

LifeSim(256, 256).start_img()
# LifeSim().start()