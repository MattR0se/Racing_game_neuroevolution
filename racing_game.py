"""
Prototype of a top-down car racing game

Controls:
    W: Accelerate forward
    S: Accelerate backwards
    A and D: Steer left and right, respectively
    H: Toggle debug mode (displays some rects and vectors)
    

Ideas for AI:
    simple:
        - "manual" placing of waypoints and steering behavior towards targets
    complicated
        - Smart Rockets
        - Neuroevolution with Raycast sensor input

"""
import pygame as pg
import traceback
from math import pi, sin, cos, inf
from pytmx.util_pygame import load_pygame
import pickle
from random import randint, random, choice
import json

W_WIDTH = 1024
W_HEIGHT = 768
TILESIZE = 64
FPS = 60
TWO_PI = pi * 2

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)


vec = pg.math.Vector2
VEC_INF = vec(inf, inf)

# ------------------ HELPER FUNCTIONS -----------------------------------------

def load_map(file):
    """
    load a Tiled map in .tmx format and return a background image Surface, 
    map objects as a TiledObjectGroup and layer_data as a list of 2D arrays
    with tile indices
    """
    tiled_map = load_pygame('assets/{}.tmx'.format(file))
    # create empty surface based on tile map dimensions
    bg_image = pg.Surface((tiled_map.width * tiled_map.tilewidth,
                          tiled_map.height * tiled_map.tileheight))
    map_objects = tiled_map.get_layer_by_name('objects_1')
    map_objects += tiled_map.get_layer_by_name('objects_2')
    # iterate through each tile layer and blit the corresponding tile
    layer_data = []
    for layer in tiled_map.layers:
        if hasattr(layer, 'data'):
            layer_data.append(layer.data)
            for x, y, image in layer.tiles():
                if image:
                    bg_image.blit(image, (x * tiled_map.tilewidth, 
                                          y * tiled_map.tileheight))
    return bg_image, map_objects, layer_data


def construct_polyeder(center, n, size, rotation=0):
    # construct a hexagon from a given center and radius
    points = []
    for i in range(n):
        angle_deg = (360 / n) * i - rotation
        angle_rad = pi / 180 * angle_deg
        points.append(vec(center.x + size * cos(angle_rad),
                         center.y + size * sin(angle_rad)))
    return points


def vec_to_int(vec):
    return (int(vec.x), int(vec.y))


def rotate_point(point, center, angle):
    # rotate a point around the center by a given angle
    point -= center
    # rotate point around the origin
    original_x = point.x
    original_y = point.y
    point.x = original_x * cos(angle) - original_y * sin(angle)
    point.y = original_y * cos(angle) + original_x * sin(angle)
    # translate back to shape's center
    point += center


def blit_alpha(target, source, location, opacity):
    # https://nerdparadise.com/programming/pygameblitopacity
    # TODO: This seems unnecessary computational intensive
    x = location[0]
    y = location[1]
    temp = pg.Surface((source.get_width(), source.get_height())).convert()
    temp.blit(target, (-x, -y))
    temp.blit(source, (0, 0))
    temp.set_alpha(opacity)        
    target.blit(temp, location)
    
    
def remap(n, start1, stop1, start2, stop2):
    # https://p5js.org/reference/#/p5/map
    newval = (n - start1) / (stop1 - start1) * (stop2 - start2) + start2
    if (start2 < stop2):
        return constrain(newval, start2, stop2)
    else:
        return constrain(newval, stop2, start2) 


def constrain(n, low, high):
    return max(min(n, high), low)


class Camera(object):
    def __init__(self, game, target):
        self.game = game
        self.offset = vec()
        self.target = target
        self.rect = self.game.screen_rect
        
    
    def update(self, dt):  
        # calculate the camera offset 
        # it is based on its target's rect plus the middle of the screen
        # multiplied by -1 because it goes in the opposite direction
        w = self.game.screen_rect.w
        h = self.game.screen_rect.h
        self.offset.x = self.target.rect.centerx + w // 2 * -1
        self.offset.y = self.target.rect.centery + h // 2 * -1
        
        # change the offset based on the target's velocity direction
        # to let the players see better where they are driving towards
        self.offset += self.target.vel * 0.4

        # camera can't go over upper and left map borders
        self.offset.x = max(self.offset.x, 0)
        self.offset.y = max(self.offset.y, 0)
        # camera can't go over bottom and right map borders
        self.offset.x = min(self.offset.x, (self.game.map_rect.w - 
                                            self.game.screen_rect.w))
        self.offset.y = min(self.offset.y, (self.game.map_rect.h - 
                                            self.game.screen_rect.h))

    
    def apply_mouse(self, m_pos):
        # currently not needed
        return m_pos - self.offset
    

    def apply_pos(self, pos):
        # translates any position vector / point for drawing
        return pos - self.offset


    def apply_rect(self, rect):
        # translates a given rect for drawing
        return pg.Rect(rect.topleft - self.offset, rect.size)


# ------------------ GAME OBJECT ----------------------------------------------

class Game:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((W_WIDTH, W_HEIGHT))
        self.screen_rect = self.screen.get_rect()
        self.clock = pg.time.Clock()
        self.running = True
        
        self.shapes = []
        self.particles = pg.sprite.Group() # probably faster for large groups than a simple list
        self.offroad = [] # contains rects that indicate off road
        
        self.track = 'track_1'
        # get tilemap, objects and tile data from tmx file
        self.map, self.map_objects, self.layer_data = load_map(self.track)
        self.map_rect = self.map.get_rect()
        
        # build checkpoints from map data
        self.checkpoints = []
        self.finish_line = None
        for obj in self.map_objects:
            if obj.name == 'checkpoint':
                self.checkpoints.append(pg.Rect(obj.x, obj.y, obj.width, obj.height))
            elif obj.name == 'finish_line':
                self.finish_line = pg.Rect(obj.x, obj.y, obj.width, obj.height)
            elif obj.name == 'grass':
                self.offroad.append(pg.Rect(obj.x, obj.y, obj.width, obj.height))

        self.car = Car(self)               
        self.car.move_to((3105, 2260)) # move the car to the start/finish line
        self.car.rotate(pi / -2)
    
        self.camera = Camera(self, self.car)
        #self.camera = Camera(self, self.shapes[0])
        
        self.round_time = 0
        self.last_round_time = 0
        self.best_round_time = inf
        self.highscores = {}
        # try accessing the highscore data from a file
        # if no file exists, set highscore to infinity
        try:
            with open('data/highscore.dat', 'rb') as file:
                self.highscores = pickle.load(file)
        except:
            self.highscores[self.track] = inf
        self.font = pg.font.SysFont('Arial', 24)
        
        self.debug_mode = False
        
        
    def events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_h:
                    self.debug_mode = not self.debug_mode
    
    
    def update(self, dt):
        pg.display.set_caption(f'FPS: {round(self.clock.get_fps(), 2)}')
        #pg.display.set_caption(f'frame: {self.camera.target.frames}')
        
        self.round_time += dt
        if self.camera.target.tile == 42:
            # 42 is the finish line
            # TODO: probably going to make this a rect object like the other checkpoints
            if len(self.camera.target.checkpoints) >= 2:
                # if player passed the checkpoints, check for highscores
                # and reset round time and checkpoints list
                self.last_round_time = self.round_time
                if self.round_time < self.best_round_time:
                    self.best_round_time = self.round_time
                if self.round_time < self.highscores[self.track]:
                    self.highscores[self.track] = self.round_time
                self.camera.target.checkpoints = []
                self.round_time = 0
                
        self.camera.update(dt)
        
        for shape in self.shapes:
            shape.update(dt)
        self.particles.update(dt)
    
    
    def draw(self):
        self.screen.blit(self.map, self.camera.apply_pos(self.map_rect.topleft))
        for p in self.particles:
            p.draw(self.screen)
        for shape in self.shapes:
            shape.draw(self.screen)
        if self.debug_mode:
            # some draw events for debugging
            """
            for check in self.checkpoints:
                pg.draw.rect(self.screen, WHITE, self.camera.apply_rect(check), 2)
            # visualise the camera offset
            v = self.camera.apply_pos(self.camera.target.center) + self.camera.target.vel * 0.4
            x = int(v.x)
            y = int(v.y)
            pg.draw.circle(self.screen, RED, (x, y), 4)
            pg.draw.line(self.screen, RED, 
                         self.camera.apply_pos(self.camera.target.center), v, 2 )
            # some lines that cut the screen rect in half
            pg.draw.line(self.screen, WHITE, (0, W_HEIGHT // 2), 
                                             (W_WIDTH, W_HEIGHT // 2), 1)
            pg.draw.line(self.screen, WHITE, (W_WIDTH // 2, 0), 
                                             (W_WIDTH // 2, W_HEIGHT), 1)
            """        
            ps = self.camera.target.intersection_points
            for p in ps:
                if p and p.length_squared() < inf:
                    p = self.camera.apply_pos(p)
                    x = int(p.x)
                    y = int(p.y)
                    pg.draw.circle(self.screen, RED, (x, y), 8)
                    
            for rect in self.offroad:
                pg.draw.rect(self.screen, WHITE, self.camera.apply_rect(rect), 2)
                     
        self.display_time()
        self.display_speed()      

        pg.display.update()
        
    
    def cleanup(self):
        # save best round time to file
        with open('data/highscore.dat', 'wb') as file:
            pickle.dump(self.highscores, file)
        # save the car's labeled actions as training data
        with open('data/car_training.json', 'w') as jsonfile:
            json.dump(self.car.training_data, jsonfile)

    
    def display_time(self):
        time = f'LAP TIME: {round(self.round_time, 3)}'
        # TODO: maybe make these inline ifs inside the f string?
        if self.last_round_time > 0:
            last_time = f'LAST LAP: {round(self.last_round_time, 3)}'
        else:
            last_time = f'LAST LAP: -----'
        if self.best_round_time < inf:
            best_time = f'BEST LAP: {round(self.best_round_time, 3)}'
        else:
            best_time = f'BEST LAP: -----'
        if self.highscores[self.track] < inf:
            record_time = f'LAP RECORD: {round(self.highscores[self.track], 3)}'
        else:
            record_time = f'LAP RECORD: -----'

        for i, t in enumerate([time, last_time, best_time, record_time]):       
            txt_surf = self.font.render(t, False, WHITE)
            txt_rect = txt_surf.get_rect()
            txt_rect.topleft = self.screen_rect.midtop
            txt_rect.y += i * 30
            self.screen.blit(txt_surf, txt_rect)
    
    
    def display_speed(self):
        speed = self.camera.target.vel_len * 0.2
        speed_str = f'SPEED: {round(speed, 1)} MP/H'
        txt_surf = self.font.render(speed_str, False, WHITE)
        txt_rect = txt_surf.get_rect()
        txt_rect.topleft = self.screen_rect.topleft
        txt_rect.x += 100
        self.screen.blit(txt_surf, txt_rect)
        
   
    def run(self):
        while self.running:
            delta_time = self.clock.tick(FPS) / 1000.0
            if delta_time < 0.5:
                # prevents the game from advancing if the window is dragged
                self.events()
                self.update(delta_time)
                self.draw()
        self.cleanup()
        pg.quit()
        

# ------------------- SPRITES -------------------------------------------------

class Shape:
    """
    This class constructs a polygon from any number of given vectors
    and can rotate the points around its center
    """
    def __init__(self, game, points, static=False):
        self.game = game
        self.game.shapes.append(self)
        self.points = points
        self.center = self.find_center()
        self.overlap = False
        self.static = static       
        self.edges = [Line(self.points[i], self.points[i + 1]) 
                      for i in range(-1, len(self.points) - 1)]
        self.diagonals = [Line(self.center, p) for p in self.points]       
        self.rect = self.construct_rect()
        
    
    def update(self):          
        # construct edges and diagonals based on the new coordinates
        self.edges = [Line(self.points[i], self.points[i + 1]) 
                      for i in range(-1, len(self.points) - 1)]
        self.diagonals = [Line(self.center, p) for p in self.points]
        self.rect = self.construct_rect()
        
        '''
        # leave that off for now
        for shape in self.game.shapes:
            # check for collisions with the other shapes
            if shape != self and self.rect.colliderect(shape.rect):
                if self.shape_overlap(shape):
                    self.overlap = True
                    shape.overlap = True
        '''
    
    def rotate(self, angle):
        # rotate the edges around the shape's center
        for point in self.points:
            # translate the center to the origin
            rotate_point(point, self.center, angle)
    
    
    def draw(self, screen):
        # mostly for debugging
        if self.overlap: 
            color = RED
        else:
            color = WHITE
        # draw the shape of the polygon
        pg.draw.polygon(screen, color, 
                        [self.game.camera.apply_pos(x) for x in self.points], 2)
        # draw the center point of the shape
        x = int(self.game.camera.apply_pos(self.center)[0])
        y = int(self.game.camera.apply_pos(self.center)[1])
        pg.draw.circle(screen, RED, (x, y), 4)
        
        '''
        for diag in self.diagonals:
            diag.draw(screen, color)
        for edge in self.edges:
            edge.draw(screen, color)'''

        pg.draw.rect(screen, WHITE, self.game.camera.apply_rect(self.rect), 1)
        # reset the overlap flag after all collisions are checked
        # TODO: This should be in a "end update" function or something
        self.overlap = False
        
        
    def find_center(self):
        # calculate geometric center (centroid) as mean of all points
        # https://en.wikipedia.org/wiki/Centroid
        p_sum = vec()
        for p in self.points:
            p_sum += p
        return p_sum / len(self.points)
    
    
    def construct_rect(self):
        # builds the smallest AABB rectangle that contains all the points
        min_x = inf
        max_x = -inf
        min_y = inf
        max_y = -inf
        
        for point in self.points:
            min_x = min(min_x, point.x)
            max_x = max(max_x, point.x)
            min_y = min(min_y, point.y)
            max_y = max(max_y, point.y)
            
        width = max_x - min_x
        height = max_y - min_y
        return pg.Rect((min_x, min_y), (width, height))
    
    
    def move(self, amount):
        # move all points of this shape by a given vector
        self.center += amount
        for point in self.points:
                point += amount
    
    
    def move_to(self, position):
        # move the center to a given position and change all points accordingly
        # just a convenience function, could possibly be refactored
        old_center = self.center
        self.center = vec(position)
        amount = position - old_center
        for point in self.points:
                point += amount
                
    
    def shape_overlap(self, other):
        # https://github.com/OneLoneCoder/olcPixelGameEngine/blob/master/OneLoneCoder_PGE_PolygonCollisions1.cpp
        # check if the diagonals of this shape overlap any of the edges of 
        # the other shape. If true, move this shape's points by the
        # displacement vector that gets modified by the intersects_line function
        for diag in self.diagonals:
            for edge in other.edges:
                displacement = vec()
                if diag.intersects_line(edge, displacement):
                    self.move(displacement)
                    if not other.static:
                        other.move(displacement * -1)
                    return True
        return False
        


class Car(Shape):
    """
    Car class that can be controlled
    maybe make this a parent class for player and enemy cars as well
    """
    def __init__(self, game, color='red'):
        # construct a rectangle that lays on its long side, right side is the front of the car
        points = [vec(64, 0), vec(64, 32), vec(0, 32), vec(0, 0)]
        super().__init__(game, points)
        
        # move the center to simulate steering with front axis
        self.center.x += 20
        
        self.image_orig = pg.image.load(f'assets/Cars/car_{color}_1.png').convert_alpha()
        self.image_orig = pg.transform.rotate(self.image_orig, 270)
        self.image_orig = pg.transform.scale(self.image_orig, self.rect.size)
        self.image = self.image_orig.copy()
        # image for tire tracks
        self.tire_image = pg.image.load('assets/tires.png').convert_alpha()
        
        self.particle_timer = 0 # timer for emitting particles (tire tracks or dust etc)
             
        self.acc = vec()
        self.vel = vec()
        self.friction = 1 #the higher, the longer the friction vector
        self.rotation = 0
        self.speed = 700
        self.steer_sensitivity = 600 # the higher the less sensitive (600 for human)
        
        self.tile = None # the tile number the car is colliding with
        self.checkpoints = [] # list of checkpoints passed
        
        # list of Line objects for Raycasting
        self.sensors = []
        self.ray_len = 100 # length of cast rays
        end_1 = (self.points[1] - self.points[2]) * self.ray_len
        self.sensors.append(Line(self.center, self.center + end_1))
        end_2 = end_1.rotate(-45)
        self.sensors.append(Line(self.center, self.center + end_2))
        end_3 = end_1.rotate(45)
        self.sensors.append(Line(self.center, self.center + end_3))

        self.intersection_points = [None for i in range(3)]
        
        self.frames = 0
        
        # data structure for neural network
        self.training_data = {'label_names:': [
                                        'steer_left',
                                        'steer_right',
                                        'acc',
                                        'break'],
                        'feature_names': [              
                                        'sensor1',
                                        'sensor2',
                                        'sensor3',
                                        'vel'],
                        'labels': [], 
                        'features': []}

    
    def rotate(self, angle):
        super().rotate(angle)  
        self.rotation += angle
        # rotate the images accordingly
        self.image = pg.transform.rotate(self.image_orig, 
                                         self.rotation * -360 / TWO_PI)
    
    
    def update(self, dt):   
        self.frames += 1
        
        # check if the car is off road (tile based collision)
        grid = self.game.layer_data[1]
        if self.game.map_rect.collidepoint(self.center):
            grid_pos_x = int(self.center.x / TILESIZE)
            grid_pos_y = int(self.center.y / TILESIZE)
            #pg.display.set_caption(f'{grid[grid_pos_y][grid_pos_x]}')
            self.tile = grid[grid_pos_y][grid_pos_x]
            
            # set friction based on tile collision
            # TODO: set these to variables
        if self.tile == 0:
            self.friction = 4
        else:
            self.friction = 1

        # check for checkpoints
        for check in self.game.checkpoints:
            if (check not in self.checkpoints and 
                self.rect.colliderect(check)):
                self.checkpoints.append(check)
        # get user inputs
        keys = pg.key.get_pressed()
        rot = keys[pg.K_d] - keys[pg.K_a]
        # backwards is only half the speed
        move = keys[pg.K_w] - (keys[pg.K_s] * 0.5)
        
        # record control data 
        self.training_data['labels'].append([
                keys[pg.K_a],
                keys[pg.K_d],
                keys[pg.K_w],
                keys[pg.K_s]
                ])
            
        # rotate
        self.vel_len = self.vel.length()
        angle = pi * rot * (self.vel_len / self.steer_sensitivity) * dt # angle in radians
        angle_deg = self.rotation * 360 / TWO_PI # angle in degrees
        self.rotate(angle)
        # move
        # calculate acceleration vector and rotate it based on the shape's angle
        self.acc.x = move * self.speed
        self.acc = self.acc.rotate(angle_deg) 
        temp_accel = self.acc - self.friction * self.vel
        self.vel += temp_accel * dt
        vel_change = self.vel * dt
        self.move(vel_change)
        self.acc *= 0
        
        # create particles to simulate tire tracks
        self.particle_timer += dt
        if self.particle_timer >= 0.01 and self.vel.length() > 0.5:
            self.particle_timer = 0
            # calculate two points that simulate the position of the tires
            p1 = self.points[0] - 0.3 * (self.center - self.points[2])
            p2 = self.points[1] - 0.3 * (self.center - self.points[3])
            leng = max(2, abs(int(self.vel_len * 0.02)))

            Particle(self.game, self.tire_image, p1, -angle_deg, (leng, 6))
            Particle(self.game, self.tire_image, p2, -angle_deg, (leng, 6))

        self.cast_rays()
        

        d1 = self.intersection_points[0] - self.center
        d2 = self.intersection_points[1] - self.center
        d3 = self.intersection_points[2] - self.center
        
        self.training_data['features'].append([
                d1.length_squared(),
                d2.length_squared(),
                d3.length_squared(),
                self.vel_len
                ])

        # after movement, update the shape's points
        super().update()
    
    
    def draw(self, screen):
        screen.blit(self.image, self.game.camera.apply_pos(self.rect.topleft))
        if self.game.debug_mode:
            super().draw(screen)
            for s in self.sensors:
                if s:
                    s.draw(screen, camera=self.game.camera)
                    
    
    def calc_dist_center(self, displacement):
        d = displacement - self.center
        return d.length_squared()
            
    
    def cast_rays(self):
        # cast 3 Lines that work as "sensors"
        start = self.center
        # front
        end_1 = (self.points[1] - self.points[2]) * self.ray_len
        self.sensors[0] = Line(start, start + end_1)
        # left
        end_2 = end_1.rotate(-45)
        self.sensors[1] = Line(start, start + end_2)
        # right
        end_3 = end_1.rotate(45)
        self.sensors[2] = Line(start, start + end_3)
        
        self.intersection_points = [VEC_INF for i in range(3)]
        for i, sensor in enumerate(self.sensors):
            intersections = [] 
            for rect in self.game.offroad:
                lines, displacements = sensor.intersects_rect(rect)
                if lines and displacements:
                    intersections += displacements
            
            if len(intersections) >= 1:
                intersections.sort(key= lambda x: self.calc_dist_center(x), reverse=True)
                d = intersections[0]
                p = sensor.end + d
                self.intersection_points[i] = p



class Line:
    '''
    custom Line class that represents a line with a start and end vector
    and provides a method for intersection checking
    '''
    def __init__(self, start, end):
        self.start = vec(start)
        self.end = vec(end)
    
    
    def draw(self, screen, color=WHITE, width=1, camera=None):
        if camera:
            pg.draw.line(screen, color, camera.apply_pos(self.start), 
                         camera.apply_pos(self.end), width)
        else:
            pg.draw.line(screen, color, self.start, self.end, width)
        
        
    def intersects_line(self, other, displacement):
        # http://www.jeffreythompson.org/collision-detection/line-rect.php
        # check if two Line objects intersect
        # if true, change the displacement vector by the distance between
        # this line's end and the intersection
        denA = ((other.end.y - other.start.y) * (self.end.x - self.start.x) - 
                (other.end.x - other.start.x) * (self.end.y - self.start.y))
        denB = ((other.end.y - other.start.y) * (self.end.x - self.start.x) - 
                (other.end.x - other.start.x) * (self.end.y - self.start.y))
        if denA == 0 or denB == 0:
            return False
        else:
            numA = ((other.end.x - other.start.x) * (self.start.y - other.start.y) - 
                    (other.end.y - other.start.y) * (self.start.x - other.start.x))
            numB = ((self.end.x - self.start.x) * (self.start.y - other.start.y) - 
                    (self.end.y - self.start.y) * (self.start.x - other.start.x))
            uA = numA / denA
            uB = numB / denB
            if (uA >= 0 and uA <= 1 and uB >= 0 and uB <= 1):
                displacement.x -= (1.0 - uA) * (self.end.x - self.start.x)
                displacement.y -= (1.0 - uA) * (self.end.y - self.start.y)
                return True
            else:
                return False
            
            
    def get_lines_from_rect(self, rect):
        l1 = Line(rect.topleft, rect.topright)
        l2 = Line(rect.topright, rect.bottomright)
        l3 = Line(rect.bottomright, rect.bottomleft)
        l4 = Line(rect.bottomleft, rect.topleft)
        return [l1, l2, l3, l4]
    
    
    def intersects_rect(self, rect):
        lines = self.get_lines_from_rect(rect)
        lines_intersect = []
        displacements = []
        for line in lines:
            displacement = vec()
            if self.intersects_line(line, displacement):
                lines_intersect.append(line)
                displacements.append(displacement)
        return lines_intersect, displacements


    def rotate(self, angle):
        # rotate the line's end point around its start point
        rotate_point(self.start, self.end, angle)
    


class Particle(pg.sprite.Sprite):
    """
    simple sprite that fades out
    TODO: this should be a more complex base class with movement, randomness etc
    """
    def __init__(self, game, image, position, rotation, size=None):
        self.game = game
        super().__init__(self.game.particles)
        self.position = position
        self.image = image.copy()
        if size:
            self.image = pg.transform.scale(self.image, size)
        self.image = pg.transform.rotate(self.image, rotation)
        self.rect = self.image.get_rect()
        self.rect.center = self.position
        self.alpha = 40
    
    
    def update(self, dt):
        self.alpha -= dt * 10
        if self.alpha <= 0:
            self.game.particles.remove(self)
            return
    
        
    def draw(self, screen):
        blit_alpha(screen, self.image, self.game.camera.apply_rect(self.rect),
                   self.alpha)


# ------------------ MAIN --------------------------------------------------------
    
if __name__ == '__main__':
    try:
        g = Game()
        g.run()
    except:
        traceback.print_exc()
        pg.quit()