#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:34:45 2023

@author: lyonnce
"""

import random
import numpy as np
import networkx as nx
import heapq
from enum import Enum


class Action(Enum):
    FORWARD = "Forward"
    TURN_LEFT = "TurnLeft"
    TURN_RIGHT = "TurnRight"
    SHOOT = "Shoot"
    GRAB = "Grab"
    CLIMB = "Climb"

class Coords:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"[{self.x},{self.y}]"

class Percept:
    def __init__(self, stench=False, breeze=False, glitter=False, bump=False, scream=False):
        self.stench = stench
        self.breeze = breeze
        self.glitter = glitter
        self.bump = bump
        self.scream = scream

class Orientation:
    def __init__(self, direction="right"):
        self.direction = direction

    def turn_left(self):
        if self.direction == "right":
            self.direction = "up"
        elif self.direction == "up":
            self.direction = "left"
        elif self.direction == "left":
            self.direction = "down"
        elif self.direction == "down":
            self.direction = "right"

    def turn_right(self):
        if self.direction == "right":
            self.direction = "down"
        elif self.direction == "down":
            self.direction = "left"
        elif self.direction == "left":
            self.direction = "up"
        elif self.direction == "up":
            self.direction = "right"


class Environment:
    def __init__(self):
        self.grid_size = 4
        self.agent_location = Coords(1, 1)
        self.wumpus_location = self.generate_random_location()
        self.pit_locations = self.generate_random_locations(0.2)
        self.gold_location = self.generate_random_location()
        self.agent_orientation = Orientation(direction="right")  # Provide a default direction argument
        self.agent_arrows = 1
        self.agent = None
        self.bumped = False
        self.scream = False


    def display_board(self):
        print("-" * 19)
        for row in self.board:
            print("|", end="")
            for col in row:
                if col == "":
                    col = " "
                print(f" {col} |", end="")
            print("\n" + "-" * 19)

    def display_agent_location(self, agent_location):
        self.board = np.zeros((4, 4), dtype=str)
        self.board[agent_location.y - 1, agent_location.x - 1] = "A"
        self.display_board()

    def display_percept(self, percept):
        print("Percept:")
        print("Stench:", percept.stench)
        print("Breeze:", percept.breeze)
        print("Glitter:", percept.glitter)
        print("Bump:", percept.bump)
        print()
    
    def get_forward_location(self, location, orientation):
        x, y = location.x, location.y

        if orientation == Orientation.UP:
            return Coords(x, y + 1)
        elif orientation == Orientation.DOWN:
            return Coords(x, y - 1)
        elif orientation == Orientation.LEFT:
            return Coords(x - 1, y)
        elif orientation == Orientation.RIGHT:
            return Coords(x + 1, y)

    def apply_action(self, agent, action):
        if action == Action.TURN_LEFT:
            self.agent_orientation.turn_left()
        elif action == Action.TURN_RIGHT:
            self.agent_orientation.turn_right()
        elif action == Action.FORWARD:
            new_location = self.get_forward_location()
            if self.is_valid_location(new_location):
                self.agent_location = new_location
        elif action == Action.SHOOT:
            if self.agent_arrows > 0:
                self.agent_arrows -= 1
                self.shoot_arrow()
        elif action == Action.GRAB:
            if self.agent_location == self.gold_location:
                self.has_gold = True
        elif action == Action.CLIMB:
            if self.agent_location == Coords(1, 1) and self.has_gold:
                self.agent_location = None  # Agent has climbed out of the cave
        else:
            raise ValueError("Invalid action")
    
        percept = self.get_percept(self.agent_location)
        return percept
    
    def get_percept(self, agent_location):
        if agent_location == self.wumpus_location or self.is_adjacent(agent_location, self.wumpus_location):
            stench = True
        else:
                stench = False

        if agent_location == self.gold_location:
            glitter = True
        else:
            glitter = False

        if self.is_adjacent_pit(agent_location):
            breeze = True
        else:
            breeze = False

        percept = Percept(stench=stench, breeze=breeze, glitter=glitter)
        return percept
    
    def is_adjacent(self, loc1, loc2):
        return (abs(loc1.x - loc2.x) == 1 and loc1.y == loc2.y) or (
            abs(loc1.y - loc2.y) == 1 and loc1.x == loc2.x
        )
    def is_pit(self, location):
        return location in self.pit_locations    
    def is_valid_location(self, location):
        return (
            1 <= location.x <= self.grid_size
            and 1 <= location.y <= self.grid_size
        )        
    def is_adjacent_pit(self, location):
        adjacent_locations = [
            Coords(location.x + 1, location.y),
            Coords(location.x - 1, location.y),
            Coords(location.x, location.y + 1),
            Coords(location.x, location.y - 1),
        ]
        for adj_location in adjacent_locations:
            if self.is_valid_location(adj_location) and self.is_pit(adj_location):
                return True
        return False
    
    def display_game_result(self, agent_location, has_gold):
        if has_gold and agent_location == Coords(1, 1):
            print("Congratulations! You climbed out of the cave with the gold!")
        else:
            print("Game Over!")

    def generate_random_location(self):
        x = random.randint(1, self.grid_size)
        y = random.randint(1, self.grid_size)
        return Coords(x, y)

    def generate_random_locations(self, probability):
        locations = []
        for x in range(1, self.grid_size + 1):
            for y in range(1, self.grid_size + 1):
                if random.random() < probability:
                    locations.append(Coords(x, y))
        return locations

    def has_wumpus(self, x, y):
        return self.wumpus_location is not None and self.wumpus_location.x == x and self.wumpus_location.y == y

    def has_pit(self, x, y):
        return any(pit.x == x and pit.y == y for pit in self.pit_locations)

    def has_gold(self, x, y):
        return self.gold_location is not None and self.gold_location.x == x and self.gold_location.y == y

    def move_forward(self):
        x, y = self.agent_location.x, self.agent_location.y
        if self.agent_orientation.direction == "right" and x < self.grid_size:
            self.agent_location.x += 1
        elif self.agent_orientation.direction == "up" and y < self.grid_size:
            self.agent_location.y += 1
        elif self.agent_orientation.direction == "left" and x > 1:
            self.agent_location.x -= 1
        elif self.agent_orientation.direction == "down" and y > 1:
            self.agent_location.y -= 1
        else:
            self.bumped = True

    def turn_left(self):
        self.agent_orientation.turn_left()

    def turn_right(self):
        self.agent_orientation.turn_right()

    def grab_gold(self):
        if self.has_gold(self.agent_location.x, self.agent_location.y):
            self.gold_location = None

    def shoot_arrow(self):
        if self.agent.has_arrow:
            self.agent.has_arrow = False
            if self.agent_orientation.direction == "right":
                for x in range(self.agent_location.x + 1, self.grid_size + 1):
                    if self.has_wumpus(x, self.agent_location.y):
                        self.wumpus_location = None
                        self.scream = True
                        break
                    elif self.has_pit(x, self.agent_location.y):
                        break
            elif self.agent_orientation.direction == "up":
                for y in range(self.agent_location.y + 1, self.grid_size + 1):
                    if self.has_wumpus(self.agent_location.x, y):
                        self.wumpus_location = None
                        self.scream = True
                        break
                    elif self.has_pit(self.agent_location.x, y):
                        break
            elif self.agent_orientation.direction == "left":
                for x in range(self.agent_location.x - 1, 0, -1):
                    if self.has_wumpus(x, self.agent_location.y):
                        self.wumpus_location = None
                        self.scream = True
                        break
                    elif self.has_pit(x, self.agent_location.y):
                        break
            elif self.agent_orientation.direction == "down":
                for y in range(self.agent_location.y - 1, 0, -1):
                    if self.has_wumpus(self.agent_location.x, y):
                        self.wumpus_location = None
                        self.scream = True
                        break
                    elif self.has_pit(self.agent_location.x, y):
                        break

    def climb_out(self):
        if self.agent_location.x == 1 and self.agent_location.y == 1:
            self.agent.score += 1000

    def run_simulation(self):
        self.agent = Agent()
        self.agent_orientation = self.agent.orientation

        while True:
            current_location = self.agent_location
            percept = self.agent.perceive(self, current_location)
            action = self.agent.choose_action()

            if action == "Forward":
                self.move_forward()
                self.agent.update_score(-1)
            elif action == "TurnLeft":
                self.turn_left()
                self.agent.update_score(-1)
            elif action == "TurnRight":
                self.turn_right()
                self.agent.update_score(-1)
            elif action == "Shoot":
                self.shoot_arrow()
                self.agent.update_score(-10)
            elif action == "Grab":
                self.grab_gold()
                self.agent.update_score(-1)
                if self.agent.has_gold:
                    self.agent.update_score(1000)
            elif action == "Climb":
                self.climb_out()
                self.agent.update_score(-1)
                if self.agent.score == 1000:
                    break

        return self.agent.score

class BeelineAgent:
    def __init__(self, environment):
        self.environment = environment
        self.agent_location = Coords(1, 1)
        self.agent_orientation = Orientation()
        self.has_gold = False
        self.safe_locations = set()
        self.graph = None
        self.path_to_home = None
        super().__init__(environment)
        self.visited_locations = set()

    def update_visited_locations(self, location):
        self.visited_locations.add(location)
    def choose_action(self, percept):
        if percept is None:
            return Action.TURN_RIGHT  # Default action when no percept is available

        if percept.glitter and not self.has_gold:
            return Action.GRAB
        elif self.has_gold and self.agent_location == Coords(1, 1):
            return Action.CLIMB
        else:
            return random.choice([Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.SHOOT])

    def update_state(self, action, percept):
        if action == Action.FORWARD and not percept.bump:
            self.agent_location = self.get_forward_location()
            self.visited_locations.add(self.agent_location)
        elif action == Action.TURN_LEFT:
            self.agent_orientation.turn_left()
        elif action == Action.TURN_RIGHT:
            self.agent_orientation.turn_right()

        if self.agent_location not in self.visited_locations:
            self.safe_locations.add(self.agent_location)

    def get_forward_location(self):
        if self.agent_orientation.direction == "right":
            return Coords(self.agent_location.x + 1, self.agent_location.y)
        elif self.agent_orientation.direction == "up":
            return Coords(self.agent_location.x, self.agent_location.y + 1)
        elif self.agent_orientation.direction == "left":
            return Coords(self.agent_location.x - 1, self.agent_location.y)
        elif self.agent_orientation.direction == "down":
            return Coords(self.agent_location.x, self.agent_location.y - 1)

    def find_escape_plan(self):
        G = self.build_graph()
        return nx.astar_path(G, self.agent_location, Coords(1, 1), self.manhattan_distance)

    def build_graph(self):
        G = nx.DiGraph()
        G.add_node(Coords(1, 1))  # Add starting node
        G.add_node(self.agent_location)  # Add current location
        G.add_edge(self.agent_location, Coords(1, 1))  # Add edge from current location to (1,1)

        for location in self.safe_locations:
            if location != self.agent_location and location != Coords(1, 1):
                G.add_node(location)

        return G

    def manhattan_distance(self, node1, node2):
        return abs(node1.x - node2.x) + abs(node1.y - node2.y)

    def get_action_to_location(self, location):
        if location == self.get_forward_location():
            return Action.FORWARD
        elif location == self.agent_location:
            return Action.TURN_LEFT
        else:
            return Action.TURN_RIGHT


# Update the main loop of the program
environment = Environment()
agent = BeelineAgent(environment)
percept = environment.get_percept(agent.agent_location)
action = agent.choose_action(percept)

while True:
    environment.display_agent_location(agent.agent_location)
    environment.display_percept(percept)

    if action == Action.CLIMB and agent.has_gold:
        break

    environment.apply_action(agent, action)
    percept = environment.get_percept(agent.agent_location)
    agent.update_state(action, percept)

    action = agent.choose_action(percept)

environment.display_agent_location(agent.agent_location)
environment.display_percept(percept)
environment.display_game_result(agent.agent_location, agent.has_gold)