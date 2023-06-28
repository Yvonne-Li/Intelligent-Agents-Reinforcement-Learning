import random

class Coords:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"[{self.x},{self.y}]"

class Percept:
    def __init__(self, stench, breeze, glitter, bump, scream):
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

class Agent:
    def __init__(self):
        self.orientation = Orientation("right")
        self.has_gold = False
        self.has_arrow = True
        self.score = 0

    def perceive(self, environment, current_location):
        x, y = current_location.x, current_location.y
        stench = environment.has_wumpus(x, y) or \
                 environment.has_wumpus(x-1, y) or \
                 environment.has_wumpus(x+1, y) or \
                 environment.has_wumpus(x, y-1) or \
                 environment.has_wumpus(x, y+1)

        breeze = environment.has_pit(x-1, y) or \
                 environment.has_pit(x+1, y) or \
                 environment.has_pit(x, y-1) or \
                 environment.has_pit(x, y+1)

        glitter = environment.has_gold(x, y)

        bump = environment.bumped

        scream = environment.scream

        return Percept(stench, breeze, glitter, bump, scream)

    def choose_action(self):
        possible_actions = ["Forward", "TurnLeft", "TurnRight", "Shoot", "Grab", "Climb"]
        return random.choice(possible_actions)

    def update_score(self, score_change):
        self.score += score_change

class Environment:
    def __init__(self):
        self.grid_size = 4
        self.agent_location = Coords(1, 1)
        self.wumpus_location = self.generate_random_location()
        self.pit_locations = self.generate_random_locations(0.2)
        self.gold_location = self.generate_random_location()
        self.agent_orientation = Orientation(direction="right")  # Provide a default direction argument

        self.agent = None
        self.bumped = False
        self.scream = False

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

# Test the environment with the NaiveAgent
simulator = Environment()
score = simulator.run_simulation()
print(f"Final Score: {score}")
