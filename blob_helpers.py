import math
import numpy as np
import cv2

import math
import numpy as np
import cv2

class floatBlob:
    def __init__(self, pos, vel, mass):
        self.pos = pos
        self.vel = vel
        self.mass = mass

    def attract(self, attractor):
        # Distance between objects - scaled by 5 because movement looks better for a window of selected size
        dist = math.sqrt((self.pos[0] - attractor.pos[0])**2 + (self.pos[1] - attractor.pos[1])**2)/2
        if dist < 20:
            dist = 20

        # Force of attraction between objets
        f = (attractor.mass * self.mass) / (dist**2)

        # Vector from attractor to self
        dir = np.subtract(attractor.pos, self.pos)
        # Normalize and then scale by force (https://math.stackexchange.com/questions/1347328/how-to-scale-a-2d-vector-and-keep-direction)
        dir = dir / dist * f

        # Update velocity based on attraction
        max_vel = 8
        self.vel = self.vel + dir
        for i in range(len(self.vel)):
            if self.vel[i] > max_vel:
                self.vel[i] = max_vel
            elif self.vel[i] < -max_vel:
                self.vel[i] = -max_vel

        # Update position
        self.pos = self.pos + self.vel
        self.pos = tuple(map(int, self.pos))

    def draw(self, frame):
        return cv2.circle(frame, self.pos, 20, (0, 0, 0), -1)

class attractor:
    def __init__(self, pos, mass):
        self.pos = (int(pos[0]), int(pos[1]))
        self.mass = mass

    def draw(self, frame):
       return cv2.circle(frame, self.pos, self.mass, (0, 0, 255), -1)


