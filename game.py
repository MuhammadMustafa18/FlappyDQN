import math
import random
import pygame
import numpy as np

# Window dimensions
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
PIPE_GAP = 100
PIPE_WIDTH = 52
BIRD_WIDTH = 34
BIRD_HEIGHT = 24
GRAVITY = 0.8
FLAP_STRENGTH = -8.0
PIPE_SPEED = 4

# Number of LIDAR rays cast from bird (like in frame.png)
NUM_RAYS = 15
# Ray angles from -75 degrees (up) to +75 degrees (down)
RAY_ANGLES = np.linspace(-math.pi / 2.5, math.pi / 2.5, NUM_RAYS)
MAX_RAY_DIST = 350.0


class FlappyBirdEnv:
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        if self.render_mode:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird AI - DQN")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 22, bold=True)

        self.reset()

    def reset(self):
        self.bird_x = 50.0
        self.bird_y = float(SCREEN_HEIGHT // 2)
        self.bird_vel = 0.0

        self.pipes = []
        self._spawn_pipe(SCREEN_WIDTH + 50)
        self._spawn_pipe(SCREEN_WIDTH + 50 + 200)

        self.score = 0
        self.frame_count = 0
        self.ray_hits = []
        self.ray_distances = [MAX_RAY_DIST] * NUM_RAYS

        return self._get_state()

    def _spawn_pipe(self, x_pos):
        # Gap center between 120 and 380
        gap_y = random.randint(120, SCREEN_HEIGHT - 120)
        top_pipe = pygame.Rect(x_pos, 0, PIPE_WIDTH, gap_y - PIPE_GAP // 2)
        bottom_pipe = pygame.Rect(
            x_pos,
            gap_y + PIPE_GAP // 2,
            PIPE_WIDTH,
            SCREEN_HEIGHT - (gap_y + PIPE_GAP // 2),
        )
        self.pipes.append({
            "top": top_pipe,
            "bottom": bottom_pipe,
            "gap_y": gap_y,
            "passed": False
        })

    def _cast_rays(self):
        bird_center = (self.bird_x + BIRD_WIDTH / 2, self.bird_y + BIRD_HEIGHT / 2)
        ray_distances = []
        ray_hits = []

        # Obstacles: pipes + ceiling + floor
        pipe_rects = []
        for p in self.pipes:
            pipe_rects.append(p["top"])
            pipe_rects.append(p["bottom"])

        floor_rect = pygame.Rect(0, SCREEN_HEIGHT - 20, SCREEN_WIDTH, 40)
        ceiling_rect = pygame.Rect(0, -20, SCREEN_WIDTH, 20)
        all_obstacles = pipe_rects + [floor_rect, ceiling_rect]

        for angle in RAY_ANGLES:
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            closest_dist = MAX_RAY_DIST
            hit_pos = (
                bird_center[0] + cos_a * MAX_RAY_DIST,
                bird_center[1] + sin_a * MAX_RAY_DIST
            )

            # Check ray intersection with obstacles
            for rect in all_obstacles:
                clipped = rect.clipline(
                    bird_center,
                    (bird_center[0] + cos_a * MAX_RAY_DIST, bird_center[1] + sin_a * MAX_RAY_DIST)
                )
                if clipped:
                    p1, _ = clipped
                    dist = math.hypot(p1[0] - bird_center[0], p1[1] - bird_center[1])
                    if dist < closest_dist:
                        closest_dist = dist
                        hit_pos = p1

            ray_distances.append(closest_dist / MAX_RAY_DIST)  # Normalize [0, 1]
            ray_hits.append(hit_pos)

        self.ray_hits = ray_hits
        self.ray_distances = ray_distances
        return ray_distances

    def _get_state(self):
        # 1. Cast LIDAR rays
        ray_dists = self._cast_rays()

        # 2. Upcoming pipe info
        next_pipe = None
        for p in self.pipes:
            if p["top"].right > self.bird_x:
                next_pipe = p
                break

        if next_pipe is None:
            dx = 1.0
            dy_gap = 0.0
        else:
            dx = (next_pipe["top"].centerx - self.bird_x) / SCREEN_WIDTH
            dy_gap = (next_pipe["gap_y"] - self.bird_y) / SCREEN_HEIGHT

        norm_bird_y = self.bird_y / SCREEN_HEIGHT
        norm_vel = self.bird_vel / 15.0  # normalize velocity

        # State representation: [ray_distances (15), bird_y, bird_vel, dist_to_pipe_x, diff_to_gap_y] -> total 19 features
        state = np.array(ray_dists + [norm_bird_y, norm_vel, dx, dy_gap], dtype=np.float32)
        return state

    def step(self, action):
        self.frame_count += 1

        # Action: 0 = Do nothing, 1 = Flap
        if action == 1:
            self.bird_vel = FLAP_STRENGTH

        # Apply gravity
        self.bird_vel += GRAVITY
        if self.bird_vel > 12.0:
            self.bird_vel = 12.0
        self.bird_y += self.bird_vel

        # Move pipes
        for p in self.pipes:
            p["top"].x -= PIPE_SPEED
            p["bottom"].x -= PIPE_SPEED

        # Remove offscreen pipes and spawn new ones
        if self.pipes and self.pipes[0]["top"].right < 0:
            self.pipes.pop(0)

        if self.pipes[-1]["top"].centerx < SCREEN_WIDTH - 150:
            self._spawn_pipe(SCREEN_WIDTH + 40)

        # Collision detection
        bird_rect = pygame.Rect(int(self.bird_x), int(self.bird_y), BIRD_WIDTH, BIRD_HEIGHT)
        done = False
        reward = 0.1  # Survival reward

        # Floor / Ceiling collision
        if self.bird_y <= 0 or self.bird_y + BIRD_HEIGHT >= SCREEN_HEIGHT - 20:
            done = True
            reward = -5.0

        # Pipe collision & passing check
        for p in self.pipes:
            if bird_rect.colliderect(p["top"]) or bird_rect.colliderect(p["bottom"]):
                done = True
                reward = -5.0
                break

            if not p["passed"] and p["top"].centerx < self.bird_x:
                p["passed"] = True
                self.score += 1
                reward = 10.0  # Big reward for passing a pipe!

        # Next state
        next_state = self._get_state()

        # Ray proximity penalty (encourages staying away from edges/pipes)
        min_ray = min(self.ray_distances)
        if min_ray < 0.1 and not done:
            reward -= 0.2

        if self.render_mode:
            self.render()

        return next_state, reward, done, {"score": self.score}

    def render(self):
        if self.screen is None:
            return

        # Handle Pygame events so window doesn't freeze
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Background (Sky blue)
        self.screen.fill((78, 192, 202))

        # Draw pipes (Classic Flappy Bird green)
        for p in self.pipes:
            pygame.draw.rect(self.screen, (115, 191, 46), p["top"])
            pygame.draw.rect(self.screen, (83, 142, 33), p["top"], 3)  # Border
            pygame.draw.rect(self.screen, (115, 191, 46), p["bottom"])
            pygame.draw.rect(self.screen, (83, 142, 33), p["bottom"], 3)

        # Draw ground
        pygame.draw.rect(self.screen, (222, 216, 149), (0, SCREEN_HEIGHT - 20, SCREEN_WIDTH, 20))
        pygame.draw.line(self.screen, (115, 191, 46), (0, SCREEN_HEIGHT - 20), (SCREEN_WIDTH, SCREEN_HEIGHT - 20), 4)

        # Draw red LIDAR rays (from frame.png)
        bird_center = (int(self.bird_x + BIRD_WIDTH / 2), int(self.bird_y + BIRD_HEIGHT / 2))
        for hit in self.ray_hits:
            pygame.draw.line(self.screen, (255, 60, 60), bird_center, (int(hit[0]), int(hit[1])), 1)

        # Draw Bird (Yellow body + eye)
        bird_rect = pygame.Rect(int(self.bird_x), int(self.bird_y), BIRD_WIDTH, BIRD_HEIGHT)
        pygame.draw.ellipse(self.screen, (244, 180, 26), bird_rect)
        pygame.draw.ellipse(self.screen, (200, 140, 20), bird_rect, 2)
        # Eye
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.bird_x + 24), int(self.bird_y + 8)), 5)
        pygame.draw.circle(self.screen, (0, 0, 0), (int(self.bird_x + 26), int(self.bird_y + 8)), 2)

        # Draw Score
        score_surf = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_surf, (SCREEN_WIDTH // 2 - score_surf.get_width() // 2, 25))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.render_mode and pygame.get_init():
            pygame.quit()
