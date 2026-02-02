# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass

import numpy as np
import pygame


@dataclass
class KeyboardConfig:
    """dataclass holding keyboard configuration data"""

    forward_key: int = pygame.K_w  # key for forward movement
    backward_key: int = pygame.K_s  # key for backward movement
    left_key: int = pygame.K_a  # key for left movement
    right_key: int = pygame.K_d  # key for right movement
    yaw_left_key: int = pygame.K_q  # key for yaw left
    yaw_right_key: int = pygame.K_e  # key for yaw right
    delta_forward_velocity: float = 0.1  # increment for forward/backward velocity
    delta_lateral_velocity: float = 0.1  # increment for lateral velocity
    delta_yaw_velocity: float = 0.1  # increment for yaw velocity
    max_forward_velocity: float = 1.0  # maximum forward velocity
    max_backward_velocity: float = 1.0  # maximum backward velocity
    max_lateral_velocity: float = 1.0  # maximum lateral velocity
    max_yaw_velocity: float = 1.0  # maximum yaw velocity
    stop_key: int = pygame.K_SPACE  # key to stop the controller


class Keyboard:

    x_vel = 0.0
    y_vel = 0.0
    yaw = 0.0
    _stopping = False
   

    def __init__(self, context, config: KeyboardConfig = None, verbose: bool = False):
        if not pygame.get_init():
            pygame.init()
        
       
        self._context = context
        self._config = config if config is not None else KeyboardConfig()
        self._verbose = verbose

        self._create_main_window()

    def _create_main_window(self):
        self._screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Keyboard Control - Keep this window focused")
        # Font for rendering text
        self._font: pygame.font.Font = pygame.font.Font(None, 36)
        self._small_font: pygame.font.Font = pygame.font.Font(None, 24)

        controls_title = self._small_font.render("Controls:", True, (255, 255, 255))
        self._screen.blit(controls_title, (250, 20))

        self._update_display()

    def _update_display(self):

        self._screen.fill((30, 30, 30))
        
        # Render velocity information
        title_text = self._font.render("Velocity Status", True, (255, 255, 255))
        self._screen.blit(title_text, (20, 20))

        x_text = self._small_font.render(f"X (Forward/Back): {self.x_vel:.2f}", True, (100, 255, 100))
        y_text = self._small_font.render(f"Y (Left/Right):   {self.y_vel:.2f}", True, (100, 200, 255))
        yaw_text = self._small_font.render(f"Yaw (Rotation):   {self.yaw:.2f}", True, (255, 200, 100))
        
        self._screen.blit(x_text, (20, 80))
        self._screen.blit(y_text, (20, 120))
        self._screen.blit(yaw_text, (20, 160))
        key_map = {
            "Fwd": self._config.forward_key,
            "Back": self._config.backward_key,
            "Left": self._config.left_key,
            "Right": self._config.right_key,
            "Yaw L": self._config.yaw_left_key,
            "Yaw R": self._config.yaw_right_key,
            "Stop": self._config.stop_key,
        }

        y_offset = 50
        for action, key in key_map.items():
            key_name = pygame.key.name(key).upper()
            text = self._small_font.render(f"{action}: {key_name}", True, (200, 200, 200))
            self._screen.blit(text, (250, y_offset))
            y_offset += 30
        
        # Update display
        pygame.display.flip()

    def listen(self):
        clock = pygame.time.Clock()
        while not self._stopping:
            # Update inputs at 10hz
            self.listen_loop()
            clock.tick(10)

    def listen_loop(self):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._stopping = True

            # Debugging: Print exactly what Pygame receives when a key goes DOWN
            if event.type == pygame.KEYDOWN and self._verbose:
                key_name = pygame.key.name(event.key).upper()
                print(f"[DEBUG] Key Pressed: {key_name} (Code: {event.key})")
                
        # Get current key states
        keys = pygame.key.get_pressed()

        # Calculate target velocities based on key presses
        self._target_x_vel = 0.0
        self._target_y_vel = 0.0
        self._target_yaw = 0.0

        # Forward/Backward
        if keys[self._config.forward_key]:
            self.x_vel += self._config.delta_forward_velocity
        if keys[self._config.backward_key]:
            self.x_vel -= self._config.delta_forward_velocity

        # Left/Right (lateral)
        if keys[self._config.left_key]:
            self.y_vel += self._config.delta_lateral_velocity
        if keys[self._config.right_key]:
            self.y_vel -= self._config.delta_lateral_velocity

        # Yaw
        if keys[self._config.yaw_left_key]:
            self.yaw += self._config.delta_yaw_velocity
        if keys[self._config.yaw_right_key]:
            self.yaw -= self._config.delta_yaw_velocity

        if keys[self._config.stop_key]:
            self.x_vel = 0.0
            self.y_vel = 0.0
            self.yaw = 0.0

        # Cap velocities
        self.x_vel = np.clip(self.x_vel, -self._config.max_backward_velocity, self._config.max_forward_velocity)
        self.y_vel = np.clip(self.y_vel, -self._config.max_lateral_velocity, self._config.max_lateral_velocity)
        self.yaw = np.clip(self.yaw, -self._config.max_yaw_velocity, self._config.max_yaw_velocity)
        
        self._context.velocity_cmd = [self.x_vel, self.y_vel, self.yaw]
        
        # Clear screen
        self._update_display()

