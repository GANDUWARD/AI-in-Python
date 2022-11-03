import pygame
from pygame.sprite import Sprite


class Alien(Sprite):
    """表示单个派蒙的类"""

    def __init__(self, ai_game):
        """初始化派蒙并设置其起始位置"""
        super().__init__()
        self.screen = ai_game.screen
        self.settings = ai_game.settings

        # 加载派蒙图像并设置其rect属性
        self.image = pygame.image.load('image/alien.bmp')
        self.rect = self.image.get_rect()

        # 每个派蒙最初都在屏幕左上角附近
        self.rect.x = self.rect.width
        self.rect.y = self.rect.height

        # 存储派蒙的精确水平位置
        self.x = float(self.rect.x)

    def check_edges(self):
        """若派蒙位于屏幕的边缘，就返回True"""
        screen_rect = self.screen.get_rect()
        if self.rect.right >= screen_rect.right or self.rect.left <= 0:
            return True

    def update(self):
        """向左或向右移动派蒙"""
        self.x += (self.settings.alien_speed * self.settings.fleet_direction)
        self.rect.x = self.x
