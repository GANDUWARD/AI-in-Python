import pygame


class Settings:
    """保存游戏中的所有设置"""

    def __init__(self):
        """屏幕设置"""
        self.screen_width = 1200
        self.screen_height = 800
        self.bg_color = (230, 230, 230)
        self.bg_image = pygame.image.load('image/alien.bmp')
        """飞船设置"""
        self.ship_speed = 1.5
        self.ship_limit = 3
        """子弹设置"""
        self.bullet_speed = 2.0
        self.bullet_width = 8
        self.bullet_height = 15
        self.bullet_color = (60, 60, 60)
        self.bullets_allowed = 6
        """派蒙设置"""
        self.alien_speed = 1.0
        self.fleet_drop_speed = 10
        # fleet_direction为1表示向右移动，为-1表示向左移动
        self.fleet_direction = 1
