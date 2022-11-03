import pygame
from pygame.sprite import Sprite


class Bullet(Sprite):
    """管理子弹的类"""

    def __init__(self, ai_game):
        """在飞船位置创建一个子弹对象"""
        super().__init__()
        self.screen = ai_game.screen
        self.settings = ai_game.settings
        self.color = self.settings.bullet_color

        # 在（0，0）处创建一个表示子弹的矩形，然后设置正确的位置
        self.rect = pygame.Rect(0, 0, self.settings.bullet_width, self.settings.bullet_height)
        self.rect.midtop = ai_game.ship.rect.midtop

        # 将子弹位置以小数位置存储
        self.y = float(self.rect.y)

    def update(self):
        """向上移动子弹"""
        # 更新表示子弹位置的小数值
        self.y -= self.settings.bullet_speed
        # 更新子弹的位置
        self.rect.y = self.y

    def draw_bullet(self):
        """在屏幕上显示子弹"""
        pygame.draw.rect(self.screen, self.color, self.rect)
