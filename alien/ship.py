import pygame


class Ship:
    """飞船控制"""

    def __init__(self, ai_game):
        """初始化飞船并设置飞船位置"""
        self.screen = ai_game.screen
        self.settings = ai_game.settings
        self.screen_rect = ai_game.screen.get_rect()

        """加载飞船图像并获取外界矩形。"""
        self.image = pygame.image.load('image/ship.bmp')
        self.rect = self.image.get_rect()
        """对于每个飞船给它放在最底下中间"""
        self.rect.midbottom = self.screen_rect.midbottom
        """储存飞船的坐标参量的小数值"""
        self.x = float(self.rect.x)
        self.y = float(self.rect.y)
        """移动标志"""
        self.moving_right = False
        self.moving_left = False
        self.moving_up = False
        self.moving_down = False
        """屏幕边界参数"""

    def update(self):
        """根据标志移动上下左右"""
        if self.moving_right and self.rect.right < self.screen_rect.right:
            self.x += self.settings.ship_speed
            # 右下
            if self.moving_right and self.moving_down and self.rect.right < self.screen_rect.right and self.rect.bottom < self.screen_rect.bottom:
                self.y += self.settings.ship_speed
            # 右上
            if self.moving_right and self.moving_up and self.rect.right < self.screen_rect.right and self.rect.top > 0:
                self.y -= self.settings.ship_speed
        elif self.moving_left and self.rect.left > 0:
            self.x -= self.settings.ship_speed
            # 左下
            if self.moving_left and self.moving_down and self.rect.left > 0 and self.rect.bottom < self.screen_rect.bottom:
                self.y += self.settings.ship_speed
            # 左上
            if self.moving_left and self.moving_up and self.rect.left > 0 and self.rect.top > 0:
                self.y -= self.settings.ship_speed
        elif self.moving_up and self.rect.top > 0:
            self.y -= self.settings.ship_speed
            # 左上
            if self.moving_left and self.moving_up and self.rect.left > 0 and self.rect.top > 0:
                self.x -= self.settings.ship_speed
            # 右上
            if self.moving_right and self.moving_up and self.rect.right < self.screen_rect.right and self.rect.top > 0:
                self.x += self.settings.ship_speed
        elif self.moving_down and self.rect.bottom < self.screen_rect.bottom:
            self.y += self.settings.ship_speed
            # 左下
            if self.moving_left and self.moving_down and self.rect.left > 0 and self.rect.bottom < self.screen_rect.bottom:
                self.x -= self.settings.ship_speed
            # 右下
            if self.moving_right and self.moving_down and self.rect.right < self.screen_rect.right and self.rect.bottom < self.screen_rect.bottom:
                self.x += self.settings.ship_speed
        self.rect.x = self.x
        self.rect.y = self.y

    def blitme(self):
        """在该区域绘制"""
        self.screen.blit(self.image, self.rect)

    def center_ship(self):
        """让旅行者在屏幕底端居中"""
        self.rect.midbottom = self.screen_rect.midbottom
        self.x = float(self.rect.x)
