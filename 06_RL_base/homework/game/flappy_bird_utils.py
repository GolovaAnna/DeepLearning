import random
import pygame

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'D:/05_Attention/12_RL_base/homework/assets/sprites/redbird-upflap.png',
        'D:/05_Attention/12_RL_base/homework/assets/sprites/redbird-midflap.png',
        'D:/05_Attention/12_RL_base/homework/assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        'D:/05_Attention/12_RL_base/homework/assets/sprites/bluebird-upflap.png',
        'D:/05_Attention/12_RL_base/homework/assets/sprites/bluebird-midflap.png',
        'D:/05_Attention/12_RL_base/homework/assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'D:/05_Attention/12_RL_base/homework/assets/sprites/yellowbird-upflap.png',
        'D:/05_Attention/12_RL_base/homework/assets/sprites/yellowbird-midflap.png',
        'D:/05_Attention/12_RL_base/homework/assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'D:/05_Attention/12_RL_base/homework/assets/sprites/background-day.png',
    'D:/05_Attention/12_RL_base/homework/assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'D:/05_Attention/12_RL_base/homework/assets/sprites/pipe-green.png',
    'D:/05_Attention/12_RL_base/homework/assets/sprites/pipe-red.png',
)


def load():
    # path of player with different states
    PLAYER_PATH = (
        'D:/05_Attention/12_RL_base/homework/assets/sprites/redbird-upflap.png',
        'D:/05_Attention/12_RL_base/homework/assets/sprites/redbird-midflap.png',
        'D:/05_Attention/12_RL_base/homework/assets/sprites/redbird-downflap.png'
    )

    # path of pipe
    PIPE_PATH = 'D:/05_Attention/12_RL_base/homework/assets/sprites/pipe-green.png'

    IMAGES, HITMASKS = {}, {}

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('D:/05_Attention/12_RL_base/homework/assets/sprites/0.png').convert_alpha(),
        pygame.image.load('D:/05_Attention/12_RL_base/homework/assets/sprites/1.png').convert_alpha(),
        pygame.image.load('D:/05_Attention/12_RL_base/homework/assets/sprites/2.png').convert_alpha(),
        pygame.image.load('D:/05_Attention/12_RL_base/homework/assets/sprites/3.png').convert_alpha(),
        pygame.image.load('D:/05_Attention/12_RL_base/homework/assets/sprites/4.png').convert_alpha(),
        pygame.image.load('D:/05_Attention/12_RL_base/homework/assets/sprites/5.png').convert_alpha(),
        pygame.image.load('D:/05_Attention/12_RL_base/homework/assets/sprites/6.png').convert_alpha(),
        pygame.image.load('D:/05_Attention/12_RL_base/homework/assets/sprites/7.png').convert_alpha(),
        pygame.image.load('D:/05_Attention/12_RL_base/homework/assets/sprites/8.png').convert_alpha(),
        pygame.image.load('D:/05_Attention/12_RL_base/homework/assets/sprites/9.png').convert_alpha()
    )

    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('D:/05_Attention/12_RL_base/homework/assets/sprites/base.png').convert_alpha()

    # select random background sprites
    randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
    IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

    # select random player sprites
    randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
    IMAGES['player'] = (
        pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
    )

    # select random pipe sprites
    pipeindex = random.randint(0, len(PIPES_LIST) - 1)
    IMAGES['pipe'] = (
        pygame.transform.flip(
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), False, True),
        pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
    )

    # hismask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    return IMAGES, HITMASKS


def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask
