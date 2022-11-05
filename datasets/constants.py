from enum import Enum


class EyePacsEnv(Enum):
    LEFT_NREF = 0
    LEFT_REF = 1
    RIGHT_NREF = 2
    RIGHT_REF = 3


EYEPACS_ENV_MAP = {
    'left_nref': EyePacsEnv.LEFT_NREF,
    'left_ref': EyePacsEnv.LEFT_REF,
    'right_nref': EyePacsEnv.RIGHT_NREF,
    'right_ref': EyePacsEnv.RIGHT_REF
}
