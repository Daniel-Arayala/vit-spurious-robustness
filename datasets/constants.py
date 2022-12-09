from enum import Enum


class EyePacsEnv(Enum):
    LEFT_NREF = 0
    LEFT_REF = 1
    RIGHT_NREF = 2
    RIGHT_REF = 3


EYEPACS_ENV_MAP = {
    'left_nref': EyePacsEnv.LEFT_NREF.value,
    'left_ref': EyePacsEnv.LEFT_REF.value,
    'right_nref': EyePacsEnv.RIGHT_NREF.value,
    'right_ref': EyePacsEnv.RIGHT_REF.value
}

EYEPACS_ENV_MAP_INV = {
    EyePacsEnv.LEFT_NREF.value: 'left_nref',
    EyePacsEnv.LEFT_REF.value: 'left_ref',
    EyePacsEnv.RIGHT_NREF.value: 'right_nref',
    EyePacsEnv.RIGHT_REF.value: 'right_ref'
}

CLASS_TYPE_MAP = {
    'bin': 'Binary',
    'mult': 'Multiclass',
    'bin_out': 'Binarized Output'
}
