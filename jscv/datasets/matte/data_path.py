"""
This file records the directory paths to the different datasets.
You will need to configure it for training the model.

All datasets follows the following format, where fgr and pha points to directory that contains jpg or png.
Inside the directory could be any nested formats, but fgr and pha structure must match. You can add your own
dataset to the list as long as it follows the format. 'fgr' should point to foreground images with RGB channels,
'pha' should point to alpha images with only 1 grey channel.
{
    'YOUR_DATASET': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        }
    }
}
"""

DATA_PATH = {
    'videomatte240k': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        }
    },
    'photomatte13k': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        }
    },
    'distinction': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        },
    },
    'adobe': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        },
    },
    'backgrounds': {
        'train': 'PATH_TO_IMAGES_DIR',
        'valid': 'PATH_TO_IMAGES_DIR'
    },
    
    'PPM100': {
        'train': {
            'fgr': 'data/matting/PPM-100/image',
            'pha': 'data/matting/PPM-100/matte',
        },
        'valid': {
            'fgr': 'data/matting/PPM-100/val/image',
            'pha': 'data/matting/PPM-100/val/matte',
        },
    },

    'Distinctions646': {
        'train': {
            'fgr': 'data/matting/Distinctions-646/Train/FG',
            'pha': 'data/matting/Distinctions-646/Train/GT',
        },
        'valid': {
            'fgr': 'data/matting/Distinctions-646/Test/FG',
            'pha': 'data/matting/Distinctions-646/Test/GT',
        },
    }
}