import os

class glb:
    AI4Mars="../AI4Mars_Data"
    pallet_nav={
        "soil" : [21, 171, 234],
        "bedrock": [191, 21, 234],
        "sand" :  [234, 84, 21],
        "big rock": [64, 234, 21], #Green
        "unlabeled":  [155,155, 155] #Grey
    }

    labels_nav = {
        0:{
            "mask_rgb": [0,0,0],
            "display_rgb": [21, 171, 234], #Light Blue
            "name": "soil"
        },
        1:{
            "mask_rgb": [1,1,1],
            "display_rgb": [191, 21, 234], #Purple
            "name": "bedrock"
        },
        2:{
            "mask_rgb": [2,2,2],
            "display_rgb": [234, 84, 21], #Light Orange
            "name": "sand"
        },
        3:{
            "mask_rgb": [3,3,3],
            "display_rgb": [64, 234, 21], #Green
            "name": "big rock"
        },
        255:{
            "mask_rgb": [255,255, 255],
            "display_rgb": [255,255, 255], #White
            "name": "unlabeled"
        },
    }
