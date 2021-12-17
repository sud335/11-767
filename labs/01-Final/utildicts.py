import numpy as np

resolution={
    "QCIF": (176, 120),
    "CIF" : (352, 240),
    "2CIF": (704, 240),
    "4CIF": (704, 480),
    "D1" : (720, 480),
    "720p": (1280, 720),
    "960p": (1280, 960),
    "1.3MP" : (1280, 1024),
    "2MP" : (1280, 1200),
    "1080p": (1920, 1080)
}

emotionLabels = {
    "happy":0, 
    "angry":1, 
    "disgust":2, 
    "fear":3, 
    "neutral":4, 
    "sad":5, 
    "surprised":6
}

activityLabels = {
    "clapping":0,
    "coughing":1,
    "drinking_sipping":2,
    "hand_saw":3,
    "keyboard_typing":4,
    "laughing":5,
    "mouse_click":6,
    "null":7,
    "pouring_water":8,
    "snoring":9,
    "vacuum_cleaner":10
}

finalLabels = {
    "excited":0, 
    "adore":1, 
    "productive":2, 
    "cheerful":3, 
    np.nan:4, 
    "dismissive":5, 
    "hate":6, 
    "non productive": 7,
    "annoyed":8, 
    "panic":9, 
    "stressed":10, 
    "calm":11, 
    "loss of appetite":12, 
    "interested":13
}