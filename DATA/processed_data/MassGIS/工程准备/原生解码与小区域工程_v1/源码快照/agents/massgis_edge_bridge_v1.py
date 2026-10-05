"""Explicit native600 -> model300 area averaging, never changes stored patches."""
from io import BytesIO
from PIL import Image
from agents.edge_cue import image_profiles,edge_features

def native_profiles(payload):
    if not isinstance(payload,bytes):raise ValueError('Only public image bytes')
    with Image.open(BytesIO(payload)) as im:
        if im.mode!='RGB' or im.size!=(600,600):raise ValueError('Native600 RGB observation required')
        model_image=im.resize((300,300),Image.Resampling.BOX)
        return image_profiles(model_image)

def native_edge_features(current_image,target_image):
    return edge_features(native_profiles(target_image),native_profiles(current_image))
