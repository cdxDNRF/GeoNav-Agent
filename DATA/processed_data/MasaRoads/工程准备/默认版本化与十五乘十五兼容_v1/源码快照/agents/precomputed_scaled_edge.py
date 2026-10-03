"""Provide only the two observed image profiles; preserve canonical policy equations."""
from hashlib import sha256
import numpy as np
from agents.scaled_edge_navigator import ScaledEdgeNavigator


class PrecomputedScaledEdgeNavigator(ScaledEdgeNavigator):
    def act_with_profiles(self,obs,current_global,current_local,target_global,target_local,current_profile,target_profile):
        # Evaluation owns image->feature lookup, just as for global/local features.
        # No bank, source id, target coordinate or unseen-image access enters this API.
        for payload,profile in ((obs.current_image,current_profile),(obs.target_image,target_profile)):
            value=np.asarray(profile,np.float32)
            if value.shape!=(4,3,64,3) or not np.isfinite(value).all():raise ValueError('invalid given-image profile')
            self.profiles[sha256(payload).hexdigest()]=value
        return super().act(obs,current_global,current_local,target_global,target_local)
