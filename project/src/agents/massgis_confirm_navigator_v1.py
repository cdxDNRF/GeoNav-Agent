"""Change protocol identity only; reuse sealed native navigation equations."""
from agents.massgis_navigator_v1 import NativeNavigator
from env.massgis_confirm_area_v1 import confirmation_spec


class ConfirmNavigator(NativeNavigator):
    def __init__(self, legacy, k, policy='M0', condition='CueFull'):
        super().__init__(legacy, k, policy, condition)
        self.spec = confirmation_spec(k)
