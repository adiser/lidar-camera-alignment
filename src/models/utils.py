from models.axis_angle_model import AxisAngleModel
from models.euler_model import EulerModel
from models.quat_model import QuatCalibratorModel


def get_model(init_extrinsics, rot_param_type):

    if rot_param_type == 'quaternion':
        calibrator_model = QuatCalibratorModel(init_extrinsics)
    elif rot_param_type == 'euler':
        calibrator_model = EulerModel(init_extrinsics)
    elif rot_param_type == 'axis_angle':
        calibrator_model = AxisAngleModel(init_extrinsics)
    else:
        raise ValueError()

    return calibrator_model
