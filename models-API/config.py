import os
from pathlib import Path
from openvino.inference_engine import IECore
import layoutparser as lp

# Global model configuration
# Setting
DEVICE = "CPU"
# 1032: 4x superresolution, 1033: 3x superresolution
MODEL_FILE = "model/super_resolution/single-image-super-resolution-1032.xml"
model_name = os.path.basename(MODEL_FILE)
model_xml_path = Path(MODEL_FILE).with_suffix(".xml")

# Load the Superresolution Model
ie = IECore()
net = ie.read_network(model=str(model_xml_path))
exec_net = ie.load_network(network=net, device_name=DEVICE)

# Load layout parser detectron deep learning model
lp_model = lp.models.Detectron2LayoutModel('lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={1:"TextRegion", 2:"ImageRegion", 3:"TableRegion", 4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"})