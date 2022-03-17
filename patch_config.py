from torch import optim


class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        # config file
        self.cfgfile = "cfg/yolov3-cowc.cfg"
        
        # weights file
        self.weightfile = "weights/yolov3-cowc-256/yolov3-cowc_best_256.weights"

        # training parameters
        self.start_learning_rate = 0.03
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)

        # self.patch_name = 'base'

class SideStreet(BaseConfig):
    """
    Patch for sidestreet scene
    """

    def __init__(self):
        super().__init__()

        # side street dataset
        self.img_dir = "#sidestreet/patch_train"
        self.lab_dir = "#sidestreet/patch_train/yolo-labels"
        self.input_size = 1088

        # patch parameters
        self.patch_name = 'sidestreet'
        self.weather_augmentations = 'off'
        
        # # on patch
        # self.patch_size = (160,200)
        # self.patch_num = 1
        # self.patch_scale = 40

        # off patch
        self.patch_size = (25,400)
        self.patch_num = 3
        self.patch_scale = 160

        # training parameters
        self.n_epochs = 1000        
        self.batch_size = 1
        self.max_lab = 10
        self.printfile = "non_printability/30values.txt"
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj

class CarPark(BaseConfig):
    """
    Patch for car park scene
    """

    def __init__(self):
        super().__init__()

        # carpack dataset
        self.img_dir = "#carpark/patch_train"
        self.lab_dir = "#carpark/patch_train/yolo-labels"
        self.input_size = 2160

        # patch parameters
        self.patch_name = 'carpark'
        self.weather_augmentations = 'on'
        
        # on patch
        self.patch_size = (240,300)
        self.patch_num = 1
        self.patch_scale = 85

        # training parameters
        self.n_epochs = 1000        
        self.batch_size = 1
        self.max_lab = 10
        self.printfile = "non_printability/30values.txt"
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj

class ExperimentX(BaseConfig):
    """
    USE THIS TO RUN YOUR OWN EXPERIMENTS
    """

    def __init__(self):
        super().__init__()

        # # dataset
        # self.img_dir = ""
        # self.lab_dir = ""
        # self.input_size = 

        # # patch parameters
        # self.patch_name = ''
        # self.weather_augmentations = 'on'
        
        # # on patch
        # self.patch_size = (,)
        # self.patch_num = 
        # self.patch_scale = 

        # # off patch
        # self.patch_size = (,)
        # self.patch_num = 
        # self.patch_scale = 

        # training parameters
        self.n_epochs = 1000        
        self.batch_size = 1
        self.max_lab = 10
        self.printfile = "non_printability/30values.txt"
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj

patch_configs = {
    "base": BaseConfig,
    "expX": ExperimentX,
    "sidestreet": SideStreet,
    "carpark": CarPark
}
