import abc

from mixformer_utils.config import Config


class TrackerWrapper:
    def __init__(self, cfg_path):
        self.cfg = Config.fromfile(cfg_path)

        self.search_factor = self.cfg.DATA.SEARCH.FACTOR
        self.search_size = self.cfg.DATA.SEARCH.SIZE 
        self.template_factor = self.cfg.DATA.TEMPLATE.FACTOR
        self.template_size = self.cfg.DATA.TEMPLATE.SIZE
        self.update_interval = self.cfg.TEST.UPDATE_INTERVALS.TRACKINGNET[0]


        self.max_pred_score = -1.0
        self.state = None
        self.name = 'base'

    @abc.abstractmethod
    def initialize(self, image, init_bbox: list) -> None:
        pass

    @abc.abstractmethod
    def track(self, image, frame_id: int) -> tuple:
        pass

    def map_box_back(self, pred_box: list, resize_factor: float) -> list:
        cx_prev = (self.state[0] + self.state[2]) * 0.5
        cy_prev = (self.state[1] + self.state[3]) * 0.5
        cx, cy, w, h = pred_box

        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)

        return [cx_real - 0.5 * w, cy_real - 0.5 * h, 
                cx_real + 0.5 * w, cy_real + 0.5 * h]
