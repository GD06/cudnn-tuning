class cudnnTrace:

    def __init__(self, IN, IC, IH, IW, OC, FH, FW,
                 pad_h, pad_w, strd_h, strd_w, *,
                 dil_h=1, dil_w=1, mode=1, conv_format=0, workspace_limit=None,
                 cudnn_selected=None):

        self.IN = IN
        self.IC = IC
        self.IH = IH
        self.IW = IW
        self.OC = OC
        self.FH = FH
        self.FW = FW
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.strd_h = strd_h
        self.strd_w = strd_w

        self.dil_h = dil_h
        self.dil_w = dil_w
        self.mode = mode
        self.conv_format = conv_format

        self.cudnn_version = None
        self.runtime_info = False
        self.workspace_dict = {}
        self.perf_dict = {}
        self.workspace_limit = workspace_limit
        self.cudnn_selected = cudnn_selected
        self.ground_truth = None

        self._derive_output_shape()
        return

    def _derive_output_shape(self):
        self.ON = self.IN
        self.OH = int((self.IH + 2 * self.pad_h - self.FH * self.dil_h) / self.strd_h + 1)
        self.OW = int((self.IW + 2 * self.pad_w - self.FW * self.dil_w) / self.strd_w + 1)
        return


