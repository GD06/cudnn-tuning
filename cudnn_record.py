import subprocess
import os

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

        self.algo_dict = self._get_algo_dict()

        self.cudnn_version = None
        self.runtime_info = False
        self.workspace_dict = {}
        self.perf_dict = {}
        self.workspace_limit = workspace_limit
        self.ground_truth = None
        self._derive_output_shape()

        self.exec_func = "./cudnn_func/cudnn_perf"

        if isinstance(cudnn_selected, str):
            self.cudnn_selected = self.algo_dict[cudnn_selected]
        else:
            self.cudnn_selected = cudnn_selected

        return

    def set_exec_func(self, exec_func):
        self.exec_func = exec_func
        return

    def collect_runtime_info(self):
        command_list = [self.exec_func]
        command_list.extend([str(self.IN), str(self.IC), str(self.IH), str(self.IW),
                             str(self.OC), str(self.FH), str(self.FW),
                             str(self.OH), str(self.OW), str(self.pad_h), str(self.pad_w),
                             str(self.strd_h), str(self.strd_w),
                             str(self.mode), str(self.conv_format)])
        proc = subprocess.Popen(command_list, cwd=os.path.realpath(__file__),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            outs, errs = proc.communicate(timeout=60)
            lines = outs.split('\n')

        except Exception as excep:
            proc.kill()

        return

    def _derive_output_shape(self):
        self.ON = self.IN
        self.OH = int((self.IH + 2 * self.pad_h - self.FH * self.dil_h) / self.strd_h + 1)
        self.OW = int((self.IW + 2 * self.pad_w - self.FW * self.dil_w) / self.strd_w + 1)
        return

    def _get_algo_dict(self):
        return {'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM': 0,
                'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM': 1,
                'CUDNN_CONVOLUTION_FWD_ALGO_GEMM': 2,
                'CUDNN_CONVOLUTION_FWD_ALGO_DIRECT': 3,
                'CUDNN_CONVOLUTION_FWD_ALGO_FFT': 4,
                'CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING': 5,
                'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD': 6,
                'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED': 7}
