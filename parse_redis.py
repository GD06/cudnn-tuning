#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from cudnn_record import cudnnTrace
import re
import struct
import pickle

def main():

    parser = argparse.ArgumentParser(
        description="parse redis log file of cudnn traces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_file', help="input_file path of the redis log file")
    parser.add_argument('--output_dir', default=None,
                        help="output directory of the parsed cudnn traces")
    parser.add_argument('--output_file', default=None,
                        help="output filename of the parsed cudnn traces")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.getenv('LOG_OUTPUT_DIR'), 'cudnn-tuning')

    if args.output_file is None:
        args.output_file = os.path.basename(args.input_file) + '.pkl'

    cudnn_trace_list = []

    with open(args.input_file, 'r') as f:
        lines = f.readlines()

    total_line_num = 0
    fwd_line_num = 0
    error_line_num = 0

    for line in lines:
        total_line_num += 1

        try:
            exconvtype, exshape = line.split('@')[1:3]
            convname = re.search(r'conv_bwd_filter|conv_fwd|conv_bwd_data', exconvtype).group()
            tmpa, tmpb = exshape.split('|')[0:2]
            aInfo = list(map(int, re.findall("(\d+)", tmpa.split(';')[0])))
            bInfo = list(map(int, re.findall("(\d+)", tmpb.split(';')[0])))
            assert len(aInfo) == 4
            assert len(bInfo) == 4
            tmpstruct = exshape.split('|')[-1]
            bitmpstruct = tmpstruct.encode()

            if convname != 'conv_fwd':
                continue

        except Exception as e:
            error_line_num += 1
            continue

        if len(bitmpstruct) == 41:
            mode, pad_h, pad_w, strd_h, strd_w, dil_h, dil_w, dtype, sparse, conv_format = \
                    struct.unpack_from('iiiiiiiiii', bitmpstruct, 0)
        elif len(bitmpstruct) == 21:
            mode, pad_h, pad_w, strd_h, strd_w = struct.unpack_from('iiiii', bitmpstruct, 0)
            conv_format = 0
            dil_h = 1
            dil_w = 1
        else:
            error_line_num += 1

        mode = int(not mode)
        fwd_line_num += 1
        cudnn_trace_list.append(cudnnTrace(aInfo[0], aInfo[1], aInfo[2], aInfo[3],
                                           bInfo[0], bInfo[2], bInfo[3], pad_h, pad_w,
                                           strd_h, strd_w, dil_h=dil_h, dil_w=dil_w, mode=mode,
                                           conv_format=conv_format))

    print('Parsed Total # of Lines:', total_line_num)
    print('Error # of Lines:', error_line_num)
    print('Forward Trace # of Lines:', fwd_line_num)

    with open(os.path.join(args.output_dir, args.output_file), 'wb') as f:
        pickle.dump(cudnn_trace_list, f)

if __name__ == '__main__':
    main()
