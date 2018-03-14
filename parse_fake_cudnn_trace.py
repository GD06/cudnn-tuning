#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from cudnn_record import cudnnTrace
import re
import pickle

def parse_trace_file(output_dir, lines):
    num_traces = 0

    start_sign_match = re.compile(r'Model (?P<model_name>[\w.]*) start running')
    stop_sign_match = re.compile(r'Model (?P<model_name>[\w.]*) stop')
    cudnnForwardMatch = re.compile(r"cudnnConvolutionForward: "
        "\{(?P<IN>\d*),(?P<IC>\d*),(?P<IH>\d*),(?P<IW>\d*)\} "
        "\{dt=\d* fmt=(?P<format>\d*) (?P<FOC>\d*),(?P<FIC>\d*),(?P<FH>\d*),(?P<FW>\d*)\} "
        "\{(?P<ON>\d*),(?P<OC>\d*),(?P<OH>\d*),(?P<OW>\d*)\} "
        "\{pad=(?P<pad_h>\d*),(?P<pad_w>\d*) strd=(?P<strd_h>\d*),(?P<strd_w>\d*) scale=(?P<scale_h>\d*),(?P<scale_w>\d*) mode=(?P<mode>\d*)\} "
        "(?P<algo>\w*) "
        "(?P<workspace>\d*.\d*)")
    model_name = None

    for line in lines:

        match_result = start_sign_match.match(line)
        if match_result is not None:
            model_name = match_result.group('model_name')
            print('{} start sign matched'.format(model_name))
            cudnn_trace_list = []
            continue

        match_result = stop_sign_match.match(line)
        if match_result is not None:
            assert model_name == match_result.group('model_name')
            print('{} stop sign matched'.format(model_name))

            output_file_name = "{}.pkl".format(model_name)
            with open(os.path.join(output_dir, output_file_name), 'wb') as f:
                pickle.dump(cudnn_trace_list, f)

            model_name = None
            continue

        if model_name is not None:
            match_result = cudnnForwardMatch.match(line)
            if match_result is not None:
                [IN, IC, IH, IW] = [int(match_result.group('IN')), int(match_result.group('IC')),
                                    int(match_result.group('IH')), int(match_result.group('IW'))]
                conv_format = int(match_result.group('format'))
                [FOC, FIC, FH, FW] = [int(match_result.group('FOC')), int(match_result.group('FIC')),
                                      int(match_result.group('FH')), int(match_result.group('FW'))]
                [ON, OC, OH, OW] = [int(match_result.group('ON')), int(match_result.group('OC')),
                                    int(match_result.group('OH')), int(match_result.group('OW'))]
                [pad_h, pad_w, strd_h, strd_w, scale_h, scale_w, mode] = [
                    int(match_result.group('pad_h')), int(match_result.group('pad_w')),
                    int(match_result.group('strd_h')), int(match_result.group('strd_w')),
                    int(match_result.group('scale_h')), int(match_result.group('scale_w')),
                    int(match_result.group('mode'))]
                algo = match_result.group('algo')
                workspace = float(match_result.group('workspace'))

                if conv_format != 0:
                    [IN, IH, IW, IC] = [IN, IC, IH, IW]
                    [ON, OH, OW, OC] = [ON, OC, OH, OW]

                assert OC == FOC
                assert IC == FIC

                cudnn_trace_list.append(cudnnTrace(IN, IC, IH, IW, OC, FH, FW,
                                                   pad_h, pad_w, strd_h, strd_w,
                                                   mode=mode, conv_format=conv_format,
                                                   workspace_limit=workspace,
                                                   cudnn_selected=algo))

                num_traces += 1

    return num_traces

def main():

    parser = argparse.ArgumentParser(
        description="parse fake cudnn traces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', help="directory contains trace files")
    parser.add_argument('--suffix', help="trace file suffix",
                        default="raw.txt")
    parser.add_argument('--output_dir', default=None,
                        help="output directory of parsed cudnn traces")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.getenv('LOG_OUTPUT_DIR'), 'cudnn-tuning')

    for root, dirs, files in os.walk(args.input_dir):
        for input_file in files:
            if args.suffix in input_file:

                with open(os.path.join(root, input_file), 'r') as f:
                    lines = f.readlines()

                print('Processing file {}'.format(input_file))
                num_traces = parse_trace_file(args.output_dir, lines)
                print('{} traces parsed'.format(num_traces))


if __name__ == '__main__':
    main()
