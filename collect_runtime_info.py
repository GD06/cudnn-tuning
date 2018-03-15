#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from cudnn_record import cudnnTrace
import pickle
from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser(
        description="collect cudnn runtime information",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', help="directory contains trace files")
    parser.add_argument('--suffix', help="trace file suffix",
                        default=".pkl")

    args = parser.parse_args()

    for root, dirs, files in os.walk(args.input_dir):
        for input_file in files:
            if args.suffix in input_file:

                with open(os.path.join(root, input_file), 'rb') as f:
                    cudnn_trace_list = pickle.load(f)

                trace_len = len(cudnn_trace_list)

                if trace_len == 0:
                    continue

                for i in tqdm(range(trace_len), desc=input_file):
                    if cudnn_trace_list[i].runtime_info is None:
                        cudnn_trace_list[i].collect_runtime_info()

                    if i % 100 == 0:
                        with open(os.path.join(root, input_file), 'wb') as f:
                            pickle.dump(cudnn_trace_list, f)

                with open(os.path.join(root, input_file), 'wb') as f:
                    pickle.dump(cudnn_trace_list, f)

if __name__ == '__main__':
    main()
