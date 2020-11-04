#!/usr/bin/env python

"""
Script to train the plate reader neural networks.
"""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--id", action="store_true", help="train the plate identification network"
    )
    parser.add_argument(
        "-r", "--read", action="store_true", help="train the plate reader network"
    )
    args = parser.parse_args()
