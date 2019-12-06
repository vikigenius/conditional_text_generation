#!/usr/bin/env python3

import mmap


def get_num_lines(file_path):
    with open(file_path, "r+") as fp:
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
    return lines
