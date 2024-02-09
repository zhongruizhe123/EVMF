import csv
import io,os
import six
from functools import partial
from torchtext.utils import download_from_url, unicode_csv_reader
from torchtext import data
from torchtext.data.example import Example

path = 'val_1_imgs_meta.csv'
# !/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
create_author : 蛙鳜鸡鹳狸猿
create_time   : 2019-03-19
program       : *_* .tsv file handler *_*
"""

import codecs


class TSV(object):
    """
    .tsv file's handler.
    """

    def __init__(self, file):
        """
        TSV init.
        :param file: .tsv file to handle.
        """
        self.file = file

    def __repr__(self):
        return "File {file} under handling......".format(file=self.file)

    def tsv(self):
        """
        .tsv file's column definition and data check.
        :return: List.
            lines data from [file] row by row in dict format.
        """
        with codecs.open(self.file, 'r', "utf-8") as f:
            line = f.readline()
            data = []
            head = []
            while line:
                if line.isspace():
                    line = f.readline()
                    continue
                elif not line.isspace():
                    # to be compatible between OS
                    head = line.rstrip("\r\n").split('\t')
                    line = f.readline()
                    break
            i=0
            while line:
                if line.isspace():
                    line = f.readline()
                    continue
                elif not line.isspace():
                    body = line.rstrip("\r\n").split('\t')
                    rows = zip(head, body)
                    tsv_dic = {}
                    for (head_sub, body_sub) in list(rows):
                        tsv_dic[head_sub] = body_sub
                        # print(head_sub, body_sub)
                    data.append(tsv_dic)

                    print(data[i])
                    i+=1
                    line = f.readline()
            return data


if __name__ == "__main__":
    # with codecs.open(path, 'r', "utf-8") as f:
    #     rows = """
    #     Id\tContent
    #     1\tContent1
    #     2\tContent2
    #     3\tContent3
    #     4\tContent4
    #     1024\tContent1024
    #     """
    #     f.writelines(rows.replace(' ', ''))
    TSV_Tester = TSV(file=path)
    TSV_Tester.tsv()
    # print(TSV_Tester.tsv())