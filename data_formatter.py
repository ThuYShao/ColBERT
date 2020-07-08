# -*- coding: utf-8 -*-
__author__ = 'yshao'

import re
import json
import os
from collections import defaultdict

MAX_QUERY_LEN = 128
MAX_PARA_LEN = 256


def train_tsv_formatter(_org_file, _out_file):
    tot_cnt = 0
    fout = open(_out_file, 'w', encoding='utf-8')
    query_dict = defaultdict()
    pos_para_dict = defaultdict(lambda: defaultdict())
    neg_para_dict = defaultdict(lambda: defaultdict())
    with open(_org_file, encoding='utf-8') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            json_obj = json.loads(line.strip())
            qid, pid = json_obj['guid'].split('_')
            if qid not in query_dict:
                query_dict[qid] = json_obj['text_a']
            if json_obj['label'] > 0:
                pos_para_dict[qid][pid] = json_obj['text_b']
            else:
                neg_para_dict[qid][pid] = json_obj['text_b']
    assert (len(query_dict) == len(pos_para_dict))
    assert (len(query_dict) == len(neg_para_dict))

    for qid in query_dict:
        query = query_dict[qid]
        query = re.sub('\t', ' ', query)
        ws = query.split(' ')
        if len(ws) > MAX_QUERY_LEN:
            query = ' '.join(ws[:MAX_QUERY_LEN])
        else:
            query = ' '.join(ws)
        for p_pid in pos_para_dict[qid]:
            p_para = pos_para_dict[qid][p_pid]
            p_para = re.sub('\t', ' ', p_para)
            p_ws = p_para.split(' ')
            if len(p_ws) > MAX_PARA_LEN:
                p_para = ' '.join(p_ws[:MAX_PARA_LEN])
            else:
                p_para = ' '.join(p_ws)
            for n_pid in neg_para_dict[qid]:
                n_para = neg_para_dict[qid][n_pid]
                n_para = re.sub('\t', ' ', n_para)
                n_ws = n_para.split(' ')
                if len(n_ws) > MAX_PARA_LEN:
                    n_para = ' '.join(n_ws[:MAX_PARA_LEN])
                else:
                    n_para = ' '.join(n_ws)
                out_line = '\t'.join([query, p_para, n_para]) + '\n'
                fout.write(out_line)
                tot_cnt += 1
                if tot_cnt % 10000 == 5:
                    print('tot_cnt=%d' % tot_cnt)
    fout.close()
    print('save train pairs in tsv, #data=%d' % tot_cnt)


if __name__ == '__main__':
    org_file = '/work/shaoyunqiu/coliee_2020/data/task2/format/train_split.json'
    out_file = './data/train_triples.tsv'
    train_tsv_formatter(org_file, out_file)

