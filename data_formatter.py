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


def dev_tsv_format(_org_file, _out_file_text, _out_file_label):
    # <query ID, passage ID, query text, passage text>
    # <query ID, 0, passage ID, 1>
    tot_cnt = 0
    l_cnt = 0
    fout_t = open(_out_file_text, 'w', encoding='utf-8')
    fout_l = open(_out_file_label, 'w', encoding='utf-8')
    with open(_org_file, encoding='utf-8') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            json_obj = json.loads(line.strip())
            qid, pid = json_obj['guid'].split('_')
            query = json_obj['text_a']
            para = json_obj['text_b']
            query = re.sub('\t', ' ', query)
            ws = query.split(' ')
            if len(ws) > MAX_QUERY_LEN:
                query = ' '.join(ws[:MAX_QUERY_LEN])
            else:
                query = ' '.join(ws)
            para = re.sub('\t', ' ', para)
            p_ws = para.split(' ')
            if len(p_ws) > MAX_PARA_LEN:
                para = ' '.join(p_ws[:MAX_PARA_LEN])
            else:
                para = ' '.join(p_ws)
            out_line = '\t'.join([qid, json_obj['guid'], query, para]) + '\n'
            fout_t.write(out_line)
            tot_cnt += 1
            if json_obj['label'] > 0:
                out_line_l = '\t'.join([qid, str(0), json_obj['guid'], str(1)]) + '\n'
                fout_l.write(out_line_l)
                l_cnt += 1
    fout_t.close()
    fout_l.close()
    print('save in file=%s, tot_cnt=%d' % (_out_file_text, tot_cnt))
    print('save label in file=%s, label_cnt=%d' % (_out_file_label, l_cnt))


def data_check(_filename):
    tot_cnt = 0
    with open(_filename, encoding='utf-8') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            tot_cnt += 1
            x = line.strip().split('\t')
            if len(x) != 3:
                print(tot_cnt, len(x))
                print(x)
                input('continue？')
    print('check_done!')


def reader_debug(_filename, _bsize, _maxsteps):
    reader = open(_filename, 'r', encoding='utf-8')
    for batch_idx in range(_maxsteps):
        Batch = [reader.readline().split('\t') for _ in range(_bsize)]
        try:
            Batch = sorted(Batch, key=lambda x: max(len(x[1]), len(x[2])))
        except Exception as e:
            print(e)
            print(batch_idx)
            print(Batch)


if __name__ == '__main__':
    org_file = '/work/shaoyunqiu/coliee_2020/data/task2/format/train_split.json'
    out_file_text = './data/train_eval.tsv'
    out_file_label = './data/train_qrel.tsv'
    dev_tsv_format(org_file, out_file_text, out_file_label)

    org_file = '/work/shaoyunqiu/coliee_2020/data/task2/format/dev_split.json'
    out_file_text = './data/dev_eval.tsv'
    out_file_label = './data/dev_qrel.tsv'
    dev_tsv_format(org_file, out_file_text, out_file_label)
    # train_tsv_formatter(org_file, out_file)
    # data_check(out_file)
    # bsize = 16
    # maxstep = 400000
    # reader_debug(out_file, _bsize=bsize, _maxsteps=maxstep)

