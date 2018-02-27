"""
会話コーパスのロード
"""

import json
import glob
import MeCab


class ShowDial():
    def show(self, fname):
        f = open(fname, "r")
        json_data = json.load(f)
        seq = []
        m = MeCab.Tagger("-Owakati")
        for turn in json_data["turns"]:
            s = m.parse(turn["utterance"])
            if len(s) > 2:
                if turn["speaker"] == "U":
                    s = " BOS " + s
                s = s[:-2]
                if s[-1] != "。":
                    s = s + " 。"
                seq.append(s)
        return seq

    def reshape(self, arr):
        li = []
        for a in arr:
            li.append(len(a.split(" ")))
        m = max(li)

        reshape_seq = []
        for a in arr:
            while len(a.split(" ")) < m:
                a += (" 。")
            reshape_seq.append(a)
        return reshape_seq

    def output_file(self, seq_set):
        f = open("for_w2v_dict.txt", "a")
        for seq in seq_set:
            f.writelines(seq)


def test():
    file_list = glob.glob('json/init100/*')
    fname = "json/init100/1407219916.log.json"
    seq = ShowDial().show(fname)
    seq = ShowDial().reshape(seq)
    # print(seq)


def main():
    file_list = glob.glob('./src/init100/*') + \
        glob.glob('../src/rest1046/*')

    s = ShowDial()
    seq_set = []
    for fname in file_list:
        seq = s.show(fname)
        seq = s.reshape(seq)
        seq_set.append(seq)
    s.output_file(seq_set)


if __name__ == "__main__":
    main()
