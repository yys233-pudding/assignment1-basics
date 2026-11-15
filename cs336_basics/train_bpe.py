import regex as re
from collections import Counter

from cs336_basics.pretokenization_example import find_chunk_boundaries

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r",encoding="utf-8") as f:
        #这里指定encoding格式为utf-8，对输入进行decode，把读入的bytes转换为str类型
        #因为不确定有哪些分隔符所以在这个文件里不进行chunk切分，暂时把整个文件作为一个chunk来处理
        data=f.read()
        regex_tokens = []
        for token in special_tokens:
            regex_tokens.append(re.escape(token))
        print(regex_tokens)
        pattern = re.compile("|".join(regex_tokens))
        chunks = re.split(pattern, data)
        counter = Counter()
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for chunk in chunks:
            for match in re.finditer(PAT, chunk):
                counter[match.group(0)] += 1
        # 统计预分词片段的次数,因为只能合并每个segment内的相邻字符不能跨切分出的segments进行合并
