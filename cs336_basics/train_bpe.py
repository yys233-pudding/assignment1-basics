import regex as re
from collections import Counter

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
        if regex_tokens:
            pattern = re.compile("|".join(regex_tokens))
        else:
            pattern = None
        if pattern!=None:
            chunks = re.split(pattern, data)
        else:
            chunks = [data] #需要处理分隔符为空的情况
        counter = Counter()
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for chunk in chunks:
            for match in re.finditer(PAT, chunk):
                counter[match.group(0)] += 1
        # 统计预分词片段的次数,因为只能合并每个segment内的相邻字符不能跨切分出的segments进行合并
        # 统计每个相邻字符对的次数
        vocab={i: bytes([i]) for i in range(256)}  # Initialize vocab with all byte values
        vocab.update({256+i: token.encode("utf-8") for i, token in enumerate(special_tokens)})  # Add special tokens to vocab
        assert vocab_size >= len(vocab), "Vocab size must be greater than length of vocab to accommodate all byte values"
        merges = []
        special_tokens_number=len(special_tokens)
        sequences = []
        for segment, count in counter.items():
                symbols = [bytes([value]) for value in segment.encode("utf-8")] ##先编码后显示类型转换来提取bytes
                sequences.append((symbols,count))  #保存每个segment转化成的symbols列表和对应的count
        for i in range(vocab_size-len(vocab)):
            pair_counts = Counter()
            for symbols, count in sequences:
                for i in range(len(symbols) - 1):          
                    pair = (symbols[i], symbols[i + 1])
                    pair_counts[pair] += count
            if not pair_counts:
                break  # No more pairs to merge
            max_pair, frequency = max(
                    pair_counts.items(),
                    key=lambda item: (item[1], item[0]),
                    )
            if frequency>0:
                merges.append(max_pair)
                new_symbol = max_pair[0] + max_pair[1]
                vocab[len(vocab)] =new_symbol
                # Update pair counts with the new symbol
                new_sequences = []
                for symbols, count in sequences:
                    new_symbols=[]
                    i=0
                    while i < len(symbols):
                        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == max_pair:
                                new_symbols.append(symbols[i]+symbols[i + 1])  
                                i=i+2  # Skip the next symbol since it's part of the merged pair 
                        else:
                            new_symbols.append(symbols[i])
                            i=i+1
                    new_sequences.append((new_symbols,count))
                sequences=new_sequences
    return vocab, merges