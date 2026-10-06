from lhotse import CutSet
import multiprocessing as mp
from itertools import islice
from tqdm import tqdm

def process_chunk(chunk):
    """处理单个块：转为eager并返回排序后的块"""
    chunk = chunk.to_eager()
    return chunk.sort_by_duration()

def chunked_iterator(cuts, chunk_size=5000):
    """将CutSet分块生成"""
    iterator = iter(cuts)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            break
        yield CutSet.from_cuts(chunk)

input_path = "/data0/questar/users/wuzhiyue/emilia/train_cuts_windows-None_min_duration-None_max_duration-40_shuffle-True.jsonl.gz"
output_path = "/data0/questar/users/wuzhiyue/emilia/sorted_train_cuts.jsonl.gz"

# 参数配置
chunk_size = 2000  # 根据内存调整块大小
num_workers = 100     # 根据CPU核心数调整

# 延迟加载原始数据
cuts = CutSet.from_jsonl_lazy(input_path)
print('load cuts done')

# 多进程分块处理
with mp.Pool(num_workers) as pool:
    # 分块并行处理（排序单个块）
    chunk_iter = chunked_iterator(cuts, chunk_size=chunk_size)
    # 计算总块数
    total_chunks = (len(cuts) + chunk_size - 1) // chunk_size
    sorted_chunks = pool.imap(
        process_chunk,
        chunk_iter,
        chunksize=100 # 每次提交100个块给worker
    )
    # 使用tqdm添加进度条
    sorted_chunks_with_progress = tqdm(sorted_chunks, total=total_chunks, desc="Processing chunks")
    # 合并并最终排序
    final_cuts = CutSet.from_cuts(
        cut for chunk in sorted_chunks_with_progress for cut in chunk
    ).sort_by_duration()

# 保存结果
final_cuts.to_file(output_path)