import math
from typing import List, Union


def get_group_ranks(
    global_ranks_or_sizes: Union[int, List[int]],
    cur_group_size: int,
    pre_group_size: int,
    allow_partial_group: bool = False,
):
    group_ranks = []

    if isinstance(global_ranks_or_sizes, list):
        global_size = len(global_ranks_or_sizes)
        global_ranks = global_ranks_or_sizes
    else:
        global_size = global_ranks_or_sizes
        global_ranks = None

    real_global_size = global_size

    if allow_partial_group:
        global_size = math.ceil(global_size / cur_group_size) * cur_group_size

    assert global_size % cur_group_size == 0, "err1"

    def _get_local_starts():
        for i in range(0, global_size, cur_group_size * pre_group_size):
            for j in range(pre_group_size):
                yield 0 + i + j

    for start in _get_local_starts():
        ranks = [
            start + i * pre_group_size for i in range(cur_group_size) if start + i * pre_group_size < real_global_size
        ]
        if global_ranks is not None:
            ranks = [global_ranks[_idx] for _idx in ranks]

        group_ranks.append(ranks)

    assert len(group_ranks) == global_size // cur_group_size, f"{group_ranks}, {global_size}, {cur_group_size}"

    return group_ranks


if __name__ == "__main__":

    def get_interleaved(windows_num, windows_size):
        # interleaved = []
        # for _i in range(windows_num):
        #     if _i % 2 == 0:
        #         interleaved.extend([_j * 2 + _i * windows_size for _j in range(windows_size)])
        #     else:
        #         interleaved.extend([_j * 2 + 1 + (_i-1) * windows_size for _j in range(windows_size)])
        return [
            j * 2 + i * windows_size if i % 2 == 0 else j * 2 + 1 + (i - 1) * windows_size
            for i in range(windows_num)
            for j in range(windows_size)
        ]

    print(get_group_ranks(16, 4, 1))
    print(get_group_ranks(16, 4, 4))
    print(get_group_ranks(8, 32, 1, allow_partial_group=True))

    global_size = 16
    windows_size = 8
    windows_num = global_size // windows_size

    # interleaved = [0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15]
    interleaved = get_interleaved(windows_num, windows_size)
    print("interleavd:", interleaved)
    print(get_group_ranks(interleaved, windows_size, 1))
    print(get_group_ranks(interleaved, windows_num, windows_size))

    # global_size = 16, window_size: 2, windows_num: 8
    # intra windows: [[0,2], [1,3], [4,6], [5,7], [8,10], [9,11], [12,14], [13,15]]
    # inter_windows: [[0,1,4,5,8,9,12,13], [2,3,6,7,10,11,14,15]]
