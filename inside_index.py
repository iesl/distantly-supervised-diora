import torch

from offset_cache import get_offset_cache


class InsideIndex(object):
    def get_pairs(self, level, i):
        pairs = []
        for constituent_num in range(0, level):
            l_level = constituent_num
            l_i = i - level + constituent_num
            r_level = level - 1 - constituent_num
            r_i = i
            pair = ((l_level, l_i), (r_level, r_i))
            pairs.append(pair)
        return pairs

    def get_all_pairs(self, level, n):
        pairs = []
        for i in range(level, n):
            pairs += self.get_pairs(level, i)
        return pairs


class InsideIndexCheck(object):
    def __init__(self, index, length, spans, siblings):
        sib_map = {}
        for x, y, n in siblings:
            sib_map[x] = (y, n)
            sib_map[y] = (x, n)

        check = {}
        for sibling, (target, name) in sib_map.items():
            xpos = target[0]
            xsize = target[1] - target[0]
            xlevel = xsize - 1
            xoffset = index.get_offset(length)[xlevel]
            xidx = xoffset + xpos

            spos = sibling[0]
            ssize = sibling[1] - sibling[0]
            slevel = ssize - 1
            soffset = index.get_offset(length)[slevel]
            sidx = soffset + spos

            check[(xidx, sidx)] = True
        self.check = check

    def is_valid(self, xidx, sidx):
        return (xidx, sidx) in self.check


def get_inside_index_unique(length, level, offset_cache=None, device=None):
    if offset_cache is None:
        offset_cache = get_offset_cache(length)
    index = InsideIndex()
    pairs = index.get_all_pairs(level, length)

    L = length - level
    n_constituents = len(pairs) // L
    idx_set = set()

    for i in range(n_constituents):
        lvl_l = i
        lvl_r = level - i - 1
        lstart, lend = 0, L
        rstart, rend = length - L - lvl_r, length - lvl_r

        if lvl_l < 0:
            lvl_l = length + lvl_l
        if lvl_r < 0:
            lvl_r = length + lvl_r

        for pos in range(lstart, lend):
            offset = offset_cache[lvl_l]
            idx = offset + pos
            idx_set.add(idx)

        for pos in range(rstart, rend):
            offset = offset_cache[lvl_r]
            idx = offset + pos
            idx_set.add(idx)

    idx_lst = torch.tensor(list(idx_set), dtype=torch.int64, device=device).flatten()
    return idx_lst


def get_inside_components(length, level, offset_cache=None):
    if offset_cache is None:
        offset_cache = get_offset_cache(length)
    index = InsideIndex()
    pairs = index.get_all_pairs(level, length)

    L = length - level
    n_constituents = len(pairs) // L
    output = []

    for i in range(n_constituents):
        index_l, index_r = [], []
        span_x, span_l, span_r = [], [], []

        l_level = i
        r_level = level - l_level - 1

        l_start = 0
        l_end = L

        r_start = length - L - r_level
        r_end = length - r_level

        if l_level < 0:
            l_level = length + l_level
        if r_level < 0:
            r_level = length + r_level

        # The span being targeted.
        for pos in range(l_start, l_end):
            span_x.append((level, pos))

        # The left child.
        for pos in range(l_start, l_end):
            offset = offset_cache[l_level]
            idx = offset + pos
            index_l.append(idx)
            span_l.append((l_level, pos))

        # The right child.
        for pos in range(r_start, r_end):
            offset = offset_cache[r_level]
            idx = offset + pos
            index_r.append(idx)
            span_r.append((r_level, pos))

        output.append((index_l, index_r, span_x, span_l, span_r))

    return output


def get_inside_index(length, level, offset_cache=None, device=None):
    components = get_inside_components(length, level, offset_cache)

    idx_l, idx_r = [], []

    for i, (index_l, index_r, _, _, _) in enumerate(components):
        idx_l.append(index_l)
        idx_r.append(index_r)

    idx_l = torch.tensor(idx_l, dtype=torch.int64, device=device
            ).transpose(0, 1).contiguous().flatten()
    idx_r = torch.tensor(idx_r, dtype=torch.int64, device=device
            ).transpose(0, 1).contiguous().flatten()

    return idx_l, idx_r

