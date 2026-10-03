"""Exact bounded clique search for provisional, mutually isolated areas."""
from pathlib import Path
import sys
import time
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import area_compatibility as p


def maximum_clique(adjacency, seconds=20):
    best, nodes = [], 0
    deadline = time.perf_counter()+seconds
    def visit(chosen, available):
        nonlocal best, nodes
        nodes += 1
        if time.perf_counter() > deadline:
            raise TimeoutError('fixed packing search time cap')
        # Proper coloring bounds the largest clique in each remaining prefix.
        order, bounds, uncolored, color = [], [], available, 0
        while uncolored:
            color += 1
            group = uncolored
            while group:
                bit = group & -group
                v = bit.bit_length()-1
                order.append(v)
                bounds.append(color)
                uncolored &= ~bit
                group &= ~bit
                group &= ~adjacency[v]
        for index in range(len(order)-1, -1, -1):
            if len(chosen)+bounds[index] <= len(best):
                return
            v = order[index]
            rest = available & adjacency[v]
            if rest:
                visit(chosen+[v], rest)
            elif len(chosen)+1 > len(best):
                best = chosen+[v]
            available &= ~(1 << v)
    completed = True
    try:
        visit([], (1 << len(adjacency))-1)
    except TimeoutError:
        completed = False
    return best, completed, nodes


def run():
    p.check()
    # Small independently enumerable graph checks: no edges, complete, and path.
    assert len(maximum_clique([0, 0, 0])[0]) == 1
    assert len(maximum_clique([6, 5, 3])[0]) == 3
    assert len(maximum_clique([2, 5, 2])[0]) == 2
    assert maximum_clique([])[0] == []
    input_path = p.OUT / '十五乘十五数据可行性_仅目录.json'
    data = p.read(input_path)
    regions = data['candidates']
    graph = [sum(1 << j for j, b in enumerate(regions) if i != j and
        p.gap(a['provisional_bounds_m'], b['provisional_bounds_m']) >= 3000-.001) for i, a in enumerate(regions)]
    selection, proved, nodes = maximum_clique(graph)
    for i, a in enumerate(selection):
        for b in selection[i+1:]:
            assert p.gap(regions[a]['provisional_bounds_m'], regions[b]['provisional_bounds_m']) >= 3000-.001
    source = Path(__file__)
    snapshot = p.OUT / '源码快照/eval' / source.name
    with snapshot.open('xb') as f:
        f.write(source.read_bytes())
    p.write(p.OUT / '目录候选空间打包复核.json', dict(filename_candidates=len(regions),
        greedy_count=data['provisional_mutually_isolated'], selected_count=len(selection), exact_upper_bound_proved=proved,
        graph_nodes_visited=nodes, fixed_time_cap_seconds=20, graph_tests_passed=4,
        selected_indices=selection, selected_provisional_regions=[regions[i] for i in selection],
        actual_TIFF_coordinates_or_quality_passed=False, real_navigation_started=False,
        input_sha256=p.digest(input_path), source_sha256={p.rel(q):p.digest(q) for q in (source,snapshot)}))
    print(dict(greedy=data['provisional_mutually_isolated'], packed=len(selection), maximum_proved=proved), flush=True)


if __name__ == '__main__':
    run()
