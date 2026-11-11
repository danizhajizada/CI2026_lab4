"""Lab 4: generalized Klotski, optimal single-cell slide solver (A* and BFS).

Python 3.10+, standard library only. Board coordinates are zero-based.
Usage: python solver.py --all
       python solver.py --all --algorithm astar
       python solver.py --problem p04 --show-boards
"""
from __future__ import annotations
import argparse
import heapq
from itertools import count
from collections import deque, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


@dataclass(frozen=True)
class Problem:
    name: str
    rows: int
    cols: int
    exit_columns: tuple[int, ...]
    target: str
    names: tuple[str, ...]
    shapes: tuple[tuple[int, int], ...]
    initial: tuple[int, ...]  # top-left offset (row * cols + col), per piece


def parse_problems(path: Path) -> dict[str, Problem]:
    """Read the benchmark file line by line.

    Comment lines (starting with '#') and blank lines are skipped, so the
    original file from the course repository (whose header mentions
    'problem <id>') can be used unchanged.
    """
    lines = [s.strip() for s in path.read_text(encoding='utf-8').splitlines()]
    lines = [s for s in lines if s and not s.startswith('#')]
    result = {}
    i = 0
    while i < len(lines):
        if not lines[i].startswith('problem '):
            i += 1
            continue
        name = lines[i].split()[1]
        header = {}
        i += 1
        while lines[i] != '=========':
            key, *values = lines[i].split()
            header[key] = values
            i += 1
        rows, cols = map(int, header['size'])
        if header['exit'][0] != 'bottom':
            raise ValueError(f'{name}: only bottom exits are supported')
        exit_cols = tuple(map(int, header['exit'][1:]))
        target = header['target'][0]
        end = lines.index('=========', i + 1)
        grid = [line.split() for line in lines[i+1:end]]
        i = end + 1
        if len(grid) != rows or any(len(row) != cols for row in grid):
            raise ValueError(f'{name}: board does not match size {rows}x{cols}')
        coords = defaultdict(list)
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell != '.':
                    coords[cell].append((r, c))
        if target not in coords:
            raise ValueError(f'{name}: target {target} is not on the board')
        names = (target,) + tuple(sorted(n for n in coords if n != target))
        shapes, initial = [], []
        for n in names:
            cells = coords[n]
            rmin, cmin = min(r for r, c in cells), min(c for r, c in cells)
            height = max(r for r, c in cells) - rmin + 1
            width = max(c for r, c in cells) - cmin + 1
            if len(cells) != height*width:
                raise ValueError(f'{name}: piece {n} is not rectangular')
            shapes.append((height, width))
            initial.append(rmin*cols+cmin)
        if len(exit_cols) != shapes[0][1]:
            raise ValueError(f'{name}: exit width does not match target')
        result[name] = Problem(name, rows, cols, exit_cols, target, names,
                               tuple(shapes), tuple(initial))
    return result


DIRECTIONS = ((-1, 0, 'up'), (1, 0, 'down'), (0, -1, 'left'), (0, 1, 'right'))


def canonical_key(p: Problem, state: tuple[int, ...]) -> tuple:
    """Collapse label permutations of same-sized, non-target pieces.

    This is safe because pieces of identical dimensions are interchangeable
    for collision avoidance and for the target goal.
    """
    groups = defaultdict(list)
    for i in range(1, len(state)):
        groups[p.shapes[i]].append(state[i])
    return (state[0], tuple((shape, tuple(sorted(positions)))
                            for shape, positions in sorted(groups.items())))


def goal_test(p: Problem, state: tuple[int, ...]) -> bool:
    target_row, target_col = divmod(state[0], p.cols)
    height, width = p.shapes[0]
    return (target_row + height == p.rows and
            tuple(range(target_col, target_col+width)) == p.exit_columns)


def occupancy(p: Problem, state: tuple[int, ...]) -> list[int]:
    cells = [-1] * (p.rows * p.cols)
    for i, (pos, (height, width)) in enumerate(zip(state, p.shapes)):
        r, c = divmod(pos, p.cols)
        for rr in range(r, r+height):
            for cc in range(c, c+width):
                idx = rr*p.cols+cc
                if cells[idx] != -1:
                    raise ValueError('Overlapping pieces')
                cells[idx] = i
    return cells


def successors(p: Problem, state: tuple[int, ...]):
    occupied = occupancy(p, state)
    for i, (pos, (height, width)) in enumerate(zip(state, p.shapes)):
        r, c = divmod(pos, p.cols)
        for dr, dc, direction in DIRECTIONS:
            nr, nc = r+dr, c+dc
            if not (0 <= nr and 0 <= nc and nr+height <= p.rows and nc+width <= p.cols):
                continue
            # Check all destination cells, allowing overlap with this same piece.
            if any(occupied[rr*p.cols+cc] not in (-1, i)
                   for rr in range(nr, nr+height)
                   for cc in range(nc, nc+width)):
                continue
            next_state = list(state)
            next_state[i] = nr*p.cols+nc
            yield tuple(next_state), (p.names[i], direction)


def goal_cells(p: Problem) -> tuple[int, int, frozenset[int]]:
    """Top-left (row, col) of the target at the goal and the cells it covers there."""
    height, width = p.shapes[0]
    row, col = p.rows - height, p.exit_columns[0]
    cells = frozenset(rr*p.cols+cc for rr in range(row, row+height)
                      for cc in range(col, col+width))
    return row, col, cells


def heuristic(p: Problem, state: tuple[int, ...], goal) -> int:
    """Admissible and consistent lower bound on the remaining slides.

    h = Manhattan distance of the target to its goal position
        + number of other pieces lying on the goal cells.
    The target needs at least one slide per unit of distance, and every
    piece on a goal cell must slide at least once to clear it; those are
    different slides of different pieces, so h never overestimates.
    One slide changes h by at most 1, so h is also consistent.
    """
    goal_row, goal_col, cells = goal
    r, c = divmod(state[0], p.cols)
    distance = abs(goal_row - r) + abs(goal_col - c)
    occupied = occupancy(p, state)
    blockers = {occupied[idx] for idx in cells} - {-1, 0}
    return distance + len(blockers)


def rebuild_path(parents: dict, key):
    moves, states = [], [parents[key][2]]
    while parents[key][0] is not None:
        prev, action, _ = parents[key]
        moves.append(action)
        key = prev
        states.append(parents[key][2])
    moves.reverse()
    states.reverse()
    return moves, states


def solve_bfs(p: Problem, max_states: int = 3_000_000):
    start_time = perf_counter()
    start = p.initial
    start_key = canonical_key(p, start)
    # Parent dictionary stores (parent canonical key, move, concrete board).
    parents = {start_key: (None, None, start)}
    frontier = deque([start_key])
    expanded = 0
    while frontier:
        key = frontier.popleft()
        state = parents[key][2]
        expanded += 1
        if goal_test(p, state):
            moves, states = rebuild_path(parents, key)
            return moves, states, expanded, len(parents), perf_counter()-start_time
        for next_state, move in successors(p, state):
            next_key = canonical_key(p, next_state)
            if next_key not in parents:
                parents[next_key] = (key, move, next_state)
                frontier.append(next_key)
                if len(parents) >= max_states:
                    raise RuntimeError(f'State cap {max_states} reached')
    return None, None, expanded, len(parents), perf_counter()-start_time


def solve_astar(p: Problem, max_states: int = 3_000_000):
    start_time = perf_counter()
    goal = goal_cells(p)
    start = p.initial
    start_key = canonical_key(p, start)
    parents = {start_key: (None, None, start)}
    best_g = {start_key: 0}
    closed = set()
    tie = count()
    # Heap entries: (f, -g, tie, key). Among equal f, deeper nodes go first.
    frontier = [(heuristic(p, start, goal), 0, next(tie), start_key)]
    expanded = 0
    while frontier:
        _, neg_g, _, key = heapq.heappop(frontier)
        g = -neg_g
        if key in closed or g > best_g[key]:
            continue  # stale heap entry
        closed.add(key)
        state = parents[key][2]
        expanded += 1
        if goal_test(p, state):
            moves, states = rebuild_path(parents, key)
            return moves, states, expanded, len(parents), perf_counter()-start_time
        for next_state, move in successors(p, state):
            next_key = canonical_key(p, next_state)
            new_g = g + 1
            if next_key in closed or new_g >= best_g.get(next_key, new_g + 1):
                continue
            best_g[next_key] = new_g
            parents[next_key] = (key, move, next_state)
            f = new_g + heuristic(p, next_state, goal)
            heapq.heappush(frontier, (f, -new_g, next(tie), next_key))
            if len(parents) >= max_states:
                raise RuntimeError(f'State cap {max_states} reached')
    return None, None, expanded, len(parents), perf_counter()-start_time


SOLVERS = {'bfs': solve_bfs, 'astar': solve_astar}
# Kept for backward compatibility with the previous version of this file.
solve = solve_bfs


def render(p: Problem, state: tuple[int, ...]) -> str:
    cells = occupancy(p, state)
    lines = []
    for r in range(p.rows):
        lines.append(' '.join('.' if cells[r*p.cols+c] == -1
                              else p.names[cells[r*p.cols+c]] for c in range(p.cols)))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Generalized Klotski A* / BFS solver')
    parser.add_argument('--problem', help='problem id, e.g. p04 (default: all)')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--algorithm', choices=['astar', 'bfs', 'both'], default='both')
    parser.add_argument('--show-boards', action='store_true')
    parser.add_argument('--max-states', type=int, default=3_000_000)
    parser.add_argument('--file', type=Path,
                        default=Path(__file__).with_name('klotski_problems.txt'))
    args = parser.parse_args()
    problems = parse_problems(args.file)
    if args.problem and args.problem not in problems:
        parser.error(f'unknown problem {args.problem}; available: {", ".join(problems)}')
    chosen = list(problems.values()) if args.all or not args.problem else [problems[args.problem]]
    algorithms = ['astar', 'bfs'] if args.algorithm == 'both' else [args.algorithm]
    summary = []
    for p in chosen:
        for algorithm in algorithms:
            try:
                moves, states, expanded, discovered, seconds = SOLVERS[algorithm](p, args.max_states)
            except RuntimeError as error:
                print(f'\n{p.name} [{algorithm}]: STOPPED ({error})')
                summary.append((p.name, algorithm, 'cap', '-', '-', '-'))
                continue
            status = 'SOLVED' if moves is not None else 'NO SOLUTION'
            print(f'\n{p.name} [{algorithm}]: {status}')
            print(f'Moves: {len(moves) if moves is not None else "-"}; '
                  f'expanded: {expanded}; discovered: {discovered}; time: {seconds:.3f}s')
            summary.append((p.name, algorithm, len(moves) if moves is not None else '-',
                            expanded, discovered, f'{seconds:.3f}'))
            if moves is not None:
                print('Sequence: ' + ', '.join(f'{name}:{direction}' for name, direction in moves))
                if args.show_boards:
                    for step, state in enumerate(states):
                        print(f'\nStep {step}: {"start" if step == 0 else moves[step-1]}\n{render(p, state)}')
    print('\nproblem  algorithm  moves  expanded  discovered  time(s)')
    for row in summary:
        print('{:<8} {:<10} {:>5} {:>9} {:>11} {:>8}'.format(*row))


if __name__ == '__main__':
    main()
