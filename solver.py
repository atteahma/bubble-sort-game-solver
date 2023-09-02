from utils import do_move, get_available_moves, board_is_done, one2one


class Solver:
    def __init__(self):
        return

    # takes in bins of balls and returns a sequence of bin moves
    def solve(self, bins):
        moves = []
        self.solution = [None for _ in range(100)]
        self.seen_depths = {}

        self.halt = False
        self.curr_depth_value = 0
        self.depth_by_call_order = []

        print(bins)

        self.recurse(moves, bins)

        print(self.solution)
        print(len(self.solution))
        return self.solution

    def recurse(self, curr_moves_seq, bins):
        if self.halt:
            return

        curr_depth = len(curr_moves_seq)

        if curr_depth >= len(self.solution):
            return

        bins_hashable = one2one(bins)
        if (
            bins_hashable in self.seen_depths
            and self.seen_depths[bins_hashable] <= curr_depth
        ):
            return
        self.seen_depths[bins_hashable] = curr_depth

        self.curr_depth_value += 1
        self.depth_by_call_order.append((self.curr_depth_value, curr_depth))

        if board_is_done(bins):
            print(
                f"found solution at depth {len(curr_moves_seq)} seen size={len(self.seen_depths):,}"
            )
            if len(curr_moves_seq) < len(self.solution):
                self.solution = curr_moves_seq.copy()
                self.delete_large()

                # if len(self.solution) < 56:
                #     self.halt = True
            return

        possible_moves = get_available_moves(bins)

        # remove reverse of previous move
        if curr_depth > 0:
            last_move = curr_moves_seq[-1]
            rev_last_move = (last_move[1], last_move[0])
            if rev_last_move in possible_moves:
                possible_moves.remove(rev_last_move)

        for possible_move in possible_moves:
            do_move(bins, possible_move[0], possible_move[1])
            curr_moves_seq.append(possible_move)
            self.recurse(curr_moves_seq, bins)
            curr_moves_seq.pop()
            do_move(bins, possible_move[1], possible_move[0])

    def delete_large(self):
        n = 0
        for bins_h, depth in tuple(self.seen_depths.items()):
            if depth > len(self.solution):
                n += 1
                del self.seen_depths[bins_h]
        print(f"deleted {n} elems")
