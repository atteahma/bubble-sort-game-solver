import numpy as np


def mask3dwith2d(im, mask):
    masked_im3d = np.zeros_like(im)
    for i in range(3):
        masked_im3d[:, :, i] = im[:, :, i] * mask
    return masked_im3d


def segment_image(im, minGap=25, pad=5, bgColor=0, axis=0):
    s = im.shape[axis]

    if len(im.shape) == 3:
        isbg = np.all(im == bgColor, axis=2)
    else:
        isbg = im == bgColor

    lines = np.nonzero(np.all(isbg, axis=~axis))[0]
    bottom_edges = (lines - 1)[1:]
    upper_edges = (lines + 1)[:-1]

    mask = (bottom_edges - minGap) > upper_edges
    upper_edges = np.clip(upper_edges[mask] - pad, 0, s)
    bottom_edges = np.clip(bottom_edges[mask] + pad, 0, s)

    seperators = np.sort(np.concatenate((upper_edges, bottom_edges)))

    return np.split(im, seperators, axis=axis)[1::2]


def board_is_done(bins):
    for b in bins:
        if not bin_is_done(b, empty_is_done=True):
            return False
    return True


def get_top_ball(bin):
    for ball in reversed(bin):
        if ball > -1:
            return ball
    return -1


def pop_top_ball(bin):
    for i, ball in reversed(tuple(enumerate(bin))):
        if ball > -1:
            bin[i] = -1
            return ball
    return -1


def set_top_ball(bin, b):
    for i, ball in enumerate(bin):
        if ball == -1:
            bin[i] = b
            return


def bin_is_done(bin, empty_is_done=False):
    if empty_is_done:
        return (bin[0] == bin[1]) and (bin[1] == bin[2]) and (bin[2] == bin[3])
    else:
        return (
            (bin[0] != -1)
            and (bin[0] == bin[1])
            and (bin[1] == bin[2])
            and (bin[2] == bin[3])
        )


def bin_is_full(bin):
    return bin[3] != -1


def do_move(bins, i: int, j: int):
    b = pop_top_ball(bins[i])
    set_top_ball(bins[j], b)


def get_available_moves(bins):
    num_bins = len(bins)
    moves = []

    for from_i in range(num_bins):
        if bin_is_done(bins[from_i]):
            continue

        from_b = get_top_ball(bins[from_i])

        if from_b == -1:
            continue

        for to_i in range(num_bins):
            if from_i == to_i:
                continue

            if bin_is_full(bins[to_i]):
                continue

            to_b = get_top_ball(bins[to_i])

            if to_b == -1:
                moves.append((from_i, to_i))
            elif from_b == to_b:
                moves.append((from_i, to_i))

    return moves


def hash_bins(bins):
    return tuple(sorted(map(tuple, bins)))


def index_to_ij(index, grid_widths):
    i = 1
    for width in grid_widths:
        if index < width:
            return i, index + 1
        index -= width
        i += 1
    return None, None
