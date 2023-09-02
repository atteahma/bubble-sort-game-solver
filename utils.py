import numpy as np
from numba import njit
from numba.typed import List


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


@njit()
def board_is_done(bins):
    for b in bins:
        if not bin_is_done(b, empty_is_done=True):
            return False
    return True


@njit()
def get_top_ball(bin):
    for ball in bin[::-1]:
        if ball > -1:
            return ball
    return -1


@njit()
def pop_top_ball(bin):
    for i, ball in enumerate(bin[::-1]):
        if ball > -1:
            bin[i] = -1
            return ball
    return -1


@njit()
def set_top_ball(bin, b):
    for i, ball in enumerate(bin):
        if ball == -1:
            bin[i] = b
            break


@njit()
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


@njit()
def do_move(bins, i: int, j: int):
    b = pop_top_ball(bins[i])
    set_top_ball(bins[j], b)


@njit()
def get_available_moves(bins):
    num_bins = len(bins)
    moves = List()

    for from_i in range(num_bins):
        if bin_is_done(bins[from_i]):
            continue

        for to_i in range(num_bins):
            if from_i == to_i:
                continue

            from_b = get_top_ball(bins[from_i])
            to_b = get_top_ball(bins[to_i])

            if from_b == -1:
                continue
            elif to_b == -1:
                moves.append((from_i, to_i))
            elif from_b == to_b:
                moves.append((from_i, to_i))

    return moves


def one2one(bins):
    return tuple(sorted(map(tuple, bins)))
