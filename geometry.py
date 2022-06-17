from shapely.geometry import Point
import numpy as np
import skimage

def projection(x1, y1, x2, y2, cx, cy):
    dx = x2 - x1
    dy = y2 - y1
    cdx = cx - x1
    cdy = cy - y1
    t = (cdx * dx + cdy * dy) / (dx ** 2 + dy ** 2)
    inshape = (t >= 0) & (t <= 1)
    px = x1 + t * dx
    py = y1 + t * dy
    r = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
    r = r[inshape]
    mask = r == np.min(r)
    px = cx + np.sum((px[inshape] - cx)[mask])
    py = cy + np.sum((py[inshape] - cy)[mask])
    r = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
    theta = np.arctan2(py - cy, px - cx)
    return r, theta, px, py

def resultant(dx, dy, theta):
    rx = dx * np.cos(theta)
    ry = dy * np.sin(theta)
    return rx + ry

def inlayer(layer, xi, yi):
    n, m = xi.shape
    xi, yi = xi.flatten(), yi.flatten()
    k = n * m
    mask = np.array([layer.contains(Point(xi[i], yi[i])).item() for i in range(k)])
    return mask.reshape(n, m)

def make_grid(y, mask):
    a = np.zeros(mask.shape) * np.nan
    a[mask] = y
    return a

def contours(z, level, left, right, top, bottom):
    ny, nx = np.shape(z)
    contour_coords = skimage.measure.find_contours(z, level)
    coords = []
    for contour in contour_coords:
        c = np.zeros(contour.shape)
        c[:, 1] = top + (contour[:, 0]/ (ny -1)) * (bottom - top)
        c[:, 0] = left + (contour[:, 1]/(nx -1)) * (right - left)
        coords.append(c)
    return coords
