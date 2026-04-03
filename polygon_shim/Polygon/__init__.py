# PaddleOCR expects "import Polygon as plg" and plg.Polygon(bbox).area(), pD & pG.
# Shapely provides the same; we add __and__ for intersection so (pD & pG).area() works.
from shapely.geometry import Polygon as _ShapelyPolygon


class Polygon(_ShapelyPolygon):
    """Drop-in for Polygon3: construct from Nx2 array/list, .area(), and __and__ for intersection."""

    def __and__(self, other):
        return self.intersection(other)
