import math
from typing import Tuple

def haversine_distance_points(point1: Tuple, point2: Tuple) -> float:
    """Helper to calculate haversine between two (lon, lat) tuples."""
    R = 3959  # Earth's radius in miles
    lon1, lat1 = point1
    lon2, lat2 = point2
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a)) 