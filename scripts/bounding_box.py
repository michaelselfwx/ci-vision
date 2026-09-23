from pyproj import Geod


def bounding_box(center_lat, center_lon, side_km):
    g = Geod(ellps="WGS84")
    half_side = side_km / 2.0

    # Calculate the coordinates at the four cardinal directions
    north_lon, north_lat, _ = g.fwd(center_lon, center_lat, 0, half_side * 1000)
    south_lon, south_lat, _ = g.fwd(center_lon, center_lat, 180, half_side * 1000)
    east_lon, east_lat, _ = g.fwd(center_lon, center_lat, 90, half_side * 1000)
    west_lon, west_lat, _ = g.fwd(center_lon, center_lat, 270, half_side * 1000)

    # Bounding box: [min_lon, max_lon, min_lat, max_lat]
    min_lat = min(south_lat, north_lat)
    max_lat = max(south_lat, north_lat)
    min_lon = min(west_lon, east_lon)
    max_lon = max(west_lon, east_lon)

    return min_lon, max_lon, min_lat, max_lat


if __name__ == "__main__":
    center_lat = 29.4720001
    center_lon = -95.0854344
    bbox = bounding_box(center_lat, center_lon, 200)
    print("Bounding box:", bbox)

# KHGK: 29.4720001, -95.0854344
# KMLB: 28.11305, -80.65444
# KJAX: 30.48463, -81.7019