"""
This script was created by Michael Self as my first research task at TAMU.
The function of this script is to pull in radar and satellite imagery from the Houston area,
overlaying them when needed. This will be used to find the boundaries between the different air masses.
"""

"""
Script to download and plot NEXRAD Level 2 radar data from AWS and GOES ABI visible satellite imagery.
Radar: KHGX (Houston)
Satellite: GOES ABI Band 2,3,1 (True Green Visible)
"""

import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import s3fs
import pyart
from goes2go import accessors, GOES
import pandas as pd
import os
import gc
from scipy.interpolate import griddata
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

RADAR_STATION = "KHGX"  # NEXRAD station
GOES_SATELLITE = 16 # GOES-19 (GOES-R series)

#DOMAIN_BOUNDS = { # KMLB bounds
#    "min_lon": -81.6721,
#    "max_lon": -79.6367,
#    "min_lat": 27.2106,
#    "max_lat": 29.0153,
#}

DOMAIN_BOUNDS = { # KHGX bounds
    "min_lon": -96.1164,
    "max_lon": -94.0545,
    "min_lat": 28.5698,
    "max_lat": 30.3741,
}

# DOMAIN_BOUNDS = { # KJAX bounds
#      "min_lon": -82.7434,
#      "max_lon": -80.6604,
#      "min_lat": 29.5825,
#      "max_lat": 31.3866,
#  }



def compute_goes_latlon(ds):
    """
    This function computes latitude and longitude from GOES-R series ABI,
    which is needed because GOES data has coordinates in x and y numbers.
    This code is from the GOES-R series Data Book
    """

    # Read in GOES ABI fixed grid projection variables and constants
    x_coordinate_1d = ds["x"][:]  # E/W scanning angle in radians
    y_coordinate_1d = ds["y"][:]  # N/S elevation angle in radians
    projection_info = ds["goes_imager_projection"]
    lon_origin = projection_info.longitude_of_projection_origin
    H = projection_info.perspective_point_height + projection_info.semi_major_axis
    r_eq = projection_info.semi_major_axis
    r_pol = projection_info.semi_minor_axis

    # Create 2D coordinate matrices from 1D coordinate vectors
    x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)

    # Equations to calculate latitude and longitude
    lambda_0 = (lon_origin * np.pi) / 180.0
    a_var = np.power(np.sin(x_coordinate_2d), 2.0) + (
        np.power(np.cos(x_coordinate_2d), 2.0)
        * (
            np.power(np.cos(y_coordinate_2d), 2.0)
            + (
                ((r_eq * r_eq) / (r_pol * r_pol))
                * np.power(np.sin(y_coordinate_2d), 2.0)
            )
        )
    )
    b_var = -2.0 * H * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
    c_var = (H**2.0) - (r_eq**2.0)
    r_s = (-1.0 * b_var - np.sqrt((b_var**2) - (4.0 * a_var * c_var))) / (2.0 * a_var)
    s_x = r_s * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
    s_y = -r_s * np.sin(x_coordinate_2d)
    s_z = r_s * np.cos(x_coordinate_2d) * np.sin(y_coordinate_2d)

    # Ignore numpy errors for sqrt of negative number; occurs for GOES-16 ABI CONUS sector data
    np.seterr(all="ignore")

    abi_lat = (180.0 / np.pi) * (
        np.arctan(
            ((r_eq * r_eq) / (r_pol * r_pol))
            * ((s_z / np.sqrt(((H - s_x) * (H - s_x)) + (s_y * s_y))))
        )
    )
    abi_lon = (lambda_0 - np.arctan(s_y / (H - s_x))) * (180.0 / np.pi)

    return abi_lat, abi_lon


def get_available_radar_times(station, start, end):
    """
    List available NEXRAD Level 2 files on AWS S3 for a given station and time range.
    Returns sorted list of datetime objects.
    """
    fs = s3fs.S3FileSystem(anon=True)
    radar_times = []
    current = start
    while current.date() <= end.date():
        date_path = current.strftime("%Y/%m/%d")
        prefix = f"unidata-nexrad-level2/{date_path}/{station}"
        files = fs.ls(prefix)

        for f in files:
            fname = f.split("/")[-1]
            ts = fname[len(station) : len(station) + 15]  # YYYYMMDD_HHMMSS
            radar_times.append(datetime.datetime.strptime(ts, "%Y%m%d_%H%M%S"))
        current += datetime.timedelta(days=1)

    # filter to range, deduplicate, sorted
    # sometimes there are duplicates, so this gets rid of that
    uniq = {t for t in radar_times if start <= t <= end}
    return sorted(uniq)


def get_available_goes_times(start, end):
    """
    List available GOES ABI files on AWS S3 for a given time range.
    Returns sorted list of datetime objects.
    """
    G = GOES(satellite=GOES_SATELLITE, product="ABI-L2-CMIPC", domain="C", bands=2)
    # make empty list to hold times
    goes_times = []
    # use goes2go df function to grab all the metadata, without downloading files
    df = G.df(start, end)  # should be super fast

    # average start and end time so the file name time makes more sense
    for index, row in df.iterrows():
        start_time = pd.to_datetime(row["start"]).to_pydatetime()
        end_time = pd.to_datetime(row["end"]).to_pydatetime()
        midpoint_time = start_time + (end_time - start_time) / 2
        goes_times.append(midpoint_time)

    # deduplicate & sort
    # same issue as radar, sometimes duplicates
    uniq = {t for t in goes_times if start <= t <= end}
    return sorted(uniq)


def find_pairs(radar_times, goes_times, max_diff_minutes=2.5):
    """
    Find pairs of radar and GOES times that are within max_diff_minutes of each other.
    Returns a dataframe of rad, sat, and paired times.
    """
    pairs = []
    max_diff = datetime.timedelta(minutes=max_diff_minutes)

    # loop through each GOES time and find a radar time within the max_diff
    for g_time in goes_times:
        for r_time in radar_times:
            if abs(g_time - r_time) <= max_diff:
                pairs.append((g_time, r_time))
                break  # just pair one GOES times with one radar time

    # Create a DataFrame from the pairs
    pairs_df = pd.DataFrame(pairs, columns=["GOES_Time", "Radar_Time"])

    # add a column for the average time of the pair
    pairs_df["Pair_Time"] = pairs_df.apply(
        lambda row: row["GOES_Time"] + (row["Radar_Time"] - row["GOES_Time"]) / 2,
        axis=1,
    )

    return pairs_df


def get_nexrad_level2_aws(station, dt):
    """
    Download NEXRAD Level 2 file from AWS S3 and return Py-ART Radar object.
    """
    fs = s3fs.S3FileSystem(anon=True)
    date_path = dt.strftime("%Y/%m/%d")
    prefix = f"unidata-nexrad-level2/{date_path}/{station}"
    files = fs.ls(prefix)
    radar_file = None
    min_diff = None  # used to find closest time
    for f in files:
        fname = f.split("/")[-1]
        try:
            ts = fname[len(station) : len(station) + 15]  # YYYYMMDD_HHMMSS
            file_dt = datetime.datetime.strptime(ts, "%Y%m%d_%H%M%S")
            diff = abs((file_dt - dt).total_seconds())  # to find closest time
            if min_diff is None or diff < min_diff:
                min_diff = diff
                radar_file = f
        except Exception:
            continue
    with fs.open(radar_file, "rb") as f:
        radar = pyart.io.read_nexrad_archive(f)

    return radar


def get_goes_imagery(dt):
    """
    Download GOES ABI imagery from AWS S3, using Goes2Go
    and return xarray Dataset for each band.
    """
    G_red = GOES(satellite=GOES_SATELLITE, product="ABI-L2-CMIPC", domain="C", bands=2)
    ds_red = G_red.nearesttime(dt, verbose=False, overwrite=True, save_dir="../../data/")

    G_green = GOES(satellite=GOES_SATELLITE, product="ABI-L2-CMIPC", domain="C", bands=3)
    ds_green = G_green.nearesttime(dt, verbose=False, overwrite=True, save_dir="../../data/")

    G_blue = GOES(satellite=GOES_SATELLITE, product="ABI-L2-CMIPC", domain="C", bands=1)
    ds_blue = G_blue.nearesttime(dt, verbose=False, overwrite=True, save_dir="../../data/")

    return ds_red, ds_green, ds_blue


def resample_to_grid(
    RGB_cropped, lat_cropped, lon_cropped, min_lon, max_lon, min_lat, max_lat, out_shape
):
    """
    Resample cropped GOES channel(s) using scipy.interpolate.griddata onto a PlateCarree grid.
    Interpolates each channel independently.
    """
    ny, nx = out_shape

    # target grid: lon increases left->right, lat decreases top->bottom for image origin='upper'
    target_lon = np.linspace(min_lon, max_lon, nx)
    target_lat = np.linspace(max_lat, min_lat, ny)  # top->bottom
    target_lon2d, target_lat2d = np.meshgrid(target_lon, target_lat)

    # ensure numpy arrays
    arr = np.asarray(RGB_cropped)
    latf = np.asarray(lat_cropped)
    lonf = np.asarray(lon_cropped)

    # flatten source coords/values and mask invalids
    pts_lon = lonf.ravel()
    pts_lat = latf.ravel()
    valid_coord = np.isfinite(pts_lon) & np.isfinite(pts_lat)
    if not np.any(valid_coord):
        # nothing to interpolate
        if arr.ndim == 3:
            return np.zeros((ny, nx, arr.shape[2]))
        return np.zeros((ny, nx))

    points = np.column_stack((pts_lon[valid_coord], pts_lat[valid_coord]))

    # function to interpolate a single flattened channel vector (individually)
    def _interp_channel(vals_flat):
        vals = np.asarray(vals_flat).ravel()
        valid = valid_coord & np.isfinite(vals)
        if valid.sum() == 0:
            return np.zeros((ny, nx))
        values = vals[valid]
        pts = points  # already filtered lon/lat for valid_coord

        # use nearest interpolation so data isnt "made up"
        grid = griddata(
            pts, values, (target_lon2d, target_lat2d), method="nearest", fill_value=0.0
        )

        return grid

    channels = []
    for c in range(arr.shape[2]):
        ch_flat = arr[..., c].ravel()
        ch_grid = _interp_channel(ch_flat)
        channels.append(ch_grid)
    rgb = np.stack(channels, axis=2)  # ny,nx,3

    return np.clip(rgb, 0.0, 1.0)


def rescale_array_to_shape(arr, target_shape):
    """
    Rescale red band shape to `target_shape` (0.5km to 1km resolution).
    Uses block-averaging and should work well for downscaling.
    """
    arr = np.asarray(arr)  # convert to numpy array if not already
    trg_r, trg_c = target_shape
    r, c = arr.shape

    # integer block reduction
    if r % trg_r == 0 and c % trg_c == 0:
        fy = r // trg_r
        fx = c // trg_c
        # trim to exact multiple just in case
        arr2 = arr[: trg_r * fy, : trg_c * fx]
        # reshape and average blocks, 1x1 to 2x2
        return arr2.reshape(trg_r, fy, trg_c, fx).mean(axis=(1, 3))
    else:
        print(f"Warning: non-integer rescale from {arr.shape} to {target_shape}")


def save_radar_png_simple(
    radar,
    min_lon,
    max_lon,
    min_lat,
    max_lat,
    nexrad_dt,
    pair_time,
    reflectivity_threshold=8,
):
    """
    Save radar-only PNG at 512x448 without anything but the reflectivity pixels.
    Adds minimal PNG metadata: true_time, paired_time, pair_diff_seconds.
    """
    # set up figure
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(5.12, 4.48),
        dpi=100,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    # create pyart GateFilter to mask reflectivity below threshold
    gf = pyart.filters.GateFilter(radar)
    gf.exclude_below("reflectivity", reflectivity_threshold)

    display = pyart.graph.RadarMapDisplay(radar)
    display.plot_ppi_map(
        "reflectivity",  # variable for ppi map
        1,  # sweep number or scan (1 is 0.5 deg)
        vmin=reflectivity_threshold,
        vmax=65,
        fig=fig,
        ax=ax,
        lat_0=radar.latitude["data"][0],  # lat of radar
        lon_0=radar.longitude["data"][0],  # lon of radar
        colorbar_flag=False,
        title_flag=False,
        add_grid_lines=False,
        embellish=False,
        gatefilter=gf,
        cmap="HomeyerRainbow",  # colorblind friendly
    )

    # set extent to research domain (200km x 200km box around KHGX)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # make sure no ticks/labels/axes are shown
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # # save to right path (/YYYYMMDD/rad/YYYYMMDD_HHMM_rad.png)
    # day_folder = nexrad_dt.strftime("%Y%m%d")
    # os.makedirs(f"/storm/self/images/{day_folder}/rad", exist_ok=True)
    # t = nexrad_dt.strftime("%H%M")
    # p = pair_time.strftime("%Y%m%d_%H%M")
    # filename = f"/storm/self/images/{day_folder}/rad/{p}_{t}_rad.png"
    # plt.savefig(filename, pad_inches=0, dpi=100)
    # plt.close(fig)
    # print(f"Saved: {filename}")

    # save to right path (/YYYYMMDD/rad/YYYYMMDD_HHMM_rad.png)
    day_folder = nexrad_dt.strftime("%Y%m%d")
    os.makedirs(f"../other_images/{RADAR_STATION}/{day_folder}/rad", exist_ok=True)
    t = nexrad_dt.strftime("%H%M")
    p = pair_time.strftime("%Y%m%d_%H%M")
    filename = f"../other_images/{RADAR_STATION}/{day_folder}/rad/{p}_{t}_rad.png"
    plt.savefig(filename, pad_inches=0, dpi=100)
    plt.close(fig)
    print(f"Saved: {filename}")


def save_satellite_png_simple(
    lat, lon, RGB, min_lon, max_lon, min_lat, max_lat, sat_dt, pair_time
):
    """Save satellite-only PNG at 512x448 without coastlines (resampled to research domain).
    Adds minimal PNG metadata: true_time, paired_time, pair_diff_seconds.
    """
    # mask pixels within bbox and crop to minimal window
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    RGB = np.asarray(RGB)  # the 3-channel array made in main block
    lat_mask = (lat >= min_lat) & (lat <= max_lat)
    lon_mask = (lon >= min_lon) & (lon <= max_lon)
    combined_mask = lat_mask & lon_mask

    rows, cols = np.where(combined_mask)
    min_row, max_row = int(rows.min()), int(rows.max())
    min_col, max_col = int(cols.min()), int(cols.max())

    # now crop to that domain
    RGB_cropped = RGB[min_row : max_row + 1, min_col : max_col + 1, :]
    lat_cropped = lat[min_row : max_row + 1, min_col : max_col + 1]
    lon_cropped = lon[min_row : max_row + 1, min_col : max_col + 1]

    # resample to 512x448 grid matching radar pixel grid
    RGB_resampled = resample_to_grid(
        RGB_cropped,
        lat_cropped,
        lon_cropped,
        min_lon,
        max_lon,
        min_lat,
        max_lat,
        out_shape=(512, 448),
    )

    # sanity check shape
    if RGB_resampled.shape != (512, 448, 3):
        print(
            f"Resampled satellite image has wrong shape at {sat_dt}: {RGB_resampled.shape}"
        )
        return

    # plot and save using imshow instead of matplotlib because of 3-channel array (RGB)
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(5.12, 4.48),
        dpi=100,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    extent = [min_lon, max_lon, min_lat, max_lat]
    ax.imshow(
        RGB_resampled, origin="upper", extent=extent, transform=ccrs.PlateCarree()
    )
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # day_folder = sat_dt.strftime("%Y%m%d")
    # os.makedirs(f"/storm/self/images/{day_folder}/sat", exist_ok=True)
    # t = sat_dt.strftime("%H%M")
    # p = pair_time.strftime("%Y%m%d_%H%M")
    # filename = f"/storm/self/images/{day_folder}/sat/{p}_{t}_sat.png"
    # plt.savefig(filename, pad_inches=0, dpi=100)
    # plt.close(fig)
    # print(f"Saved: {filename}")
    
    day_folder = sat_dt.strftime("%Y%m%d")
    os.makedirs(f"../other_images/{RADAR_STATION}/{day_folder}/sat", exist_ok=True)
    t = sat_dt.strftime("%H%M")
    p = pair_time.strftime("%Y%m%d_%H%M")
    filename = f"../other_images/{RADAR_STATION}/{day_folder}/sat/{p}_{t}_sat.png"
    plt.savefig(filename, pad_inches=0, dpi=100)
    plt.close(fig)
    print(f"Saved: {filename}")

def process_single_radar(radar_dt, pair_time):
    min_lon, max_lon, min_lat, max_lat = DOMAIN_BOUNDS["min_lon"], DOMAIN_BOUNDS["max_lon"], DOMAIN_BOUNDS["min_lat"], DOMAIN_BOUNDS["max_lat"]

    try:

        # start with getting radar image from the specific time
        radar = get_nexrad_level2_aws(RADAR_STATION, radar_dt)

        # plot the radar image and save it
        save_radar_png_simple(
            radar, min_lon, max_lon, min_lat, max_lat, radar_dt, pair_time
        )

    except Exception as e:
        print(f"Error processing radar at {radar_dt}: {e}")

    # close anything open still and delete radar object
    plt.close("all")
    del radar
    gc.collect()


def download_single_satellite(goes_dt, pair_time):
    """
    Download GOES satellite data and return processed RGB array with metadata.
    This runs in parallel.
    """
    min_lon, max_lon, min_lat, max_lat = DOMAIN_BOUNDS["min_lon"], DOMAIN_BOUNDS["max_lon"], DOMAIN_BOUNDS["min_lat"], DOMAIN_BOUNDS["max_lat"]

    # start with getting the 3 band datasets (red, green, blue) from the specific time
    ds_red, ds_green, ds_blue = get_goes_imagery(dt=goes_dt)

    # turn these datasets into numpy arrays (for the CMI variable, which is reflectance scaled 0-1)
    R = np.asarray(ds_red["CMI"].values)
    G = np.asarray(ds_green["CMI"].values)
    B = np.asarray(ds_blue["CMI"].values)

    # determine a consistent target shape for all arrays, using green's shape
    target_shape = (G.shape[0], G.shape[1])  # should be 3000 x 5000 (1km resolution)

    # compute lat/lon from green band dataset
    lat, lon = compute_goes_latlon(ds_green)

    # rescale red channel to target shape (green and blue shape)
    if R.shape[:2] != target_shape:
        R = rescale_array_to_shape(R, target_shape)

    # apply clipping and gamma as in previous true-color implementation
    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    gamma = 2.2
    R_lin = np.power(R, 1 / gamma)
    G_lin = np.power(G, 1 / gamma)
    B_lin = np.power(B, 1 / gamma)

    # make the "true-green" composite
    G_true = 0.45 * R_lin + 0.1 * G_lin + 0.45 * B_lin
    G_true = np.clip(G_true, 0, 1)
    RGB = np.dstack([R_lin, G_true, B_lin])

    # clean up datasets
    del R, G, B, R_lin, G_lin, B_lin, G_true, ds_red, ds_green, ds_blue
    gc.collect()

    return (lat, lon, RGB, goes_dt, pair_time)


def create_satellite_image(sat_data):
    """
    Create and save satellite image from downloaded data.
    This runs sequentially.
    """
    lat, lon, RGB, goes_dt, pair_time = sat_data
    min_lon, max_lon, min_lat, max_lat = DOMAIN_BOUNDS["min_lon"], DOMAIN_BOUNDS["max_lon"], DOMAIN_BOUNDS["min_lat"], DOMAIN_BOUNDS["max_lat"]

    # now we can plot and save the satellite image
    save_satellite_png_simple(
        lat, lon, RGB, min_lon, max_lon, min_lat, max_lat, goes_dt, pair_time
    )

    # close figures and delete variables
    plt.close("all")
    del lat, lon, RGB
    gc.collect()


def process_single_satellite(goes_dt, pair_time):
    """
    Download and process a single GOES satellite image.
    """
    min_lon, max_lon, min_lat, max_lat = DOMAIN_BOUNDS["min_lon"], DOMAIN_BOUNDS["max_lon"], DOMAIN_BOUNDS["min_lat"], DOMAIN_BOUNDS["max_lat"]

    # start with getting the 3 band datasets (red, green, blue) from the specific time
    ds_red, ds_green, ds_blue = get_goes_imagery(dt=goes_dt)

    # turn these datasets into numpy arrays (for the CMI variable, which is reflectance scaled 0-1)
    R = np.asarray(ds_red["CMI"].values)
    G = np.asarray(ds_green["CMI"].values)
    B = np.asarray(ds_blue["CMI"].values)

    # determine a consistent target shape for all arrays, using green's shape
    target_shape = (G.shape[0], G.shape[1])  # should be 3000 x 5000 (1km resolution)

    # compute lat/lon from green band dataset
    lat, lon = compute_goes_latlon(ds_green)

    # rescale red channel to target shape (green and blue shape)
    if R.shape[:2] != target_shape:
        R = rescale_array_to_shape(R, target_shape)

    # apply clipping and gamma as in previous true-color implementation
    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    gamma = 2.2
    R_lin = np.power(R, 1 / gamma)
    G_lin = np.power(G, 1 / gamma)
    B_lin = np.power(B, 1 / gamma)

    # make the "true-green" composite
    G_true = 0.45 * R_lin + 0.1 * G_lin + 0.45 * B_lin
    G_true = np.clip(G_true, 0, 1)
    RGB = np.dstack([R_lin, G_true, B_lin])

    # now we can plot and save the satellite image
    save_satellite_png_simple(
        lat, lon, RGB, min_lon, max_lon, min_lat, max_lat, goes_dt, pair_time
    )

    # close figures and delete variables
    plt.close("all")
    del R, G, B, R_lin, G_lin, B_lin, G_true, RGB, lat, lon, ds_red, ds_green, ds_blue
    gc.collect()


if __name__ == "__main__":
    # define the date range (inclusive). Only the date part matters; time window is fixed per day.
    start_date = datetime.date(2022, 9, 11)
    end_date = datetime.date(2022, 9, 11)
    day = start_date

    while day <= end_date:
        start = datetime.datetime(day.year, day.month, day.day, 14, 0)
        end = datetime.datetime(day.year, day.month, day.day, 23, 59)

        print(f"Processing day: {day.isoformat()} (14:00–23:59 UTC)")

        # find all the radar and satellite data times for this day
        radar_times = get_available_radar_times(RADAR_STATION, start, end)
        goes_times = get_available_goes_times(start, end)

        # now pair the sat data with radar data that is within 2 or 2.5 minutes of it
        pairs_df = find_pairs(radar_times, goes_times)
        print(f"Found {len(pairs_df)} GOES-radar pairs within 2.5 minutes.")

        # Use ProcessPoolExecutor for radar (proven to work)
        print("Processing radar data...")
        with ProcessPoolExecutor(max_workers=32) as executor:
            list(
                executor.map(
                    process_single_radar,
                    pairs_df["Radar_Time"].tolist(),
                    pairs_df["Pair_Time"].tolist(),
                )
            )

        # Use ThreadPoolExecutor for satellite downloads (I/O-bound)
        print("Downloading satellite data...")
        with ThreadPoolExecutor(max_workers=16) as executor:
            satellite_data = list(
                executor.map(
                    download_single_satellite,
                    pairs_df["GOES_Time"].tolist(),
                    pairs_df["Pair_Time"].tolist(),
                )
            )

        # Process satellite images sequentially (avoids parallel plotting issues)
        print("Creating satellite images...")
        for sat_data in satellite_data:
            create_satellite_image(sat_data)

        # delete the downloaded satellite data to free memory
        del satellite_data
        gc.collect()

        print(f"Finished day: {day.isoformat()}")
        day += datetime.timedelta(days=1)

    print("All done.")
