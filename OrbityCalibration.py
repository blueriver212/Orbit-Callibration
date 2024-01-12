# All functions
import sp3
from astropy.coordinates import GCRS, ITRS, SkyCoord, TEME
from astropy import units
from astropy.time import Time
import numpy as np
np.set_printoptions(precision=15, suppress=True)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sgp4.api import Satrec, jday
from datetime import datetime
from sgp4.earth_gravity import wgs84
import json
from datetime import datetime, timedelta
from dateutil.tz import tzutc
from Utilities import utc_to_jd 
import os
import pickle
from datetime import datetime, timedelta, timezone
import numpy as np
from astropy.time import Time
from astropy.coordinates import GCRS, ITRS
from astropy import units

def process_raw_sp3_vectorized(product):
    # Extract positions and velocities
    positions = np.array([record.position for record in product.satellites[0].records])
    velocities = np.array([record.velocity for record in product.satellites[0].records])

    # Convert positions and velocities to Astropy Quantity objects
    positions = positions * units.meter
    velocities = velocities * units.meter / units.second

    # Convert times and adjust for leap seconds
    times = Time([record.time for record in product.satellites[0].records]) - timedelta(seconds=18)

    # Perform the transformation in a vectorized manner
    itrs = ITRS(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], 
                v_x=velocities[:, 0], v_y=velocities[:, 1], v_z=velocities[:, 2], 
                representation_type='cartesian', differential_type='cartesian', obstime=times)
    gcrs = itrs.transform_to(GCRS(obstime=times))

    # Extract transformed positions and velocities
    transformed_positions = np.vstack([gcrs.cartesian.x.value, gcrs.cartesian.y.value, gcrs.cartesian.z.value]).T
    transformed_velocities = np.vstack([gcrs.velocity.d_x.value * 1000, gcrs.velocity.d_y.value * 1000, gcrs.velocity.d_z.value * 1000]).T

    # Construct the ephemeris dictionary
    sp3_ephemeris = [{'time': time, 'position': pos, 'velocity': vel} 
                     for time, pos, vel in zip(times, transformed_positions, transformed_velocities)]

    return sp3_ephemeris

def utc_to_jd(time_stamps):
    """
    Convert UTC datetime or string representations to Julian Date (JD).

    This function takes in either a single datetime, string representation of a datetime, or a list of them. It then
    converts each of these into its corresponding Julian Date (JD) value. If a list is provided, it returns a list of JD
    values. If a single datetime or string is provided, it returns a single JD value.

    :param time_stamps: The datetime object(s) or string representation(s) of dates/times to be converted to Julian Date.
                        Strings should be in the format '%Y-%m-%d' or '%Y-%m-%d %H:%M:%S'.
    :type time_stamps: datetime.datetime, str or list of datetime.datetime/str
    :return: The corresponding Julian Date (JD) value(s) for the provided datetime(s) or string representation(s).
             Returns a single float if the input is a single datetime or string, and a list of floats if the input is a list.
    :rtype: float or list of float

    .. note:: The function uses the 'astropy' library for the conversion, so ensure that 'astropy' is installed and available.
    """
    if not isinstance(time_stamps, list):
        time_stamps = [time_stamps]

    UTC_string = []
    for ts in time_stamps:
        try:
            UTC_string.append(ts.strftime('%Y-%m-%d %H:%M:%S'))
        except:
            time_str = datetime.datetime.strptime(ts, '%Y-%m-%d')
            UTC_string.append(time_str.strftime('%Y-%m-%d %H:%M:%S'))

    t = Time(UTC_string, format='iso', scale='utc')  # astropy time object
    jd = t.to_value('jd', 'long')  # convert to jd

    jd_vals = [float(j) for j in jd]

    # If the input was a single datetime, then return a single value. Otherwise, return the list.
    return jd_vals[0] if len(jd_vals) == 1 else jd_vals

def tle_propagate_using_sp3_time(input_tles, sp3_time):
    tle_ephemeris = []
    for _sp3_time in sp3_time:
        most_recent_tle = None
        for tle in reversed(input_tles):
            tle_time = datetime.strptime(tle['EPOCH'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=tzutc())
            if tle_time <= _sp3_time:
                most_recent_tle = tle
                break

        if not most_recent_tle:
            # Handle case where no valid TLE is found for _sp3_time
            continue

        # Propagate using SGP4 for that specific timestep with the found TLE:
        satellite = Satrec.twoline2rv(most_recent_tle['TLE_LINE1'], most_recent_tle['TLE_LINE2'])
        jd, fr = jday(_sp3_time.year, _sp3_time.month, _sp3_time.day, _sp3_time.hour, _sp3_time.minute, _sp3_time.second)
        e, r, v = satellite.sgp4(jd, fr)

        x,y,z= r[0], r[1], r[2]
        u,v,w= v[0], v[1], v[2]

        # tle data is in the TEME frame, this will need to be converted to J2000
        time = utc_to_jd(_sp3_time)
        astropy_time = Time(time, format="jd")
        #astropy_time = Time(_sp3_time)

        # convert to astropy skycoord object
        skycoord = SkyCoord(x, y, z, unit='km', representation_type='cartesian', frame=TEME(obstime=astropy_time))

        # convert to GCRS frame
        gcrs = skycoord.transform_to(GCRS(obstime=astropy_time))

        # convert to cartesian coordinates
        x, y, z = gcrs.cartesian.xyz.to(units.m)

        # convert to astropy skycoord object
        skycoord = SkyCoord(u, v, w, unit='km/s', representation_type='cartesian', frame=TEME(obstime=astropy_time))

        # convert to GCRS frame
        gcrs = skycoord.transform_to(GCRS(obstime=astropy_time))

        # convert to cartesian coordinates
        u, v, w = gcrs.cartesian.xyz.to(units.m/units.s)

        # Store the result in the ephemeris list:
        tle_ephemeris.append({
            'time': _sp3_time,
            'position': np.array([x.value, y.value, z.value]),
            'velocity': np.array([u.value, v.value, w.value]),
        })

    return tle_ephemeris

def HCL_diff(eph1, eph2):
    H_diffs = []
    C_diffs = []
    L_diffs = []
    time = []
    positions = []
    xyz_correction = []
    velocity = []

    for i in range(0, len(eph1), 1):
        
        # charles 2023
        r1 = np.array(eph1[i]['position'])
        r2 = np.array(eph2[i]['position'])

        v1 = np.array(eph1[i]['velocity'])
        v2 = np.array(eph2[i]['velocity'])

        unit_radial = r1/np.linalg.norm(r1)
        unit_cross_track = np.cross(r1, v1)/np.linalg.norm(np.cross(r1, v1))
        unit_along_track = np.cross(unit_radial, unit_cross_track)

        #put the three unit vectors into a matrix
        unit_vectors = np.array([unit_radial, unit_cross_track, unit_along_track])

        #subtract the two position vectors
        r_diff = r1 - r2

        #relative position in HCL frame
        r_diff_HCL = np.matmul(unit_vectors, r_diff)

        #height, cross track and along track differences
        h_diff = r_diff_HCL[0]
        c_diff = r_diff_HCL[1]
        l_diff = r_diff_HCL[2]

        H_diffs.append(h_diff)
        C_diffs.append(c_diff)
        L_diffs.append(l_diff)
        time.append(eph1[i]['time'])
        positions.append(r1)
        xyz_correction.append(r_diff)
        velocity.append(v1)
        

    return [time, H_diffs, C_diffs, L_diffs, positions, velocity, xyz_correction] # also return the position and velocity of the TLE ephemeris for the correction mapping


############################################################################################
# GRACE 1
############################################################################################
# chose the files that you want to use
RawDataFolder = 'v1'
data_format = 'sp3'

# load the sp3 file
if data_format == 'sp3':
    raw_data_path = os.path.join(os.getcwd(), 'RawData', RawDataFolder)
    sp3_file = None
    for file in os.listdir(raw_data_path):
        if 'L64' in file and file.endswith('.sp3'):
            sp3_file = file
            break
    
    if sp3_file is None:
        raise Exception("No SP3 file found in the directory")  
    else: 
        full_path = os.path.join(raw_data_path, sp3_file)
        product = sp3.Product.from_file(full_path)

if data_format == 'pickle':
    # open the pickle file
    with open(os.path.join(os.getcwd(), 'RawData', RawDataFolder, 'L64_sp3_data.pkl'), 'rb') as file:
        product = pickle.load(file)

# Create some buffer to ensure a TLE capture the entire timeframe
start_time = product.satellites[0].records[0].time
end_time = product.satellites[0].records[-1].time
start_time_range = start_time - timedelta(days=2)
end_time_range = start_time + timedelta(days=2)
print('Start date: {}'.format(start_time))
print('End date: {}'.format(end_time))

with open(os.path.join(os.getcwd(), 'RawData', RawDataFolder, 'grace-fo-1-tle-43476.json'), "r") as file:
    tle_raw_l64 = json.load(file)

tles = [entry for entry in tle_raw_l64 if start_time_range <= datetime.fromisoformat(entry['EPOCH']).replace(tzinfo=tzutc()) <= end_time_range]

print("Number of tles available in the time range: ", len(tles))

## PLOT THE SPREAD OF TLEs BETWEEEN THE TIME RANGES
epoch_times = [datetime.fromisoformat(entry['EPOCH']) for entry in tle_raw_l64]
time_diffs = [1 for _ in epoch_times] # in hours


if data_format == 'sp3':
    sp3_time = [result.time for result in product.satellites[0].records]
if data_format == 'pickle':
    sp3_time = [result.time for result in product]

from datetime import timedelta
from astropy.utils import iers
iers.conf.auto_download = True  

sp3_ephemeris = process_raw_sp3_vectorized(product)

# dump the sp3_ephemeris to a pickle file
if data_format == 'pickle':
    # save the pickle file as it is usually massive
    with open(os.path.join(os.getcwd(), 'RawData', RawDataFolder, 'sp3_ephemeris.pkl'), 'wb') as file:
        pickle.dump(sp3_ephemeris, file)

tle_ephemeris = tle_propagate_using_sp3_time(tles, sp3_time)

grace1_ephemeris = HCL_diff(tle_ephemeris, sp3_ephemeris)

############################################################################################
# GRACE 2 / CORRECTIONS
############################################################################################

# run the code above with grace 2 data
with open(os.path.join(os.getcwd(), 'RawData', RawDataFolder, 'grace-fo-2-tle-43477.json'), "r") as file:
    grace2_tle_raw = json.load(file)

# then compare the difference to the sp3 data of grace 1 tle to see if we can reduce the errors in the hcl
grace2_tles = [entry for entry in grace2_tle_raw if start_time_range <= datetime.fromisoformat(entry['EPOCH']).replace(tzinfo=tzutc()) <= end_time_range]
grace2_tle_ephemeris = tle_propagate_using_sp3_time(grace2_tles, sp3_time)

# if the datetime is the same, then apply the xyz_corection to the grace 2 tle data
def apply_correction(grace1_corrections, grace2_tle_ephemeris):
    count = 0
    for i in range(0, len(grace1_corrections[0]), 1):
        for j in range(0, len(grace2_tle_ephemeris), 1):
            if grace1_corrections[0][i] == grace2_tle_ephemeris[j]['time']:
                grace2_tle_ephemeris[j]['corrected_position'] = grace2_tle_ephemeris[j]['position'] - grace1_corrections[6][i]
                count += 1
                break
    print('Corrections applied: ', count)
    return grace2_tle_ephemeris

grace2_tle_corrected_ephemeris = apply_correction(grace1_ephemeris, grace2_tle_ephemeris) # grace1 ephemeris has the corrections

# now take the new data and then do a HCL analysis between the new data and the sp3 data
if data_format == 'sp3':
    raw_data_path = os.path.join(os.getcwd(), 'RawData', RawDataFolder)
    sp3_file = None
    for file in os.listdir(raw_data_path):
        if 'L65' in file and file.endswith('.sp3'):
            sp3_file = file
            break
    
    if sp3_file is None:
        raise Exception("No SP3 file found in the directory")  
    else: 
        full_path = os.path.join(raw_data_path, sp3_file)
        product_grace2 = sp3.Product.from_file(full_path)

start_time = product.satellites[0].records[0].time
end_time = product.satellites[0].records[-1].time
start_time_range = start_time - timedelta(days=1) # need some buffer room
end_time_range = start_time + timedelta(days=2)
grace2_sp3_ephemeris = process_raw_sp3_vectorized(product_grace2)

# Calculate altitudes and extract times
tle_times = [entry['time'] for entry in grace2_tle_corrected_ephemeris]
grace2_sp3 = [np.linalg.norm(entry['position']) for entry in grace2_sp3_ephemeris]
grace2_tle = [np.linalg.norm(entry['position']) for entry in grace2_tle_ephemeris]
grace2_tle_corrected = [np.linalg.norm(entry['corrected_position']) for entry in grace2_tle_corrected_ephemeris]

# plot difference between grace2_tle and sp3
grace2_tle_sp3_diff = []
for i in range(0, len(grace2_tle), 1):
    grace2_tle_sp3_diff.append(grace2_tle[i] - grace2_sp3[i])

# plot difference between grace2_tle_corrected and sp3
grace2_tle_corrected_sp3_diff = []
for i in range(0, len(grace2_tle_corrected), 1):
    grace2_tle_corrected_sp3_diff.append(grace2_tle_corrected[i] - grace2_sp3[i])

# Plot using matplotlib
plt.figure(figsize=(12, 6))
plt.plot(tle_times, grace2_tle_sp3_diff, linestyle='-', color='blue', label='DIFF TLE - SP3')
plt.plot(tle_times, grace2_tle_corrected_sp3_diff, linestyle='-', color='red', label='DIFF TLE Corrected - SP3')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.gcf().autofmt_xdate()  # for slanting date labels
plt.title("Difference Between TLE and SP3 Altitudes")
plt.xlabel("UTC Time")
plt.ylabel("Altitude Difference (m)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.xticks(rotation=45)
plt.legend()
plt.show()
