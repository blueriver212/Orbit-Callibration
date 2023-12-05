from datetime import datetime, timedelta, timezone
import numpy as np
from astropy.time import Time
from astropy.coordinates import GCRS, ITRS
from astropy import units

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