import zCurve as z
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('macOSX')
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import chain
import time
from numba import jit, uint64

# Combine the 3D point (2, 16, 8) into a Morton code.
morton_code = z.interlace(2, 16, 8)
assert morton_code == 10248, "morton_code should be 10248"
#print( morton_code )
morton_code1 = z.interlace(57772400, 12765000, dims=2)
assert morton_code1 == 1606428908008832, "morton_code1 should be 1606428908008832"

morton_code2 = z.interlace(57773800, 12772000, dims=2)
assert morton_code2 == 1606429041286208, "morton_code2 should be 1606429041286208"

#print( str(morton_code1) + "\n" + str(morton_code2) )
def calculateMortonFromTwoLatLonFloats_with_zCurve(x, y):
    # Cap floating point numbers to six decimal places
    x_int = int( round( (x + 90.0), 6 ) * 100000 )
    y_int = int( round( (y + 180.0), 6 ) * 100000 )
    value = z.interlace(x_int, y_int, dims=2)
    return value

morton_code1_f = calculateMortonFromTwoLatLonFloats_with_zCurve(60.734398, 14.768745)
assert morton_code1_f == 664749436223965, "morton_code1_f should be 664749436223965"
#print( "Two positive floats: " + str(morton_code1_f) )

morton_code2_f = calculateMortonFromTwoLatLonFloats_with_zCurve(38.969745, -77.201958)
assert morton_code2_f == 231657429695220, "morton_code2_f should be 231657429695220"
#print( "One positive/one negative float: " + str(morton_code2_f) )

morton_code3_f = calculateMortonFromTwoLatLonFloats_with_zCurve(-34.619055, -58.364067)
assert morton_code3_f == 171054631290070, "morton_code3_f should be 171054631290070"
#print( "Two negative floats: " + str(morton_code3_f) )

morton_code4_f = calculateMortonFromTwoLatLonFloats_with_zCurve(-33.956603, 150.949719)
assert morton_code4_f == 769185334910863, "morton_code4_f should be 769185334910863"
#print( "One negative/one positive float floats: " + str(morton_code4_f) )

@jit(nopython=True)
def mortonEncode2D(a, b):
    x = uint64(a)
    y = uint64(b)

    x = (x | (x << uint64(16))) & uint64(0x0000FFFF0000FFFF)
    x = (x | (x << uint64(8)))  & uint64(0x00FF00FF00FF00FF)
    x = (x | (x << uint64(4)))  & uint64(0x0F0F0F0F0F0F0F0F)
    x = (x | (x << uint64(2)))  & uint64(0x3333333333333333)
    x = (x | (x << uint64(1)))  & uint64(0x5555555555555555)

    y = (y | (y << uint64(16))) & uint64(0x0000FFFF0000FFFF)
    y = (y | (y << uint64(8)))  & uint64(0x00FF00FF00FF00FF)
    y = (y | (y << uint64(4)))  & uint64(0x0F0F0F0F0F0F0F0F)
    y = (y | (y << uint64(2)))  & uint64(0x3333333333333333)
    y = (y | (y << uint64(1)))  & uint64(0x5555555555555555)

    return uint64(x | (y << uint64(1)))

def mortonExtractEvenBits(a):
    x = np.uint64(a)

    x = x & np.uint64(0x5555555555555555)
    x = (x | (x >> np.uint64(1))) & np.uint64(0x3333333333333333)
    x = (x | (x >> np.uint64(2))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = (x | (x >> np.uint64(4))) & np.uint64(0x00FF00FF00FF00FF)
    x = (x | (x >> np.uint64(8))) & np.uint64(0x0000FFFF0000FFFF)
    x = (x | (x >> np.uint64(16))) & np.uint64(0x00000000FFFFFFFF)

    return x.astype(np.uint32)

def mortonDecode2D(a):
    _a = np.uint64(a)

    x = mortonExtractEvenBits(_a)
    y = mortonExtractEvenBits(_a >> np.uint64(1))

    return (x, y)




morton_code1 = mortonEncode2D(57772400, 12765000)
assert morton_code1 == 1606428908008832, "morton_code1 should be 1606428908008832"
#print(morton_code1)

decode_morton_code1 = mortonDecode2D(1606428908008832)
assert decode_morton_code1 == (57772400, 12765000), "decode_morton_code1 should be (57772400, 12765000)"
#print(decode_morton_code1)
morton_code1_3rd = z.interlace(57772400, 12765000, dims=2)
morton_code1_own = mortonEncode2D(57772400, 12765000)

#print( morton_code1_3rd - morton_code1_own )
def calculateMortonFromTwoLatLonFloats(x, y):
    _x = np.float32(x)
    _y = np.float32(y)
    # Cap floating point numbers to six decimal places.
    x_int = np.uint32( np.round( (_x + np.float32(90.0) ), 6 ) * np.uint32(100000) )
    y_int = np.uint32( np.round( (_y + np.float32(180.0) ), 6 ) * np.uint32(100000) )
    value = mortonEncode2D(x_int, y_int)
    return value

p1 = (60.734398, 14.768745)
morton_code1 = calculateMortonFromTwoLatLonFloats(p1[0], p1[1])
assert morton_code1 == 664749436224642, "morton_code1 should be 664749436224642"
#print( "Two positive floats: " + str(morton_code1) )
#print( "Delta to 3rd party library: " + str(morton_code1 - morton_code1_f) )

p2 = (38.969745, -77.201958)
morton_code2 = calculateMortonFromTwoLatLonFloats(p2[0], p2[1])
assert morton_code2 == 231657429695220, "morton_code2 should be 231657429695220"
#print( "One positive/one negative float: " + str(morton_code2) )
#print( "Delta to 3rd party library: " + str(morton_code2 - morton_code2_f) )

p3 = (-34.619055, -58.364067)
morton_code3 = calculateMortonFromTwoLatLonFloats(p3[0], p3[1])
assert morton_code3 == 171054631290070, "morton_code3 should be 171054631290070"
#print( "Two negative floats: " + str(morton_code3) )
#print( "Delta to 3rd party library: " + str(morton_code3 - morton_code3_f) )

p4 = (-33.956603, 150.949719)
morton_code4 = calculateMortonFromTwoLatLonFloats(p4[0], p4[1])
assert morton_code4 == 769185334910861, "morton_code4 should be 769185334910861"
#print( "One negative/one positive float floats: " + str(morton_code4) )
#print( "Delta to 3rd party library: " + str(morton_code4 - morton_code4_f) )

def calculateTwoLatLonFloatsFromMorton(a):
    pair = mortonDecode2D(a)
    _x = np.float32(pair[0] / np.float32(100000.0) - np.float32(90.0))
    _y = np.float32(pair[1] / np.float32(100000.0) - np.float32(180.0))
    return (_x, _y)

eps = 0.00001
decode_morton_code1 = calculateTwoLatLonFloatsFromMorton(morton_code1)
#print( "{:.6f}".format(decode_morton_code1[0]) + " " + "{:.6f}".format(decode_morton_code1[1]))
#print( "Delta: " + "{:.6f}".format(decode_morton_code1[0] - p1[0]) + " " + "{:.6f}".format(decode_morton_code1[1] - p1[1]))
assert np.fabs(decode_morton_code1[0] - p1[0]) < eps , "Delta for p1 after decoding too large"

decode_morton_code2 = calculateTwoLatLonFloatsFromMorton(morton_code2)
#print( "{:.6f}".format(decode_morton_code2[0]) + " " + "{:.6f}".format(decode_morton_code2[1]))
#print( "Delta: " + "{:.6f}".format(decode_morton_code2[0] - p2[0]) + " " + "{:.6f}".format(decode_morton_code2[1] - p2[1]))
assert np.fabs(decode_morton_code2[0] - p2[0]) < eps , "Delta for p2 after decoding too large"

decode_morton_code3 = calculateTwoLatLonFloatsFromMorton(morton_code3)
#print( "{:.6f}".format(decode_morton_code3[0]) + " " + "{:.6f}".format(decode_morton_code3[1]))
#print( "Delta: " + "{:.6f}".format(decode_morton_code3[0] - p3[0]) + " " + "{:.6f}".format(decode_morton_code3[1] - p3[1]))
assert np.fabs(decode_morton_code3[0] - p3[0]) < eps , "Delta for p3 after decoding too large"

decode_morton_code4 = calculateTwoLatLonFloatsFromMorton(morton_code4)
#print( "{:.6f}".format(decode_morton_code4[0]) + " " + "{:.6f}".format(decode_morton_code4[1]))
#print( "Delta: " + "{:.6f}".format(decode_morton_code4[0] - p4[0]) + " " + "{:.6f}".format(decode_morton_code4[1] - p4[1]))
assert np.fabs(decode_morton_code4[0] - p4[0]) < eps , "Delta for p4 after decoding too large"

# First, we read the CSV file.
df = pd.read_csv('emergency_brakingx100.csv', sep=';')

# Next, we print the column headers to know what we have access to.
cols = df.columns.values.tolist()
#print( "Column headers: " + str(cols) )
df['timestamp'] = df['sampleTimeStamp.seconds'] * 1000000 + df['sampleTimeStamp.microseconds']
threshold = -5.4
times_for_maneuver = []
start_time = time.time()
for _, row in df.iterrows():
    # The following line has been commented out for readability purposes.
    #print( str(row['timestamp']) + ": "+ str(row['accel_x']) )
    if row['accel_x'] < threshold:
        times_for_maneuver.append(row['timestamp'])
end_time = time.time()
print(f"Execution time cell 15: {end_time - start_time} seconds")
#print( "Length of maneuver: " + str(len(times_for_maneuver)) )

def calculateMortonFromXYAccelerationFloats(x, y):
    _x = np.float32(x)
    _y = np.float32(y)
    # Cap floating point numbers to six decimal places.
    x_int = np.uint32( np.round( (_x + np.float32(10.0) ), 6 ) * np.uint32(100) )
    y_int = np.uint32( np.round( (_y + np.float32(10.0) ), 6 ) * np.uint32(100) )
    value = mortonEncode2D(x_int, y_int)
    return value

def calculateXYAccelerationFloatsFromMorton(a):
    pair = mortonDecode2D(a)
    _x = np.float32(pair[0] / np.float32(100.0) - np.float32(10.0))
    _y = np.float32(pair[1] / np.float32(100.0) - np.float32(10.0))
    return (_x, _y)

counter = 0
y = 0
for _, row in df.iterrows():
    morton = calculateMortonFromXYAccelerationFloats(row['accel_x'], y)
   # print( str(row['timestamp']) + ": (" + str(row['accel_x']) + ",0) --> " + str(morton) )
    counter = counter + 1

lst = []
y = 0
start_time = time.time()
for _, row in df.iterrows():
    morton = calculateMortonFromXYAccelerationFloats(row['accel_x'], y)
    lst.append({'morton': morton, 'timestamp': row['timestamp']})

df2 = pd.DataFrame(lst, columns=['morton', 'timestamp'])
end_time = time.time()
print(f"Execution time cell 18: {end_time - start_time} seconds")
#print( df2 )
data = df2['morton']
group_A = []
group_B = []
for _, row in df2.iterrows():
    morton = row['morton']
    timestamp = row['timestamp']
    if timestamp in times_for_maneuver:
        group_A.append(morton)
    else:
        group_B.append(morton)

data = [group_A, group_B]
data_colors = ['red', 'lightgray']
group_A = []
group_B = []
for _, row in df2.iterrows():
    morton = row['morton']
    timestamp = row['timestamp']
    if timestamp in times_for_maneuver:
        group_A.append(morton)
    else:
        group_B.append(morton)

data = [group_A, group_B]
data_colors = ['red', 'lightgray']
df2 = df2.sort_values(by=['morton'])
#print( df2 )
dict_of_morton_to_timestamps = dict(zip(df2.morton, df2.timestamp))
matches = list(filter( lambda k: k < 0.78459e6, dict_of_morton_to_timestamps.keys()) )
#print( "Length of maneuver: " + str(len(matches)) + ", entries: " + str(matches))
# The standard dictionary in Python does not allow duplicated keys and hence, we use the defaultdict based on list.
defaultdict_of_morton_to_timestamps = defaultdict(list)
for k, v in zip(df2.morton, df2.timestamp):
    defaultdict_of_morton_to_timestamps[k].append(v)

# Filter the keys between the lower and upper limit.
matches = list(filter( lambda k: k < 0.78459e6, defaultdict_of_morton_to_timestamps.keys()) )

# Finally, we need to "flatten" the matches to reduce the set of duplicated values for some keys:
_timepoints = []
for e in matches:
    _timepoints.append(defaultdict_of_morton_to_timestamps[e])
timepoints_for_maneuver_morton_setting_y_to_0 = sorted(list(chain.from_iterable(_timepoints)))
#print( "Length of maneuver: " + str(len(timepoints_for_maneuver_morton_setting_y_to_0)) + ", entries: " + str(timepoints_for_maneuver_morton_setting_y_to_0))
start_time = time.time()
#print( str(times_for_maneuver) )
#print( str(timepoints_for_maneuver_morton_setting_y_to_0) )
end_time = time.time()
print(f"Execution time cell 25: {end_time - start_time} seconds")
counter = 0
for _, row in df.iterrows():
    morton = calculateMortonFromXYAccelerationFloats(row['accel_x'], row['accel_y'])
    counter = counter + 1
lst = []
start_time = time.time()
for _, row in df.iterrows():
    morton = calculateMortonFromXYAccelerationFloats(row['accel_x'], row['accel_y'])
    lst.append({'morton': morton, 'timestamp': row['timestamp']})
df2 = pd.DataFrame(lst, columns=['morton', 'timestamp'])
end_time = time.time()
print(f"Execution time cell 27: {end_time - start_time} seconds")

data = df2['morton']

group_A = []
group_B = []
for _, row in df2.iterrows():
    morton = row['morton']
    timestamp = row['timestamp']
    if timestamp in times_for_maneuver:
        group_A.append(morton)
    else:
        group_B.append(morton)

data = [group_A, group_B]
data_colors = ['red', 'lightgray']

df2 = df2.sort_values(by=['morton'])

dict_of_morton_to_timestamps = dict(zip(df2.morton, df2.timestamp))
matches = list(filter( lambda k: k < 0.78459e6, dict_of_morton_to_timestamps.keys()) )

dict_of_morton_to_timestamps = dict(zip(df2.morton, df2.timestamp))
matches = list(filter( lambda k: k < 0.8e6, dict_of_morton_to_timestamps.keys()) )

for morton in matches:
    acc = calculateXYAccelerationFloatsFromMorton(morton)

times_for_maneuver_morton = []
for morton in matches:
    acc = calculateXYAccelerationFloatsFromMorton(morton)
    if acc[0] < threshold:
        times_for_maneuver_morton.append(dict_of_morton_to_timestamps[morton])
times_for_maneuver.sort()
times_for_maneuver_morton.sort()

threshold = 5.4
times_for_maneuver = []
for _, row in df.iterrows():
# print( str(row['timestamp']) + ": "+ str(row['accel_x']) )
    if row['accel_x'] < threshold:
        # print( "Maneuver at " + str(row['timestamp']) )
        times_for_maneuver.append(row['timestamp'])

my_dict = dict(zip(df2.morton, df2.timestamp))
l = list(filter( lambda k: k < 0.8e6, my_dict.keys()) )
#print(str(l))

lst = []
for _, row in df.iterrows():
    morton = calculateMortonFromXYAccelerationFloats(row['accel_z'], row['accel_y'])
    lst.append({'morton': morton, 'timestamp': row['timestamp']})

df3 = pd.DataFrame(lst, columns=['morton', 'timestamp'])
print( df3 )

data = df3['morton']

# First, we read the CSV file.
df_lane_change = pd.read_csv('lane_changesx100.csv', sep=';')

# Next, we print the column headers to know what we have access to.
cols = df_lane_change.columns.values.tolist()
#print( "Column headers: " + str(cols) )

# Plot the accelerations but this time using the correct resolution of time.
df_lane_change['timestamp'] = df_lane_change['sampleTimeStamp.seconds'] * 1000000 + df_lane_change['sampleTimeStamp.microseconds']

# Plot the accelerations but this time using the correct resolution of time and subtract by the first value to let the X values start at 0.
offset = df_lane_change['sampleTimeStamp.seconds'][0] * 1000000 + df_lane_change['sampleTimeStamp.microseconds'][0]
df_lane_change['timestamp'] = df_lane_change['sampleTimeStamp.seconds'] * 1000000 + df_lane_change['sampleTimeStamp.microseconds'] - offset

counter = 0

lst = []
lst_morton_points = []
longitudinal_acceleration = 0.0
for _, row in df_lane_change.iterrows():
    # Compute the Morton value but keep the longitudinal acceleration const to 0.0
    morton = calculateMortonFromXYAccelerationFloats(longitudinal_acceleration, row['accel_y'])
    lst.append({'morton': morton, 'timestamp': row['timestamp']})

    # Scale the Morton values so that we can create an intuitive overlay plot.
    lst_morton_points.append({'timestamp': row['timestamp'], 'morton': (morton - 2.1e6) / 1e6})

    # Print raw Morton values.
    #print(str(row['timestamp']) + ": (" + str(row['accel_x']) + "," + str(row['accel_y']) + ") --> " + str(morton))
    counter = counter + 1

df_lane_change_morton = pd.DataFrame(lst, columns=['morton', 'timestamp'])

df_lane_change_morton_scaled = pd.DataFrame(lst_morton_points, columns=['morton', 'timestamp'])

offset = df_lane_change['sampleTimeStamp.seconds'][0] * 1000000 + df_lane_change['sampleTimeStamp.microseconds'][0]
df_lane_change['timestamp'] = df_lane_change['sampleTimeStamp.seconds'] * 1000000 + df_lane_change['sampleTimeStamp.microseconds'] - offset

my_dict = dict(zip(df_lane_change_morton.morton, df_lane_change_morton.timestamp))

area_a = list(filter( lambda k: k > 8.9e5 and k < 9.1e5, my_dict.keys()) )
print( "Area A: " + str(area_a))

area_c = list(filter( lambda k: k > 2.488e6 and k < 2.6e6, my_dict.keys()) )
print( "Area C: " + str(area_c))

# Turn list of Morton codes back to time points:
time_points_area_a = []
time_points_area_c = []

for a in area_a:
    time_points_area_a.append(my_dict[a])
for c in area_c:
    time_points_area_c.append(my_dict[c])

# Sort by time.
time_points_area_a = sorted( time_points_area_a )
time_points_area_c = sorted( time_points_area_c )

# Heuristically determined; given in µs.
minimum_duration =  40000
maximum_duration = 200000

# This function returns a list of tuples (a, b) where (b-a) is the largest range within [minimum_duration, maximum_duration] from a list of timepoints:
def getRanges(listOfTimePoints, minDuration, maxDuration):
    assert len(listOfTimePoints) > 1, "listOfTimePoints must contain minimally two items"
    listOfTuples = []

    lastIdx = 0
    # Once a tuple has been added to the list of matching temporal ranges, start *after* the end point of the last tuple as long as there is more to process.
    while lastIdx < len(listOfTimePoints) and ((len(listOfTimePoints) - lastIdx) > 1):
        idx = lastIdx
        t1 = 0
        t2 = 0
        # Skip all time points that have been processed so far already (ie., slicing from idx:).
        for t in listOfTimePoints[idx:]:
            idx = idx + 1
            if t1 == 0:
                t1 = t
                continue
            # Check whether we find matching durations:
            if (t - t1) > minimum_duration and (t - t1) < maximum_duration:
                t2 = t
            # Stop criterion once we exceed the maximum duration:
            if (t - t1) > maximum_duration:
                break
        # Only add a tuple when we have valid start and end time points:
        if t1 > 0 and t2 > 0:
            listOfTuples.append((t1, t2))
            print("Range " + str(len(listOfTuples)) + ": " + str(t1) + " --> " + str(t2))
        lastIdx = idx
    return listOfTuples
start_time = time.time()
ranges_a = getRanges(time_points_area_a, minimum_duration, maximum_duration)
end_time = time.time()
print(f"Execution time cell 43: {end_time - start_time} seconds")
print(str(ranges_a))

start_time = time.time()
ranges_c = getRanges(time_points_area_c, minimum_duration, maximum_duration)
end_time = time.time()
print(f"Execution time cell 43: {end_time - start_time} seconds")
#print(str(ranges_c))
# Heuristically determined; given in µs.
minimum_event_duration = 2000000
maximum_event_duration = 5000000

# Starting from the first list, we add elements from the other lists as long as R1 and R2 are fulfilled:
listOfEvents = []
for a in ranges_a:
    for c in ranges_c:
        # Check for t2 < t5
        if a[1] > c[0]:
            continue
        if ((c[1] - a[0]) > minimum_event_duration) and ((c[1] - a[0]) < maximum_event_duration):
            listOfEvents.append( (a, c) )
#print( str(listOfEvents) )

# Plot the accelerations but this time using the correct resolution of time and subtract by the first value to let the X values start at 0.
my_dict = dict(zip(df_lane_change.timestamp, df_lane_change.accel_y))

lst_timestamp_accelerations = []
for e in listOfEvents:
    firstTimeStamp_boxA = e[0][0]
    firstTimeStamp_boxC = e[1][0]

    lst_timestamp_accelerations.append({'timestamp': firstTimeStamp_boxA, 'accel_y': my_dict[firstTimeStamp_boxA]})
    lst_timestamp_accelerations.append({'timestamp': firstTimeStamp_boxC, 'accel_y': my_dict[firstTimeStamp_boxC]})

# dict_timestamp_accelerations contains now all pairs (timeStamp, accel_y) that match the query for the lane change.
dict_timestamp_accelerations = pd.DataFrame(lst_timestamp_accelerations, columns=['timestamp', 'accel_y'])

# First, we read the CSV file.
df_lane_change = pd.read_csv('multiple_lane_changesx100.csv', sep=';')

# Next, we print the column headers to know what we have access to.
cols = df_lane_change.columns.values.tolist()
#print( "Column headers: " + str(cols) )
offset = df_lane_change['sampleTimeStamp.seconds'][0] * 1000000 + df_lane_change['sampleTimeStamp.microseconds'][0]
df_lane_change['timestamp'] = df_lane_change['sampleTimeStamp.seconds'] * 1000000 + df_lane_change['sampleTimeStamp.microseconds'] - offset
# Heuristically determined.
area_a_lower = 8.9e5
area_a_upper = 9.1e5
area_c_lower = 2.44e6
area_c_upper = 2.6e6

# Heuristically determined; given in µs.
area_minimum_duration =  10000
area_maximum_duration = 400000

# Heuristically determined; given in µs.
minimum_event_duration = 1000000
maximum_event_duration = 5000000

lst = []
lst_morton_points = []
longitudinal_acceleration = 0.0
start_time = time.time()
for _, row in df_lane_change.iterrows():
    # Compute the Morton value but keep the longitudinal acceleration const to 0.0
    morton = calculateMortonFromXYAccelerationFloats(longitudinal_acceleration, row['accel_y'])
    lst.append({'morton': morton, 'timestamp': row['timestamp']})

    # Scale the Morton values so that we can create an intuitive overlay plot.
    lst_morton_points.append({'timestamp': row['timestamp'], 'morton': (morton - 2.1e6) / 1e6})

    # Print raw Morton values.
    #print(str(row['timestamp']) + ": (" + str(row['accel_x']) + "," + str(row['accel_y']) + ") --> " + str(morton))
    counter = counter + 1

df_lane_change_morton = pd.DataFrame(lst, columns=['morton', 'timestamp'])

df_lane_change_morton_scaled = pd.DataFrame(lst_morton_points, columns=['morton', 'timestamp'])
#print(df_lane_change_morton_scaled)
end_time = time.time()
print(f"Execution time cell 49: {end_time - start_time} seconds")

my_dict = dict(zip(df_lane_change_morton.morton, df_lane_change_morton.timestamp))

area_a = list(filter( lambda k: k > area_a_lower and k < area_a_upper, my_dict.keys()) )
#print( "Area A: " + str(area_a))

area_c = list(filter( lambda k: k > area_c_lower and k < area_c_upper, my_dict.keys()) )
#print( "Area C: " + str(area_c))


# Turn list of Morton codes back to time points:
time_points_area_a = []
time_points_area_c = []

for a in area_a:
    time_points_area_a.append(my_dict[a])
for c in area_c:
    time_points_area_c.append(my_dict[c])

# Sort by time.
time_points_area_a = sorted( time_points_area_a )
time_points_area_c = sorted( time_points_area_c )

# This function returns a list of tuples (a, b) where (b-a) is the largest range within [minimum_duration, maximum_duration] from a list of timepoints:
def getRanges(listOfTimePoints, minDuration, maxDuration):
    assert len(listOfTimePoints) > 1, "listOfTimePoints must contain minimally two items"
    listOfTuples = []
    lastIdx = 0
    # Once a tuple has been added to the list of matching temporal ranges, start *after* the end point of the last tuple as long as there is more to process.
    while lastIdx < len(listOfTimePoints) and ((len(listOfTimePoints) - lastIdx) > 1):
        idx = lastIdx
        t1 = 0
        t2 = 0

        # Skip all time points that have been processed so far already (ie., slicing from idx:).
        for t in listOfTimePoints[idx:]:
            idx = idx + 1
            if t1 == 0:
                t1 = t
                continue
            # Check whether we find matching durations:
            if (t - t1) > minimum_duration and (t - t1) < maximum_duration:
                t2 = t
            # Stop criterion once we exceed the maximum duration:
            if (t - t1) > maximum_duration:
                break
        # Only add a tuple when we have valid start and end time points:
        if t1 > 0 and t2 > 0:
            listOfTuples.append((t1, t2))
            #print("Range " + str(len(listOfTuples)) + ": " + str(t1) + " --> " + str(t2))
        lastIdx = idx
    return listOfTuples

start_time = time.time()
ranges_a = getRanges(time_points_area_a, area_minimum_duration, area_maximum_duration)
#print(str(ranges_a))
end_time = time.time()
print(f"Execution time cell 50A: {end_time - start_time} seconds")
start_time = time.time()
ranges_c = getRanges(time_points_area_c, area_minimum_duration, area_maximum_duration)
end_time = time.time()
print(f"Execution time cell 50C: {end_time - start_time} seconds")
#print(str(ranges_c))

# Starting from the first list, we add elements from the other lists as long as R1 and R2 are fulfilled:
listOfEvents = []
for a in ranges_a:
    for c in ranges_c:
        # Check for t2 < t5
        if a[1] > c[0]:
            continue
        if ((c[1] - a[0]) > minimum_event_duration) and ((c[1] - a[0]) < maximum_event_duration):
            listOfEvents.append( (a, c) )
#print( str(listOfEvents) )

# Plot the accelerations but this time using the correct resolution of time and subtract by the first value to let the X values start at 0.
my_dict = dict(zip(df_lane_change.timestamp, df_lane_change.accel_y))

lst_timestamp_accelerations = []
for e in listOfEvents:
    firstTimeStamp_boxA = e[0][0]
    firstTimeStamp_boxC = e[1][0]

    lst_timestamp_accelerations.append({'timestamp': firstTimeStamp_boxA, 'accel_y': my_dict[firstTimeStamp_boxA]})
    lst_timestamp_accelerations.append({'timestamp': firstTimeStamp_boxC, 'accel_y': my_dict[firstTimeStamp_boxC]})

# dict_timestamp_accelerations contains now all pairs (timeStamp, accel_y) that match the query for the lane change.
dict_timestamp_accelerations = pd.DataFrame(lst_timestamp_accelerations, columns=['timestamp', 'accel_y'])

# Create a list of accelerations from -1g to +1g in 0.1 resolution:
longitudinal_acceleration = 0.0
lst_accelerations_vs_morton = []
for x in range(-98, +98, 1):
    a = x/10.0
    morton = calculateMortonFromXYAccelerationFloats(longitudinal_acceleration, a)
    lst_accelerations_vs_morton.append({'acceleration': a, 'morton': morton})

df_accelerations_vs_morton = pd.DataFrame(lst_accelerations_vs_morton, columns=['acceleration', 'morton'])







