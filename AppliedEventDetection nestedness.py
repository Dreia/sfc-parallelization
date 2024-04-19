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
df = pd.read_csv('emergency_braking_x500.csv', sep=';')

# Next, we print the column headers to know what we have access to.
cols = df.columns.values.tolist()
#print( "Column headers: " + str(cols) )
df['timestamp'] = df['sampleTimeStamp.seconds'] * 1000000 + df['sampleTimeStamp.microseconds']
threshold = -5.4
times_for_maneuver = []
for _, row in df.iterrows():
    # The following line has been commented out for readability purposes.
    #print( str(row['timestamp']) + ": "+ str(row['accel_x']) )
    if row['accel_x'] < threshold:
        times_for_maneuver.append(row['timestamp'])
#print( "Length of maneuver: " + str(len(times_for_maneuver)) )

@jit(nopython=True)
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
# We introduce nestedness here
start_time = time.time()
for _, row in df.iterrows():
    morton = calculateMortonFromXYAccelerationFloats(row['accel_x'], y)
    if morton % 2 == 0:
        for j in range(10):  # Second level of nested loop
            secondary_morton = calculateMortonFromXYAccelerationFloats(row['accel_x'] + j, y)
            for k in range(5):  # Adding a third level of nested loop
                tertiary_morton = calculateMortonFromXYAccelerationFloats(row['accel_x'] + j + k, y)

df2 = pd.DataFrame(lst, columns=['morton', 'timestamp'])
end_time = time.time()
print(f"Execution time cell 18: {end_time - start_time} seconds")
