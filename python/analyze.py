import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

def distance(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    # d = ds.cityblock(p1, p2)
    return d


org_data = []
with open('yolo/results.csv') as csvf:
    reader = csv.reader(csvf)
    org_data = [row for row in reader]

df_data = pd.DataFrame(org_data)
npdata = df_data.to_numpy()

#print(npdata)
#m = len(npdata)
m = int(npdata[-1,0])

print(m)

J = np.zeros(shape=(m, 7))
J1 = np.zeros(shape=(m, 7))
B = np.zeros(shape=(m, 2))
C = np.zeros(shape=(m, 2))

print(npdata[npdata[:,0] == '387'])
for i in range(m):
    float_i = float(i + 1)
    np_data = npdata[npdata[:, 0] == str(i+1)]
    if len(np_data) != 0:
        J[i, 0] = len(np_data) # number of birds
        len(np_data)

    if len(np_data) == 1:
        J[i, 1] = np_data[0, 1]
        J[i, 2] = np_data[0, 2]
        J[i, 3] = np_data[0, 3]

    elif len(np_data) == 2:
        if int(np_data[0, 1]) < int(np_data[1, 1]):
            J[i, 1] = np_data[0, 1]
            J[i, 2] = np_data[0, 2]
            J[i, 3] = np_data[0, 3]

            J[i, 4] = np_data[1, 1]
            J[i, 5] = np_data[1, 2]
            J[i, 6] = np_data[1, 3]
            print('Testring :' + str(np_data[0,1]) + ' < ' + str(np_data[1,1]))
            print(J[i,])
        else:
            J[i, 1] = np_data[1, 1]
            J[i, 2] = np_data[1, 2]
            J[i, 3] = np_data[1, 3]

            J[i, 4] = np_data[0, 1]
            J[i, 5] = np_data[0, 2]
            J[i, 6] = np_data[0, 3]
            print('Best !!!! :' + str(np_data[0,1]) + ' > ' + str(np_data[1,1]))
            print(J[i,])




figflag2 = 1
figflag3 = 1
figflag4 = 1

m2 = len(J)
s1 = J[0, 1:4]
s2 = J[0, 4:7]
print(s1)
print(s2)

B[0, 0] = J[0, 2]
B[0, 1] = J[0, 3]
C[0, 0] = J[0, 5]
C[0, 1] = J[0, 6]

for i in range(1, m2):
    L = []
    d1 = distance(s1[1:3], J[i, 2: 4])
    d2 = distance(s1[1:3], J[i, 5: 7])
    d3 = distance(s2[1:3], J[i, 2: 4])
    d4 = distance(s2[1:3], J[i, 5: 7])
    L.append(d1)
    L.append(d2)
    L.append(d3)
    L.append(d4)
    loc = L.index(np.min(L))

    if loc == 0 or loc == 3:
        s1 = J[i, 1:4]
        s2 = J[i, 4:7]
        if np.any(J[:, 5]) > 0:
            B[i, 0] = J[i, 2]
            B[i, 1] = J[i, 3]
            C[i, 0] = J[i, 5]
            C[i, 1] = J[i, 6]
        else:
            B[i, 0] = J[i, 2]
            B[i, 1] = J[i, 3]
    else:
        s1 = J[i, 4:7]
        s2 = J[i, 1:4]

        if np.any(J[:, 5]) > 0:
            B[i, 0] = J[i, 5]
            B[i, 1] = J[i, 6]
            C[i, 0] = J[i, 2]
            C[i, 1] = J[i, 3]
        else:
            B[i, 0] = J[i, 2]
            B[i, 1] = J[i, 3]

#print(B)
#print(C)

L3 = np.zeros(shape=(m, 4))
t2 = np.arange(1, m + 1)
L3[:, 0] = t2
for i in range(len(t2)):
    if B[i, 0] > 0:
        L3[i, 1] = 1
    else:
        L3[i, 1] = 0

    if C[i, 0] > 0:
        L3[i, 2] = 1
    else:
        L3[i, 2] = 0


from scipy.signal import medfilt

L3[:, 3] = L3[:, 1] + L3[:, 2]
#print('L3 : ----->')
#print(L3)
#L3[:, 3] = medfilt(L3[:, 3], kernel_size=5)
#print('L3 : ----->')
#print(L3)

if figflag4 == 1:
    plt.figure(1)
    #plt.plot(L3[:, 0], L3[:, 3], '.-', label="Total")
    plt.plot(L3[:, 0], L3[:, 3], '.-', label="Total")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Bird Count with Median filter")
    plt.axis([0, m, 0, 3])

if figflag2 == 1:
    plt.figure(2)
    plt.plot(B[:, 0], 1 - B[:, 1], '.', label="bird1")
    plt.plot(C[:, 0], 1 - C[:, 1], '.', label="bird2")
    plt.legend()
    plt.axis([0, 1, 0, 1])

flag1 = 0
b = {}
for i in range(m):
    if B[i, 0] > 0:
        b[i] = 1
        flag1 = 1
    elif (i > 0 and B[i - 1, 0] > 0 and B[i, 0] == 0 and B[i - 1, 0] < 0.6 and B[i - 2, 1] < 0.5) and flag1 == 1:
        b[i] = 1
        flag1 = 1
        B[i, :] = B[i - 1, :]
    else:
        b[i] = 0
        flag1 = 0

flag2 = 0
c = {}
for i in range(m):
    if C[i, 0] > 0:
        c[i] = 1
        flag2 = 1
    elif (i > 0 and C[i - 1, 0] > 0 and C[i, 0] == 0 and C[i - 1, 0] < 0.6 and C[i - 2, 1] < 0.5) and flag2 == 1:
        c[i] = 1
        flag2 = 1
        C[i, :] = C[i - 1, :]
    else:
        c[i] = 0
        flag2 = 0

L = np.zeros(shape=(m, 4))
L1 = np.zeros(shape=(m, 4), dtype=str)
t = np.arange(1, m + 1)

L[:, 0] = t

# median_filter(B, size=3, cval=0, mode='constant')
# median_filter(C, size=3, cval=0, mode='constant')

#print(len(t))
for i in range(len(t)):
    if B[i, 0] > 0:
        L[i, 1] = 1
    else:
        L[i, 1] = 0

    if C[i, 0] > 0:
        L[i, 2] = 1
    else:
        L[i, 2] = 0

# L[:, 1] = median_filter(L[:, 1], size=3, mode='constant')
# L[:, 2] = median_filter(L[:, 2], size=3, mode='constant')

L[:, 3] = L[:, 1] + L[:, 2]

# median_filter(C, size=3, cval=0, mode='constant')

t1 = []
t1_str = []
for i in range(len(t)):
    second = t[i] * 0.1
    td = timedelta(seconds=second)
    t_str = str(td)
    t1.append(t_str)
    t1_str.append(t_str)

L1[:, 0] = t1
#print(L1)
L1[:, 1:4] = L[:, 1: 4]
print(L1)

t1_str_pd = pd.DataFrame(t1_str)
L1_pd = pd.DataFrame(L1[:, 1:4])
L1_data_pd = pd.concat([t1_str_pd, L1_pd], axis=1)
L1_data_pd.to_csv('yolo/birdstatus.csv', header=False, index=False)

if figflag3 == 1:
    plt.figure(3)
    plt.plot(L[:, 0], L[:, 1] + 0.03, '.-', label="bird1")
    plt.plot(L[:, 0], L[:, 2], '.-', label="bird2")
    plt.plot(L[:, 0], L[:, 3], '.-', label="Total")
    plt.legend()
    plt.xlabel("frame")
    plt.ylabel("bird_count")
    plt.axis([0, m, 0, 3])
    plt.show()

