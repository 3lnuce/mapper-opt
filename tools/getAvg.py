import os
import sys

f = open(sys.argv[1])
lines = f.readlines()
f.close()

def getAvg(keyword, lines):
    time_v = []
    for idx, line in enumerate(lines):
        if (keyword in line):
            line = line.strip().split("  ")
            time_v.append(float(line[1]))
    print ("Num: ", keyword, len(time_v))
    print ("Avg: ", keyword, sum(time_v) / len(time_v))
    print ("Tot: ", keyword, sum(time_v))

# def getAvg(keyword, lines):
#     time_v = []
#     for idx, line in enumerate(lines):
#         if (keyword in line):
#             line = line.strip().split("  ")
#             if (len(line) != 2):
#                 continue
#             if ("Backend" in keyword and float(line[1]) <= 300.00):
#                 continue
#             time_v.append(float(line[1]))
#     print ("Num: ", keyword, len(time_v))
#     print ("Avg: ", keyword, sum(time_v) / len(time_v))
#     print ("Tot: ", keyword, sum(time_v))

getAvg("Frontend duration:  ", lines)
getAvg("Backend duration:  ", lines)
