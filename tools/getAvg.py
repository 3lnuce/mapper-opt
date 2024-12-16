import os
import sys

f = open(sys.argv[1])
lines = f.readlines()
f.close()

def getAvg(keyword, lines):
    time_v = []
    for idx, line in enumerate(lines):
        if (keyword in line):
            line = line.strip().split(" ")
            print (idx, line)
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

print (len(sys.argv))
if (len(sys.argv) == 2):
    getAvg("[Frontend] [Duration]:  ", lines)
    getAvg("[Frontend] [tot_forward]:  ", lines)
    getAvg("[Frontend] [tot_bckward]:  ", lines)

    getAvg("[Backend] [Duration]:  ", lines)
    getAvg("[Backend] [tot_forward]:  ", lines)
    getAvg("[Backend] [tot_bckward]:  ", lines)
else:
    getAvg(sys.argv[-1], lines)
