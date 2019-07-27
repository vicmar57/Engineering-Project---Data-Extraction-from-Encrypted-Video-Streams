import csv
import numpy as np

for z in range(5):
    print(z)
    if z>=0 and z < 5:
        window_size = 2
        if z>=0:
            f = open("C:/Users/nimrod/Desktop/nimrod/Experiments/exp_22_1_4_parts_rand/window_2/test/part_"+str(z)+".csv", "r")
            with open("C:/Users/nimrod/Desktop/nimrod/Experiments/exp_22_1_4_parts_rand/window_2/test/part_"+str(z)+"_features.csv", "a+") as ofile:
                reader = csv.reader(f)
                first_round = 1
                starting_time = 0
                num_packets_in_windows = 0
                max_packet_size = 1440
                packets = []
                for row in reader:
                    packets.append(row)
                for i, packet in enumerate(packets):
                    #time diff arrival between packets and next packets
                    small_packet_time_to_next_packet =[]
                    arr_small_packet_size =[]
                    arr_small_packet_diff_arrival_time =[]
                    starting_time = float(packet[1])
                    arr_packet_diff_arrival_time = [float(packet[2])]
                    arr_packet_size = [float(packet[0])]
                    if float(packet[0])<max_packet_size:
                        arr_small_packet_diff_arrival_time.append(float(packet[2]))
                        arr_small_packet_size.append(float(packet[0]))
                        if i<len(packets)-1:
                            small_packet_time_to_next_packet.append((float(packets[i+1][2])))
                    j = i+1
                    #print ("no starting {0}".format((float(packets[j][1]))))
                    #print ("starting time {0}".format((starting_time)))
                    while j<len(packets) and (float(packets[j][1])-starting_time < window_size):
                        arr_packet_diff_arrival_time.append(float(packets[j][2]))
                        arr_packet_size.append(float(packets[j][0]))
                        if float(packets[j][0])<max_packet_size:
                            arr_small_packet_diff_arrival_time.append(float(packets[j][2]))
                            arr_small_packet_size.append(float(packets[j][0]))
                            if j < len(packets) - 1:
                                small_packet_time_to_next_packet.append((float(packets[j+1][2])))
                        j+=1
                        num_packets_in_windows = num_packets_in_windows+1
                    if j == len(packets):
                        break
                        #       arr_packet_size = [float(i) for i in arr_packet_size]
                        #       arr_packet_diff_arrival_time = [float(i) for i in arr_packet_diff_arrival_time]
                    a = float(np.sum(arr_packet_size))
                    b = float(np.std(arr_packet_size))
                    c = float(np.mean(arr_packet_size))
                    d =  float(np.std(arr_packet_diff_arrival_time))
                    e = float(np.mean(arr_packet_diff_arrival_time))
                    f = num_packets_in_windows
                    ofile.write("%.6f" %a + ",%.6f" % b+ ",%.6f" % c + ",%.6f" %d+ ",%.6f" % e + ",%.6f" % f+",%.6f" % z+ "\n")
                    num_packets_in_windows = 0


