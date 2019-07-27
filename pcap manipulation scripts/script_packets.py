from scapy.all import *

for i in range(5):
	if i>=0 and i<5:
		with PcapReader("C:/Users/nimrod/Desktop/nimrod/Experiments/exp_22_1_4_parts_rand/first/"+str(i)+".pcap") as pcap_reader:
			j=0
			flag_first=1
			time_zero=0
			p2=0
			sum_data=0
			l=[]
			for packet in pcap_reader.read_all():
				if flag_first == 1:
					p2 = packet.time
					time_zero=packet.time
					flag_first=0
				p = packet.time
				sum_data=sum_data+len(packet)
				l.append((j, len(packet), packet.time, packet.time - p2, packet.time - time_zero, sum_data))
				p2 = packet.time   
				j=j+1
			with open("part_"+str(i)+".csv", "w+") as f:
				for res in l:
					f.write(str(res[1])+  ",%.6f" % res[4] + ",%.6f\n" % res[3])

	   
	   
