
###Copy table from runs database to a .csv file and insert file name below###
###adds 2nd column "Numbers" with condensed date info###

fileName = "run_14.csv" #specify read file name here

fl = open(fileName) 

listdb = []
for line in fl:
	l = line.split(",")
	listdb.append(l)

fl.close()

listdb[0].insert(1, "Number")
for i in range(1, len(listdb)):
	format = listdb[i][0].split("_")
	listdb[i].insert(1, format[1] + format[2])
	listdb[i] = ",".join(listdb[i])
listdb[0] = ",".join(listdb[0])



fl.close()

fl = open("XENON100_14.csv", 'w') #specify write file name here

for line in listdb:
	fl.write(line)

fl.close()