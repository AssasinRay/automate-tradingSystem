import os
import operator

file_list = os.listdir("scripts")
#print file_list

def read_matrix(file_name):
	f = open ( file_name , 'r')
	l = [ map(float,line.split(',')) for line in f ]
	#print l
	return l
#print file_list[0]
#read_matrix("./scripts/" + file_list[2])

file_dict = {}
for i in xrange(2,len(file_list)):
	m = read_matrix("./scripts/" + file_list[i])
	file_dict[file_list[i]] = [m[3][0],m[4][1],m[5][2]]

#print file_dict
best_acc0 = 0
best_acc1 = 0
best_acc2 = 0
best_total = [0,0,0]
best_1_2   = [0,0]
best_combo0 = " "
best_combo1 = " "
best_combo2 = " "
best_combototal = " "
best1_2_combo  = " "
for key,val in file_dict.iteritems():
	if val[0] >= best_acc0:
		best_acc0 = val[0]
		best_combo0 = key
	if val[1] >= best_acc1:
		best_acc1 = val[1]
		best_combo1 = key
	if val[2] >= best_acc2:
		best_acc2 = val[2]
		best_combo2 = key

	if val[1] + val[2] >= best_1_2[1] + best_1_2[0]:
		best_1_2 = [val[1] , val[2]]
		best_1_2combo = key

	total_pro = val[0]+val[1]+val[2]
	best_total_val = best_total[0]+best_total[1]+best_total[2]
	if  total_pro >= best_total_val:
		best_total = [val[0],val[1],val[2]]
		best_combototal = key

print best_acc0 
print best_combo0 
print best_acc1 
print best_combo1 
print best_acc2 
print best_combo2 
print best_total
print best_combototal
print best_1_2
print best_1_2combo