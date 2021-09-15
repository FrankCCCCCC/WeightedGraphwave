# -*- coding: utf-8 -*-

import csv

with open("newcsv.csv","w") as csvfile: 
    writer = csv.writer(csvfile)
    num = [1,2,3,4]
    for i in num:
        writer.writerow([i,i+1,i+2])

