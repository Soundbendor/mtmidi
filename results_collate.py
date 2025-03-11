import csv, os, sys, time

cur_time = int(time.time() * 1000)
outfname = f'{cur_time}-combined.csv'
outf = open(os.path.join('res_csv',outfname), 'w')
csvw = csv.writer(outf)
keys = None
first_file = False
for fi,f in enumerate(os.listdir('res_csv')):
    if 'combined' in f:
        continue
    iptf = open(os.path.join('res_csv', f), 'r')
    csvr = csv.DictReader(iptf)
    for row in csvr:
        if first_file == False:
            keys = list(row.keys())
            csvw.writerow(keys)
            first_file = True
        cur_row = [row[k] for k in keys]
        csvw.writerow(cur_row)
    iptf.close()
outf.close()

    


