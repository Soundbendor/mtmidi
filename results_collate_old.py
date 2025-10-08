import csv, os, sys, time,argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-pf", "--prefix", type=int, default=333, help="specify a prefix > 0 for save files (db, etc.) for potential reloading (if file exists)")

args = parser.parse_args()
arg_dict = vars(args)

prefix = args.prefix
cur_time = int(time.time() * 1000)
outfname = f'{prefix}-{cur_time}-combined.csv'
outf = open(os.path.join('res_csv',outfname), 'w')
csvw = csv.writer(outf)
keys = None
first_file = False
pfix_str = f'{prefix}-'
len_pfixstr = len(pfix_str)
for fi,f in enumerate(os.listdir('res_csv')):
    if 'combined' in f or pfix_str != f[:len_pfixstr]:
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

    


