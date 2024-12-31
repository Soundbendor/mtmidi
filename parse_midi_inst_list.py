import csv

d = {}
with open('midi_inst_list.txt', 'r') as f:
    cur_cat = ""
    for l in f.readlines():
        l2 = l.strip()
        is_cat = ":" in l2
        if is_cat == True:
            cat = l2[:-1]
            if cat == "Ethnic":
                cat = "Other"
            cur_cat = cat
            if cur_cat not in d.keys():
                d[cur_cat] = {}
            print(f"cat: {cat}")
        elif len(l2) > 0:
            l3 = l2.split(' ')
            inst_num = int(l3[0]) -1
            inst_name = ' '.join(l3[1:])
            d[cur_cat][inst_num] = inst_name
            print(f"{inst_num}: {inst_name}")
        
with open('inst_list.csv', 'w') as f:
    csvw = csv.writer(f, delimiter=',')
    csvw.writerow(['program_number','instrument_name','category'])
    for inst_cat, cat_dict in d.items():
        for inst_num,inst_name in cat_dict.items():
            cur = [inst_num, inst_name, inst_cat]
            csvw.writerow(cur)


