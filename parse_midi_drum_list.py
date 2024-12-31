import csv

d = {}
with open('midi_drum_list.txt', 'r') as f:
    for l in f.readlines():
        l2 = l.strip()
        psplit = l2.split('(')
        is_gm2 = False
        # detect lines ending with (GM2)
        if ')' in psplit[-1]:
            is_gm2 = True
        psplit2 = psplit[0].strip().split(' ')
        note_num = int(psplit2[0])
        inst_name = ' '.join(psplit2[1:])
        d[note_num] = {}
        d[note_num]['note_num'] = note_num
        d[note_num]['inst_name'] = inst_name
        d[note_num]['is_gm2'] = 1 if is_gm2 == True else 0

print(d)
with open('drum_list.csv', 'w') as f:
    csvw = csv.writer(f, delimiter=',')
    csvw.writerow(['program_number', 'note_number','instrument_name','GM2'])
    for note_num, inst_dict in d.items():
        cur = [1, note_num, inst_dict['inst_name'], inst_dict['is_gm2']]
        csvw.writerow(cur)


