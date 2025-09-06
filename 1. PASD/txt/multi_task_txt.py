
def sorted_dict(path):
    # path = r'E:\pyWorkSpace\PA\txt\multi_task\MR_symptoms.txt'

    f = open(path, 'r')

    patient = dict()

    for l in f.readlines():
        id, label_0, label_1, label_2, label_3, label_4, label_5, label_6 = l.split()
        # print(id, label_0, label_1, label_2, label_3, label_4, label_5, label_6)
        patient[id] = [label_0, label_1, label_2, label_3, label_4, label_5, label_6]
    # print(patient)
    f.close()
    # print()
    # sorted(patient.items(), key=lambda kv:(kv[1], kv[0]))
    patient_1 = sorted(patient.items(), key=lambda s:s[0])
    # print(patient_1[0][1])
    return patient_1

symptoms_path = r'E:\pyWorkSpace\PA\txt\multi_task\multi_task_MRsymptoms.txt'
_tuple = sorted_dict(symptoms_path)

# for p in _tuple:
write_path = r'E:\pyWorkSpace\PA\txt\multi_task\sym_6.txt'
with open(write_path, 'w') as f:
    for p in _tuple:
        symptom = p[1][6]
        f.write(p[0] + '\t' + symptom + '\n')
f.close()