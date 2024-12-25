from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
    txt_path = '/data1/jianghai/DECT/txt/binary/train.txt'
    id_list, label_list = [], []
    f = open(txt_path, 'r')
    for line in f:
        id_list.append(line.split()[0])
        label_list.append(line.split()[1])
    # print(imgs_info)
    
    skf = StratifiedKFold(n_splits=4)
    # for i in imgs_info:
    #     print(i, imgs_info[i])
    #     print(type(i), type(imgs_info[i]))
    # label_list = []
    # for i in imgs_info:
    #     label_list.append(imgs_info[i])
    # print(label_list)
    for i, (train_index, val_index) in enumerate(skf.split(id_list, label_list)):
        print('Fold ' + str(i))
        # print('\t', imgs[train_index])
        # print('\t', imgs[val_index])
        # print(train_index)
        # print(val_index)
        # print(train_index)
        train_id, train_label = [], []
        val_id, val_label = [], []
        for i in train_index:
            print(id_list[i], label_list[i])
            train_id.append(id_list[i])
            train_label.append(label_list[i])
        for i in val_index:
            print(id_list[i], label_list[i])
            val_id.append(id_list[i])
            val_label.append(label_list[i])

        



        