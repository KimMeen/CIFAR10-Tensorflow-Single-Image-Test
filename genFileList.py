import os

img_root = "./cifar-10/"

# get file list in source dir
def get_file_list(src_dir):
    file_list = []
    for filename in os.listdir(src_dir):
        file_list.append(filename)
        # print(filename)
    return file_list

def get_label(imgs_dir):
    path_name = os.path.abspath(imgs_dir)
    sp = path_name.split('/')
    return sp[-1]

def gen_label_file(imgs_dir):
    rows = []
    file_list = get_file_list(imgs_dir)
    label = get_label(imgs_dir)
    for file in file_list:
        rows.append(file + ' ' + label + '\n')
    return rows

def files_labels2txt():
    train_path = img_root + "train/"
    test_path = img_root + "test/"
    tmp_train = []
    tmp_test = []
    # gen train file list
    for i in range(10):
        train_dir = train_path + str(i) + '/'
        tmp_train = tmp_train + gen_label_file(train_dir)
        test_dir = test_path + str(i) + '/'
        tmp_test = tmp_test + gen_label_file(test_dir)
    open(train_path + 'train.txt', "w").writelines(tmp_train)
    open(test_path + 'test.txt', "w").writelines(tmp_test)


if __name__ == "__main__":
    #get_label(img_root + "mnist_test/0/")
    #gen_label_file(img_root + "mnist_test/0/")
    files_labels2txt()
