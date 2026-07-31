import os
import shutil
from multiprocessing import Pool


def copy_file_to_folder(_target_file):
    source_file = r'Q:\pychem_project\BIMContextNet_Master\data\REVERSE_IFCSchemaGraph.bin'

    # 如果目标文件夹中不存在相同的文件，则复制
    if not os.path.exists(_target_file):
        shutil.copy(source_file, _target_file)
        print("Copied to {}".format(_target_file))
    else:
        print("File already exists in {}".format(_target_file))


def delete_file_from_folder(_target_file):

    if os.path.exists(_target_file):
        os.remove(_target_file)
        print("Deleted {}".format(_target_file))
    else:
        print("File does not exist in {}".format(_target_file))


if __name__ == '__main__':
    target_file = []

    # use sampled dataset
    root_path = r"Q:\context500\test"
    for category in os.listdir(root_path):
        category_path = os.path.join(root_path, category)
        for instance in os.listdir(category_path):
            instance_path = os.path.join(root_path, category, instance)
            target_file_path = os.path.join(instance_path, 'GRAPH', "REVERSE_IFCSchemaGraph.bin")
            target_file.append(target_file_path)

    # 创建进程池
    num_processes = 24
    with Pool(num_processes) as pool:
        pool.map(delete_file_from_folder, target_file)

