import os

def get_all_fjsp_instances():
    '''Lists all FJSP instances'''
    matching_files = []
    root_dir = "../../benchmarks/fjsp"
    target_string = ".fjs"

    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(foldername, filename)
            if target_string in filepath:
                matching_files.append(filepath)

    return matching_files

with open('fjsp_details.csv', 'w') as info_file:
    info_file.write('instance,jobs,machines,best_makespan\n')
    for i, instance in enumerate(get_all_fjsp_instances()):
        with open(instance, 'r') as instance_file:
            header = instance_file.readline().removesuffix('\n').split()
            jobs = int(header[0])
            machines = int(header[1])
            info_file.write(f"{instance.split('/').pop()},{jobs},{machines},\n")
