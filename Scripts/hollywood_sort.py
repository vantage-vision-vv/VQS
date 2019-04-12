import os

classes = [
    'AnswerPhone',
    'DriveCar',
    'Eat',
    'FightPerson',
    'GetOutCar',
    'HandShake',
    'HugPerson',
    'Kiss',
    'Run',
    'SitDown',
    'SitUp',
    'StandUp'
]

file_path = '/home/user/Downloads/Hollywood2/ClipSets/'
vid_src = '/home/user/Downloads/Hollywood2/AVIClips/'
dest_path = '/tmp/Hollywood/'

for c in classes:
    os.system("mkdir " + dest_path + c)

    path = file_path + c + '_autotrain.txt'
    with open(path, 'r') as myfile:
        vid_list = myfile.readlines()

    for v in vid_list:
        flag = int((v.split('\n')[0]).split(' ')[-1])
        v_name = (v.split('\n')[0]).split(' ')[0]

        if flag == 1:
            path = vid_src + v_name + '.avi'
            os.system("cp " + path + " " + dest_path + c + "/")
