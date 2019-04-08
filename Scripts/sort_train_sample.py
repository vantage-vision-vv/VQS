import os

dest_path = "/tmp/Virat"
src_path = "/tmp/Virat_Trimed"

files = os.listdir(src_path)

map_class ={1:"Load",2:"UnLoad",3:"Open",4:"Close",5:"Into_Vehicle",6:"Outof_Vehicle",7:"Gesturing",8:"Dig",9:"Carry",10:"Run",11:"Enter",12:"Exit"}

for i in range(1,13):
    os.system("mkdir " + dest_path + map_class[i])

for item in files:
    label = item.split("_")[0]
    os.system("cp "+src_path+item+" "+dest_path+map_class[int(label)]+"/")