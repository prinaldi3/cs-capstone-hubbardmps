import os

mpsdir = "./Data/Tenpy/Basic/"
for file in os.listdir(mpsdir):
    sname = file.split("-")
    if sname[0] != "times":
        temp = sname[-1]
        sname = sname[:-2]
        sname[-1] += "." + temp.split(".")[-1]
        new_name = "-".join(sname)
        os.rename(mpsdir + file, mpsdir + new_name)
