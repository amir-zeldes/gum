import os, re

os.chdir(r'C:\Users\logan\Dropbox\GUM\amir_gumdev\_build\target\dep\ud')

f = open('validate_errors_v7.txt','r')
flist = f.readlines()
fw = open('validate_errors_sorted1_v7.txt','w')
for line in flist:
    line = re.sub(":", ":\t", line)
    if line.startswith('GUM_'):
        id = line[:-1]
    elif line!="":
        if line.startswith("["):
            fw.write(line[0]+id+'\\'+line[1:])
        else:
            fw.write(id + '\\' + line)

f.close()
fw.close()