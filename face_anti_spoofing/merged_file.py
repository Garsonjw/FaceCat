# 指定要合并的两个文件名
file1 = "/data/home/scv7305/run/chenjiawei/ddpm-cd/log/b+c+d_log/1/None/test_path.txt"  # None Co Replay Paper Paperglass
file2 = "/data/home/scv7305/run/chenjiawei/ddpm-cd/baseline/face_anti_spoofing/log/epoch_18_score_test.txt"
#CDCN_log/2/Co
# 读取第一个文件的内容
f1 = open(file1, 'r')
i=0
with open(file2,'r') as f:
    lines = f.readlines()
    for f in f1:
        f_new = f.strip() + ' ' + lines[i]
        #print(f_new)
        i+=1
        with open("my_log_1_None.txt", 'a') as f:
            f.write(f_new)
