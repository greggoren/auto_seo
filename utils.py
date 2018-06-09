import math
import subprocess

# def cosine_similarity(v1,v2):
#     sumxx, sumxy, sumyy = 0, 0, 0
#     for i in range(len(v1)):
#         x = v1[i]; y = v2[i]
#         sumxx += x*x
#         sumyy += y*y
#         sumxy += x*y
#     return sumxy/math.sqrt(sumxx*sumyy)

def cosine_similarity(v1,v2):
    keys1 = set(v1.keys())
    keys2 = set(v2.keys())
    mutual_keys = keys1.intersection(keys2)
    sum = 0
    for key in mutual_keys:
        x=v1[key]
        y=v2[key]
        sum+=x*y
    norm1=compact_norm(v1)
    norm2=compact_norm(v2)
    denominator = norm1*norm2
    if denominator==0:
        return 0
    return sum/denominator

def compact_norm(v1):
    sum=0
    for id in v1:
        x=v1[id]
        sum+=x*x
    return math.sqrt(sum)



def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
    return iter(p.stdout.readline, b'')



def run_bash_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, shell=True)
    try:

        subprocess.check_call(command,shell=True)
    except subprocess.CalledProcessError as e:
        print(e)

    out, err = p.communicate()
    return out