import numpy as np
def demo1(a):
    if len(a()) <=10:
        print('aa')
    else:
        print('大于10')
@demo1
def demo():
    print('123')
    return 'good evening'
demo()