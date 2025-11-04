#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   logger.py
@Time    :   2023/08/23 21:59:59
@Author  :   Cao Zhanxiang 
@Version :   1.0
@Contact :   caozx1110@163.com
@License :   (C)Copyright 2023
@Desc    :   None
'''

class SimpleWriter(dict):
    def __init__(self) -> None:
        super().__init__()

    def log(self, key, value):
        if key not in self:
            self[key] = []
        
        self[key].append(value)
            
    def get(self, key):
        return self[key]
    
    def clear(self):
        self.clear()
    
    
if __name__ == '__main__':
    sl = SimpleWriter()
    sl.log('a', 1)
    sl.log('a', 2)
    sl.log('b', 3)
    sl.log('b', 4)
    print(sl)
