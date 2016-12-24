import threading

def task(f,params):
    t = threading.Thread(target=f,args=params)
    return t

def fib(x):
    return 1 if x<=2 else fib(x-1)+fib(x-2)

import threadpool as tp
pool = tp.ThreadPool(8)

# async map function with multithreading support.
# returned mapresult is a map with integer indices.
def amap(f,plist):
    mapresult = {}

    def wrapper(idx):
        param = plist[idx]
        return idx,f(param)

    idxlist = range(len(plist))

    def taskend(request,result):
        idx,res = result
        mapresult[idx] = res

    reqs = tp.makeRequests(wrapper, idxlist, taskend)
    #构建请求，get_title为要运行的函数，data为要多线程执行函数的参数，最后这个print_result是可选的，是对前两个函数运行结果的操作
    [pool.putRequest(req) for req in reqs]  #多线程一块执行
    pool.wait()  #线程挂起，直到结束

    return mapresult
