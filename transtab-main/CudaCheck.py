import os

def check_cuda_installation():
    cuda_home = os.getenv('CUDA_HOME')
    print(cuda_home)
    if cuda_home and cuda_home == '/usr/local/cuda':
        print("CUDA 工具包已安装，安装路径为: /usr/local/cuda")
    else:
        print("CUDA 工具包未安装或未设置 CUDA_HOME 环境变量")

if __name__ == "__main__":
    check_cuda_installation()
