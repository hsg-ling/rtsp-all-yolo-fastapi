import torch

if __name__ == '__main__':
    print("PyTorch版本:", torch.__version__)
    print("PyTorch编译时的CUDA版本:", torch.version.cuda)
    print("CUDA可用:", torch.cuda.is_available())
    # 查看torch安装路径（判断是否是旧版本）
    print("torch模块路径:", torch.__file__)
    print(f'CUDA可用: {torch.cuda.is_available()}, GPU数量: {torch.cuda.device_count()}')
