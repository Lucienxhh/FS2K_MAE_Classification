# FS2K_Classification

## 数据集转换
进入data_transform目录下，运行train_val.py得到标准格式的训练集和验证集，运行test.py得到标准格式的测试集。

## 预训练
根据链接下载预训练模型至根目录。

https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth

## 微调
运行main_finetune.py, 根据分类的属性，修改120行的路径名和122行的分类数。 模型默认保存至output_dir/checkpoint-39.pth。

## 测试
运行main_test.py， 根据分类的属性，修改39-43行， 其中41行是导入微调后的模型的路径。
