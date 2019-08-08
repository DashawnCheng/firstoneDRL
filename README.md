# firstoneDRL
DRL学习实录


#tensorflow 

% use tensorflow to print session

hello.py
深度学习框架

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TensorFlow = Tensor +Flow 

 Tensor 张量
	数据结构：多维数组
 Flow 流
	计算模型：张量之间通过计算而转换的过程
TensorFlow 是一个通过计算图的形式表述计算的编程系统

计算图是一个有向图， 由一组节点，一组有向边组成
TensorFlow有两种边：
常规边（实线）：代表数据依赖关系。
特殊边（虚线）：不携带值，表示两个节点之间的控制关系。（happens-before）

张量是一个标量，张量保存的是计算的过程，内容需要调用会话窗口。

张量：
名字(name): "node : src_output"  节点名称 节点的第几个输出
形状(shape) 张量的维度信息 shape=()表示是标量
类型(type) 每一个张量都会有一个唯一的类型， tensorflow会对参与运算的所有张量进行类型检查

三个属于描述张量的维度 阶 形状 维数

TensorFlow支持14种不同的类型 
实数 float32 float64
整数 int8 int16 int32 int64 uint8
布尔 bool
复数 complex64 complex128
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
