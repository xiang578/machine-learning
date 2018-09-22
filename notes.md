- `line.strip()` 截取掉回车符
- `line.split()` 分割
- `list.extend()` 方法实现用新列表扩展原来的列表
```python
>>> a = [1,2,3]
>>> a.append([1,2,3])
>>> a
[1, 2, 3, [1, 2, 3]]
>>> a.extend([1,2,3])
>>> a
[1, 2, 3, [1, 2, 3], 1, 2, 3]
```

- `randindex = int(random.uniform(0,len(trainingset)))`: 取一个范围内的随机数

- `transpose()` numpy 矩阵转置

- `getA()` numpy 矩阵转成数组
- 声明主函数方法：`__name__ == '__main__'`
- numpy.array
- 初始化d维array为0 `zeros(d)`
- random 使用方法`idx = range(n) idx = random.sample(idx, n)`
- inner 内积，对于两个一维数组，计算的是这两个数组对应下标元素的乘积和
- 注意整数除法
