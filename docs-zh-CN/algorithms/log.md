# Log

Log 是 LibLLIE 中已经实现的对数变换增强算法。

## 链接

| 类型 | URL |
| --- | --- |
| 论文链接 | None |
| 官方源码 | None |
| 官方项目页 | None |

## 算法简介

对数变换是一种经典的灰度级增强方法，常用于压缩高亮区域并提升暗区域的可见性。其基本形式为：

```text
output = gain * log(1 + input)
```

由于低强度像素在对数函数中会获得相对更大的增益，因此该方法适合用作简单、快速的低光增强基线。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `libllie/traditional/algorithms/Log.py` |
| 算法类名 | `Log` |
| 注册名称 | `log` |
| 基类 | `libllie/traditional/algorithms/BaseEnhancer.py` 中的 `LLIEnhancer` |

## 实现说明

LibLLIE 中的 Log 方法会先将图像转换到浮点范围，再执行对数映射和归一化，最后恢复到原始 dtype。

主要参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `gain` | `float` | `10.0` | 对数增强增益，必须为正数 |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "log",
    "input.jpg",
    output="results/log/output.jpg",
    gain=10.0,
)
```
