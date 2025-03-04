# binding whisper.cpp and python

将whisper.cpp编译成动态库，通过pybind11导出Python调用数据结构和接口。


## 编译

修改Makefile，新增如下内容
```makefile
PYBIND11FLAGS = -shared -std=c++11 -fPIC $(shell python3 -m pybind11 --includes) $(shell python3-config --cflags)
pywhisper.so: examples/whisper.py/pywhisper.cpp $(SRC_COMMON) $(WHISPER_OBJ)
	$(CXX) $(PYBIND11FLAGS) $(CXXFLAGS) examples/whisper.py/pywhisper.cpp $(SRC_COMMON) $(WHISPER_OBJ) -o pywhisper.so $(LDFLAGS)
```

```bash
# cd 项目whisper.cpp目录下，执行下面的命令
WHISPER_CUBLAS=1 make pywhisper.so
```

## 数据结构和接口
数据结构：

```python
 from typing import List
 
 
 class segment_t:
    def __init__(self, start_tm: int, end_tm: int, text: str) -> None:
        self.start_tm = start_tm
        self.end_tm = end_tm
        self.text = text


class transcribe_result_t:
    def __init__(self, success: bool, segments: List[segment_t]) -> None:
        self.success = success
        self.segments = segments
```

接口列表：
- 是否已经初始化，`is_initialized() -> bool`
- 加载模型，`load(model_path: str) -> int`
- 转写，`transcribe(audio_path: str, langugage: str = 'auto', beam_size: int = 5) -> transcribe_result_t` 
- 销毁上下文，`destroy()`

## 示例

```python
# 编译成动态库后，导入pywhisper包
import pywhisper

# 加载模型
pywhisper.load("/data/models/whisper/ggml/base.bin")

# 检查是否初始化，返回结果是true
pywhisper.is_initialized()

# 转写
result = pywhisper.transcribe("/home/rd/Downloads/en-cn.wav")
for seg in result.segments:
   print(f"start_tm: {round(seg.start_tm / 1000, 2)}, end_tm: {round(seg.end_tm / 1000, 2)}, text: {seg.text}")
# start_tm: 0.0, end_tm: 4.98, text:  In a world, we should do everything to add to the greening of our cities.

# 销毁上下文
pywhisper.destroy()
```