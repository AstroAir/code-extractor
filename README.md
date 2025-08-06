# pysearch

pysearch 是一个面向 Python 代码库的高性能、上下文感知搜索引擎，支持文本/正则/AST/语义搜索，提供 CLI 与可编程 API，适用于大型多模块项目的工程化检索。

特性概览
- 代码块匹配：函数、类、装饰器、导入、字符串/注释、任意代码片段
- 上下文感知：返回匹配代码并带可配置上下文行数
- 全项目搜索：高效索引与缓存，适配大型代码库
- 多种匹配：正则、AST 结构化、语义（轻量向量/符号特征）
- 强定制化：包含/排除目录、文件类型、上下文窗口、过滤器（函数名、类名、装饰器、导入等）
- 多格式输出：plain text、JSON、带高亮的控制台输出
- 评分排序：可配置的结果打分规则
- 性能指标：检索时间、扫描文件数、命中数量等
- CLI 与 API：命令行操作与 Python 内嵌调用
- 测试与基准：pytest 覆盖率 > 90%，附带基准脚本

## 安装

建议使用 Python 3.10+。
```
pip install -e .
```

开发依赖：
```
pip install -e ".[dev]"
```

## 快速开始

CLI
```
pysearch find \
  --pattern "requests.get" \
  --path . \
  --regex \
  --context 3 \
  --format text
```

API
```python
from pysearch.api import PySearch
from pysearch.config import SearchConfig

engine = PySearch(SearchConfig(paths=["."], include=["**/*.py"], context=2))
results = engine.search(pattern="def main", regex=True)
for r in results.items:
    print(r.file, r.lines)
```

## 核心能力

- 文本/正则搜索：基于 `regex` 提供更强正则能力，支持多行模式与命名分组
- AST 搜索：基于 `ast` 和自定义匹配器，按函数/类/装饰器/导入过滤或定位节点
- 语义搜索：轻量向量+符号特征，考虑结构与标识符语义（无需外部模型）
- 索引与缓存：记录文件 mtime、哈希、大小，实现增量更新
- 输出与高亮：`rich`/`pygments` 控制台高亮，`orjson` 快速 JSON 输出

## 典型用例

- 查找使用特定装饰器的所有函数
- 定位引入特定模块的文件，并显示上下文
- 搜索所有包含某正则模式的代码块
- 基于 AST 查找所有名为 X 的类/函数定义
- 跨项目统计匹配结果与性能指标

## CLI 使用

```
pysearch find \
  --path src tests \
  --include "**/*.py" \
  --exclude "*/.venv/*" "*/build/*" \
  --pattern "def .*_handler" \
  --regex \
  --context 4 \
  --format json \
  --filter-func-name ".*handler" \
  --filter-decorator "lru_cache" \
  --rank "ast_weight:2,text_weight:1"
```

主要参数
- --path: 搜索路径（可多个）
- --include/--exclude: 包含/排除的 glob 模式
- --pattern: 文本/正则模式或语义查询
- --regex: 启用正则匹配
- --context: 上下文行数
- --format: 输出格式 text/json/highlight
- --filter-func-name/--filter-class-name/--filter-decorator/--filter-import: AST 过滤器
- --rank: 排序权重配置
- --docstrings/--comments/--strings: 是否搜索文档字符串、注释、字符串字面量
- --stats: 打印性能统计

## 编程接口

```python
from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.types import Query, OutputFormat

cfg = SearchConfig(
    paths=["."],
    include=["**/*.py"],
    exclude=["**/.venv/**", "**/build/**"],
    context=3,
    output_format=OutputFormat.JSON,
    enable_docstrings=True,
    enable_comments=True,
    enable_strings=True,
)

engine = PySearch(cfg)
res = engine.run(Query(pattern="ClassName", use_regex=False, use_ast=True))
print(res.stats, len(res.items))
```

## 测试与基准

运行测试与覆盖率：
```
pytest
```

运行基准：
```
pytest tests/benchmarks -k benchmark -q
```

## 路线图

- 更强的语义检索（可选外部嵌入后端）
- IDE/编辑器集成（VS Code/JetBrains）协议化输出
- 并行与分布式索引
- 更精细的语法高亮与差异化展示

## 许可证

MIT