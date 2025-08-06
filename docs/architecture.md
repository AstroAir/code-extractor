# 架构设计

本节概述 pysearch 的核心组件与数据流，帮助贡献者快速理解内部实现。

## 模块划分

- 核心模块
  - `pysearch.indexer`：文件扫描、哈希/mtime 缓存、增量索引
  - `pysearch.matchers`：文本/正则匹配、AST 节点过滤，基础语义信号抽取
  - `pysearch.scorer`：结果评分与排序，基于多信号（位置、频次、结构等）
  - `pysearch.formatter`：输出渲染（text/json/highlight）
  - `pysearch.api`：对外 API 封装（`PySearch`）、统一入口
  - `pysearch.cli`：命令行接口（click）
  - `pysearch.config`：`SearchConfig` 配置对象定义与校验
  - `pysearch.utils` / `pysearch.types`：通用工具与类型声明
  - `pysearch.ide_hooks`：IDE 集成相关输出（如后续扩展 LSP/编辑器协议）

- 外部依赖
  - `regex`：增强版正则库
  - `rich/pygments`：高亮与控制台美化
  - `orjson`：快速 JSON 序列化
  - `click`：CLI 框架
  - `pydantic`：配置/类型校验

## 数据流与执行路径（CLI）

1. `pysearch.cli:find_cmd` 解析用户参数，构造 `SearchConfig` 和（可选）`ASTFilters`
2. 构造 `Query` 并调用 `PySearch.run`
3. `PySearch` 协调 `indexer` 扫描与缓存、`matchers` 执行匹配
4. 结果交由 `scorer` 打分排序
5. 通过 `formatter` 输出（text/json/highlight），并可选打印统计信息

## 索引与缓存

- 最小化设计：按需遍历路径，结合 mtime/大小/hash 等元信息进行增量更新
- 可后续扩展：
  - 多进程/多线程并行索引
  - 跨仓库/多项目索引
  - 持久化缓存（磁盘/SQLite）

## AST 与语义信号

- AST：基于 `ast` 模块解析，提供函数/类/装饰器/导入级别过滤
- 语义：当前为轻量特征（结构/标识符等），后续可引入可选嵌入后端（保持可拔插）

## 可观测性与性能

- CLI `--stats` 输出：文件扫描数/命中/耗时/索引命中
- `pytest-benchmark`：提供基准能力（tests/benchmarks 目录可扩充）
- 后续可加入跨度跟踪与更细粒度的 profiling hooks

## 错误处理与日志

- CLI：对常见输入错误进行友好提示，非零退出码
- API：抛出清晰的异常类型或返回包含错误上下文的结构
- 可选：引入标准化日志（logging）与调试开关

## 兼容性与扩展

- Python 3.10+，优先使用类型注解与现代语法
- 配置面向扩展：新增过滤器/匹配器时尽量保持后向兼容
- 输出格式可扩展：markdown、HTML、高亮文件等