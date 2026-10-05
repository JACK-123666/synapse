# Synapse 插件开发指南

Synapse 支持两种扩展方式：

| 方式 | 适合场景 | 配置位置 |
|------|----------|----------|
| 本地 Python 插件 | 需要写业务逻辑、调用内部系统 | `plugins/<插件名>/` 目录 |
| MCP 服务 | 已有现成的 MCP Server（文件系统、数据库、浏览器……） | `POST /plugins/mcp/servers` |

两种方式注册的工具都会进入工具注册表，受角色白名单约束；内置能力（知识库、记忆、网页、代码仓库、定时任务）也是用同一套 `Capability` 接口实现的。

## 一、本地 Python 插件

### 1. 目录结构

```text
plugins/
└── my_plugin/
    ├── plugin.yaml
    └── __init__.py
```

### 2. plugin.yaml

```yaml
name: my_plugin            # 插件名：字母、数字、下划线、横线
version: 1.0.0
description: 我的插件
author: me
entry: __init__:MyPlugin   # 模块:类名（模块相对插件目录）；省略时自动查找唯一的 Capability 子类
permissions: [read]        # 声明 write 才允许注册写操作工具
```

### 3. 入口代码

```python
from langchain_core.tools import tool

from app.capabilities.base import Capability, CapabilityTool
from app.core.context import get_request_context
from app.intent.catalog import IntentSpec


@tool("query_order")
async def query_order(order_id: str) -> str:
    """查询订单状态。

    Args:
        order_id: 订单号
    """
    user = get_request_context()          # 当前用户：user_id / role / session_id
    return f"订单 {order_id} 已发货（查询人 {user.username}）"


class MyPlugin(Capability):
    name = "my_plugin"
    description = "订单查询"

    def tools(self):
        return [CapabilityTool(query_order, tags=("order",))]

    def intents(self):
        return [IntentSpec(
            name="order_query",
            description="查询订单状态、物流信息",
            keywords=["订单", "物流", "发货"],
            examples=["帮我查一下订单 123 的状态", "我的快递到哪了"],
        )]
```

可覆写的方法：

| 方法 | 作用 |
|------|------|
| `tools()` | 返回 LangChain 工具，或用 `CapabilityTool(tool, tags=..., write=..., roles=...)` 附带元数据 |
| `intents()` | 返回 `IntentSpec`，自动加入三路意图识别（LLM 分类 prompt、向量示例、关键词） |
| `agents()` | 返回专属 Agent（继承 `LangChainAgent`，设置 `tool_tags` 与 `build_system_prompt`） |
| `routes()` | 意图到 Agent 的路由，如 `{"order_query": ["order_agent", "general_agent", "fallback_agent"]}` |
| `startup()` / `shutdown()` | 加载 / 卸载时的初始化与清理 |

只声明 `intents()` 而不提供 `agents()` / `routes()` 时，平台会自动生成 `<插件名>_agent`（只使用本插件的工具），并路由为 `[<插件名>_agent, general_agent, fallback_agent]`。

### 4. 工具编写约定

- 优先写 `async` 工具；docstring 第一行是工具描述，`Args:` 部分说明参数，模型据此决定何时调用
- 通过 `get_request_context()` 获取当前用户，按用户做数据隔离
- 有外部副作用的工具标记 `write=True`，并在 `plugin.yaml` 中声明 `permissions: [write]`
- 出错时返回可读的错误文本，而不是抛出异常，模型可以据此向用户解释

### 5. 管理接口（仅管理员）

| 接口 | 说明 |
|------|------|
| `GET /plugins` | 插件列表（版本、启停、加载错误、注册的工具和意图） |
| `POST /plugins/{name}/enable` / `disable` | 启用 / 停用（状态持久化） |
| `POST /plugins/{name}/reload` | 热重载：修改代码后无需重启服务 |
| `POST /plugins/reload` | 重新扫描插件目录 |

参考示例：[plugins/example_plugin](../plugins/example_plugin/__init__.py)。

## 二、MCP 服务

```bash
# 远程 HTTP 服务
curl -X POST http://localhost:8000/plugins/mcp/servers \
  -H "Authorization: Bearer <管理员 token>" -H "Content-Type: application/json" \
  -d '{"name": "docs", "transport": "streamable_http",
       "config": {"url": "https://mcp.example.com/mcp", "headers": {"Authorization": "Bearer xxx"}}}'

# 本地 stdio 服务（在服务器上启动子进程）
curl -X POST http://localhost:8000/plugins/mcp/servers \
  -H "Authorization: Bearer <管理员 token>" -H "Content-Type: application/json" \
  -d '{"name": "fs", "transport": "stdio",
       "config": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"],
                  "intent": {"description": "读取服务器 /data 目录下的文件", "keywords": ["文件", "目录"]}}}'
```

- 工具名统一为 `mcp_<服务名>_<工具名>`，避免与内置工具冲突
- 默认由通用 Agent（`general_task` 意图）使用；配置 `intent` 后会注册专属意图与 Agent
- MCP 工具注解 `destructiveHint=true` 或 `readOnlyHint=false` 的工具会被标记为写操作
- `GET /plugins/mcp/servers` 查看连接状态与错误；`POST /plugins/mcp/reload` 重新连接

## 三、权限

- 插件与 MCP 的管理接口仅管理员可用
- `PUT /plugins/policies/{role}` 为角色设置工具白名单（fnmatch 通配），如普通用户只允许 `["web_*", "knowledge_*", "memory_*"]`
- `GET /plugins/tools` 查看当前用户可用的工具
