# ResolveFlow AI 前端

Vue 3 + Vite 构建的售后 Agent 演示工作台。页面已收口为单一 ResolveFlow 产品，不再向用户暴露后端语言选择。

页面用于展示：

- 用户多轮对话和 Agent 回复；
- 意图与主辅 Agent 路由；
- “收集—核验—检索—诊断—决策—执行”六步任务计划；
- 已核验业务事实、待确认写操作和决策依据；
- 工具调用名称、参数、成功状态和耗时；
- 知识库检索、Skills 与端到端评测。

用户回复与内部决策信息分层展示：对话区只显示对用户有用的结论和下一步；路由、风险级别、工具参数和交接信息收起在“运行详情”中，供产品演示和问题定位使用。

## 完整产品运行

推荐直接从后端目录启动一体化 Docker 环境：

```bash
cd "../后端"
docker compose up -d --build
```

打开 `http://localhost/`。该入口同时提供页面、`/api/python` API 代理与 `/docs` Swagger 文档。

## 前端开发运行

先确保后端已在 `http://localhost:8000` 启动，然后执行：

```bash
npm ci
npm run dev
```

打开 `http://localhost:5173`。Vite 会把 `/api/python` 请求代理到本地后端。

## 检查

```bash
npm test
npm run build
```

2026-10-30 实测：26 项前端测试通过，生产构建成功。

本项目使用本地演示业务数据，不代表已接入真实品牌的售后平台。
