# ResolveFlow AI Skills

ResolveFlow AI 启动时从 `RESOLVEFLOW_SKILLS_DIR` 读取业务 Skills，并根据用户消息和 Agent 角色注入 system prompt。Skills 用于维护售后 SOP、信息收集、诊断边界、工单确认和人工升级规则；真实工具权限仍由 Python 白名单控制。

当前内置四类 Skills：

```text
skills/general_customer_service/SKILL.md  # intake：接待、澄清和分诊
skills/technical_support/SKILL.md         # diagnostic：设备诊断和安全边界
skills/billing_support/SKILL.md           # resolution：保修、方案和工单
skills/escalation_handoff/SKILL.md        # escalation：安全停止和人工交接
```

## Front matter

```markdown
---
name: 消费电子设备诊断规范
description: 设备故障检索、低风险诊断和安全升级规范
keywords: 故障,无法充电,连不上,鼓包,冒烟
agents: diagnostic
enabled: true
---
```

- `name`：Skill 展示名称；
- `description`：用途说明；
- `keywords`：用户消息命中后注入；
- `agents`：`intake`、`diagnostic`、`resolution`、`escalation`；
- `enabled`：是否启用。

## 设计边界

- Skill 负责业务规范和表达边界，不直接赋予工具权限；
- 事实必须来自用户输入、结构化状态、RAG 或业务工具；
- 写操作必须由程序检查确认、幂等和重试安全；
- 重要规则放在文档前半部分，避免超过注入预算；
- 修改后调用 `POST /skills/reload` 热加载，再用 `GET /skills` 检查结果。
