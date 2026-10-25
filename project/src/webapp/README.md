# 当前Web运行入口

唯一入口为workbench_server_v21（GeoNav4.13.1），由平台/启动平台.cmd调用。其server6至12等祖先和live2/3/4属于明确内部依赖，并非多套平行平台。static_v10为当前主视图，static_ground_v1为已接入地面任务视图，static_v7仅保必要地形资产。

空间证据使用spatial_evidence_v2.js；场景、地面实时与人工组件按当前绑定组合。旧绑定源码留本地复核，公开发布由repository_publication_v1和发布清单筛选，不能仅按最高编号删除依赖。页面可用不代表地面目标自主算法通过。
