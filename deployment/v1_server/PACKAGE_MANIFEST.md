# 部署包内容

## 包含

- `app.py`：V1.0 正式首页与导航入口。
- `pages/`：正式单鱼测量、批量测量、结果导出、高级工具以及保留的 V1 页面代码。
- `siganusmorph/`：校准、分割、关键点预标注、几何测量、导出和前端组件核心代码。
- `developer_tools/`：保留的旧开发后台代码，不作为默认公网入口。
- `assets/`：关键点说明和界面所需图像资源。
- 三个 `preannotation_candidate.pt`：V1 推荐流程实际使用的 v0.1、v0.5 heatmap 和 v0.3 YOLO 权重。
- `keypoint_source_recommendation.csv`：v0.6 keypoint-wise selector 的推荐来源表。
- `requirements-server.txt`、`install_server.sh`、`verify_server_runtime.py`：Linux CPU 服务器环境安装与检查。
- `config.toml`：`/SiganusMorph` 子路径的 Streamlit 配置模板。
- `siganusmorph-v1.service`：systemd 服务模板。
- `nginx-location.conf`：Nginx 子路径反向代理模板。

## 不包含

- V2.0 React/FastAPI/Tauri 源码；
- datasets、runs、训练日志和非运行候选权重；
- corrected_keypoints；
- 历史 batch measurement 和 final analysis；
- 本机绝对路径或本机 Python 虚拟环境。
