# SiganusMorph Local V1.0 服务器部署

本压缩包只包含 V1.0 服务器运行所需的代码、正式页面、模型权重和部署模板。它不包含训练数据、V2.0、历史 corrected keypoints 或历史批量分析结果。

## 1. 解压位置

```bash
sudo mkdir -p /opt/siganusmorph-v1
sudo unzip SiganusMorph_V1_Server_20260821.zip -d /opt/siganusmorph-v1
sudo useradd --system --home-dir /opt/siganusmorph-v1 --shell /usr/sbin/nologin siganusmorph || true
sudo chown -R siganusmorph:siganusmorph /opt/siganusmorph-v1
```

解压后入口文件应直接位于 `/opt/siganusmorph-v1/app.py`。

## 2. 安装系统依赖

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip nginx unzip libgl1 libglib2.0-0 libgomp1
```

2GB 内存服务器建议额外设置 4GB swap。

## 3. 安装 Python 依赖

```bash
sudo -u siganusmorph -H bash -c 'cd /opt/siganusmorph-v1 && bash install_server.sh'
```

## 4. 安装 Streamlit 配置

```bash
sudo -u siganusmorph mkdir -p /opt/siganusmorph-v1/.streamlit
sudo cp /opt/siganusmorph-v1/config.toml /opt/siganusmorph-v1/.streamlit/config.toml
sudo chown -R siganusmorph:siganusmorph /opt/siganusmorph-v1/.streamlit
```

## 5. 启用 systemd

```bash
sudo cp /opt/siganusmorph-v1/siganusmorph-v1.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now siganusmorph-v1
sudo systemctl status siganusmorph-v1
curl http://127.0.0.1:8501/SiganusMorph/_stcore/health
```

## 6. 配置 Nginx

把 `nginx-location.conf` 的内容加入现有 `yangzz.online` 的 `server` 块，然后执行：

```bash
sudo nginx -t
sudo systemctl reload nginx
```

访问地址：

```text
https://yangzz.online/SiganusMorph/
```

## 7. 日志与输出

```bash
sudo journalctl -u siganusmorph-v1 -f
```

新保存的用户结果只写入：

```text
/opt/siganusmorph-v1/results/formal_v1_user_outputs/
```
