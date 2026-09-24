# V1.0 Editorial Style Hotfix

本补丁只更新正式公开前台的视觉样式和首页布局，不修改测量算法、远程计算、人工标注或历史结果。

## 更新内容

- 黑白灰编辑式界面；
- 首页虹彩媒体图；
- 直角卡片、输入框和表格；
- 全圆角黑白按钮；
- 无阴影细边框布局；
- 首页功能网格和五步流程重新排版；
- 保留成功、需复核和失败的必要语义状态色。

## 云服务器安装

在正式目录执行：

```bash
cd ~/yangzz/SiganusMorph
sudo systemctl stop siganusmorph-v1

mkdir -p ~/yangzz/SiganusMorph_style_backup_20260824
cp -a app.py siganusmorph/ui_styles.py ~/yangzz/SiganusMorph_style_backup_20260824/

unzip -o SiganusMorph_V1_EditorialStyle_Hotfix_20260824.zip
python3 -m py_compile app.py siganusmorph/ui_styles.py

sudo systemctl start siganusmorph-v1
curl -fsS http://127.0.0.1:8501/SiganusMorph/_stcore/health
```

浏览器端建议执行强制刷新：Windows 使用 `Ctrl+F5`。
