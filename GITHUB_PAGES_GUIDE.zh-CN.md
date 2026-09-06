# GitHub Pages 发布步骤

网页已经制作在 `docs/` 中，无需安装前端工具或执行构建。

## 1. 填写网页信息

打开 `docs/page-config.js`，修改以下五项；修改时保留英文引号和逗号：

| 字段 | 填写内容 |
| --- | --- |
| `authors` | 真实作者名单；留空则不显示 |
| `affiliations` | 作者单位；留空则不显示 |
| `paper` | 正式论文或预印本链接 |
| `code` | 您已创建的 GitHub 仓库完整链接 |
| `dataset` | Google Drive 数据集公开分享链接 |

网页中的资源按钮会统一读取这个文件。目前的三个资源地址是示例链接，需要替换后才能访问真实资源。作者和单位未虚构。

网页信息与仓库 README 独立：网页只需修改这一个配置文件；仓库中英文 README、`data/README.md` 和根目录 `links.json` 的示例链接也应同步替换。

## 2. 上传仓库内容

将 `PDP-Net` 文件夹里的文件和子文件夹上传到您现有仓库的根目录。首页应能直接看到 `README.md`、两个 Python 文件及 `docs/`。

不要把整个项目套在远程仓库的另一个 `PDP-Net/` 子文件夹里。使用网页上传时不要上传本地 `.git` 文件夹。图片数据仍通过 Google Drive 分发。

网页必需文件：

```text
docs/
├── index.html
├── style.css
├── page-config.js
├── page.js
├── .nojekyll
└── assets/
    ├── architecture.png
    ├── architecture.pdf
    ├── dataset_visualization.png
    └── dataset_visualization.pdf
```

`docs/USAGE.md` 是代码运行说明，与网页共存，不影响发布。

## 3. 开启 GitHub Pages

在 GitHub 仓库中依次选择：

1. **Settings → Pages**。
2. 在 **Build and deployment** 下，将 **Source** 设为 **Deploy from a branch**。
3. **Branch** 选择 `main`（如果您的文件上传到了另一个分支，请选择实际分支）。
4. 文件夹选择 **/docs**，点击 **Save**。
5. 等待部署完成，在 Pages 设置页点击 **Visit site**。

普通项目仓库的默认网址为 `https://用户名.github.io/仓库名/`。如果仓库本身名为 `用户名.github.io`，则使用 `https://用户名.github.io/`。请以 GitHub Pages 设置页显示的地址为准。

免费账户可使用公开仓库发布 Pages；私有仓库的可用性取决于账户套餐。参见 [GitHub Pages 官方说明](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)。

## 4. 检查发布结果

打开网站，确认两张图显示正常，Paper、Code、Dataset 按钮指向真实地址。Google Drive 链接应能让目标读者访问。

在仓库首页右侧 **About → 齿轮 → Website** 中填入网页地址；也可以把项目主页链接加入 README 顶部。

后续修改 `docs/` 内文件并上传到选定分支，网页会重新部署。若页面暂未更新，可查看仓库 **Actions** 中的 Pages 部署状态。

配置参考：[GitHub Pages 发布来源](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)。
