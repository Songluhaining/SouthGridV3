# v3 模型交付包（GitHub Release 附件）

训练好的 pi0.5+LoRA 权重（`pi05_g1_button_lora` / `g1_button_local_v3/29999`）体积 6.45 GB，超出 git 范畴，
以 Release 附件形式提供，切成 4 个分片（每片 ≤ 1.99 GB）。

## 下载与还原
```bash
# 从本仓库 Releases 页面下载以下文件到同一目录：
#   g1_button_v3_bundle.tar.zst.part00 ~ part03、SHA256SUMS、README_BUNDLE.md
cat g1_button_v3_bundle.tar.zst.part* > g1_button_v3_bundle.tar.zst
sha256sum -c SHA256SUMS                             # 全部 OK 才继续
zstd -d -c g1_button_v3_bundle.tar.zst | tar -x     # 需要 zstd
```

## 校验和
```
a8ee70b0bc8ce1df23a188988cee2294f4b132af41b1cb06abcb78498066e18e  g1_button_v3_bundle.tar.zst
7a3afe61201fbde57d31627f399962fa5afe57554513a8e518fdcde8bd2c57a4  g1_button_v3_bundle.tar.zst.part00
8a778cb79036afd41ebae7adbf72e319bac7b4b42a47804c6ea8ec07097d404c  g1_button_v3_bundle.tar.zst.part01
d20c3389ccecb9143d0cb946969e18fc2662045d15cb42f081038982ea3da05d  g1_button_v3_bundle.tar.zst.part02
8b9b2551076c1ecfedf5260ff62b0e97fa221eb85e30ffbeb89bff5618f7ec7b  g1_button_v3_bundle.tar.zst.part03
```

## 包内容与使用
解包后见 `README_BUNDLE.md`：`checkpoint/`（权重 + 归一化统计）、`openpi_data/`（tokenizer）、
`SouthGrid/`（本分支源码快照）与 `SouthGrid.git.bundle`。策略服务与评测步骤见
`openpi_patches/g1_button/README.md` 和 `docs/g1_omnipicker_inference.md`。
