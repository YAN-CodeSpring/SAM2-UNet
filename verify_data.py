import os
import argparse
from pathlib import Path
import sys

def verify_dataset(img_dir, mask_dir, output_txt):
    # 1. 检查路径是否存在
    if not os.path.exists(img_dir):
        print(f"❌ 错误: 图片路径不存在 -> {img_dir}")
        sys.exit(1)
    if not os.path.exists(mask_dir):
        print(f"❌ 错误: Mask路径不存在 -> {mask_dir}")
        sys.exit(1)

    print(f"正在扫描图片: {img_dir} ...")
    print(f"正在扫描Mask: {mask_dir} ...")

    # 2. 支持的图片格式 (可以根据需要添加)
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    # 3. 获取文件列表 (保存完整文件名)
    # 过滤掉非图片文件（如 .DS_Store 或 .txt）
    img_files = [f for f in os.listdir(img_dir) if Path(f).suffix.lower() in valid_exts]
    mask_files = [f for f in os.listdir(mask_dir) if Path(f).suffix.lower() in valid_exts]

    # 4. 建立映射：文件名(不含后缀) -> 完整文件名
    # 例如: 'covid_19' -> 'covid_19.png'
    # 这样即使图片是.jpg，mask是.png，也能匹配上
    img_map = {Path(f).stem: f for f in img_files}
    mask_map = {Path(f).stem: f for f in mask_files}

    img_stems = set(img_map.keys())
    mask_stems = set(mask_map.keys())

    # 5. 计算交集和差集
    matches = img_stems & mask_stems  # 匹配成功的
    missing_masks = img_stems - mask_stems  # 有图没Mask
    orphaned_masks = mask_stems - img_stems  # 有Mask没图

    # 6. 写入报告
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("="*40 + "\n")
        f.write("       数据集校验报告 (QaTa-COV19)\n")
        f.write("="*40 + "\n\n")
        
        f.write(f"图片路径: {img_dir}\n")
        f.write(f"Mask路径: {mask_dir}\n")
        f.write(f"输出报告: {output_txt}\n\n")

        f.write("-" * 20 + " 统计信息 " + "-" * 20 + "\n")
        f.write(f"图片总数: {len(img_files)}\n")
        f.write(f"Mask总数: {len(mask_files)}\n")
        f.write(f"✅ 完美匹配对数: {len(matches)}\n")
        
        # 检查重复文件名（防止同一个文件夹里有同名不同后缀的文件，导致混淆）
        if len(img_files) != len(img_stems):
            f.write(f"⚠️ 警告: 图片文件夹中存在同名不同后缀的文件 (实际文件{len(img_files)} != 唯一文件名{len(img_stems)})\n")

        f.write("\n" + "-" * 20 + " 详细错误 " + "-" * 20 + "\n")

        has_error = False

        # 记录有图没Mask的情况
        if missing_masks:
            has_error = True
            f.write(f"\n❌ 发现 {len(missing_masks)} 张图片缺少对应的Mask:\n")
            for stem in sorted(list(missing_masks)):
                f.write(f"  [图片存在] {img_map[stem]}  -->  [Mask缺失]\n")
        
        # 记录有Mask没图的情况
        if orphaned_masks:
            has_error = True
            f.write(f"\n⚠️ 发现 {len(orphaned_masks)} 个Mask缺少对应的图片 (可能是多余文件):\n")
            for stem in sorted(list(orphaned_masks)):
                f.write(f"  [图片缺失]  <--  [Mask存在] {mask_map[stem]}\n")

        if not has_error:
            f.write("\n✨ 恭喜！所有图片和Mask一一对应，未发现问题。\n")

    print("-" * 50)
    print(f"校验完成！")
    if len(matches) > 0 and not missing_masks:
        print(f"✅ 成功匹配 {len(matches)} 对数据。")
    else:
        print(f"❌ 发现问题，请查看报告。")
    print(f"📄 详细结果已保存在: {output_txt}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify QaTa-COV19 Dataset Alignment")
    parser.add_argument('--img_dir', type=str, required=True, help='Path to Images folder')
    parser.add_argument('--mask_dir', type=str, required=True, help='Path to Ground-truths folder')
    parser.add_argument('--output', type=str, default='verify_result.txt', help='Path to save output report')
    
    args = parser.parse_args()
    
    verify_dataset(args.img_dir, args.mask_dir, args.output)