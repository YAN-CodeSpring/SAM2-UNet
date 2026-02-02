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

    # 2. 支持的图片格式
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    # 3. 获取文件列表
    img_files = [f for f in os.listdir(img_dir) if Path(f).suffix.lower() in valid_exts]
    mask_files = [f for f in os.listdir(mask_dir) if Path(f).suffix.lower() in valid_exts]

    # 4. 建立映射：核心文件名 -> 完整文件名
    
    # 图片处理：直接用文件名(无后缀)作为 Key
    img_map = {Path(f).stem: f for f in img_files}
    
    # Mask处理：【核心修改】去除 'mask_' 前缀
    mask_map = {}
    for f in mask_files:
        stem = Path(f).stem
        # 如果文件名以 mask_ 开头，则切掉前5个字符
        if stem.startswith("mask_"):
            core_name = stem[5:]  # 去掉 'mask_'
        else:
            core_name = stem      # 假如有的没有前缀，就保持原样
        
        mask_map[core_name] = f

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
        
        f.write("\n" + "-" * 20 + " 详细错误 " + "-" * 20 + "\n")

        has_error = False

        # 记录有图没Mask的情况
        if missing_masks:
            has_error = True
            f.write(f"\n❌ 发现 {len(missing_masks)} 张图片缺少对应的Mask:\n")
            # 显示前10个例子，防止刷屏
            count = 0
            for stem in sorted(list(missing_masks)):
                f.write(f"  [图片: {img_map[stem]}] --> 找不到对应Mask (预期: mask_{stem}.png)\n")
                count += 1
                if count >= 20:
                    f.write(f"  ... 以及其他 {len(missing_masks)-20} 张 ...\n")
                    break
        
        # 记录有Mask没图的情况
        if orphaned_masks:
            has_error = True
            f.write(f"\n⚠️ 发现 {len(orphaned_masks)} 个Mask缺少对应的图片:\n")
            count = 0
            for stem in sorted(list(orphaned_masks)):
                f.write(f"  [Mask: {mask_map[stem]}] --> 找不到对应图片 (预期: {stem}.png/jpg)\n")
                count += 1
                if count >= 20:
                    f.write(f"  ... 以及其他 {len(orphaned_masks)-20} 个 ...\n")
                    break

        if not has_error:
            f.write("\n✨ 恭喜！所有图片和Mask一一对应（已自动处理 mask_ 前缀）。\n")

    print("-" * 50)
    print(f"校验完成！(已处理 'mask_' 前缀)")
    if len(matches) > 0 and not missing_masks:
        print(f"✅ 成功匹配 {len(matches)} 对数据。")
    else:
        print(f"❌ 发现问题，请打开txt查看详细报告。")
    print(f"📄 详细结果已保存在: {output_txt}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify QaTa-COV19 Dataset Alignment")
    parser.add_argument('--img_dir', type=str, required=True, help='Path to Images folder')
    parser.add_argument('--mask_dir', type=str, required=True, help='Path to Ground-truths folder')
    parser.add_argument('--output', type=str, default='verify_result.txt', help='Path to save output report')
    
    args = parser.parse_args()
    
    verify_dataset(args.img_dir, args.mask_dir, args.output)