import os
import numpy as np
from PIL import Image
# from PIL.UnidentifiedImageError import UnidentifiedImageError

# 目标文件夹路径（注意路径中的空格要保留）
TARGET_FOLDER = "/root/autodl-tmp/COVID/output1_predict_results"

# 支持的图片格式（小写，避免格式判断错误）
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

# 初始化全局像素值统计变量
global_min_pixel = None
global_max_pixel = None
all_pixel_values = set()  # 存储所有出现过的像素值（去重）

def get_image_info(image_path):
    """
    获取单张图片的详细信息
    :param image_path: 图片文件路径
    :return: 包含图片信息的字典，失败返回None
    """
    try:
        # 打开图片（避免PIL自动旋转图片）
        with Image.open(image_path) as img:
            # 基本信息
            img_name = os.path.basename(image_path)
            width, height = img.size  # 宽度、高度（像素）
            mode = img.mode  # 图片模式（如L=灰度，RGB=三通道，RGBA=四通道）
            
            # 计算通道数
            if mode == 'L':
                channels = 1
            elif mode == 'RGB':
                channels = 3
            elif mode == 'RGBA':
                channels = 4
            elif mode == 'CMYK':
                channels = 4
            else:
                channels = len(img.getbands())  # 通用方式获取通道数
            
            # 转换为numpy数组，方便计算像素值
            img_array = np.array(img)
            # 单张图片的像素最小值、最大值
            img_min = np.min(img_array)
            img_max = np.max(img_array)
            # 提取该图片的所有唯一像素值
            img_unique_pixels = set(img_array.flatten())
            
            return {
                'name': img_name,
                'width': width,
                'height': height,
                'channels': channels,
                'mode': mode,
                'pixel_min': img_min,
                'pixel_max': img_max,
                'unique_pixels': img_unique_pixels
            }
    except UnidentifiedImageError:
        print(f"❌ 文件 {os.path.basename(image_path)} 不是有效图片，跳过")
        return None
    except PermissionError:
        print(f"❌ 没有权限访问文件 {os.path.basename(image_path)}，跳过")
        return None
    except Exception as e:
        print(f"❌ 处理图片 {os.path.basename(image_path)} 时出错：{str(e)}，跳过")
        return None

def main():
    global global_min_pixel, global_max_pixel, all_pixel_values
    
    # 检查目标文件夹是否存在
    if not os.path.exists(TARGET_FOLDER):
        print(f"❌ 目标文件夹不存在：{TARGET_FOLDER}")
        return
    
    # 遍历文件夹中的所有文件
    file_list = os.listdir(TARGET_FOLDER)
    if not file_list:
        print(f"⚠️  目标文件夹 {TARGET_FOLDER} 中没有任何文件")
        return
    
    print(f"📁 开始处理文件夹：{TARGET_FOLDER}")
    print(f"🔍 共发现 {len(file_list)} 个文件，正在筛选图片文件...\n")
    
    # 统计有效图片数量
    valid_image_count = 0
    
    # 遍历每个文件
    for filename in file_list:
        # 跳过隐藏文件（以.开头）
        if filename.startswith('.'):
            continue
        
        # 拼接完整路径
        file_path = os.path.join(TARGET_FOLDER, filename)
        
        # 只处理文件（跳过子文件夹）
        if not os.path.isfile(file_path):
            continue
        
        # 筛选支持的图片格式（不区分大小写）
        if not filename.lower().endswith(SUPPORTED_FORMATS):
            continue
        
        # 获取图片信息
        img_info = get_image_info(file_path)
        if img_info is None:
            continue
        
        valid_image_count += 1
        
        # 打印单张图片的信息
        print(f"=== 图片 {valid_image_count}：{img_info['name']} ===")
        print(f"尺寸（宽×高）：{img_info['width']} × {img_info['height']} 像素")
        print(f"通道数：{img_info['channels']}（模式：{img_info['mode']}）")
        print(f"像素值范围：{img_info['pixel_min']} ~ {img_info['pixel_max']}")
        print(f"该图片唯一像素值数量：{len(img_info['unique_pixels'])}\n")
        
        # 更新全局像素值统计
        all_pixel_values.update(img_info['unique_pixels'])
        # 更新全局最小/最大像素值
        if global_min_pixel is None or img_info['pixel_min'] < global_min_pixel:
            global_min_pixel = img_info['pixel_min']
        if global_max_pixel is None or img_info['pixel_max'] > global_max_pixel:
            global_max_pixel = img_info['pixel_max']
    
    # 打印汇总信息
    print("="*50)
    print("📊 所有图片汇总信息")
    print("="*50)
    print(f"有效图片数量：{valid_image_count}")
    if valid_image_count > 0:
        print(f"全局像素值范围：{global_min_pixel} ~ {global_max_pixel}")
        print(f"所有出现过的唯一像素值总数：{len(all_pixel_values)}")
        # 可选：打印所有像素值（如果数量少的话，数量多建议注释）
        # print(f"所有像素值：{sorted(list(all_pixel_values))}")
    else:
        print("⚠️  未找到任何有效图片")

if __name__ == "__main__":
    main()