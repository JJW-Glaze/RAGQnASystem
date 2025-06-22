import os
from PIL import Image

def convert_webp_to_jpg(directory):
    # 确保目录存在
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在")
        return
    
    # 获取目录中的所有文件
    files = os.listdir(directory)
    
    # 计数器
    converted = 0
    failed = 0
    
    # 遍历所有文件
    for file in files:
        if file.endswith('.webp'):
            try:
                # 构建完整的文件路径
                webp_path = os.path.join(directory, file)
                jpg_path = os.path.join(directory, file.replace('.webp', '.jpg'))
                
                # 打开webp图片
                with Image.open(webp_path) as img:
                    # 转换为RGB模式（以防有透明通道）
                    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                        img = img.convert('RGB')
                    
                    # 保存为jpg
                    img.save(jpg_path, 'JPEG', quality=95)
                
                # 删除原始webp文件
                os.remove(webp_path)
                converted += 1
                print(f"已转换: {file}")
                
            except Exception as e:
                failed += 1
                print(f"转换失败 {file}: {str(e)}")
    
    print(f"\n转换完成！")
    print(f"成功转换: {converted} 个文件")
    print(f"转换失败: {failed} 个文件")

if __name__ == "__main__":
    # 指定海报目录
    posters_dir = "data/posters"
    convert_webp_to_jpg(posters_dir) 